"""
Flask API for Diabetes Prediction.

Professional API with error handling, validation, logging, and health checks.
Uses application factory pattern for testability and configuration injection.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

# Handle path for imports - add parent directory for local modules FIRST
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import Flask - the api_consctructor module handles the import correctly
# Import Pydantic models for request validation
# Use importlib to avoid conflict with Flask package
import importlib.util

import pandas as pd

import mlflow
from config.settings import Settings, settings
from src.api_consctructor import (
    Flask,
    InputValidator,
    ModelManager,
    handle_errors,
    jsonify,
    log_request,
    logger,
    request,
)

_models_path = os.path.join(os.path.dirname(__file__), "models.py")
_spec = importlib.util.spec_from_file_location("flask_models", _models_path)
if _spec is None:
    raise ImportError("Failed to load flask_models spec")
_flask_models = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
if _spec.loader is None:
    raise ImportError("Failed to get loader for flask_models")
_spec.loader.exec_module(_flask_models)  # type: ignore[union-attr]
PredictionRequest = _flask_models.PredictionRequest
BatchPredictionRequest = _flask_models.BatchPredictionRequest
validate_prediction_request = _flask_models.validate_prediction_request
validate_batch_request = _flask_models.validate_batch_request


def create_app(config: Settings | None = None) -> Flask:
    """Application factory for Flask app.

    Creates and configures a Flask application with the provided settings.
    Initializes ModelManager with injected configuration for testability.

    Args:
        config: Optional settings override for testing. If None, uses global settings.

    Returns:
        Configured Flask application.
    """
    cfg = config or settings
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Store config in app for access in routes
    app.config["ML_SETTINGS"] = cfg

    # Initialize model manager with injected config
    try:
        model_manager = ModelManager(
            tracking_uri=cfg.mlflow_tracking_uri, experiment_id=cfg.mlflow_experiment_id, model_name=cfg.model_name
        )
        app.config["MODEL_MANAGER"] = model_manager
        logger.info("API initialized successfully")
    except Exception as e:
        logger.critical(f"Critical failure initializing API: {e}")
        app.config["MODEL_MANAGER"] = None
        app.config["MODEL_INIT_ERROR"] = str(e)

    # Register routes
    _register_routes(app)
    _register_error_handlers(app)
    _register_hooks(app)

    return app


def _register_routes(app: Flask) -> None:
    """Register all API routes on the Flask app."""

    @app.route("/health", methods=["GET"])
    @log_request
    def health_check() -> tuple[Any, int]:
        """Health check endpoint."""
        cfg: Settings = app.config["ML_SETTINGS"]
        model_manager: ModelManager | None = app.config.get("MODEL_MANAGER")

        try:
            if model_manager:
                model_health = model_manager.is_healthy()
            else:
                model_health = {
                    "model_loaded": False,
                    "error": app.config.get("MODEL_INIT_ERROR", "Model not initialized"),
                }

            mlflow_healthy = False
            mlflow_error = None
            try:
                mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
                client = mlflow.client.MlflowClient()
                if cfg.mlflow_experiment_id:
                    client.get_experiment(cfg.mlflow_experiment_id)
                mlflow_healthy = True
            except Exception as e:
                mlflow_error = str(e)
                logger.error(f"MLflow not accessible: {e}")

            all_healthy = model_health.get("model_loaded", False) and mlflow_healthy
            status_code = 200 if all_healthy else 503

            response = {
                "status": "healthy" if all_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "api": "running",
                    "mlflow": {
                        "status": "connected" if mlflow_healthy else "disconnected",
                        "tracking_uri": cfg.mlflow_tracking_uri,
                        "error": mlflow_error,
                    },
                    "model": model_health,
                },
            }

            return jsonify(response), status_code

        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return jsonify({"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}), 500

    @app.route("/predict", methods=["POST"])
    @log_request
    @handle_errors
    def predict() -> tuple[Any, int]:
        """Prediction endpoint."""
        model_manager: ModelManager | None = app.config.get("MODEL_MANAGER")

        if model_manager is None:
            return jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model is not loaded. Check MLflow connectivity.",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 503

        if not request.is_json:
            return jsonify(
                {
                    "error": "Invalid Content-Type",
                    "message": "Content-Type must be 'application/json'",
                    "received": request.content_type,
                }
            ), 400

        data = request.get_json(silent=True)

        feature_names: list[str] = model_manager.feature_names or []
        is_valid, error_message = InputValidator.validate_prediction_input(data, feature_names)

        if not is_valid:
            logger.warning(f"Validation failed: {error_message}")
            return jsonify(
                {
                    "error": "Validation Error",
                    "message": error_message,
                    "expected_features": model_manager.feature_names,
                    "received_features": list(data.keys()) if data else [],
                }
            ), 400

        df = pd.DataFrame([data])
        score = model_manager.predict(df)
        prediction_label = "diabetes" if score >= 0.5 else "no_diabetes"

        response = {
            "score": round(score, 4),
            "prediction": prediction_label,
            "confidence": round(score if score >= 0.5 else 1 - score, 4),
            "model_version": model_manager.model_version,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Prediction: {prediction_label} (score: {score:.4f})")
        return jsonify(response), 200

    @app.route("/predict/batch", methods=["POST"])
    @log_request
    @handle_errors
    def predict_batch() -> tuple[Any, int]:
        """Batch prediction endpoint."""
        model_manager: ModelManager | None = app.config.get("MODEL_MANAGER")

        if model_manager is None:
            return jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model is not loaded. Check MLflow connectivity.",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 503

        if not request.is_json:
            return jsonify({"error": "Invalid Content-Type", "message": "Content-Type must be 'application/json'"}), 400

        data = request.get_json(silent=True)

        if not data or "instances" not in data:
            return jsonify(
                {
                    "error": "Invalid Input",
                    "message": "Request body must contain 'instances' (list of objects)",
                    "example": {"instances": [{"Glucose": 148, "BMI": 33.6}]},
                }
            ), 400

        instances = data["instances"]

        if not isinstance(instances, list) or len(instances) == 0:
            return jsonify({"error": "Invalid Input", "message": "'instances' must be a non-empty list"}), 400

        if len(instances) > 1000:
            return jsonify(
                {
                    "error": "Batch Too Large",
                    "message": f"Maximum 1000 instances per batch (received: {len(instances)})",
                }
            ), 400

        logger.info(f"Processing batch with {len(instances)} instances")

        predictions = []
        errors = []

        for idx, instance in enumerate(instances):
            try:
                feature_names_batch: list[str] = model_manager.feature_names or []
                is_valid, error_message = InputValidator.validate_prediction_input(instance, feature_names_batch)

                if not is_valid:
                    errors.append({"instance_index": idx, "error": error_message})
                    continue

                df = pd.DataFrame([instance])
                score = model_manager.predict(df)
                prediction_label = "diabetes" if score >= 0.5 else "no_diabetes"

                predictions.append(
                    {
                        "score": round(score, 4),
                        "prediction": prediction_label,
                        "confidence": round(score if score >= 0.5 else 1 - score, 4),
                        "instance_index": idx,
                    }
                )

            except Exception as e:
                logger.error(f"Error in instance {idx}: {e}")
                errors.append({"instance_index": idx, "error": str(e)})

        response = {
            "predictions": predictions,
            "total": len(predictions),
            "errors": errors if errors else None,
            "model_version": model_manager.model_version,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Batch completed: {len(predictions)}/{len(instances)} successes")
        return jsonify(response), 200

    @app.route("/model/info", methods=["GET"])
    @log_request
    def model_info() -> tuple[Any, int]:
        """Return detailed model information."""
        cfg: Settings = app.config["ML_SETTINGS"]
        model_manager: ModelManager | None = app.config.get("MODEL_MANAGER")

        if model_manager is None:
            return jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model is not loaded",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 503

        return jsonify(
            {
                "model_name": cfg.model_name,
                "model_version": model_manager.model_version,
                "features": model_manager.feature_names,
                "feature_count": len(model_manager.feature_names) if model_manager.feature_names else 0,
                "loaded_at": model_manager.model_loaded_at.isoformat() if model_manager.model_loaded_at else None,
                "mlflow_tracking_uri": cfg.mlflow_tracking_uri,
                "experiment_id": cfg.mlflow_experiment_id,
            }
        ), 200

    @app.route("/model/reload", methods=["POST"])
    @log_request
    @handle_errors
    def reload_model() -> tuple[Any, int]:
        """Reload model from MLflow."""
        model_manager: ModelManager | None = app.config.get("MODEL_MANAGER")

        if model_manager is None:
            return jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model manager not initialized",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 503

        reload_info = model_manager.reload_model()
        return jsonify({"message": "Model reloaded successfully", **reload_info}), 200


def _register_error_handlers(app: Flask) -> None:
    """Register error handlers on the Flask app."""

    @app.errorhandler(404)
    def not_found(e: Exception) -> tuple[Any, int]:
        logger.warning(f"Route not found: {request.path}")
        return jsonify(
            {
                "error": "Not Found",
                "message": f"Endpoint '{request.path}' does not exist",
                "available_endpoints": [
                    "GET  /health",
                    "GET  /model/info",
                    "POST /predict",
                    "POST /predict/batch",
                    "POST /model/reload",
                ],
            }
        ), 404

    @app.errorhandler(405)
    def method_not_allowed(e: Exception) -> tuple[Any, int]:
        logger.warning(f"Method not allowed: {request.method} {request.path}")
        return jsonify(
            {
                "error": "Method Not Allowed",
                "message": f"Method {request.method} is not allowed for {request.path}",
                "allowed_methods": getattr(e, "valid_methods", []),
            }
        ), 405

    @app.errorhandler(500)
    def internal_error(e: Exception) -> tuple[Any, int]:
        logger.error(f"Internal error: {e}")
        return jsonify(
            {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Check server logs.",
                "timestamp": datetime.now().isoformat(),
            }
        ), 500


def _register_hooks(app: Flask) -> None:
    """Register request hooks on the Flask app."""

    @app.before_request
    def before_request() -> None:
        request.start_time = datetime.now()

    @app.after_request
    def after_request(response: Any) -> Any:
        if hasattr(request, "start_time"):
            elapsed = (datetime.now() - request.start_time).total_seconds()
            response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response


# Create default app instance for backward compatibility
app = create_app()


if __name__ == "__main__":
    cfg = settings
    model_manager = app.config.get("MODEL_MANAGER")

    logger.info("=" * 80)
    logger.info("Starting Diabetes Prediction API")
    logger.info("=" * 80)
    logger.info(f"Server: http://{cfg.api_host}:{cfg.api_port}")
    logger.info(f"MLflow: {cfg.mlflow_tracking_uri}")
    logger.info(f"Model: {cfg.model_name}")
    if model_manager:
        logger.info(f"Model Version: {model_manager.model_version}")
        logger.info(f"Features: {len(model_manager.feature_names) if model_manager.feature_names else 0}")
    logger.info("=" * 80)

    app.run(host=cfg.api_host, port=cfg.api_port, debug=cfg.api_debug)
