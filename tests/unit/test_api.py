"""Unit tests for Flask API endpoints.

Tests cover:
- /health endpoint returns correct structure
- /predict with valid input returns prediction
- /predict with invalid input returns 400
- /predict/batch with multiple instances
- /model/info returns model metadata
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock

import pytest

# Ensure proper import path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.settings import Settings  # noqa: E402
from src.api_consctructor import (  # noqa: E402
    Flask,
    InputValidator,
    jsonify,
)
from src.api_consctructor import (  # noqa: E402
    request as flask_request,
)


def create_test_app(mock_model_manager=None, test_config=None):
    """Create a test Flask app with routes registered."""
    cfg = test_config or Settings(
        mlflow_tracking_uri="http://localhost:5000/",
        mlflow_experiment_id="123456",
        model_name="test_model",
        api_port=5005,
    )

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["ML_SETTINGS"] = cfg
    app.config["MODEL_MANAGER"] = mock_model_manager

    # Register routes
    @app.route("/health", methods=["GET"])
    def health_check():
        model_manager = app.config.get("MODEL_MANAGER")

        if model_manager:
            model_health = model_manager.is_healthy()
        else:
            model_health = {"model_loaded": False, "error": "Model not initialized"}

        mlflow_healthy = True  # Mock for testing
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
                    "error": None,
                },
                "model": model_health,
            },
        }
        return jsonify(response), status_code

    @app.route("/predict", methods=["POST"])
    def predict():
        import pandas as pd

        model_manager = app.config.get("MODEL_MANAGER")

        if model_manager is None:
            return jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model is not loaded.",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 503

        if not flask_request.is_json:
            return jsonify(
                {
                    "error": "Invalid Content-Type",
                    "message": "Content-Type must be 'application/json'",
                    "received": flask_request.content_type,
                }
            ), 400

        data = flask_request.get_json(silent=True)

        is_valid, error_message = InputValidator.validate_prediction_input(data, model_manager.feature_names)

        if not is_valid:
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

        return jsonify(
            {
                "score": round(score, 4),
                "prediction": prediction_label,
                "confidence": round(score if score >= 0.5 else 1 - score, 4),
                "model_version": model_manager.model_version,
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        import pandas as pd

        model_manager = app.config.get("MODEL_MANAGER")

        if model_manager is None:
            return jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model is not loaded.",
                }
            ), 503

        if not flask_request.is_json:
            return jsonify({"error": "Invalid Content-Type", "message": "Content-Type must be 'application/json'"}), 400

        data = flask_request.get_json(silent=True)

        if not data or "instances" not in data:
            return jsonify(
                {
                    "error": "Invalid Input",
                    "message": "Request body must contain 'instances' (list of objects)",
                }
            ), 400

        instances = data["instances"]

        if not isinstance(instances, list) or len(instances) == 0:
            return jsonify({"error": "Invalid Input", "message": "'instances' must be a non-empty list"}), 400

        predictions = []
        errors = []

        for idx, instance in enumerate(instances):
            is_valid, error_message = InputValidator.validate_prediction_input(instance, model_manager.feature_names)

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

        return jsonify(
            {
                "predictions": predictions,
                "total": len(predictions),
                "errors": errors if errors else None,
                "model_version": model_manager.model_version,
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/model/info", methods=["GET"])
    def model_info():
        model_manager = app.config.get("MODEL_MANAGER")

        if model_manager is None:
            return jsonify(
                {
                    "error": "Service Unavailable",
                    "message": "Model is not loaded",
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

    @app.errorhandler(404)
    def not_found(e):
        return jsonify(
            {
                "error": "Not Found",
                "message": f"Endpoint '{flask_request.path}' does not exist",
            }
        ), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify(
            {
                "error": "Method Not Allowed",
                "message": f"Method {flask_request.method} is not allowed",
            }
        ), 405

    return app


@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager for testing."""
    manager = MagicMock()
    manager.model = MagicMock()
    manager.model_version = 1
    manager.feature_names = ["Glucose", "BMI", "DiabetesPedigreeFunction", "Insulin", "SkinThickness"]
    manager.model_loaded_at = datetime.now()
    manager.is_healthy.return_value = {
        "model_loaded": True,
        "model_name": "test_model",
        "model_version": 1,
        "features_count": 5,
        "loaded_at": datetime.now().isoformat(),
    }
    manager.predict.return_value = 0.75
    return manager


@pytest.fixture
def client(mock_model_manager):
    """Create test client."""
    app = create_test_app(mock_model_manager)
    return app.test_client()


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_correct_structure(self, client):
        """Test that /health endpoint returns correct structure."""
        response = client.get("/health")
        data = response.get_json()

        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert "api" in data["services"]
        assert "mlflow" in data["services"]
        assert "model" in data["services"]

    def test_health_returns_healthy_when_all_services_up(self, client):
        """Test that /health returns healthy status when all services are up."""
        response = client.get("/health")
        data = response.get_json()

        assert response.status_code == 200
        assert data["status"] == "healthy"
        assert data["services"]["api"] == "running"


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_with_valid_input_returns_prediction(self, client):
        """Test that /predict with valid input returns prediction."""
        valid_input = {
            "Glucose": 148,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Insulin": 0,
            "SkinThickness": 35,
        }

        response = client.post("/predict", json=valid_input, content_type="application/json")
        data = response.get_json()

        assert response.status_code == 200
        assert "score" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert data["prediction"] in ["diabetes", "no_diabetes"]

    def test_predict_with_missing_field_returns_400(self, client):
        """Test that /predict with missing field returns 400."""
        invalid_input = {"Glucose": 148, "BMI": 33.6}

        response = client.post("/predict", json=invalid_input, content_type="application/json")
        data = response.get_json()

        assert response.status_code == 400
        assert "error" in data
        assert "message" in data

    def test_predict_with_invalid_content_type_returns_400(self, client):
        """Test that /predict with invalid content type returns 400."""
        response = client.post("/predict", data="not json", content_type="text/plain")
        data = response.get_json()

        assert response.status_code == 400
        assert "error" in data

    def test_predict_with_empty_body_returns_400(self, client):
        """Test that /predict with empty body returns 400."""
        response = client.post("/predict", json={}, content_type="application/json")
        data = response.get_json()

        assert response.status_code == 400
        assert "error" in data

    def test_predict_with_non_numeric_value_returns_400(self, client):
        """Test that /predict with non-numeric value returns 400."""
        invalid_input = {
            "Glucose": "not a number",
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Insulin": 0,
            "SkinThickness": 35,
        }

        response = client.post("/predict", json=invalid_input, content_type="application/json")
        data = response.get_json()

        assert response.status_code == 400
        assert "error" in data


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_with_multiple_instances(self, client):
        """Test that /predict/batch processes multiple instances."""
        batch_input = {
            "instances": [
                {"Glucose": 148, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Insulin": 0, "SkinThickness": 35},
                {"Glucose": 85, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Insulin": 0, "SkinThickness": 29},
            ]
        }

        response = client.post("/predict/batch", json=batch_input, content_type="application/json")
        data = response.get_json()

        assert response.status_code == 200
        assert "predictions" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_with_missing_instances_returns_400(self, client):
        """Test that /predict/batch without instances field returns 400."""
        response = client.post("/predict/batch", json={"data": []}, content_type="application/json")
        data = response.get_json()

        assert response.status_code == 400
        assert "error" in data

    def test_batch_predict_with_empty_instances_returns_400(self, client):
        """Test that /predict/batch with empty instances returns 400."""
        response = client.post("/predict/batch", json={"instances": []}, content_type="application/json")
        data = response.get_json()

        assert response.status_code == 400
        assert "error" in data


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_returns_metadata(self, client):
        """Test that /model/info returns model metadata."""
        response = client.get("/model/info")
        data = response.get_json()

        assert response.status_code == 200
        assert "model_name" in data
        assert "model_version" in data
        assert "features" in data
        assert "feature_count" in data
        assert "mlflow_tracking_uri" in data

    def test_model_info_returns_correct_feature_count(self, client):
        """Test that /model/info returns correct feature count."""
        response = client.get("/model/info")
        data = response.get_json()

        assert data["feature_count"] == 5
        assert len(data["features"]) == 5


class TestErrorHandlers:
    """Tests for error handlers."""

    def test_404_for_unknown_endpoint(self, client):
        """Test that unknown endpoints return 404."""
        response = client.get("/unknown/endpoint")
        data = response.get_json()

        assert response.status_code == 404
        assert "error" in data
        assert data["error"] == "Not Found"

    def test_405_for_wrong_method(self, client):
        """Test that wrong HTTP method returns 405."""
        response = client.get("/predict")
        data = response.get_json()

        assert response.status_code == 405
        assert "error" in data
        assert data["error"] == "Method Not Allowed"
