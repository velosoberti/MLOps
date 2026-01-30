"""API constructor module for Flask prediction service.

This module provides the ModelManager class for loading and managing ML models
from MLflow, and the InputValidator class for validating prediction requests.
"""

import logging
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar

import pandas as pd

import mlflow
from config.settings import Settings, settings

# Flask imports - no longer need workaround since api/ folder doesn't conflict
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(settings.log_file)],
)
logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


class ModelManager:
    """Manager for MLflow models with validation and caching.

    Handles loading, caching, and reloading of ML models from MLflow.
    Provides prediction functionality with feature validation.

    Attributes:
        model: The loaded sklearn model.
        model_version: Version number of the loaded model.
        feature_names: List of feature names expected by the model.
        model_loaded_at: Timestamp when the model was loaded.
    """

    def __init__(
        self, tracking_uri: str | None = None, experiment_id: str | None = None, model_name: str | None = None
    ) -> None:
        """Initialize ModelManager with configuration.

        Args:
            tracking_uri: MLflow tracking server URI. Defaults to settings value.
            experiment_id: MLflow experiment ID. Defaults to settings value.
            model_name: Name of the registered model. Defaults to settings value.
        """
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.experiment_id = experiment_id or settings.mlflow_experiment_id
        self.model_name = model_name or settings.model_name

        self.model: Any | None = None
        self.model_version: int | None = None
        self.feature_names: list[str] | None = None
        self.model_loaded_at: datetime | None = None

        self.load_model()

    def load_model(self) -> None:
        """Load model from MLflow with error handling.

        Raises:
            ValueError: If no model versions are found.
            Exception: If model loading fails.
        """
        try:
            logger.info("🔄 Carregando modelo do MLflow...")

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_id=self.experiment_id)

            client = mlflow.client.MlflowClient()
            versions = client.get_latest_versions(self.model_name)

            if not versions:
                raise ValueError(f"Nenhuma versão encontrada para o modelo '{self.model_name}'")

            self.model_version = max([int(v.version) for v in versions])
            model_uri = f"models:/{self.model_name}/{self.model_version}"

            self.model = mlflow.sklearn.load_model(model_uri)
            self.feature_names = list(self.model.feature_names_in_)
            self.model_loaded_at = datetime.now()

            logger.info(f"Modelo carregado com sucesso: {self.model_name} v{self.model_version}")
            logger.info(f"Features esperadas: {self.feature_names}")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            logger.error(traceback.format_exc())
            raise

    def reload_model(self) -> dict[str, Any]:
        """Reload model and return version information.

        Returns:
            Dictionary with previous version, current version, and reload timestamp.
        """
        logger.info("🔄 Recarregando modelo...")
        old_version = self.model_version
        self.load_model()
        return {
            "previous_version": old_version,
            "current_version": self.model_version,
            "reloaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None,
        }

    def predict(self, df: pd.DataFrame) -> float:
        """Make prediction with feature validation.

        Args:
            df: DataFrame with features for prediction.

        Returns:
            Probability of positive class as float.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Modelo não carregado")

        if self.feature_names is None:
            raise RuntimeError("Feature names não disponíveis")

        X = df[self.feature_names]
        pred = self.model.predict_proba(X)[:, 1]
        return float(pred[0])

    def is_healthy(self) -> dict[str, Any]:
        """Check model health status.

        Returns:
            Dictionary with model health information.
        """
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "features_count": len(self.feature_names) if self.feature_names else 0,
            "loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None,
        }


class InputValidator:
    """Validator for prediction input data.

    Provides static methods for validating prediction request data
    against expected feature requirements.
    """

    @staticmethod
    def validate_prediction_input(data: dict[str, Any] | None, required_features: list[str]) -> tuple[bool, str | None]:
        """Validate prediction input data.

        Args:
            data: Input data dictionary to validate.
            required_features: List of required feature names.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if not data:
            return False, "Request body está vazio"

        if not isinstance(data, dict):
            return False, "Request body deve ser um objeto JSON"

        missing_features = set(required_features) - set(data.keys())
        if missing_features:
            return False, f"Features faltantes: {sorted(missing_features)}"

        extra_features = set(data.keys()) - set(required_features)
        if extra_features:
            logger.warning(f"Features extras ignoradas: {sorted(extra_features)}")

        for feature in required_features:
            value = data.get(feature)
            if value is None:
                return False, f"Feature '{feature}' não pode ser null"

            if not isinstance(value, (int, float)):
                return False, f"Feature '{feature}' deve ser numérica (recebido: {type(value).__name__})"

        return True, None


def handle_errors(f: F) -> F:
    """Decorator for centralized error handling.

    Catches and formats exceptions into appropriate HTTP responses.

    Args:
        f: Function to wrap.

    Returns:
        Wrapped function with error handling.
    """

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Erro de validação: {e}")
            return jsonify(
                {"error": "Validation Error", "message": str(e), "timestamp": datetime.now().isoformat()}
            ), 400
        except RuntimeError as e:
            logger.error(f"Erro de runtime: {e}")
            return jsonify({"error": "Runtime Error", "message": str(e), "timestamp": datetime.now().isoformat()}), 500
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            logger.error(traceback.format_exc())
            return jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "Ocorreu um erro inesperado. Verifique os logs.",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 500

    return decorated_function  # type: ignore[return-value]


def log_request(f: F) -> F:
    """Decorator for request logging.

    Logs incoming requests and response times.

    Args:
        f: Function to wrap.

    Returns:
        Wrapped function with request logging.
    """

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        start_time = datetime.now()
        logger.info(f"{request.method} {request.path} - IP: {request.remote_addr}")

        try:
            response = f(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()

            status = response[1] if isinstance(response, tuple) else 200
            logger.info(f"{request.method} {request.path} - Status: {status} - Tempo: {elapsed:.3f}s")

            return response
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"{request.method} {request.path} - Erro após {elapsed:.3f}s: {e}")
            raise

    return decorated_function  # type: ignore[return-value]


def create_app(config: Settings | None = None) -> Any:
    """Application factory for Flask app.

    Creates and configures a Flask application with the provided settings.

    Args:
        config: Optional settings override for testing.

    Returns:
        Configured Flask application.
    """
    cfg = config or settings
    app = Flask(__name__)

    # Store config in app for access in routes
    app.config["ML_SETTINGS"] = cfg

    return app
