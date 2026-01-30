"""Property-based tests for API configuration injection.

# Feature: ml-pipeline-refactor-cicd, Property 2: API Configuration Injection

**Validates: Requirements 6.1**

This module tests that the Flask API factory correctly accepts configuration
injection and uses those configuration values for MLflow tracking URI,
model name, and API settings.
"""

import os
import sys

# Ensure proper import path - add project root
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


from hypothesis import given  # noqa: E402
from hypothesis import settings as hypothesis_settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from config.settings import Settings  # noqa: E402

# Import create_app from src.api_consctructor which has the correct Flask import
from src.api_consctructor import create_app  # noqa: E402

# Strategy for valid URIs
valid_uri_values = st.from_regex(r"https?://[a-z0-9.-]+:[0-9]{1,5}/", fullmatch=True)

# Strategy for valid model names (alphanumeric with underscores)
valid_model_names = st.from_regex(r"[a-z][a-z0-9_]{2,30}", fullmatch=True)

# Strategy for valid port numbers
valid_port_values = st.integers(min_value=1024, max_value=65535)

# Strategy for valid experiment IDs
valid_experiment_ids = st.from_regex(r"[0-9]{1,20}", fullmatch=True)


class TestAPIConfigurationInjection:
    """Property 2: API Configuration Injection.

    *For any* valid Settings configuration provided to the Flask API factory,
    the created application SHALL use those configuration values for MLflow
    tracking URI, model name, and API settings.
    """

    @given(tracking_uri=valid_uri_values)
    @hypothesis_settings(max_examples=100)
    def test_app_uses_injected_mlflow_tracking_uri(self, tracking_uri: str) -> None:
        """Test that create_app() uses the injected MLflow tracking URI.

        **Validates: Requirements 6.1**
        """
        config = Settings(mlflow_tracking_uri=tracking_uri, mlflow_experiment_id="123456", model_name="test_model")

        app = create_app(config)

        stored_config = app.config.get("ML_SETTINGS")
        assert stored_config is not None, "ML_SETTINGS should be stored in app config"
        assert stored_config.mlflow_tracking_uri == tracking_uri, (
            f"Expected mlflow_tracking_uri to be '{tracking_uri}', but got '{stored_config.mlflow_tracking_uri}'"
        )

    @given(model_name=valid_model_names)
    @hypothesis_settings(max_examples=100)
    def test_app_uses_injected_model_name(self, model_name: str) -> None:
        """Test that create_app() uses the injected model name.

        **Validates: Requirements 6.1**
        """
        config = Settings(
            mlflow_tracking_uri="http://localhost:5000/", mlflow_experiment_id="123456", model_name=model_name
        )

        app = create_app(config)

        stored_config = app.config.get("ML_SETTINGS")
        assert stored_config is not None
        assert stored_config.model_name == model_name, (
            f"Expected model_name to be '{model_name}', but got '{stored_config.model_name}'"
        )

    @given(experiment_id=valid_experiment_ids)
    @hypothesis_settings(max_examples=100)
    def test_app_uses_injected_experiment_id(self, experiment_id: str) -> None:
        """Test that create_app() uses the injected experiment ID.

        **Validates: Requirements 6.1**
        """
        config = Settings(
            mlflow_tracking_uri="http://localhost:5000/", mlflow_experiment_id=experiment_id, model_name="test_model"
        )

        app = create_app(config)

        stored_config = app.config.get("ML_SETTINGS")
        assert stored_config is not None
        assert stored_config.mlflow_experiment_id == experiment_id, (
            f"Expected mlflow_experiment_id to be '{experiment_id}', but got '{stored_config.mlflow_experiment_id}'"
        )

    @given(tracking_uri=valid_uri_values, model_name=valid_model_names, experiment_id=valid_experiment_ids)
    @hypothesis_settings(max_examples=100)
    def test_different_configs_produce_different_app_configs(
        self, tracking_uri: str, model_name: str, experiment_id: str
    ) -> None:
        """Test that different configs produce different app configurations.

        **Validates: Requirements 6.1**
        """
        config1 = Settings(mlflow_tracking_uri=tracking_uri, mlflow_experiment_id=experiment_id, model_name=model_name)

        config2 = Settings(
            mlflow_tracking_uri="http://different:9999/", mlflow_experiment_id="999999", model_name="different_model"
        )

        app1 = create_app(config1)
        app2 = create_app(config2)

        stored_config1 = app1.config.get("ML_SETTINGS")
        stored_config2 = app2.config.get("ML_SETTINGS")

        assert (
            stored_config1.mlflow_tracking_uri != stored_config2.mlflow_tracking_uri
            or stored_config1.model_name != stored_config2.model_name
            or stored_config1.mlflow_experiment_id != stored_config2.mlflow_experiment_id
        ), "Different configs should produce different app configurations"

    def test_default_config_used_when_none_provided(self) -> None:
        """Test that default settings are used when no config is provided.

        **Validates: Requirements 6.1**
        """
        from config.settings import settings as default_settings

        app = create_app()

        stored_config = app.config.get("ML_SETTINGS")
        assert stored_config is not None
        assert stored_config.mlflow_tracking_uri == default_settings.mlflow_tracking_uri
        assert stored_config.model_name == default_settings.model_name

    @given(port=valid_port_values)
    @hypothesis_settings(max_examples=100)
    def test_app_stores_api_port_from_config(self, port: int) -> None:
        """Test that API port from config is stored in app.

        **Validates: Requirements 6.1**
        """
        config = Settings(
            mlflow_tracking_uri="http://localhost:5000/",
            mlflow_experiment_id="123456",
            model_name="test_model",
            api_port=port,
        )

        app = create_app(config)

        stored_config = app.config.get("ML_SETTINGS")
        assert stored_config is not None
        assert stored_config.api_port == port, f"Expected api_port to be {port}, but got {stored_config.api_port}"

    def test_no_hardcoded_values_override_injected_config(self) -> None:
        """Test that no hardcoded values override the injected configuration.

        **Validates: Requirements 6.1**
        """
        custom_uri = "http://custom-mlflow-server:12345/"
        custom_model = "very_custom_model_name"
        custom_experiment = "987654321"

        config = Settings(
            mlflow_tracking_uri=custom_uri, mlflow_experiment_id=custom_experiment, model_name=custom_model
        )

        app = create_app(config)

        stored_config = app.config.get("ML_SETTINGS")

        assert stored_config.mlflow_tracking_uri == custom_uri
        assert stored_config.model_name == custom_model
        assert stored_config.mlflow_experiment_id == custom_experiment
