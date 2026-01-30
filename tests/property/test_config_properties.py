"""Property-based tests for configuration loading.

# Feature: ml-pipeline-refactor-cicd, Property 1: Configuration Loading with Fallbacks

**Validates: Requirements 1.1**

This module tests that the Configuration_Manager correctly loads settings from
environment variables with fallback to default values.
"""

import os

from hypothesis import given, settings
from hypothesis import strategies as st

from config.settings import Settings

# Strategy for generating valid string configuration values
valid_string_values = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")), min_size=1, max_size=100
).filter(lambda x: x.strip() != "")

# Strategy for valid URIs
valid_uri_values = st.from_regex(r"https?://[a-z0-9.-]+:[0-9]{1,5}/", fullmatch=True)

# Strategy for valid port numbers
valid_port_values = st.integers(min_value=1, max_value=65535)

# Strategy for valid log levels
valid_log_levels = st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

# Strategy for positive integers
positive_integers = st.integers(min_value=1, max_value=10000)


class TestConfigurationLoadingWithFallbacks:
    """Property 1: Configuration Loading with Fallbacks.

    *For any* valid combination of environment variables, the Configuration_Manager
    SHALL correctly load values from environment variables when present, and fall
    back to default values when environment variables are not set.
    """

    @given(uri=valid_uri_values)
    @settings(max_examples=100)
    def test_env_var_overrides_default_for_mlflow_uri(self, uri: str) -> None:
        """Test that ML_MLFLOW_TRACKING_URI env var overrides the default value.

        **Validates: Requirements 1.1**
        """
        # Set environment variable
        os.environ["ML_MLFLOW_TRACKING_URI"] = uri

        try:
            # Create new settings instance (bypasses cached global)
            config = Settings()

            # Assert env var value is used
            assert config.mlflow_tracking_uri == uri, (
                f"Expected mlflow_tracking_uri to be '{uri}' from env var, but got '{config.mlflow_tracking_uri}'"
            )
        finally:
            # Clean up
            del os.environ["ML_MLFLOW_TRACKING_URI"]

    @given(model_name=valid_string_values)
    @settings(max_examples=100)
    def test_env_var_overrides_default_for_model_name(self, model_name: str) -> None:
        """Test that ML_MODEL_NAME env var overrides the default value.

        **Validates: Requirements 1.1**
        """
        os.environ["ML_MODEL_NAME"] = model_name

        try:
            config = Settings()
            assert config.model_name == model_name, (
                f"Expected model_name to be '{model_name}' from env var, but got '{config.model_name}'"
            )
        finally:
            del os.environ["ML_MODEL_NAME"]

    @given(port=valid_port_values)
    @settings(max_examples=100)
    def test_env_var_overrides_default_for_api_port(self, port: int) -> None:
        """Test that ML_API_PORT env var overrides the default value.

        **Validates: Requirements 1.1**
        """
        os.environ["ML_API_PORT"] = str(port)

        try:
            config = Settings()
            assert config.api_port == port, f"Expected api_port to be {port} from env var, but got {config.api_port}"
        finally:
            del os.environ["ML_API_PORT"]

    @given(log_level=valid_log_levels)
    @settings(max_examples=100)
    def test_env_var_overrides_default_for_log_level(self, log_level: str) -> None:
        """Test that ML_LOG_LEVEL env var overrides the default value.

        **Validates: Requirements 1.1**
        """
        os.environ["ML_LOG_LEVEL"] = log_level

        try:
            config = Settings()
            assert config.log_level == log_level.upper(), (
                f"Expected log_level to be '{log_level.upper()}' from env var, but got '{config.log_level}'"
            )
        finally:
            del os.environ["ML_LOG_LEVEL"]

    @given(n_patients=positive_integers)
    @settings(max_examples=100)
    def test_env_var_overrides_default_for_n_patients(self, n_patients: int) -> None:
        """Test that ML_N_PATIENTS env var overrides the default value.

        **Validates: Requirements 1.1**
        """
        os.environ["ML_N_PATIENTS"] = str(n_patients)

        try:
            config = Settings()
            assert config.n_patients == n_patients, (
                f"Expected n_patients to be {n_patients} from env var, but got {config.n_patients}"
            )
        finally:
            del os.environ["ML_N_PATIENTS"]

    def test_missing_optional_fields_use_defaults(self) -> None:
        """Test that missing optional fields fall back to default values.

        **Validates: Requirements 1.1**
        """
        # Clear any env vars that might be set
        env_vars_to_clear = [
            "ML_MLFLOW_TRACKING_URI",
            "ML_MODEL_NAME",
            "ML_API_PORT",
            "ML_API_HOST",
            "ML_API_DEBUG",
            "ML_LOG_LEVEL",
            "ML_N_PATIENTS",
            "ML_FEAST_FEATURE_VIEW",
        ]

        saved_values: dict[str, str] = {}
        for var in env_vars_to_clear:
            if var in os.environ:
                saved_values[var] = os.environ[var]
                del os.environ[var]

        try:
            config = Settings()

            # Verify defaults are used
            assert config.mlflow_tracking_uri == "http://127.0.0.1:5000/"
            assert config.model_name == "diabete_model"
            assert config.api_port == 5005
            assert config.api_host == "0.0.0.0"
            assert config.api_debug is False
            assert config.log_level == "INFO"
            assert config.n_patients == 50
            assert config.feast_feature_view == "predictors_df_feature_view"
        finally:
            # Restore saved values
            for var, value in saved_values.items():
                os.environ[var] = value

    @given(
        uri=valid_uri_values,
        model_name=valid_string_values,
        port=valid_port_values,
    )
    @settings(max_examples=100)
    def test_multiple_env_vars_override_correctly(self, uri: str, model_name: str, port: int) -> None:
        """Test that multiple env vars can be set simultaneously.

        **Validates: Requirements 1.1**
        """
        os.environ["ML_MLFLOW_TRACKING_URI"] = uri
        os.environ["ML_MODEL_NAME"] = model_name
        os.environ["ML_API_PORT"] = str(port)

        try:
            config = Settings()

            assert config.mlflow_tracking_uri == uri
            assert config.model_name == model_name
            assert config.api_port == port
        finally:
            del os.environ["ML_MLFLOW_TRACKING_URI"]
            del os.environ["ML_MODEL_NAME"]
            del os.environ["ML_API_PORT"]

    @given(debug_value=st.booleans())
    @settings(max_examples=100)
    def test_boolean_env_var_parsing(self, debug_value: bool) -> None:
        """Test that boolean env vars are parsed correctly.

        **Validates: Requirements 1.1**
        """
        # Pydantic accepts various boolean representations
        str_value = "true" if debug_value else "false"
        os.environ["ML_API_DEBUG"] = str_value

        try:
            config = Settings()
            assert config.api_debug == debug_value, (
                f"Expected api_debug to be {debug_value} from env var '{str_value}', but got {config.api_debug}"
            )
        finally:
            del os.environ["ML_API_DEBUG"]
