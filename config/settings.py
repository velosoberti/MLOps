"""Centralized configuration management using Pydantic settings.

All configuration values can be overridden via environment variables with the ML_ prefix.
For example, ML_MLFLOW_TRACKING_URI will override mlflow_tracking_uri.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration schema.

    Configuration values are loaded from environment variables with the ML_ prefix.
    Required fields must be set via environment variables or will raise validation errors.
    Optional fields have sensible defaults for development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ML_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(default="http://127.0.0.1:5000/", description="MLflow tracking server URI")
    mlflow_experiment_id: str = Field(default="", description="MLflow experiment ID for tracking runs")

    # Feast Configuration
    feast_repo_path: str = Field(default="", description="Path to Feast feature repository")
    feast_feature_view: str = Field(default="predictors_df_feature_view", description="Name of the Feast feature view")
    feast_dataset_name: str = Field(default="my_training_dataset", description="Name of the saved Feast dataset")

    # Data Paths
    data_base_path: str = Field(default="", description="Base path for data files")
    artifacts_path: str = Field(default="", description="Path for output artifacts")
    input_file: str = Field(default="", description="Path to input CSV file (e.g., diabetes.csv)")
    output_dir: str = Field(default="", description="Directory for output files")

    # Prediction Configuration
    predictions_output_dir: str = Field(default="", description="Directory for prediction output files")
    historical_predictions_file: str = Field(default="", description="Path to historical predictions parquet file")
    n_patients: int = Field(default=50, description="Number of patients to process in prediction pipeline")

    # Model Configuration
    model_name: str = Field(default="diabete_model", description="Name of the registered model in MLflow")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="Host address for Flask API")
    api_port: int = Field(default=5005, description="Port for Flask API")
    api_debug: bool = Field(default=False, description="Enable Flask debug mode")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_file: str = Field(default="api.log", description="Path to log file")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level '{v}'. Must be one of: {', '.join(sorted(valid_levels))}")
        return upper_v

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, v: int) -> int:
        """Validate that API port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"API port must be between 1 and 65535, got {v}")
        return v

    @field_validator("n_patients")
    @classmethod
    def validate_n_patients(cls, v: int) -> int:
        """Validate that n_patients is positive."""
        if v < 1:
            raise ValueError(f"n_patients must be at least 1, got {v}")
        return v

    def get_input_file_path(self) -> str:
        """Get the full path to the input file.

        Returns input_file if set, otherwise constructs from data_base_path.
        """
        if self.input_file:
            return self.input_file
        if self.data_base_path:
            return str(Path(self.data_base_path) / "diabetes.csv")
        return "data/diabetes.csv"

    def get_artifacts_path(self) -> str:
        """Get the full path to artifacts directory.

        Returns artifacts_path if set, otherwise constructs from data_base_path.
        """
        if self.artifacts_path:
            return self.artifacts_path
        if self.data_base_path:
            return str(Path(self.data_base_path) / "artifacts")
        return "data/artifacts"

    def get_predictions_output_dir(self) -> str:
        """Get the full path to predictions output directory."""
        if self.predictions_output_dir:
            return self.predictions_output_dir
        return str(Path(self.get_artifacts_path()) / "predictions")

    def get_historical_predictions_file(self) -> str:
        """Get the full path to historical predictions file."""
        if self.historical_predictions_file:
            return self.historical_predictions_file
        return str(Path(self.get_predictions_output_dir()) / "predictions_history.parquet")


# Global settings instance - loaded once at module import
settings = Settings()
