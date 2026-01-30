"""Shared test fixtures for the ML pipeline test suite."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config.settings import Settings


@pytest.fixture
def sample_diabetes_data() -> pd.DataFrame:
    """Sample diabetes dataset for testing.

    Returns a small DataFrame with the same structure as the real
    diabetes dataset, suitable for unit testing ETL functions.
    """
    return pd.DataFrame(
        {
            "Pregnancies": [6, 1, 8, 1, 0],
            "Glucose": [148, 85, 183, 89, 137],
            "BloodPressure": [72, 66, 64, 66, 40],
            "SkinThickness": [35, 29, 0, 23, 35],
            "Insulin": [0, 0, 0, 94, 168],
            "BMI": [33.6, 26.6, 23.3, 28.1, 43.1],
            "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288],
            "Age": [50, 31, 32, 21, 33],
            "Outcome": [1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Test configuration with temporary paths.

    Creates a Settings instance with paths pointing to temporary
    directories for isolated testing.
    """
    return Settings(
        mlflow_tracking_uri="http://localhost:5000/",
        data_base_path=str(tmp_path / "data"),
        artifacts_path=str(tmp_path / "artifacts"),
        input_file=str(tmp_path / "data" / "diabetes.csv"),
        output_dir=str(tmp_path / "output"),
    )


class MockModelVersion:
    """Mock MLflow model version object."""

    def __init__(self, version: str = "1", name: str = "diabete_model") -> None:
        self.version = version
        self.name = name
        self.current_stage = "Production"
        self.creation_timestamp = int(datetime.now().timestamp() * 1000)
        self.last_updated_timestamp = int(datetime.now().timestamp() * 1000)
        self.run_id = "mock_run_id_12345"
        self.source = f"models:/{name}/{version}"
        self.status = "READY"


@pytest.fixture
def mock_mlflow_client() -> MagicMock:
    """Mock MLflow client for testing without MLflow server.

    Provides a mock MlflowClient that simulates:
    - get_latest_versions: Returns mock model versions
    - get_model_version: Returns a specific model version
    - search_model_versions: Returns list of model versions
    - get_run: Returns mock run information

    This fixture allows testing MLflow-dependent code without
    requiring a running MLflow server.
    """
    client = MagicMock()

    # Mock get_latest_versions - returns list of model versions
    mock_versions = [MockModelVersion(version="1"), MockModelVersion(version="2")]
    client.get_latest_versions.return_value = mock_versions

    # Mock get_model_version - returns a single version
    client.get_model_version.return_value = MockModelVersion(version="2")

    # Mock search_model_versions - returns list of versions
    client.search_model_versions.return_value = mock_versions

    # Mock get_run - returns run info
    mock_run = MagicMock()
    mock_run.info.run_id = "mock_run_id_12345"
    mock_run.info.experiment_id = "467326610704772702"
    mock_run.info.status = "FINISHED"
    mock_run.info.start_time = int(datetime.now().timestamp() * 1000)
    mock_run.info.end_time = int(datetime.now().timestamp() * 1000)
    mock_run.data.params = {"penalty": "l2", "solver": "lbfgs", "max_iter": "100"}
    mock_run.data.metrics = {"acc_train": 0.85, "acc_test": 0.82}
    mock_run.data.tags = {"model_type": "classification"}
    client.get_run.return_value = mock_run

    # Mock list_experiments
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "467326610704772702"
    mock_experiment.name = "diabetes_experiment"
    client.list_experiments.return_value = [mock_experiment]

    return client
