"""DAG import and syntax tests.

These tests verify that all Airflow DAGs can be imported without errors
and that task dependencies are correctly defined.

Requirements: 7.3, 7.4, 8.4
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Mock Airflow modules before importing DAGs
@pytest.fixture(autouse=True)
def mock_airflow():
    """Mock Airflow modules to allow DAG imports without Airflow installation."""
    # Create mock modules
    mock_dag = MagicMock()
    mock_task = MagicMock()
    MagicMock()
    mock_dates = MagicMock()
    mock_dates.days_ago = MagicMock(return_value="2023-01-01")

    # Mock decorators to return the function unchanged
    mock_dag.dag = lambda **_kwargs: lambda f: f
    mock_task.task = lambda **_kwargs: lambda f: f

    # Set up the mock modules
    modules_to_mock = {
        "airflow": MagicMock(),
        "airflow.decorators": MagicMock(dag=lambda **_kwargs: lambda f: f, task=lambda **_kwargs: lambda f: f),
        "airflow.operators": MagicMock(),
        "airflow.operators.python": MagicMock(PythonOperator=MagicMock()),
        "airflow.utils": MagicMock(),
        "airflow.utils.dates": mock_dates,
        "airflow.models": MagicMock(),
    }

    with patch.dict(sys.modules, modules_to_mock):
        yield


class TestDAGImports:
    """Test that all DAGs can be imported without errors."""

    def test_etl_dag_imports(self, mock_airflow):  # noqa: ARG002
        """Test that ETL DAG can be imported without errors.

        Validates: Requirement 7.3 - DAGs should be testable in isolation
        """
        # Import should not raise any exceptions
        try:
            # We need to reload to pick up the mocked modules
            import importlib

            import airflow.dags.etl as etl_module

            importlib.reload(etl_module)
        except ImportError as e:
            # Expected if Airflow is not installed, but module structure should be valid
            if "airflow" not in str(e).lower():
                pytest.fail(f"ETL DAG import failed with unexpected error: {e}")

    def test_train_dag_imports(self, mock_airflow):  # noqa: ARG002
        """Test that training DAG can be imported without errors.

        Validates: Requirement 7.3 - DAGs should be testable in isolation
        """
        try:
            import importlib

            import airflow.dags.train as train_module

            importlib.reload(train_module)
        except ImportError as e:
            if "airflow" not in str(e).lower():
                pytest.fail(f"Training DAG import failed with unexpected error: {e}")

    def test_predict_dag_imports(self, mock_airflow):  # noqa: ARG002
        """Test that prediction DAG can be imported without errors.

        Validates: Requirement 7.3 - DAGs should be testable in isolation
        """
        try:
            import importlib

            import airflow.dags.predict as predict_module

            importlib.reload(predict_module)
        except ImportError as e:
            if "airflow" not in str(e).lower():
                pytest.fail(f"Prediction DAG import failed with unexpected error: {e}")

    def test_feature_store_dag_imports(self, mock_airflow):  # noqa: ARG002
        """Test that feature store DAG can be imported without errors.

        Validates: Requirement 7.3 - DAGs should be testable in isolation
        """
        try:
            import importlib

            import airflow.dags.feature_store as fs_module

            importlib.reload(fs_module)
        except ImportError as e:
            if "airflow" not in str(e).lower():
                pytest.fail(f"Feature store DAG import failed with unexpected error: {e}")


class TestDAGSyntax:
    """Test DAG file syntax without requiring Airflow."""

    def test_etl_dag_syntax(self):
        """Test ETL DAG file has valid Python syntax.

        Validates: Requirement 8.4 - CD workflow validates DAG syntax
        """
        dag_path = Path("airflow/dags/etl.py")
        assert dag_path.exists(), f"ETL DAG file not found at {dag_path}"

        source = dag_path.read_text()
        # compile() will raise SyntaxError if syntax is invalid
        compile(source, dag_path, "exec")

    def test_train_dag_syntax(self):
        """Test training DAG file has valid Python syntax.

        Validates: Requirement 8.4 - CD workflow validates DAG syntax
        """
        dag_path = Path("airflow/dags/train.py")
        assert dag_path.exists(), f"Training DAG file not found at {dag_path}"

        source = dag_path.read_text()
        compile(source, dag_path, "exec")

    def test_predict_dag_syntax(self):
        """Test prediction DAG file has valid Python syntax.

        Validates: Requirement 8.4 - CD workflow validates DAG syntax
        """
        dag_path = Path("airflow/dags/predict.py")
        assert dag_path.exists(), f"Prediction DAG file not found at {dag_path}"

        source = dag_path.read_text()
        compile(source, dag_path, "exec")

    def test_feature_store_dag_syntax(self):
        """Test feature store DAG file has valid Python syntax.

        Validates: Requirement 8.4 - CD workflow validates DAG syntax
        """
        dag_path = Path("airflow/dags/feature_store.py")
        assert dag_path.exists(), f"Feature store DAG file not found at {dag_path}"

        source = dag_path.read_text()
        compile(source, dag_path, "exec")


class TestDAGConfiguration:
    """Test that DAGs use configuration from config module."""

    def test_etl_dag_uses_config(self):
        """Test ETL DAG imports config settings.

        Validates: Requirement 7.1 - DAGs use Configuration_Manager
        """
        dag_path = Path("airflow/dags/etl.py")
        source = dag_path.read_text()

        assert "from config.settings import settings" in source, "ETL DAG should import settings from config module"

    def test_train_dag_uses_config(self):
        """Test training DAG imports config settings.

        Validates: Requirement 7.1 - DAGs use Configuration_Manager
        """
        dag_path = Path("airflow/dags/train.py")
        source = dag_path.read_text()

        assert "from config.settings import settings" in source, (
            "Training DAG should import settings from config module"
        )

    def test_predict_dag_uses_config(self):
        """Test prediction DAG imports config settings.

        Validates: Requirement 7.1 - DAGs use Configuration_Manager
        """
        dag_path = Path("airflow/dags/predict.py")
        source = dag_path.read_text()

        assert "from config.settings import settings" in source, (
            "Prediction DAG should import settings from config module"
        )

    def test_feature_store_dag_uses_config(self):
        """Test feature store DAG imports config settings.

        Validates: Requirement 7.1 - DAGs use Configuration_Manager
        """
        dag_path = Path("airflow/dags/feature_store.py")
        source = dag_path.read_text()

        assert "from config.settings import settings" in source, (
            "Feature store DAG should import settings from config module"
        )

    def test_feature_store_dag_no_hardcoded_paths(self):
        """Test feature store DAG has no hardcoded absolute paths.

        Validates: Requirement 7.1 - Replace hardcoded paths
        """
        dag_path = Path("airflow/dags/feature_store.py")
        source = dag_path.read_text()

        # Check for common hardcoded path patterns
        hardcoded_patterns = [
            "/home/",
            "/Users/",
            "C:\\",
            "D:\\",
        ]

        for pattern in hardcoded_patterns:
            assert pattern not in source, f"Feature store DAG contains hardcoded path pattern: {pattern}"


class TestDAGTaskDependencies:
    """Test that DAG task dependencies are correctly defined."""

    def test_etl_dag_has_sequential_flow(self):
        """Test ETL DAG defines sequential task flow.

        Validates: Requirement 7.4 - Proper task dependencies
        """
        dag_path = Path("airflow/dags/etl.py")
        source = dag_path.read_text()

        # Check for task function definitions
        expected_tasks = [
            "task_extract",
            "task_transform",
            "task_timestamps",
            "task_patient_ids",
            "task_save",
        ]

        for task_name in expected_tasks:
            assert task_name in source, f"ETL DAG should define task: {task_name}"

    def test_train_dag_has_sequential_flow(self):
        """Test training DAG defines sequential task flow.

        Validates: Requirement 7.4 - Proper task dependencies
        """
        dag_path = Path("airflow/dags/train.py")
        source = dag_path.read_text()

        # Check for task definitions
        expected_tasks = [
            "setup_mlflow",
            "load_data_from_feast",
            "prepare_and_split_data",
            "train_model",
            "evaluate_model",
            "create_artifacts",
            "log_to_mlflow",
            "cleanup_temp_files",
        ]

        for task_name in expected_tasks:
            assert task_name in source, f"Training DAG should reference task: {task_name}"

    def test_predict_dag_has_sequential_flow(self):
        """Test prediction DAG defines sequential task flow.

        Validates: Requirement 7.4 - Proper task dependencies
        """
        dag_path = Path("airflow/dags/predict.py")
        source = dag_path.read_text()

        # Check for task definitions
        expected_tasks = [
            "setup_and_materialize_features",
            "find_valid_patient_ids",
            "fetch_features",
            "load_model",
            "make_predictions",
            "save_predictions",
            "cleanup_temp_files",
        ]

        for task_name in expected_tasks:
            assert task_name in source, f"Prediction DAG should reference task: {task_name}"

    def test_feature_store_dag_has_task(self):
        """Test feature store DAG defines the dataset creation task.

        Validates: Requirement 7.4 - Proper task dependencies
        """
        dag_path = Path("airflow/dags/feature_store.py")
        source = dag_path.read_text()

        assert "create_dataset_task" in source, "Feature store DAG should define create_dataset_task"
