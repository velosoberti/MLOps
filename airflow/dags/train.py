"""
ML Training Pipeline DAG with Feast and MLflow.

This DAG orchestrates the ML training workflow:
1. Setup MLflow tracking
2. Load data from Feast feature store
3. Prepare and split data
4. Train logistic regression model
5. Evaluate model performance
6. Create artifacts (confusion matrix, feature list)
7. Log experiment to MLflow
8. Cleanup temporary files

Configuration is loaded from the config module via environment variables.
"""
import logging
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from config.settings import settings
from src.training import (
    cleanup_temp_files,
    create_artifacts,
    evaluate_model,
    load_data_from_feast,
    log_to_mlflow,
    prepare_and_split_data,
    setup_mlflow,
    train_model,
)

logger = logging.getLogger(__name__)

# Log configuration at DAG load time
logger.info(
    f"Training DAG loaded with config: "
    f"mlflow_uri={settings.mlflow_tracking_uri}, "
    f"feast_repo={settings.feast_repo_path}, "
    f"model_name={settings.model_name}"
)

# ===================== DAG DEFINITION =====================

default_args = {
    "owner": "data_science_team",
    "depends_on_past": False,
    "email": ["data-team@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "ml_training_pipeline",
    default_args=default_args,
    description="ML training pipeline with Feast and MLflow",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["machine_learning", "feast", "mlflow", "diabetes", "training"],
)

# ===================== TASK DEFINITIONS =====================

task_setup_mlflow = PythonOperator(
    task_id="setup_mlflow",
    python_callable=setup_mlflow,
    dag=dag,
)

task_load_data = PythonOperator(
    task_id="load_data_from_feast",
    python_callable=load_data_from_feast,
    dag=dag,
)

task_prepare_data = PythonOperator(
    task_id="prepare_and_split_data",
    python_callable=prepare_and_split_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    dag=dag,
)

task_artifacts = PythonOperator(
    task_id="create_artifacts",
    python_callable=create_artifacts,
    dag=dag,
)

task_log_mlflow = PythonOperator(
    task_id="log_to_mlflow",
    python_callable=log_to_mlflow,
    dag=dag,
)

task_cleanup = PythonOperator(
    task_id="cleanup_temp_files",
    python_callable=cleanup_temp_files,
    trigger_rule="all_done",  # Execute always, even if there are failures
    dag=dag,
)

# ===================== FLOW DEFINITION =====================

(
    task_setup_mlflow
    >> task_load_data
    >> task_prepare_data
    >> task_train
    >> task_evaluate
    >> task_artifacts
    >> task_log_mlflow
    >> task_cleanup
)