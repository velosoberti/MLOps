"""
ML Prediction Pipeline DAG for incremental predictions.

This DAG orchestrates the prediction workflow:
1. Setup MLflow and materialize Feast features
2. Find valid patient IDs with complete feature data
3. Fetch features from Feast online store
4. Load the latest model from MLflow
5. Make predictions
6. Save predictions to permanent storage
7. Cleanup temporary files

Configuration is loaded from the config module via environment variables.
"""
import logging
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from config.settings import settings
from src.prediction import (
    cleanup_temp_files,
    fetch_features,
    find_valid_patient_ids,
    load_model,
    make_predictions,
    save_predictions,
    setup_and_materialize_features,
)

logger = logging.getLogger(__name__)

# Log configuration at DAG load time
logger.info(
    f"Prediction DAG loaded with config: "
    f"mlflow_uri={settings.mlflow_tracking_uri}, "
    f"feast_repo={settings.feast_repo_path}, "
    f"model_name={settings.model_name}, "
    f"n_patients={settings.n_patients}"
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
    "ml_prediction_pipeline",
    default_args=default_args,
    description="Incremental prediction pipeline with result persistence",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["machine_learning", "prediction", "feast", "mlflow"],
)

# ===================== TASK DEFINITIONS =====================

task_setup = PythonOperator(
    task_id="setup_and_materialize_features",
    python_callable=setup_and_materialize_features,
    dag=dag,
)

task_find_patients = PythonOperator(
    task_id="find_valid_patient_ids",
    python_callable=find_valid_patient_ids,
    dag=dag,
)

task_fetch = PythonOperator(
    task_id="fetch_features",
    python_callable=fetch_features,
    dag=dag,
)

task_load = PythonOperator(
    task_id="load_model",
    python_callable=load_model,
    dag=dag,
)

task_predict = PythonOperator(
    task_id="make_predictions",
    python_callable=make_predictions,
    dag=dag,
)

task_save_pred = PythonOperator(
    task_id="save_predictions",
    python_callable=save_predictions,
    dag=dag,
)

task_cleanup = PythonOperator(
    task_id="cleanup_temp_files",
    python_callable=cleanup_temp_files,
    trigger_rule="all_done",
    dag=dag,
)

# ===================== FLOW DEFINITION =====================

(
    task_setup
    >> task_find_patients
    >> task_fetch
    >> task_load
    >> task_predict
    >> task_save_pred
    >> task_cleanup
)