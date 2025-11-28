"""
DAG Airflow para predição incremental
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta, datetime
import socket
import pandas as pd
from feast import FeatureStore
import mlflow
import os
from pathlib import Path
from framework.prediction import *


# ===================== DEFINIÇÃO DA DAG =====================

default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email': ['data-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_prediction_pipeline',
    default_args=default_args,
    description='Pipeline de predição incremental com salvamento de resultados',
    schedule_interval='@daily',  # Executar diariamente
    start_date=days_ago(1),
    catchup=False,
    tags=['machine_learning', 'prediction'],
)

# ===================== DEFINIÇÃO DAS TASKS =====================

task_setup = PythonOperator(
    task_id='setup_and_materialize_features',
    python_callable=setup_and_materialize_features,
    dag=dag,
)

task_find_patients = PythonOperator(
    task_id='find_valid_patient_ids',
    python_callable=find_valid_patient_ids,
    dag=dag,
)

task_fetch = PythonOperator(
    task_id='fetch_features',
    python_callable=fetch_features,
    dag=dag,
)

task_load = PythonOperator(
    task_id='load_model',
    python_callable=load_model,
    dag=dag,
)

task_predict = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    dag=dag,
)

task_save_pred = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions,
    dag=dag,
)

task_cleanup = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    trigger_rule='all_done',
    dag=dag,
)

# ===================== DEFINIÇÃO DO FLUXO =====================

task_setup >> task_find_patients >> task_fetch >> task_load >> task_predict >> task_save_pred >> task_cleanup