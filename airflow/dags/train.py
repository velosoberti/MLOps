"""
DAG Airflow para treinamento de modelo ML com Feast e MLflow
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import socket
import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import matplotlib.pyplot as plt
from framework.training import *



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
    'ml_training_pipeline',
    default_args=default_args,
    description='Pipeline de treinamento ML com Feast e MLflow',
    schedule_interval='@weekly',  # Executar semanalmente
    start_date=days_ago(1),
    catchup=False,
    tags=['machine_learning', 'feast', 'mlflow', 'diabetes'],
)

# ===================== DEFINIÇÃO DAS TASKS =====================

task_setup_mlflow = PythonOperator(
    task_id='setup_mlflow',
    python_callable=setup_mlflow,
    dag=dag,
)

task_load_data = PythonOperator(
    task_id='load_data_from_feast',
    python_callable=load_data_from_feast,
    dag=dag,
)

task_prepare_data = PythonOperator(
    task_id='prepare_and_split_data',
    python_callable=prepare_and_split_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

task_artifacts = PythonOperator(
    task_id='create_artifacts',
    python_callable=create_artifacts,
    dag=dag,
)

task_log_mlflow = PythonOperator(
    task_id='log_to_mlflow',
    python_callable=log_to_mlflow,
    dag=dag,
)

task_cleanup = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    trigger_rule='all_done',  # Executar sempre, mesmo se houver falhas
    dag=dag,
)

# ===================== DEFINIÇÃO DO FLUXO =====================

task_setup_mlflow >> task_load_data >> task_prepare_data >> task_train >> task_evaluate >> task_artifacts >> task_log_mlflow >> task_cleanup