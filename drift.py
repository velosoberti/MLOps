#### Prediction DAG

# Caminho base do volume montado (visível tanto no Host quanto no Docker)
BASE_PATH = "/home/luisveloso/MLOps_projects"

try:
    hostname = 'host.docker.internal'
    host_ip = socket.gethostbyname(hostname)
    MLFLOW_TRACKING_URI = f"http://{host_ip}:5000/"
except socket.gaierror:
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"

FEAST_REPO_PATH = f"{BASE_PATH}/feature_store/feature_repo"
FEATURE_VIEW = "predictors_df_feature_view"
MODEL_NAME = "diabete_model"
N_PATIENTS = 50

# Diretório de saída para predições
OUTPUT_DIR = f"{BASE_PATH}/data/artifacts/predictions"

# Caminho para o histórico acumulado de predições
HISTORICAL_PREDICTIONS_FILE = f"{BASE_PATH}/data/artifacts/predictions/predictions_history.parquet"


# ===================== FUNÇÕES DAS TASKS =====================

def setup_and_materialize_features(**context):
    """Task 1: Setup MLflow e materializa features do Feast"""
    print("🔧 Configurando MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    print("🔄 Materializando features no Feast...")
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    store.materialize_incremental(end_date=datetime.now())
    
    context['ti'].xcom_push(key='mlflow_uri', value=MLFLOW_TRACKING_URI)
    context['ti'].xcom_push(key='materialization_timestamp', value=datetime.now().isoformat())
    
    print("✅ MLflow configurado e features materializadas!")


def find_valid_patient_ids(**context):
    """Task 2: Busca patient IDs válidos (sem NaN)"""
    print(f"🔍 Buscando os últimos {N_PATIENTS} pacientes válidos...")
    
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    
    # Features usadas no treinamento (sem Glucose)
    feast_features = [
        f"{FEATURE_VIEW}:DiabetesPedigreeFunction",
        f"{FEATURE_VIEW}:BMI",
        f"{FEATURE_VIEW}:SkinThickness",
        f"{FEATURE_VIEW}:Insulin"
    ]
    
    valid_patient_ids = []
    
    # Buscar de trás para frente (IDs mais recentes)
    for patient_id in range(1000, 0, -1):
        if len(valid_patient_ids) >= N_PATIENTS:
            break
        
        try:
            features = store.get_online_features(
                features=feast_features,
                entity_rows=[{"patient_id": patient_id}]
            ).to_dict()
            
            features_df = pd.DataFrame.from_dict(features)
            
            # Verificar se há NaN
            if not features_df.drop(columns=["patient_id"]).isna().any().any():
                valid_patient_ids.append(patient_id)
                
        except Exception:
            continue
    
    if not valid_patient_ids:
        raise ValueError("❌ Nenhum paciente válido encontrado!")
    
    # Ordenar em ordem decrescente (mais recentes primeiro)
    valid_patient_ids = sorted(valid_patient_ids, reverse=True)
    
    print(f"✅ Encontrados {len(valid_patient_ids)} pacientes válidos")
    print(f"   IDs: {valid_patient_ids[:10]}... (primeiros 10)")
    
    # Salvar IDs
    context['ti'].xcom_push(key='valid_patient_ids', value=valid_patient_ids)
    context['ti'].xcom_push(key='n_valid_patients', value=len(valid_patient_ids))


def fetch_features(**context):
    """Task 3: Busca features dos pacientes válidos"""
    print("📥 Buscando features dos pacientes...")
    
    # Recuperar IDs válidos
    patient_ids = context['ti'].xcom_pull(key='valid_patient_ids', task_ids='find_valid_patient_ids')
    
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    
    # Features usadas no treinamento (sem Glucose)
    feast_features = [
        f"{FEATURE_VIEW}:DiabetesPedigreeFunction",
        f"{FEATURE_VIEW}:BMI",
        f"{FEATURE_VIEW}:SkinThickness",
        f"{FEATURE_VIEW}:Insulin"
    ]
    
    entity_rows = [{"patient_id": pid} for pid in patient_ids]
    
    features = store.get_online_features(
        features=feast_features,
        entity_rows=entity_rows
    ).to_dict()
    
    features_df = pd.DataFrame.from_dict(features)
    
    print(f"✅ Features carregadas: {features_df.shape}")
    print(f"   Colunas: {features_df.columns.tolist()}")
    
    # Salvar features
    features_df.to_parquet('/tmp/features_for_prediction.parquet', index=False)
    
    context['ti'].xcom_push(key='features_shape', value=features_df.shape)


def load_model(**context):
    """Task 4: Carrega última versão do modelo"""
    print(f"🤖 Carregando modelo '{MODEL_NAME}'...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.client.MlflowClient()
    
    latest_versions = client.get_latest_versions(MODEL_NAME)
    
    if not latest_versions:
        raise ValueError(f"❌ Nenhuma versão encontrada para o modelo '{MODEL_NAME}'")
    
    latest_version = max([int(v.version) for v in latest_versions])
    
    print(f"📦 Versão encontrada: {latest_version}")
    
    model_uri = f"models:/{MODEL_NAME}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Extrair feature names do modelo
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_.tolist()
        print(f"📋 Features do modelo: {model_features}")
        context['ti'].xcom_push(key='model_features', value=model_features)
    else:
        print("⚠️ Modelo não possui feature_names_in_, usando features padrão")
    
    # Salvar modelo temporariamente
    import joblib
    joblib.dump(model, '/tmp/loaded_model.pkl')
    
    print("✅ Modelo carregado!")
    
    context['ti'].xcom_push(key='model_version', value=latest_version)
    context['ti'].xcom_push(key='model_uri', value=model_uri)


def make_predictions(**context):
    """Task 5: Faz predições"""
    print("🔮 Fazendo predições...")
    
    import joblib
    
    # Carregar modelo e features
    model = joblib.load('/tmp/loaded_model.pkl')
    features_df = pd.read_parquet('/tmp/features_for_prediction.parquet')
    
    # Verificar se temos as feature names do modelo
    ti = context['ti']
    model_features = ti.xcom_pull(key='model_features', task_ids='load_model')
    
    # Preparar features
    X = features_df.drop(columns=["patient_id"])
    
    # Se temos as features do modelo, garantir que usamos apenas essas
    if model_features:
        # Verificar se todas as features necessárias estão disponíveis
        missing_features = set(model_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"❌ Features faltando: {missing_features}")
        
        # Selecionar apenas as features do modelo na ordem correta
        X = X[model_features]
        print(f"✅ Usando features do modelo: {model_features}")
    else:
        # Fallback: ordenar alfabeticamente
        X = X[sorted(X.columns)]
        print(f"⚠️ Usando features ordenadas: {sorted(X.columns)}")
    
    # Predições
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Criar DataFrame de resultados
    results_df = features_df.copy()
    results_df['prediction'] = predictions
    results_df['probability_class_0'] = probabilities[:, 0]
    results_df['probability_class_1'] = probabilities[:, 1]
    results_df['prediction_timestamp'] = datetime.now()
    
    print(f"✅ Predições concluídas!")
    print(f"   Total: {len(predictions)}")
    print(f"   Classe 0 (Não diabético): {sum(predictions == 0)}")
    print(f"   Classe 1 (Diabético): {sum(predictions == 1)}")
    
    # Salvar predições
    results_df.to_parquet('/tmp/predictions_results.parquet', index=False)
    
    context['ti'].xcom_push(key='n_predictions', value=len(predictions))
    context['ti'].xcom_push(key='n_class_0', value=int(sum(predictions == 0)))
    context['ti'].xcom_push(key='n_class_1', value=int(sum(predictions == 1)))


def save_predictions(**context):
    """Task 6: Salva predições em arquivo permanente e histórico acumulado"""
    print("💾 Salvando predições...")
    
    # Criar diretório
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(HISTORICAL_PREDICTIONS_FILE)).mkdir(parents=True, exist_ok=True)
    
    # Carregar predições atuais
    predictions_df = pd.read_parquet('/tmp/predictions_results.parquet')
    
    # Adicionar ID único para cada batch de predição
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_df['batch_id'] = batch_id
    
    # Adicionar versão do modelo
    ti = context['ti']
    model_version = ti.xcom_pull(key='model_version', task_ids='load_model')
    predictions_df['model_version'] = model_version
    
    # === SALVAR PREDIÇÕES INDIVIDUAIS (OPCIONAL) ===
    # Nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.parquet"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Salvar Parquet
    predictions_df.to_parquet(filepath, index=False)
    
    # Salvar CSV para fácil visualização
    csv_filepath = filepath.replace('.parquet', '.csv')
    predictions_df.to_csv(csv_filepath, index=False)
    
    print(f"✅ Predições individuais salvas:")
    print(f"   Parquet: {filepath}")
    print(f"   CSV: {csv_filepath}")
    
    # === ACUMULAR NO HISTÓRICO ===
    if os.path.exists(HISTORICAL_PREDICTIONS_FILE):
        # Carregar histórico existente
        historical_df = pd.read_parquet(HISTORICAL_PREDICTIONS_FILE)
        print(f"📚 Histórico encontrado: {len(historical_df)} predições anteriores")
        
        # Concatenar com novas predições
        updated_history = pd.concat([historical_df, predictions_df], ignore_index=True)
    else:
        # Criar novo histórico
        print(f"📚 Criando novo arquivo de histórico")
        updated_history = predictions_df
    
    # Salvar histórico atualizado
    updated_history.to_parquet(HISTORICAL_PREDICTIONS_FILE, index=False)
    print(f"✅ Histórico atualizado: {len(updated_history)} predições totais")
    
    # Também salvar CSV do histórico
    historical_csv = HISTORICAL_PREDICTIONS_FILE.replace('.parquet', '.csv')
    updated_history.to_csv(historical_csv, index=False)
    
    # === ESTATÍSTICAS DO HISTÓRICO ===
    print(f"\n📊 Estatísticas do Histórico:")
    print(f"   Total de predições: {len(updated_history)}")
    print(f"   Total de batches: {updated_history['batch_id'].nunique()}")
    print(f"   Primeira predição: {updated_history['prediction_timestamp'].min()}")
    print(f"   Última predição: {updated_history['prediction_timestamp'].max()}")
    print(f"   Proporção classe 1 (histórico): {updated_history['prediction'].mean():.3f}")
    
    context['ti'].xcom_push(key='predictions_file', value=filepath)
    context['ti'].xcom_push(key='predictions_csv_file', value=csv_filepath)
    context['ti'].xcom_push(key='historical_file', value=HISTORICAL_PREDICTIONS_FILE)
    context['ti'].xcom_push(key='total_historical_predictions', value=len(updated_history))


def cleanup_temp_files(**context):
    """Task 7: Limpa arquivos temporários"""
    print("🧹 Limpando arquivos temporários...")
    
    temp_files = [
        '/tmp/features_for_prediction.parquet',
        '/tmp/loaded_model.pkl',
        '/tmp/predictions_results.parquet'
    ]
    
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   Removido: {file_path}")
        except Exception as e:
            print(f"   ⚠️ Erro ao remover {file_path}: {e}")
    
    print("✅ Limpeza concluída!")




### training dag

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

# ===================== CONFIGURAÇÕES =====================
try:
    hostname = 'host.docker.internal'
    host_ip = socket.gethostbyname(hostname)
    MLFLOW_TRACKING_URI = f"http://{host_ip}:5000/"
    print(f"✅ IP resolvido para MLflow: {MLFLOW_TRACKING_URI}")
except socket.gaierror:
    # Fallback caso algo dê errado com o DNS do Docker
    print("⚠️ Falha ao resolver host.docker.internal, tentando localhost...")
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"

MLFLOW_EXPERIMENT_ID = '467326610704772702'
FEAST_REPO_PATH = "/home/luisveloso/MLOps_projects/feature_store/feature_repo"
DATASET_NAME = "my_training_dataset"


# ===================== FUNÇÕES DAS TASKS =====================

def setup_mlflow(**context):
    """Task 1: Configura MLflow tracking"""
    print("🔧 Configurando MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)
    
    # Pushando configuração para XCom
    context['ti'].xcom_push(key='mlflow_uri', value=MLFLOW_TRACKING_URI)
    context['ti'].xcom_push(key='experiment_id', value=MLFLOW_EXPERIMENT_ID)
    
    print(f"✅ MLflow configurado: {MLFLOW_TRACKING_URI}")


def load_data_from_feast(**context):
    """Task 2: Carrega dados do Feast"""
    print(f"📦 Carregando dados do Feast...")
    
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    training_data = store.get_saved_dataset(name=DATASET_NAME).to_df()
    
    print(f"✅ Dados carregados: {training_data.shape}")
    print(f"   Colunas: {training_data.columns.tolist()}")
    
    # Salvar dados no XCom (para datasets pequenos) ou em arquivo temporário
    context['ti'].xcom_push(key='data_shape', value=training_data.shape)
    context['ti'].xcom_push(key='data_path', value='/tmp/training_data.parquet')
    
    # Salvar em arquivo para não sobrecarregar XCom
    training_data.to_parquet('/tmp/training_data.parquet', index=False)


def prepare_and_split_data(**context):
    """Task 3: Prepara features e divide dados"""
    print("🔧 Preparando features e dividindo dados...")
    
    # Carregar dados
    training_data = pd.read_parquet('/tmp/training_data.parquet')
    
    # Preparar features
    y = training_data["Outcome"]
    X = training_data.drop(columns=["Outcome", "event_timestamp", "patient_id"])
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        stratify=y, 
        test_size=0.25,
        random_state=42
    )
    
    print(f"📊 Dados divididos:")
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")
    
    # Salvar splits
    X_train.to_parquet('/tmp/X_train.parquet', index=False)
    X_test.to_parquet('/tmp/X_test.parquet', index=False)
    y_train.to_frame().to_parquet('/tmp/y_train.parquet', index=False)
    y_test.to_frame().to_parquet('/tmp/y_test.parquet', index=False)
    
    # Pushando metadados
    context['ti'].xcom_push(key='n_features', value=X_train.shape[1])
    context['ti'].xcom_push(key='n_train_samples', value=X_train.shape[0])
    context['ti'].xcom_push(key='n_test_samples', value=X_test.shape[0])
    context['ti'].xcom_push(key='train_positive_ratio', value=float(y_train.sum() / len(y_train)))
    context['ti'].xcom_push(key='feature_names', value=sorted(X_train.columns.tolist()))


def train_model(**context):
    """Task 4: Treina o modelo"""
    print("🤖 Treinando modelo...")
    
    # Carregar dados
    X_train = pd.read_parquet('/tmp/X_train.parquet')
    y_train = pd.read_parquet('/tmp/y_train.parquet')['Outcome']
    
    # Ordenar colunas
    X_train_sorted = X_train[sorted(X_train.columns)]
    
    # Treinar modelo
    model = LogisticRegression()
    model.fit(X_train_sorted, y_train)
    
    print(f"✅ Modelo treinado!")
    
    # Salvar modelo temporariamente
    import joblib
    joblib.dump(model, '/tmp/model.pkl')
    
    # Pushando parâmetros do modelo
    context['ti'].xcom_push(key='model_penalty', value=model.penalty)
    context['ti'].xcom_push(key='model_solver', value=model.solver)
    context['ti'].xcom_push(key='model_max_iter', value=model.max_iter)


def evaluate_model(**context):
    """Task 5: Avalia o modelo"""
    print("📈 Avaliando modelo...")
    
    import joblib
    
    # Carregar modelo e dados
    model = joblib.load('/tmp/model.pkl')
    X_train = pd.read_parquet('/tmp/X_train.parquet')
    y_train = pd.read_parquet('/tmp/y_train.parquet')['Outcome']
    X_test = pd.read_parquet('/tmp/X_test.parquet')
    y_test = pd.read_parquet('/tmp/y_test.parquet')['Outcome']
    
    # Ordenar colunas
    X_train_sorted = X_train[sorted(X_train.columns)]
    X_test_sorted = X_test[sorted(X_test.columns)]
    
    # Predições
    y_train_pred = model.predict(X_train_sorted)
    y_test_pred = model.predict(X_test_sorted)
    
    # Métricas
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    
    print(f"   Acurácia (treino): {acc_train:.4f}")
    print(f"   Acurácia (teste): {acc_test:.4f}")
    
    # Salvar predições
    pd.DataFrame({'y_pred': y_test_pred}).to_parquet('/tmp/y_test_pred.parquet', index=False)
    
    # Pushando métricas
    context['ti'].xcom_push(key='acc_train', value=float(acc_train))
    context['ti'].xcom_push(key='acc_test', value=float(acc_test))


def create_artifacts(**context):
    """Task 6: Cria artefatos (matriz de confusão, lista de features)"""
    print("🎨 Criando artefatos...")
    
    # Carregar dados
    y_test = pd.read_parquet('/tmp/y_test.parquet')['Outcome']
    y_test_pred = pd.read_parquet('/tmp/y_test_pred.parquet')['y_pred']
    feature_names = context['ti'].xcom_pull(key='feature_names', task_ids='prepare_and_split_data')
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('/tmp/confusion_matrix.png')
    plt.close()
    print("✅ Matriz de confusão salva")
    
    # Lista de features
    with open('/tmp/features.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    print("✅ Lista de features salva")


def log_to_mlflow(**context):
    """Task 7: Registra tudo no MLflow"""
    print("📝 Registrando experimento no MLflow...")
    
    import joblib
    
    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)
    
    # Carregar modelo
    model = joblib.load('/tmp/model.pkl')
    
    # Puxar informações do XCom
    ti = context['ti']
    
    with mlflow.start_run(run_name="logistic_regression_airflow"):
        # Tags
        mlflow.set_tags({
            "model_type": "classification",
            "algorithm": "logistic_regression",
            "dataset": "diabetes_dataset",
            "developer": "airflow_pipeline",
            "environment": "production",
            "orchestrator": "airflow"
        })
        
        # Autolog
        mlflow.sklearn.autolog(
            log_models=True,
            log_input_examples=True,
            log_model_signatures=True,
            log_datasets=False
        )
        
        # Parâmetros
        params = {
            "penalty": ti.xcom_pull(key='model_penalty', task_ids='train_model'),
            "solver": ti.xcom_pull(key='model_solver', task_ids='train_model'),
            "max_iter": ti.xcom_pull(key='model_max_iter', task_ids='train_model'),
            "n_features": ti.xcom_pull(key='n_features', task_ids='prepare_and_split_data'),
            "n_train_samples": ti.xcom_pull(key='n_train_samples', task_ids='prepare_and_split_data'),
            "n_test_samples": ti.xcom_pull(key='n_test_samples', task_ids='prepare_and_split_data'),
            "train_positive_ratio": ti.xcom_pull(key='train_positive_ratio', task_ids='prepare_and_split_data')
        }
        mlflow.log_params(params)
        
        # Métricas
        metrics = {
            "acc_train": ti.xcom_pull(key='acc_train', task_ids='evaluate_model'),
            "acc_test": ti.xcom_pull(key='acc_test', task_ids='evaluate_model')
        }
        mlflow.log_metrics(metrics)
        
        # Artefatos
        mlflow.log_artifact('/tmp/confusion_matrix.png')
        mlflow.log_artifact('/tmp/features.txt')
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, "model")
        
        print("✅ Experimento registrado no MLflow!")


def cleanup_temp_files(**context):
    """Task 8: Limpa arquivos temporários"""
    print("🧹 Limpando arquivos temporários...")
    
    import os
    
    temp_files = [
        '/tmp/training_data.parquet',
        '/tmp/X_train.parquet',
        '/tmp/X_test.parquet',
        '/tmp/y_train.parquet',
        '/tmp/y_test.parquet',
        '/tmp/y_test_pred.parquet',
        '/tmp/model.pkl',
        '/tmp/confusion_matrix.png',
        '/tmp/features.txt'
    ]
    
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   Removido: {file_path}")
        except Exception as e:
            print(f"   ⚠️ Erro ao remover {file_path}: {e}")
    
    print("✅ Limpeza concluída!")


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



## API constructor
"""
API Flask Profissional para Predição de Diabetes
Com tratamento de erros, validação, logging e health check completo
"""

from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from functools import wraps
import traceback
from framework.api_consctructor import *
# ================== CONFIGURAÇÃO DE LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# ================== CONFIGURAÇÕES ==================
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"
EXPERIMENT_ID = "467326610704772702"
MODEL_NAME = "diabete_model"

# ================== INICIALIZAÇÃO DO MODELO ==================
class ModelManager:
    """Gerenciador do modelo MLflow com validação e cache."""
    
    def __init__(self):
        self.model = None
        self.model_version = None
        self.feature_names = None
        self.model_loaded_at = None
        self.load_model()
    
    def load_model(self) -> None:
        """Carrega modelo do MLflow com tratamento de erros."""
        try:
            logger.info("🔄 Carregando modelo do MLflow...")
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
            
            client = mlflow.client.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME)
            
            if not versions:
                raise ValueError(f"Nenhuma versão encontrada para o modelo '{MODEL_NAME}'")
            
            self.model_version = max([int(v.version) for v in versions])
            model_uri = f"models:/{MODEL_NAME}/{self.model_version}"
            
            self.model = mlflow.sklearn.load_model(model_uri)
            self.feature_names = list(self.model.feature_names_in_)
            self.model_loaded_at = datetime.now()
            
            logger.info(f"✅ Modelo carregado com sucesso: {MODEL_NAME} v{self.model_version}")
            logger.info(f"📋 Features esperadas: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def reload_model(self) -> Dict[str, Any]:
        """Recarrega modelo e retorna informações."""
        logger.info("🔄 Recarregando modelo...")
        old_version = self.model_version
        self.load_model()
        return {
            "previous_version": old_version,
            "current_version": self.model_version,
            "reloaded_at": self.model_loaded_at.isoformat()
        }
    
    def predict(self, df: pd.DataFrame) -> float:
        """Faz predição com validação de features."""
        if self.model is None:
            raise RuntimeError("Modelo não carregado")
        
        # Valida e ordena features
        X = df[self.feature_names]
        pred = self.model.predict_proba(X)[:, 1]
        return float(pred[0])
    
    def is_healthy(self) -> Dict[str, Any]:
        """Verifica saúde do modelo."""
        return {
            "model_loaded": self.model is not None,
            "model_name": MODEL_NAME,
            "model_version": self.model_version,
            "features_count": len(self.feature_names) if self.feature_names else 0,
            "loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None
        }


# ================== VALIDAÇÃO DE ENTRADA ==================
class InputValidator:
    """Validador de entrada de dados."""
    
    @staticmethod
    def validate_prediction_input(data: Dict[str, Any], required_features: list) -> tuple[bool, Optional[str]]:
        """
        Valida entrada de predição.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not data:
            return False, "Request body está vazio"
        
        if not isinstance(data, dict):
            return False, "Request body deve ser um objeto JSON"
        
        # Verifica features faltantes
        missing_features = set(required_features) - set(data.keys())
        if missing_features:
            return False, f"Features faltantes: {sorted(missing_features)}"
        
        # Verifica features extras
        extra_features = set(data.keys()) - set(required_features)
        if extra_features:
            logger.warning(f"⚠️  Features extras ignoradas: {sorted(extra_features)}")
        
        # Valida tipos de dados
        for feature in required_features:
            value = data.get(feature)
            if value is None:
                return False, f"Feature '{feature}' não pode ser null"
            
            if not isinstance(value, (int, float)):
                return False, f"Feature '{feature}' deve ser numérica (recebido: {type(value).__name__})"
        
        return True, None


# ================== DECORADORES ==================
def handle_errors(f):
    """Decorator para tratamento centralizado de erros."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"❌ Erro de validação: {e}")
            return jsonify({
                "error": "Validation Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 400
        except RuntimeError as e:
            logger.error(f"❌ Erro de runtime: {e}")
            return jsonify({
                "error": "Runtime Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Internal Server Error",
                "message": "Ocorreu um erro inesperado. Verifique os logs.",
                "timestamp": datetime.now().isoformat()
            }), 500
    return decorated_function


def log_request(f):
    """Decorator para logging de requisições."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"📥 {request.method} {request.path} - IP: {request.remote_addr}")
        
        try:
            response = f(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            status = response[1] if isinstance(response, tuple) else 200
            logger.info(f"✅ {request.method} {request.path} - Status: {status} - Tempo: {elapsed:.3f}s")
            
            return response
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ {request.method} {request.path} - Erro após {elapsed:.3f}s: {e}")
            raise
    
    return decorated_function


# ================== INICIALIZAÇÃO ==================
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Inicializa gerenciador de modelo
try:
    model_manager = ModelManager()
    logger.info("✅ API inicializada com sucesso")
except Exception as e:
    logger.critical(f"💥 Falha crítica ao inicializar API: {e}")
    sys.exit(1)


# ================== ENDPOINTS ==================

@app.route("/health", methods=['GET'])
@log_request
def health_check():
    """
    Health check completo da API.
    
    Verifica:
    - Status da API
    - Conexão com MLflow
    - Status do modelo
    - Disponibilidade de features
    
    Returns:
        JSON com status detalhado
    """
    try:
        # Verifica modelo
        model_health = model_manager.is_healthy()
        
        # Verifica conexão com MLflow
        mlflow_healthy = False
        mlflow_error = None
        try:
            client = mlflow.client.MlflowClient()
            client.get_experiment(EXPERIMENT_ID)
            mlflow_healthy = True
        except Exception as e:
            mlflow_error = str(e)
            logger.error(f"❌ MLflow não acessível: {e}")
        
        # Determina status geral
        all_healthy = model_health["model_loaded"] and mlflow_healthy
        status_code = 200 if all_healthy else 503
        
        response = {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "running",
                "mlflow": {
                    "status": "connected" if mlflow_healthy else "disconnected",
                    "tracking_uri": MLFLOW_TRACKING_URI,
                    "error": mlflow_error
                },
                "model": model_health
            }
        }
        
        if all_healthy:
            logger.info("✅ Health check: HEALTHY")
        else:
            logger.warning("⚠️  Health check: UNHEALTHY")
        
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"❌ Erro no health check: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/predict", methods=['POST'])
@log_request
@handle_errors
def predict():
    """
    Endpoint de predição.
    
    Request Body (JSON):
        {
            "Glucose": 148,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Insulin": 0,
            "SkinThickness": 35
        }
    
    Response (JSON):
        {
            "score": 0.6523,
            "prediction": "diabetes",
            "confidence": 0.6523,
            "model_version": 3,
            "timestamp": "2025-11-26T10:30:00"
        }
    """
    # Valida Content-Type
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type",
            "message": "Content-Type deve ser 'application/json'",
            "received": request.content_type
        }), 400
    
    # Obtém dados
    data = request.get_json(silent=True)
    
    # Valida entrada
    is_valid, error_message = InputValidator.validate_prediction_input(
        data, 
        model_manager.feature_names
    )
    
    if not is_valid:
        logger.warning(f"⚠️  Validação falhou: {error_message}")
        return jsonify({
            "error": "Invalid Input",
            "message": error_message,
            "expected_features": model_manager.feature_names,
            "received_features": list(data.keys()) if data else []
        }), 400
    
    # Cria DataFrame
    df = pd.DataFrame([data])
    
    # Faz predição
    score = model_manager.predict(df)
    prediction_label = "diabetes" if score >= 0.5 else "no_diabetes"
    
    # Monta resposta
    response = {
        "score": round(score, 4),
        "prediction": prediction_label,
        "confidence": round(score if score >= 0.5 else 1 - score, 4),
        "model_version": model_manager.model_version,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Predição: {prediction_label} (score: {score:.4f})")
    
    return jsonify(response), 200


@app.route("/predict/batch", methods=['POST'])
@log_request
@handle_errors
def predict_batch():
    """
    Endpoint de predição em batch.
    
    Request Body (JSON):
        {
            "instances": [
                {
                    "Glucose": 148,
                    "BMI": 33.6,
                    ...
                },
                {
                    "Glucose": 85,
                    "BMI": 26.6,
                    ...
                }
            ]
        }
    
    Response (JSON):
        {
            "predictions": [
                {
                    "score": 0.6523,
                    "prediction": "diabetes",
                    "instance_index": 0
                },
                ...
            ],
            "total": 2,
            "model_version": 3,
            "timestamp": "2025-11-26T10:30:00"
        }
    """
    # Valida Content-Type
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type",
            "message": "Content-Type deve ser 'application/json'"
        }), 400
    
    data = request.get_json(silent=True)
    
    if not data or 'instances' not in data:
        return jsonify({
            "error": "Invalid Input",
            "message": "Request body deve conter 'instances' (lista de objetos)",
            "example": {
                "instances": [
                    {"Glucose": 148, "BMI": 33.6, "...": "..."}
                ]
            }
        }), 400
    
    instances = data['instances']
    
    if not isinstance(instances, list) or len(instances) == 0:
        return jsonify({
            "error": "Invalid Input",
            "message": "'instances' deve ser uma lista não vazia"
        }), 400
    
    if len(instances) > 1000:
        return jsonify({
            "error": "Batch Too Large",
            "message": f"Máximo de 1000 instâncias por batch (recebido: {len(instances)})"
        }), 400
    
    logger.info(f"📦 Processando batch com {len(instances)} instâncias")
    
    predictions = []
    errors = []
    
    for idx, instance in enumerate(instances):
        try:
            # Valida instância
            is_valid, error_message = InputValidator.validate_prediction_input(
                instance,
                model_manager.feature_names
            )
            
            if not is_valid:
                errors.append({
                    "instance_index": idx,
                    "error": error_message
                })
                continue
            
            # Predição
            df = pd.DataFrame([instance])
            score = model_manager.predict(df)
            prediction_label = "diabetes" if score >= 0.5 else "no_diabetes"
            
            predictions.append({
                "score": round(score, 4),
                "prediction": prediction_label,
                "confidence": round(score if score >= 0.5 else 1 - score, 4),
                "instance_index": idx
            })
            
        except Exception as e:
            logger.error(f"❌ Erro na instância {idx}: {e}")
            errors.append({
                "instance_index": idx,
                "error": str(e)
            })
    
    response = {
        "predictions": predictions,
        "total": len(predictions),
        "errors": errors if errors else None,
        "model_version": model_manager.model_version,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Batch concluído: {len(predictions)}/{len(instances)} sucessos")
    
    return jsonify(response), 200


@app.route("/model/info", methods=['GET'])
@log_request
def model_info():
    """
    Retorna informações detalhadas do modelo.
    
    Response (JSON):
        {
            "model_name": "log_reg_diabetes_predict",
            "model_version": 3,
            "features": [...],
            "feature_count": 5,
            "loaded_at": "2025-11-26T10:00:00",
            "mlflow_tracking_uri": "http://127.0.0.1:5000/"
        }
    """
    return jsonify({
        "model_name": MODEL_NAME,
        "model_version": model_manager.model_version,
        "features": model_manager.feature_names,
        "feature_count": len(model_manager.feature_names),
        "loaded_at": model_manager.model_loaded_at.isoformat() if model_manager.model_loaded_at else None,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_id": EXPERIMENT_ID
    }), 200


@app.route("/model/reload", methods=['POST'])
@log_request
@handle_errors
def reload_model():
    """
    Recarrega modelo do MLflow (útil após novo treinamento).
    
    Response (JSON):
        {
            "message": "Modelo recarregado com sucesso",
            "previous_version": 3,
            "current_version": 4,
            "reloaded_at": "2025-11-26T10:30:00"
        }
    """
    reload_info = model_manager.reload_model()
    
    return jsonify({
        "message": "Modelo recarregado com sucesso",
        **reload_info
    }), 200


# ================== ERROR HANDLERS ==================

@app.errorhandler(404)
def not_found(e):
    """Handler para rotas não encontradas."""
    logger.warning(f"⚠️  Rota não encontrada: {request.path}")
    return jsonify({
        "error": "Not Found",
        "message": f"Endpoint '{request.path}' não existe",
        "available_endpoints": [
            "GET  /health",
            "GET  /model/info",
            "POST /predict",
            "POST /predict/batch",
            "POST /model/reload"
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handler para métodos não permitidos."""
    logger.warning(f"⚠️  Método não permitido: {request.method} {request.path}")
    return jsonify({
        "error": "Method Not Allowed",
        "message": f"Método {request.method} não é permitido para {request.path}",
        "allowed_methods": e.valid_methods if hasattr(e, 'valid_methods') else []
    }), 405


@app.errorhandler(500)
def internal_error(e):
    """Handler para erros internos."""
    logger.error(f"💥 Erro interno: {e}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "Ocorreu um erro interno. Verifique os logs do servidor.",
        "timestamp": datetime.now().isoformat()
    }), 500


# ================== STARTUP ==================

@app.before_request
def before_request():
    """Hook executado antes de cada requisição."""
    request.start_time = datetime.now()


@app.after_request
def after_request(response):
    """Hook executado após cada requisição."""
    if hasattr(request, 'start_time'):
        elapsed = (datetime.now() - request.start_time).total_seconds()
        response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
    
    # Adiciona headers de segurança
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    
    return response


# ================== MAIN ==================

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("🚀 Iniciando API de Predição de Diabetes")
    logger.info("=" * 80)
    logger.info(f"📍 Servidor: http://0.0.0.0:5002")
    logger.info(f"🔗 MLflow: {MLFLOW_TRACKING_URI}")
    logger.info(f"📊 Modelo: {MODEL_NAME} v{model_manager.model_version}")
    logger.info(f"📋 Features: {len(model_manager.feature_names)}")
    logger.info("=" * 80)
    logger.info("Endpoints disponíveis:")
    logger.info("  GET  /health")
    logger.info("  GET  /model/info")
    logger.info("  POST /predict")
    logger.info("  POST /predict/batch")
    logger.info("  POST /model/reload")
    logger.info("=" * 80)
    
    app.run(
        host='0.0.0.0',
        port=5005,
        debug=False  # True em desenvolvimento
    )


## api request
"""
Suite de Testes para API de Predição de Diabetes
Testa todos os endpoints, validações e tratamentos de erro
"""

import requests
import json
from typing import Dict, Any

API_BASE_URL = "http://localhost:5002"


class TestRunner:
    """Executor de testes da API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name: str, func):
        """Executa um teste."""
        print(f"\n{'='*80}")
        print(f"🧪 {name}")
        print('='*80)
        
        try:
            func()
            self.passed += 1
            print(f"✅ PASSOU")
            self.tests.append((name, True, None))
        except AssertionError as e:
            self.failed += 1
            print(f"❌ FALHOU: {e}")
            self.tests.append((name, False, str(e)))
        except Exception as e:
            self.failed += 1
            print(f"💥 ERRO: {e}")
            self.tests.append((name, False, str(e)))
    
    def summary(self):
        """Imprime resumo dos testes."""
        print(f"\n{'='*80}")
        print("📊 RESUMO DOS TESTES")
        print('='*80)
        
        for name, passed, error in self.tests:
            status = "✅" if passed else "❌"
            print(f"{status} {name}")
            if error:
                print(f"   └─ {error}")
        
        print(f"\n{'='*80}")
        total = self.passed + self.failed
        print(f"Total: {total} | Passaram: {self.passed} | Falharam: {self.failed}")
        
        if self.failed == 0:
            print("🎉 TODOS OS TESTES PASSARAM!")
        else:
            print(f"⚠️  {self.failed} teste(s) falharam")
        print('='*80)


# ================== HELPER FUNCTIONS ==================

def make_request(method: str, endpoint: str, **kwargs) -> requests.Response:
    """Faz requisição HTTP."""
    url = f"{API_BASE_URL}{endpoint}"
    return requests.request(method, url, **kwargs)


def valid_prediction_data() -> Dict[str, Any]:
    """Retorna dados válidos para predição."""
    return {
        "Glucose": 148,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Insulin": 0,
        "SkinThickness": 35
    }


# ================== TESTES ==================

def test_health_check(runner: TestRunner):
    """Testa endpoint de health check."""
    
    def run():
        response = make_request("GET", "/health")
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        
        assert response.status_code in [200, 503], f"Status esperado 200 ou 503, recebido {response.status_code}"
        assert "status" in data, "Resposta deve conter 'status'"
        assert "services" in data, "Resposta deve conter 'services'"
        assert data["status"] in ["healthy", "unhealthy"], f"Status inválido: {data['status']}"
    
    runner.test("Health Check", run)


def test_model_info(runner: TestRunner):
    """Testa endpoint de informações do modelo."""
    
    def run():
        response = make_request("GET", "/model/info")
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        
        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert "model_name" in data, "Resposta deve conter 'model_name'"
        assert "model_version" in data, "Resposta deve conter 'model_version'"
        assert "features" in data, "Resposta deve conter 'features'"
        assert isinstance(data["features"], list), "Features deve ser uma lista"
        assert len(data["features"]) > 0, "Features não pode estar vazia"
    
    runner.test("Model Info", run)


def test_valid_prediction(runner: TestRunner):
    """Testa predição com dados válidos."""
    
    def run():
        data = valid_prediction_data()
        response = make_request("POST", "/predict", json=data)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert "score" in result, "Resposta deve conter 'score'"
        assert "prediction" in result, "Resposta deve conter 'prediction'"
        assert "confidence" in result, "Resposta deve conter 'confidence'"
        assert "model_version" in result, "Resposta deve conter 'model_version'"
        
        assert 0 <= result["score"] <= 1, f"Score deve estar entre 0 e 1, recebido {result['score']}"
        assert result["prediction"] in ["diabetes", "no_diabetes"], f"Prediction inválida: {result['prediction']}"
    
    runner.test("Predição Válida", run)


def test_missing_features(runner: TestRunner):
    """Testa predição com features faltantes."""
    
    def run():
        data = {"Glucose": 148, "BMI": 33.6}  # Faltam features
        response = make_request("POST", "/predict", json=data)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
        assert "missing" in result["message"].lower() or "faltantes" in result["message"].lower(), \
            "Mensagem deve indicar features faltantes"
    
    runner.test("Features Faltantes", run)


def test_invalid_feature_type(runner: TestRunner):
    """Testa predição com tipo de feature inválido."""
    
    def run():
        data = valid_prediction_data()
        data["Glucose"] = "invalid"  # String ao invés de número
        response = make_request("POST", "/predict", json=data)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
    
    runner.test("Tipo de Feature Inválido", run)


def test_null_feature(runner: TestRunner):
    """Testa predição com feature null."""
    
    def run():
        data = valid_prediction_data()
        data["Glucose"] = None
        response = make_request("POST", "/predict", json=data)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
        assert "null" in result["message"].lower(), "Mensagem deve indicar valor null"
    
    runner.test("Feature Null", run)


def test_empty_body(runner: TestRunner):
    """Testa predição com body vazio."""
    
    def run():
        response = make_request("POST", "/predict", json={})
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
    
    runner.test("Body Vazio", run)


def test_invalid_content_type(runner: TestRunner):
    """Testa predição com Content-Type inválido."""
    
    def run():
        data = "not a json"
        response = make_request(
            "POST", 
            "/predict", 
            data=data,
            headers={"Content-Type": "text/plain"}
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
    
    runner.test("Content-Type Inválido", run)


def test_batch_prediction_valid(runner: TestRunner):
    """Testa predição em batch com dados válidos."""
    
    def run():
        data = {
            "instances": [
                valid_prediction_data(),
                {**valid_prediction_data(), "Glucose": 85},
                {**valid_prediction_data(), "BMI": 26.6}
            ]
        }
        response = make_request("POST", "/predict/batch", json=data)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert "predictions" in result, "Resposta deve conter 'predictions'"
        assert "total" in result, "Resposta deve conter 'total'"
        assert len(result["predictions"]) == 3, f"Esperado 3 predições, recebido {len(result['predictions'])}"
        
        for pred in result["predictions"]:
            assert "score" in pred, "Cada predição deve conter 'score'"
            assert "prediction" in pred, "Cada predição deve conter 'prediction'"
            assert "instance_index" in pred, "Cada predição deve conter 'instance_index'"
    
    runner.test("Batch Válido", run)


def test_batch_missing_instances(runner: TestRunner):
    """Testa batch sem campo 'instances'."""
    
    def run():
        response = make_request("POST", "/predict/batch", json={})
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
    
    runner.test("Batch sem Instances", run)


def test_batch_empty_instances(runner: TestRunner):
    """Testa batch com lista vazia."""
    
    def run():
        response = make_request("POST", "/predict/batch", json={"instances": []})
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
    
    runner.test("Batch Vazio", run)


def test_batch_partial_errors(runner: TestRunner):
    """Testa batch com algumas instâncias inválidas."""
    
    def run():
        data = {
            "instances": [
                valid_prediction_data(),
                {"Glucose": 85},  # Faltam features
                valid_prediction_data()
            ]
        }
        response = make_request("POST", "/predict/batch", json=data)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert result["total"] == 2, f"Esperado 2 sucessos, recebido {result['total']}"
        assert result["errors"] is not None, "Deve conter erros"
        assert len(result["errors"]) == 1, f"Esperado 1 erro, recebido {len(result['errors'])}"
    
    runner.test("Batch com Erros Parciais", run)


def test_endpoint_not_found(runner: TestRunner):
    """Testa endpoint inexistente."""
    
    def run():
        response = make_request("GET", "/invalid")
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 404, f"Status esperado 404, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
        assert "available_endpoints" in result, "Deve listar endpoints disponíveis"
    
    runner.test("Endpoint Inexistente", run)


def test_method_not_allowed(runner: TestRunner):
    """Testa método HTTP não permitido."""
    
    def run():
        response = make_request("GET", "/predict")  # POST esperado
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 405, f"Status esperado 405, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
    
    runner.test("Método Não Permitido", run)


def test_response_headers(runner: TestRunner):
    """Testa headers de segurança na resposta."""
    
    def run():
        response = make_request("GET", "/health")
        
        print(f"Status: {response.status_code}")
        print("\nHeaders:")
        for key, value in response.headers.items():
            if key.startswith('X-'):
                print(f"  {key}: {value}")
        
        assert "X-Response-Time" in response.headers, "Deve conter header X-Response-Time"
        assert "X-Content-Type-Options" in response.headers, "Deve conter header X-Content-Type-Options"
        assert "X-Frame-Options" in response.headers, "Deve conter header X-Frame-Options"
    
    runner.test("Headers de Segurança", run)


# ================== TESTES DE PERFORMANCE ==================

def test_response_time(runner: TestRunner):
    """Testa tempo de resposta."""
    
    def run():
        import time
        
        data = valid_prediction_data()
        start = time.time()
        response = make_request("POST", "/predict", json=data)
        elapsed = time.time() - start
        
        print(f"Status: {response.status_code}")
        print(f"Tempo de resposta: {elapsed:.3f}s")
        
        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert elapsed < 2.0, f"Tempo de resposta muito alto: {elapsed:.3f}s (limite: 2s)"
        
        # Verifica header de tempo
        if "X-Response-Time" in response.headers:
            header_time = response.headers["X-Response-Time"]
            print(f"Header X-Response-Time: {header_time}")
    
    runner.test("Tempo de Resposta", run)


# ================== MAIN ==================

def run_all_tests():
    """Executa todos os testes."""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              SUITE DE TESTES - API DE PREDIÇÃO DE DIABETES                ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    runner = TestRunner()
    
    # Verifica se API está rodando
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        print(f"✅ API está rodando em {API_BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f"❌ API não está rodando em {API_BASE_URL}")
        print("   Por favor, inicie a API antes de executar os testes:")
        print("   python flask_api.py")
        return
    
    print("\n🚀 Iniciando testes...\n")
    
    # Testes básicos
    test_health_check(runner)
    test_model_info(runner)
    
    # Testes de predição válida
    test_valid_prediction(runner)
    
    # Testes de validação
    test_missing_features(runner)
    test_invalid_feature_type(runner)
    test_null_feature(runner)
    test_empty_body(runner)
    test_invalid_content_type(runner)
    
    # Testes de batch
    test_batch_prediction_valid(runner)
    test_batch_missing_instances(runner)
    test_batch_empty_instances(runner)
    test_batch_partial_errors(runner)
    
    # Testes de erros HTTP
    test_endpoint_not_found(runner)
    test_method_not_allowed(runner)
    
    # Testes de segurança e performance
    test_response_headers(runner)
    test_response_time(runner)
    
    # Resumo
    runner.summary()


if __name__ == "__main__":
    run_all_tests()