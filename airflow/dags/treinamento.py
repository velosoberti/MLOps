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