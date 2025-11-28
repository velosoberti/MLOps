from datetime import timedelta, datetime
import socket
import pandas as pd
from feast import FeatureStore
import mlflow
import os
from pathlib import Path

BASE_PATH = "/home/luisveloso/MLOps_projects"

hostname = 'host.docker.internal'
host_ip = socket.gethostbyname(hostname)
MLFLOW_TRACKING_URI = f"http://{host_ip}:5000/"


FEAST_REPO_PATH = f"{BASE_PATH}/feature_store/feature_repo"
FEATURE_VIEW = "predictors_df_feature_view"
MODEL_NAME = "diabete_model"
N_PATIENTS = 50

# Diretório de saída para predições
OUTPUT_DIR = f"{BASE_PATH}/data/artifacts/predictions"

# Caminho para o histórico acumulado de predições
HISTORICAL_PREDICTIONS_FILE = f"{BASE_PATH}/data/artifacts/predictions/predictions_history.parquet"


def setup_and_materialize_features(**context):
    """Task 1: Setup MLflow e materializa features do Feast"""
    print("Configurando MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    print("Materializando features no Feast...")
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    store.materialize_incremental(end_date=datetime.now())
    
    context['ti'].xcom_push(key='mlflow_uri', value=MLFLOW_TRACKING_URI)
    context['ti'].xcom_push(key='materialization_timestamp', value=datetime.now().isoformat())
    
    print("MLflow configurado e features materializadas!")


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
        raise ValueError("Nenhum paciente válido encontrado!")
    
    # Ordenar em ordem decrescente (mais recentes primeiro)
    valid_patient_ids = sorted(valid_patient_ids, reverse=True)
    
    print(f"Encontrados {len(valid_patient_ids)} pacientes válidos")
    print(f"   IDs: {valid_patient_ids[:10]}... (primeiros 10)")
    
    # Salvar IDs
    context['ti'].xcom_push(key='valid_patient_ids', value=valid_patient_ids)
    context['ti'].xcom_push(key='n_valid_patients', value=len(valid_patient_ids))


def fetch_features(**context):
    """Task 3: Busca features dos pacientes válidos"""
    print("Buscando features dos pacientes...")
    
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

    print(f"Features carregadas: {features_df.shape}")
    print(f"   Colunas: {features_df.columns.tolist()}")
    
    # Salvar features
    features_df.to_parquet('/tmp/features_for_prediction.parquet', index=False)
    
    context['ti'].xcom_push(key='features_shape', value=features_df.shape)


def load_model(**context):
    """Task 4: Carrega última versão do modelo"""
    print(f"Carregando modelo '{MODEL_NAME}'...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.client.MlflowClient()
    
    latest_versions = client.get_latest_versions(MODEL_NAME)
    
    if not latest_versions:
        raise ValueError(f"Nenhuma versão encontrada para o modelo '{MODEL_NAME}'")
    
    latest_version = max([int(v.version) for v in latest_versions])
    
    print(f"Versão encontrada: {latest_version}")
    
    model_uri = f"models:/{MODEL_NAME}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Extrair feature names do modelo
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_.tolist()
        print(f"Features do modelo: {model_features}")
        context['ti'].xcom_push(key='model_features', value=model_features)
    else:
        print("Modelo não possui feature_names_in_, usando features padrão")
    
    # Salvar modelo temporariamente
    import joblib
    joblib.dump(model, '/tmp/loaded_model.pkl')
    
    print("Modelo carregado!")
    
    context['ti'].xcom_push(key='model_version', value=latest_version)
    context['ti'].xcom_push(key='model_uri', value=model_uri)


def make_predictions(**context):
    """Task 5: Faz predições"""
    print("Fazendo predições...")
    
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
        print(f"Usando features do modelo: {model_features}")
    else:
        # Fallback: ordenar alfabeticamente
        X = X[sorted(X.columns)]
        print(f"Usando features ordenadas: {sorted(X.columns)}")
    
    # Predições
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Criar DataFrame de resultados
    results_df = features_df.copy()
    results_df['prediction'] = predictions
    results_df['probability_class_0'] = probabilities[:, 0]
    results_df['probability_class_1'] = probabilities[:, 1]
    results_df['prediction_timestamp'] = datetime.now()
    
    print(f"   Predições concluídas!")
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
    print("Salvando predições...")
    
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
    
    print(f"   Predições individuais salvas:")
    print(f"   Parquet: {filepath}")
    print(f"   CSV: {csv_filepath}")
    
    # === ACUMULAR NO HISTÓRICO ===
    if os.path.exists(HISTORICAL_PREDICTIONS_FILE):
        # Carregar histórico existente
        historical_df = pd.read_parquet(HISTORICAL_PREDICTIONS_FILE)
        print(f"Histórico encontrado: {len(historical_df)} predições anteriores")
        
        # Concatenar com novas predições
        updated_history = pd.concat([historical_df, predictions_df], ignore_index=True)
    else:
        # Criar novo histórico
        print(f"Criando novo arquivo de histórico")
        updated_history = predictions_df
    
    # Salvar histórico atualizado
    updated_history.to_parquet(HISTORICAL_PREDICTIONS_FILE, index=False)
    print(f"Histórico atualizado: {len(updated_history)} predições totais")
    
    # Também salvar CSV do histórico
    historical_csv = HISTORICAL_PREDICTIONS_FILE.replace('.parquet', '.csv')
    updated_history.to_csv(historical_csv, index=False)
    
    # === ESTATÍSTICAS DO HISTÓRICO ===
    print(f"\n Estatísticas do Histórico:")
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
    print("Limpando arquivos temporários...")
    
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
            print(f"   Erro ao remover {file_path}: {e}")
    
    print("Limpeza concluída!")