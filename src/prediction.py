"""Prediction module for ML pipeline.

This module contains functions for making predictions using trained models
from MLflow and features from Feast feature store.
"""

import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from feast import FeatureStore

import mlflow
from config.settings import settings


def _resolve_mlflow_tracking_uri() -> str:
    """Resolve MLflow tracking URI, attempting Docker host resolution first.

    Returns:
        str: The resolved MLflow tracking URI.
    """
    if settings.mlflow_tracking_uri:
        return settings.mlflow_tracking_uri

    try:
        hostname = "host.docker.internal"
        host_ip = socket.gethostbyname(hostname)
        uri = f"http://{host_ip}:5000/"
        print(f"IP resolvido para MLflow: {uri}")
        return uri
    except socket.gaierror:
        print("Falha ao resolver host.docker.internal, tentando localhost...")
        return "http://127.0.0.1:5000/"


def setup_and_materialize_features(**context: Any) -> None:
    """Set up MLflow and materialize features from Feast.

    Configures MLflow tracking and triggers incremental materialization
    of features in the Feast feature store.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Configurando MLflow...")
    tracking_uri = _resolve_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    print("Materializando features no Feast...")
    store = FeatureStore(repo_path=settings.feast_repo_path)
    store.materialize_incremental(end_date=datetime.now())

    context["ti"].xcom_push(key="mlflow_uri", value=tracking_uri)
    context["ti"].xcom_push(key="materialization_timestamp", value=datetime.now().isoformat())

    print("MLflow configurado e features materializadas!")


def get_feast_features() -> list[str]:
    """Get the list of Feast feature names for prediction.

    Returns:
        List of fully qualified feature names.
    """
    feature_view = settings.feast_feature_view
    return [
        f"{feature_view}:DiabetesPedigreeFunction",
        f"{feature_view}:BMI",
        f"{feature_view}:SkinThickness",
        f"{feature_view}:Insulin",
    ]


def find_valid_patient_ids(**context: Any) -> None:
    """Find valid patient IDs that have complete feature data.

    Searches for patients with non-null feature values in the Feast
    online store, starting from the most recent IDs.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    n_patients = settings.n_patients
    print(f"🔍 Buscando os últimos {n_patients} pacientes válidos...")

    store = FeatureStore(repo_path=settings.feast_repo_path)
    feast_features = get_feast_features()

    valid_patient_ids: list[int] = []

    for patient_id in range(1000, 0, -1):
        if len(valid_patient_ids) >= n_patients:
            break

        try:
            features = store.get_online_features(
                features=feast_features, entity_rows=[{"patient_id": patient_id}]
            ).to_dict()

            features_df = pd.DataFrame.from_dict(features)

            if not features_df.drop(columns=["patient_id"]).isna().any().any():
                valid_patient_ids.append(patient_id)

        except Exception:
            continue

    if not valid_patient_ids:
        raise ValueError("Nenhum paciente válido encontrado!")

    valid_patient_ids = sorted(valid_patient_ids, reverse=True)

    print(f"Encontrados {len(valid_patient_ids)} pacientes válidos")
    print(f"   IDs: {valid_patient_ids[:10]}... (primeiros 10)")

    context["ti"].xcom_push(key="valid_patient_ids", value=valid_patient_ids)
    context["ti"].xcom_push(key="n_valid_patients", value=len(valid_patient_ids))


def fetch_features(**context: Any) -> None:
    """Fetch features for valid patients from Feast.

    Retrieves online features for all valid patient IDs and saves
    them to a temporary parquet file.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Buscando features dos pacientes...")

    patient_ids: list[int] = context["ti"].xcom_pull(key="valid_patient_ids", task_ids="find_valid_patient_ids")

    store = FeatureStore(repo_path=settings.feast_repo_path)
    feast_features = get_feast_features()

    entity_rows = [{"patient_id": pid} for pid in patient_ids]

    features = store.get_online_features(features=feast_features, entity_rows=entity_rows).to_dict()

    features_df = pd.DataFrame.from_dict(features)

    print(f"Features carregadas: {features_df.shape}")
    print(f"   Colunas: {features_df.columns.tolist()}")

    features_df.to_parquet("/tmp/features_for_prediction.parquet", index=False)

    context["ti"].xcom_push(key="features_shape", value=features_df.shape)


def load_model(**context: Any) -> None:
    """Load the latest version of the model from MLflow.

    Retrieves the latest registered model version and saves it
    to a temporary file for prediction.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    model_name = settings.model_name
    print(f"Carregando modelo '{model_name}'...")

    tracking_uri = _resolve_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient()

    latest_versions = client.get_latest_versions(model_name)

    if not latest_versions:
        raise ValueError(f"Nenhuma versão encontrada para o modelo '{model_name}'")

    latest_version = max([int(v.version) for v in latest_versions])

    print(f"Versão encontrada: {latest_version}")

    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)

    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_.tolist()
        print(f"Features do modelo: {model_features}")
        context["ti"].xcom_push(key="model_features", value=model_features)
    else:
        print("Modelo não possui feature_names_in_, usando features padrão")

    joblib.dump(model, "/tmp/loaded_model.pkl")

    print("Modelo carregado!")

    context["ti"].xcom_push(key="model_version", value=latest_version)
    context["ti"].xcom_push(key="model_uri", value=model_uri)


def prepare_features_for_prediction(features_df: pd.DataFrame, model_features: list[str] | None = None) -> pd.DataFrame:
    """Prepare features DataFrame for model prediction.

    Args:
        features_df: DataFrame with raw features including patient_id.
        model_features: Optional list of feature names expected by the model.

    Returns:
        DataFrame with features ready for prediction.

    Raises:
        ValueError: If required features are missing.
    """
    X = features_df.drop(columns=["patient_id"])

    if model_features:
        missing_features = set(model_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"❌ Features faltando: {missing_features}")
        X = X[model_features]
    else:
        X = X[sorted(X.columns)]

    return X


def make_predictions(**context: Any) -> None:
    """Make predictions using the loaded model.

    Loads features and model from temporary files, makes predictions,
    and saves results with probabilities.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Fazendo predições...")

    model = joblib.load("/tmp/loaded_model.pkl")
    features_df = pd.read_parquet("/tmp/features_for_prediction.parquet")

    ti = context["ti"]
    model_features: list[str] | None = ti.xcom_pull(key="model_features", task_ids="load_model")

    X = prepare_features_for_prediction(features_df, model_features)

    if model_features:
        print(f"Usando features do modelo: {model_features}")
    else:
        print(f"Usando features ordenadas: {sorted(X.columns.tolist())}")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results_df = features_df.copy()
    results_df["prediction"] = predictions
    results_df["probability_class_0"] = probabilities[:, 0]
    results_df["probability_class_1"] = probabilities[:, 1]
    results_df["prediction_timestamp"] = datetime.now()

    print("   Predições concluídas!")
    print(f"   Total: {len(predictions)}")
    print(f"   Classe 0 (Não diabético): {sum(predictions == 0)}")
    print(f"   Classe 1 (Diabético): {sum(predictions == 1)}")

    results_df.to_parquet("/tmp/predictions_results.parquet", index=False)

    context["ti"].xcom_push(key="n_predictions", value=len(predictions))
    context["ti"].xcom_push(key="n_class_0", value=int(sum(predictions == 0)))
    context["ti"].xcom_push(key="n_class_1", value=int(sum(predictions == 1)))


def save_predictions(**context: Any) -> None:
    """Save predictions to permanent storage and update history.

    Saves predictions to individual files and appends to the historical
    predictions file for tracking over time.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Salvando predições...")

    output_dir = settings.get_predictions_output_dir()
    historical_file = settings.get_historical_predictions_file()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(historical_file)).mkdir(parents=True, exist_ok=True)

    predictions_df = pd.read_parquet("/tmp/predictions_results.parquet")

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_df["batch_id"] = batch_id

    ti = context["ti"]
    model_version: int = ti.xcom_pull(key="model_version", task_ids="load_model")
    predictions_df["model_version"] = model_version

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.parquet"
    filepath = os.path.join(output_dir, filename)

    predictions_df.to_parquet(filepath, index=False)

    csv_filepath = filepath.replace(".parquet", ".csv")
    predictions_df.to_csv(csv_filepath, index=False)

    print("   Predições individuais salvas:")
    print(f"   Parquet: {filepath}")
    print(f"   CSV: {csv_filepath}")

    if os.path.exists(historical_file):
        historical_df = pd.read_parquet(historical_file)
        print(f"Histórico encontrado: {len(historical_df)} predições anteriores")
        updated_history = pd.concat([historical_df, predictions_df], ignore_index=True)
    else:
        print("Criando novo arquivo de histórico")
        updated_history = predictions_df

    updated_history.to_parquet(historical_file, index=False)
    print(f"Histórico atualizado: {len(updated_history)} predições totais")

    historical_csv = historical_file.replace(".parquet", ".csv")
    updated_history.to_csv(historical_csv, index=False)

    print("\n Estatísticas do Histórico:")
    print(f"   Total de predições: {len(updated_history)}")
    print(f"   Total de batches: {updated_history['batch_id'].nunique()}")
    print(f"   Primeira predição: {updated_history['prediction_timestamp'].min()}")
    print(f"   Última predição: {updated_history['prediction_timestamp'].max()}")
    print(f"   Proporção classe 1 (histórico): {updated_history['prediction'].mean():.3f}")

    context["ti"].xcom_push(key="predictions_file", value=filepath)
    context["ti"].xcom_push(key="predictions_csv_file", value=csv_filepath)
    context["ti"].xcom_push(key="historical_file", value=historical_file)
    context["ti"].xcom_push(key="total_historical_predictions", value=len(updated_history))


def cleanup_temp_files(**context: Any) -> None:
    """Clean up temporary files created during prediction.

    Removes all temporary parquet files and model files created
    during the prediction pipeline.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Limpando arquivos temporários...")

    temp_files = ["/tmp/features_for_prediction.parquet", "/tmp/loaded_model.pkl", "/tmp/predictions_results.parquet"]

    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   Removido: {file_path}")
        except Exception as e:
            print(f"   Erro ao remover {file_path}: {e}")

    print("Limpeza concluída!")
