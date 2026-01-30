"""Training module for ML pipeline.

This module contains functions for training machine learning models using data from Feast
feature store and logging experiments to MLflow.
"""

import os
import socket
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from feast import FeatureStore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

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


def setup_mlflow(**context: Any) -> None:
    """Configure MLflow tracking for the training pipeline.

    Sets up MLflow tracking URI and experiment ID, then pushes configuration
    to Airflow XCom for downstream tasks.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Configurando MLflow...")
    tracking_uri = _resolve_mlflow_tracking_uri()
    experiment_id = settings.mlflow_experiment_id

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)

    context["ti"].xcom_push(key="mlflow_uri", value=tracking_uri)
    context["ti"].xcom_push(key="experiment_id", value=experiment_id)

    print(f"MLflow configurado: {tracking_uri}")


def load_data_from_feast(**context: Any) -> None:
    """Load training data from Feast feature store.

    Retrieves the saved dataset from Feast and saves it to a temporary
    parquet file for downstream processing.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Carregando dados do Feast...")

    store = FeatureStore(repo_path=settings.feast_repo_path)
    training_data = store.get_saved_dataset(name=settings.feast_dataset_name).to_df()

    print(f"Dados carregados: {training_data.shape}")
    print(f"   Colunas: {training_data.columns.tolist()}")

    context["ti"].xcom_push(key="data_shape", value=training_data.shape)
    context["ti"].xcom_push(key="data_path", value="/tmp/training_data.parquet")

    training_data.to_parquet("/tmp/training_data.parquet", index=False)


def prepare_features(
    training_data: pd.DataFrame, target_column: str = "Outcome", columns_to_drop: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from training data.

    Args:
        training_data: Raw training DataFrame.
        target_column: Name of the target column.
        columns_to_drop: Additional columns to drop from features.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    if columns_to_drop is None:
        columns_to_drop = ["event_timestamp", "patient_id"]

    y = training_data[target_column]
    drop_cols = [target_column] + [c for c in columns_to_drop if c in training_data.columns]
    X = training_data.drop(columns=drop_cols)

    return X, y


def prepare_and_split_data(**context: Any) -> None:
    """Prepare features and split data into train/test sets.

    Loads data from temporary file, prepares features, and splits into
    training and test sets with stratification.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("🔧 Preparando features e dividindo dados...")

    training_data = pd.read_parquet("/tmp/training_data.parquet")

    X, y = prepare_features(training_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    print("📊 Dados divididos:")
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")

    X_train.to_parquet("/tmp/X_train.parquet", index=False)
    X_test.to_parquet("/tmp/X_test.parquet", index=False)
    y_train.to_frame().to_parquet("/tmp/y_train.parquet", index=False)
    y_test.to_frame().to_parquet("/tmp/y_test.parquet", index=False)

    context["ti"].xcom_push(key="n_features", value=X_train.shape[1])
    context["ti"].xcom_push(key="n_train_samples", value=X_train.shape[0])
    context["ti"].xcom_push(key="n_test_samples", value=X_test.shape[0])
    context["ti"].xcom_push(key="train_positive_ratio", value=float(y_train.sum() / len(y_train)))
    context["ti"].xcom_push(key="feature_names", value=sorted(X_train.columns.tolist()))


def train_model(**context: Any) -> None:
    """Train a logistic regression model.

    Loads training data from temporary files, trains a LogisticRegression model,
    and saves the model to a temporary file.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("🤖 Treinando modelo...")

    X_train = pd.read_parquet("/tmp/X_train.parquet")
    y_train = pd.read_parquet("/tmp/y_train.parquet")["Outcome"]

    X_train_sorted = X_train[sorted(X_train.columns)]

    model = LogisticRegression()
    model.fit(X_train_sorted, y_train)

    print("Modelo treinado!")

    joblib.dump(model, "/tmp/model.pkl")

    context["ti"].xcom_push(key="model_penalty", value=model.penalty)
    context["ti"].xcom_push(key="model_solver", value=model.solver)
    context["ti"].xcom_push(key="model_max_iter", value=model.max_iter)


def calculate_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate accuracy score between true and predicted values.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Accuracy score as a float.
    """
    return float(accuracy_score(y_true, y_pred))


def evaluate_model(**context: Any) -> None:
    """Evaluate the trained model on train and test sets.

    Loads the model and data from temporary files, makes predictions,
    and calculates accuracy metrics.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Avaliando modelo...")

    model = joblib.load("/tmp/model.pkl")
    X_train = pd.read_parquet("/tmp/X_train.parquet")
    y_train = pd.read_parquet("/tmp/y_train.parquet")["Outcome"]
    X_test = pd.read_parquet("/tmp/X_test.parquet")
    y_test = pd.read_parquet("/tmp/y_test.parquet")["Outcome"]

    X_train_sorted = X_train[sorted(X_train.columns)]
    X_test_sorted = X_test[sorted(X_test.columns)]

    y_train_pred = model.predict(X_train_sorted)
    y_test_pred = model.predict(X_test_sorted)

    acc_train = calculate_accuracy(y_train, y_train_pred)
    acc_test = calculate_accuracy(y_test, y_test_pred)

    print(f"   Acurácia (treino): {acc_train:.4f}")
    print(f"   Acurácia (teste): {acc_test:.4f}")

    pd.DataFrame({"y_pred": y_test_pred}).to_parquet("/tmp/y_test_pred.parquet", index=False)

    context["ti"].xcom_push(key="acc_train", value=acc_train)
    context["ti"].xcom_push(key="acc_test", value=acc_test)


def create_artifacts(**context: Any) -> None:
    """Create training artifacts including confusion matrix and feature list.

    Generates a confusion matrix plot and saves the list of features used
    for training.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("🎨 Criando artefatos...")

    y_test = pd.read_parquet("/tmp/y_test.parquet")["Outcome"]
    y_test_pred = pd.read_parquet("/tmp/y_test_pred.parquet")["y_pred"]
    feature_names: list[str] = context["ti"].xcom_pull(key="feature_names", task_ids="prepare_and_split_data")

    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("/tmp/confusion_matrix.png")
    plt.close()
    print("Matriz de confusão salva")

    with open("/tmp/features.txt", "w") as f:
        f.write("\n".join(feature_names))
    print("Lista de features salva")


def log_to_mlflow(**context: Any) -> None:
    """Log training run to MLflow.

    Records parameters, metrics, artifacts, and the trained model to MLflow
    for experiment tracking.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Registrando experimento no MLflow...")

    tracking_uri = _resolve_mlflow_tracking_uri()
    experiment_id = settings.mlflow_experiment_id

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)

    model = joblib.load("/tmp/model.pkl")

    ti = context["ti"]

    with mlflow.start_run(run_name="logistic_regression_airflow"):
        mlflow.set_tags(
            {
                "model_type": "classification",
                "algorithm": "logistic_regression",
                "dataset": "diabetes_dataset",
                "developer": "airflow_pipeline",
                "environment": "production",
                "orchestrator": "airflow",
            }
        )

        mlflow.sklearn.autolog(log_models=True, log_input_examples=True, log_model_signatures=True, log_datasets=False)

        params: dict[str, Any] = {
            "penalty": ti.xcom_pull(key="model_penalty", task_ids="train_model"),
            "solver": ti.xcom_pull(key="model_solver", task_ids="train_model"),
            "max_iter": ti.xcom_pull(key="model_max_iter", task_ids="train_model"),
            "n_features": ti.xcom_pull(key="n_features", task_ids="prepare_and_split_data"),
            "n_train_samples": ti.xcom_pull(key="n_train_samples", task_ids="prepare_and_split_data"),
            "n_test_samples": ti.xcom_pull(key="n_test_samples", task_ids="prepare_and_split_data"),
            "train_positive_ratio": ti.xcom_pull(key="train_positive_ratio", task_ids="prepare_and_split_data"),
        }
        mlflow.log_params(params)

        metrics: dict[str, float] = {
            "acc_train": ti.xcom_pull(key="acc_train", task_ids="evaluate_model"),
            "acc_test": ti.xcom_pull(key="acc_test", task_ids="evaluate_model"),
        }
        mlflow.log_metrics(metrics)

        mlflow.log_artifact("/tmp/confusion_matrix.png")
        mlflow.log_artifact("/tmp/features.txt")

        mlflow.sklearn.log_model(model, "model")

        print("Experimento registrado no MLflow!")


def cleanup_temp_files(**context: Any) -> None:
    """Clean up temporary files created during training.

    Removes all temporary parquet files, model files, and artifacts
    created during the training pipeline.

    Args:
        **context: Airflow task context containing task instance for XCom.
    """
    print("Limpando arquivos temporários...")

    temp_files = [
        "/tmp/training_data.parquet",
        "/tmp/X_train.parquet",
        "/tmp/X_test.parquet",
        "/tmp/y_train.parquet",
        "/tmp/y_test.parquet",
        "/tmp/y_test_pred.parquet",
        "/tmp/model.pkl",
        "/tmp/confusion_matrix.png",
        "/tmp/features.txt",
    ]

    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   Removido: {file_path}")
        except Exception as e:
            print(f"   Erro ao remover {file_path}: {e}")

    print(" Limpeza concluída!")
