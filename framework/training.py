"""
Funções modulares para treinamento de modelo com Feast e MLflow
"""
import socket
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import matplotlib.pyplot as plt
import pandas as pd


def setup_mlflow_tracking():
    """
    Configura o URI de tracking do MLflow com fallback para Docker
    
    Returns:
        str: URI do MLflow tracking configurado
    """
    try:
        hostname = 'host.docker.internal'
        host_ip = socket.gethostbyname(hostname)
        mlflow_uri = f"http://{host_ip}:5000/"
        print(f"✅ IP resolvido para MLflow: {mlflow_uri}")
    except socket.gaierror:
        print("⚠️ Falha ao resolver host.docker.internal, tentando localhost...")
        mlflow_uri = "http://127.0.0.1:5000/"
    
    mlflow.set_tracking_uri(mlflow_uri)
    return mlflow_uri


def load_data_from_feast(repo_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Carrega dados do Feature Store do Feast
    
    Args:
        repo_path: Caminho do repositório Feast
        dataset_name: Nome do dataset salvo
        
    Returns:
        DataFrame com os dados de treinamento
    """
    print(f"📦 Carregando dados do Feast: {dataset_name}")
    store = FeatureStore(repo_path=repo_path)
    training_data = store.get_saved_dataset(name=dataset_name).to_df()
    print(f"✅ Dados carregados: {training_data.shape}")
    return training_data


def prepare_features(df: pd.DataFrame, target_col: str = "Outcome", 
                     exclude_cols: list = None) -> tuple:
    """
    Prepara features e target para treinamento
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
        exclude_cols: Lista de colunas para excluir
        
    Returns:
        Tuple com (X, y)
    """
    if exclude_cols is None:
        exclude_cols = ["event_timestamp", "patient_id"]
    
    y = df[target_col]
    cols_to_drop = [target_col] + exclude_cols
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    print(f"🔧 Features preparadas: {X.shape[1]} features, {X.shape[0]} amostras")
    print(f"   Features: {sorted(X.columns.tolist())}")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, 
               stratify: bool = True, random_state: int = None) -> tuple:
    """
    Divide dados em treino e teste
    
    Args:
        X: Features
        y: Target
        test_size: Proporção do conjunto de teste
        stratify: Se deve estratificar a divisão
        random_state: Seed para reprodutibilidade
        
    Returns:
        Tuple com (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        stratify=stratify_param, 
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"📊 Dados divididos:")
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")
    print(f"   Proporção positiva (treino): {y_train.sum() / len(y_train):.3f}")
    print(f"   Proporção positiva (teste): {y_test.sum() / len(y_test):.3f}")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                              **model_params) -> LogisticRegression:
    """
    Treina modelo de Regressão Logística
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        **model_params: Parâmetros adicionais para o modelo
        
    Returns:
        Modelo treinado
    """
    print(f"🤖 Treinando Regressão Logística...")
    
    # Ordena colunas para consistência
    X_train_sorted = X_train[sorted(X_train.columns)]
    
    model = LogisticRegression(**model_params)
    model.fit(X_train_sorted, y_train)
    
    print(f"✅ Modelo treinado!")
    return model


def evaluate_model(model: LogisticRegression, X_train: pd.DataFrame, 
                   y_train: pd.Series, X_test: pd.DataFrame, 
                   y_test: pd.Series) -> dict:
    """
    Avalia o modelo e retorna métricas
    
    Args:
        model: Modelo treinado
        X_train: Features de treino
        y_train: Target de treino
        X_test: Features de teste
        y_test: Target de teste
        
    Returns:
        Dicionário com as métricas
    """
    # Ordena colunas
    X_train_sorted = X_train[sorted(X_train.columns)]
    X_test_sorted = X_test[sorted(X_test.columns)]
    
    # Predições
    y_train_pred = model.predict(X_train_sorted)
    y_test_pred = model.predict(X_test_sorted)
    
    # Métricas
    metrics = {
        "acc_train": accuracy_score(y_train, y_train_pred),
        "acc_test": accuracy_score(y_test, y_test_pred)
    }
    
    print(f"📈 Métricas:")
    print(f"   Acurácia (treino): {metrics['acc_train']:.4f}")
    print(f"   Acurácia (teste): {metrics['acc_test']:.4f}")
    
    return metrics, y_test_pred


def save_confusion_matrix(y_true, y_pred, filename: str = "confusion_matrix.png"):
    """
    Salva matriz de confusão como imagem
    
    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos
        filename: Nome do arquivo para salvar
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(filename)
    plt.close()
    print(f"💾 Matriz de confusão salva: {filename}")


def save_feature_list(features: list, filename: str = "features.txt"):
    """
    Salva lista de features em arquivo
    
    Args:
        features: Lista de nomes de features
        filename: Nome do arquivo para salvar
    """
    with open(filename, "w") as f:
        f.write("\n".join(sorted(features)))
    print(f"💾 Lista de features salva: {filename}")


def log_to_mlflow(model, X_train, y_train, X_test, y_test, metrics_dict, 
                  experiment_id: str, run_name: str = "logistic_regression_baseline",
                  tags: dict = None):
    """
    Registra experimento completo no MLflow
    
    Args:
        model: Modelo treinado
        X_train: Features de treino
        y_train: Target de treino
        X_test: Features de teste
        y_test: Target de teste
        metrics_dict: Dicionário com métricas calculadas
        experiment_id: ID do experimento MLflow
        run_name: Nome da run
        tags: Tags adicionais para a run
    """
    mlflow.set_experiment(experiment_id=experiment_id)
    
    default_tags = {
        "model_type": "classification",
        "algorithm": "logistic_regression",
        "dataset": "diabetes_dataset",
        "developer": "data_scientist",
        "environment": "development"
    }
    
    if tags:
        default_tags.update(tags)
    
    with mlflow.start_run(run_name=run_name):
        print(f"📝 Registrando no MLflow...")
        
        # Tags
        mlflow.set_tags(default_tags)
        
        # Autolog do sklearn
        mlflow.sklearn.autolog(
            log_models=True,
            log_input_examples=True,
            log_model_signatures=True,
            log_datasets=False
        )
        
        # Parâmetros do modelo
        params = {
            "penalty": model.penalty,
            "solver": model.solver,
            "max_iter": model.max_iter,
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0],
            "train_positive_ratio": y_train.sum() / len(y_train)
        }
        mlflow.log_params(params)
        
        # Métricas
        mlflow.log_metrics(metrics_dict)
        
        # Artefatos
        X_test_sorted = X_test[sorted(X_test.columns)]
        y_test_pred = model.predict(X_test_sorted)
        
        save_confusion_matrix(y_test, y_test_pred)
        mlflow.log_artifact("confusion_matrix.png")
        
        save_feature_list(X_train.columns.tolist())
        mlflow.log_artifact("features.txt")
        
        print(f"✅ Experimento registrado no MLflow!")


def train_pipeline(repo_path: str, dataset_name: str, experiment_id: str,
                   run_name: str = "logistic_regression_baseline",
                   test_size: float = 0.25, random_state: int = None,
                   model_params: dict = None, tags: dict = None):
    """
    Pipeline completo de treinamento
    
    Args:
        repo_path: Caminho do repositório Feast
        dataset_name: Nome do dataset no Feast
        experiment_id: ID do experimento MLflow
        run_name: Nome da run no MLflow
        test_size: Proporção do conjunto de teste
        random_state: Seed para reprodutibilidade
        model_params: Parâmetros para o modelo
        tags: Tags para o MLflow
    """
    print("=" * 70)
    print("🚀 INICIANDO PIPELINE DE TREINAMENTO")
    print("=" * 70)
    
    # 1. Setup MLflow
    mlflow_uri = setup_mlflow_tracking()
    
    # 2. Carregar dados
    df = load_data_from_feast(repo_path, dataset_name)
    
    # 3. Preparar features
    X, y = prepare_features(df)
    
    # 4. Dividir dados
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 5. Treinar modelo
    if model_params is None:
        model_params = {}
    model = train_logistic_regression(X_train, y_train, **model_params)
    
    # 6. Avaliar modelo
    metrics_dict, _ = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # 7. Registrar no MLflow
    log_to_mlflow(
        model, X_train, y_train, X_test, y_test, 
        metrics_dict, experiment_id, run_name, tags
    )
    
    print("=" * 70)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    
    return model, metrics_dict


# Exemplo de uso
if __name__ == "__main__":
    # Configuração
    REPO_PATH = "/home/luisveloso/MLOps_projects/feature_store/feature_repo"
    DATASET_NAME = "my_training_dataset"
    EXPERIMENT_ID = "467326610704772702"
    
    # Executar pipeline
    model, metrics = train_pipeline(
        repo_path=REPO_PATH,
        dataset_name=DATASET_NAME,
        experiment_id=EXPERIMENT_ID,
        run_name="logistic_regression_baseline",
        test_size=0.25,
        random_state=42,
        tags={"developer": "luis_veloso"}
    )