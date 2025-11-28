from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
import os

# Caminhos (ajustados para o ambiente Docker/Airflow se necessário)
REPO_PATH = "/home/luisveloso/MLOps_projects/feature_store/feature_repo/"
ENTITY_PATH = "/home/luisveloso/MLOps_projects/data/artifacts/target.parquet"
DATASET_OUTPUT_PATH = "/home/luisveloso/MLOps_projects/feature_store/data/my_training_dataset2.parquet"

@dag(
    dag_id="feature_store_cre",
    schedule_interval="@daily",
    start_date=datetime(2023, 8, 1),
    catchup=False,
    tags=['mlops', 'feast']
)
def feature_store_dag():

    @task()
    def create_dataset_task():
        store = FeatureStore(repo_path=REPO_PATH)
            
        print(f"Lendo entidades de: {ENTITY_PATH}")
        entity_df = pd.read_parquet(ENTITY_PATH)

        print("Recuperando features históricas...")
            # 1. Obtenha o JOB (não converta para DF ainda)
        retrieval_job = store.get_historical_features(
                entity_df=entity_df,
                features=[
                    "predictors_df_feature_view:DiabetesPedigreeFunction",
                    "predictors_df_feature_view:BMI",
                    "predictors_df_feature_view:SkinThickness",
                    "predictors_df_feature_view:Insulin",
                    "predictors_df_feature_view:Glucose"
                ],
            )

            # Opcional: Se você quiser ver o shape no log, pode converter uma cópia,
            # mas NÃO use essa cópia no create_saved_dataset
            # print(f"Preview: {retrieval_job.to_df().head()}") 

            # 2. Criar o diretório de saída
        os.makedirs(os.path.dirname(DATASET_OUTPUT_PATH), exist_ok=True)

        print("Persistindo Dataset via Feast...")
        store.create_saved_dataset(
                from_=retrieval_job,  # <--- O Segredo está aqui
                name="my_training_dataset-2",
                storage=SavedDatasetFileStorage(
                    path=DATASET_OUTPUT_PATH
                )
            )
            
        print(f"Dataset salvo e registrado em: {DATASET_OUTPUT_PATH}")
        return DATASET_OUTPUT_PATH

    # Execução
    create_dataset_task()

fs_dag = feature_store_dag()