from airflow.decorators import dag, task
import pandas as pd
from etl.etl import extract, transform, create_timestamps, create_patient_ids, save_parquet

@dag(dag_id="etl_pipeline_final", schedule_interval="@daily", start_date=pd.Timestamp(2023, 8, 1), catchup=False)
def etl_flow():

    @task()
    def task_extract():
        return extract()

    @task(multiple_outputs=True)
    def task_transform(df):
        # Desempacota a tupla que vem do etl.py
        p, t = transform(df)
        # Retorna um dict para o Airflow gerenciar as saídas
        return {"predictor": p, "target": t}

    @task(multiple_outputs=True)
    def task_timestamps(predictor, target):
        p, t = create_timestamps(predictor, target)
        return {"predictor": p, "target": t}

    @task(multiple_outputs=True)
    def task_patient_ids(predictor, target):
        p, t = create_patient_ids(predictor, target)
        return {"predictor": p, "target": t}

    @task()
    def task_save(predictor, target):
        save_parquet(predictor, target)

    # --- DEFINIÇÃO DO FLUXO ---

    # 1. Extrair
    df = task_extract()

    # 2. Transformar (Saída é um dict)
    data_transformed = task_transform(df)

    # 3. Criar Timestamps (Usa as chaves do dict anterior)
    data_with_time = task_timestamps(
        predictor=data_transformed['predictor'],
        target=data_transformed['target']
    )

    # 4. Criar IDs
    data_final = task_patient_ids(
        predictor=data_with_time['predictor'],
        target=data_with_time['target']
    )

    # 5. Salvar
    task_save(
        predictor=data_final['predictor'],
        target=data_final['target']
    )

dag_instance = etl_flow()