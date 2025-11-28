import pandas as pd
import os

# Ajuste os caminhos conforme necessário
INPUT_FILE = "/home/luisveloso/MLOps_projects/data/diabetes.csv"
OUTPUT_FILE = "/home/luisveloso/MLOps_projects/data/artifacts/"

# --- REMOVIDA A LINHA: len = pd.read_csv(...) ---

def extract():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    return df

def transform(df):
    predictor = df.loc[:, df.columns != "Outcome"]
    target = df['Outcome']
    return predictor, target

def create_timestamps(predictor, target):
    # Agora len() funciona porque não foi sobrescrito
    timestamps = pd.date_range(
        end=pd.Timestamp.now(), 
        periods=len(predictor), 
        freq='D'
    ).to_frame(name="event_timestamp", index=False)

    predictor = pd.concat([predictor, timestamps], axis=1)
    target = pd.concat([target, timestamps], axis=1)
    return predictor, target

def create_patient_ids(predictor, target):
    dataLen = len(predictor) # Agora funciona
    idsList = list(range(dataLen))
    patient_ids = pd.DataFrame(idsList, columns=['patient_id'])

    predictor = pd.concat([predictor, patient_ids], axis=1)
    target = pd.concat([target, patient_ids], axis=1)
    return predictor, target

def save_parquet(predictor, target):
    # Garante que o diretório existe antes de salvar
    os.makedirs(OUTPUT_FILE, exist_ok=True)
    
    predictor.to_parquet(path=f"{OUTPUT_FILE}/predictor.parquet", engine="pyarrow")
    target.to_parquet(path=f"{OUTPUT_FILE}/target.parquet", engine="pyarrow")
    print(f"Snapshot criado em: {OUTPUT_FILE}")