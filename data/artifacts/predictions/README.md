# Predictions Directory

Este diretório contém as predições individuais de cada execução do pipeline.

## Arquivos:
- `predictions_YYYYMMDD_HHMMSS.parquet`: Predições de uma execução específica (formato Parquet)
- `predictions_YYYYMMDD_HHMMSS.csv`: Predições de uma execução específica (formato CSV)

## Estrutura:
Cada arquivo contém:
- patient_id
- Features (BMI, DiabetesPedigreeFunction, Insulin, SkinThickness)
- prediction (0 ou 1)
- probability_class_0, probability_class_1
- prediction_timestamp
- batch_id

## Retenção:
Arquivos individuais são mantidos por 30 dias. Dados completos estão em `predictions_history.parquet`.
