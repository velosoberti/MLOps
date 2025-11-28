# 📦 Artifacts - Histórico de Predições e Monitoramento

Este diretório contém todos os artefatos gerados pelo pipeline de predição e monitoramento.

## 📁 Estrutura

```
artifacts/
├── predictions/              # Predições individuais por batch
├── predictions_history.*     # Histórico ACUMULADO de todas as predições
├── monitoring/              # Estatísticas e drift detection
└── analysis/                # Análises e visualizações
```

## 🔑 Arquivos Principais

### Histórico de Predições
- **`predictions_history.parquet`**: Todas as predições acumuladas (formato eficiente)
- **`predictions_history.csv`**: Mesmas predições em formato CSV

### Histórico de Monitoramento
- **`monitoring/monitoring_history.jsonl`**: Estatísticas de todas as execuções
- **`monitoring/monitoring_summary.csv`**: Resumo tabular
- **`monitoring/drift_history.jsonl`**: Histórico de detecções de drift

## 📊 Como Usar

### Carregar histórico completo:
```python
import pandas as pd

# Todas as predições
df = pd.read_parquet('predictions_history.parquet')

# Filtrar período
df = df[df['prediction_timestamp'] > '2025-01-01']

# Estatísticas por batch
stats = df.groupby('batch_id')['prediction'].mean()
```

### Analisar monitoramento:
```python
import json

# Carregar histórico
monitoring = []
with open('monitoring/monitoring_history.jsonl', 'r') as f:
    for line in f:
        monitoring.append(json.loads(line))

# Converter para DataFrame
df = pd.DataFrame(monitoring)
```

### Gerar análises e gráficos:
```bash
python history_analysis.py
```

## 📈 Retenção de Dados

- **Históricos acumulados** (`.parquet`, `.jsonl`): Mantidos indefinidamente
- **Arquivos individuais**: Últimos 30 dias (auditoria)
- **Análises**: Regeneradas conforme necessário

## 🔄 Versionamento

- Históricos `.jsonl` e `.csv`: **Versionados no Git** (pequenos)
- Predições `.parquet`: **NÃO versionados** (grandes, use DVC se necessário)
- Gráficos: **NÃO versionados** (regeráveis)

## 📝 Notas

1. Nunca deletar arquivos `*_history.*`
2. Fazer backup regular dos históricos
3. Usar DVC para versionar dados grandes (opcional)
4. Monitorar crescimento dos arquivos

---

Gerado automaticamente pelo pipeline de MLOps.
