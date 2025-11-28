#!/usr/bin/env python3
"""
Script para criar a estrutura de diretórios de artifacts
"""
import os
from pathlib import Path

# Configuração base
BASE_DIR = "/home/luisveloso/MLOps_projects"
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data/artifacts")

# Estrutura de diretórios
DIRECTORIES = [
    "predictions",
    "monitoring"
]

# Arquivos .gitkeep para manter estrutura no git
GITKEEP_DIRS = [
    "predictions",
    "monitoring"
]

# README para cada pasta
README_CONTENTS = {
    "predictions": """# Predictions Directory

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
""",
    
    "monitoring": """# Monitoring Directory

Este diretório contém estatísticas de monitoramento e detecção de drift.

## Arquivos Principais:
- `monitoring_history.jsonl`: Histórico acumulado de todas as execuções (JSON Lines)
- `monitoring_summary.csv`: Resumo tabular do histórico
- `drift_history.jsonl`: Histórico de detecções de drift

## Arquivos Individuais:
- `monitoring_stats_YYYYMMDD_HHMMSS.json`: Estatísticas de uma execução
- `drift_report_YYYYMMDD_HHMMSS.json`: Relatório de drift de uma execução

## Uso:
Os arquivos `.jsonl` são append-only e contêm todo o histórico de monitoramento.
"""
}


def create_directory_structure():
    """Cria estrutura completa de diretórios"""
    print("=" * 70)
    print("🏗️  CRIANDO ESTRUTURA DE DIRETÓRIOS")
    print("=" * 70)
    
    # Criar diretório base
    artifacts_path = Path(ARTIFACTS_DIR)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Diretório base criado: {ARTIFACTS_DIR}")
    
    # Criar subdiretórios
    for directory in DIRECTORIES:
        dir_path = artifacts_path / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Criado: {directory}/")
        
        # Adicionar .gitkeep
        if directory in GITKEEP_DIRS:
            gitkeep_path = dir_path / ".gitkeep"
            gitkeep_path.touch()
            print(f"   📌 Adicionado .gitkeep")
        
        # Adicionar README
        if directory in README_CONTENTS:
            readme_path = dir_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(README_CONTENTS[directory])
            print(f"   📄 Adicionado README.md")
    
    print("\n" + "=" * 70)
    print("✅ ESTRUTURA CRIADA COM SUCESSO!")
    print("=" * 70)


def create_gitignore():
    """Cria arquivo .gitignore na pasta artifacts"""
    print("\n📝 Criando .gitignore...")
    
    gitignore_content = """# =============================================================================
# .gitignore para data/artifacts
# =============================================================================

# ============================
# ARQUIVOS GRANDES DE PREDIÇÕES
# ============================
predictions/*.parquet
predictions/*.csv
predictions_history.parquet
predictions_history.csv

# ============================
# MONITORAMENTO (Manter históricos)
# ============================
monitoring/monitoring_stats_*.json
# Históricos acumulados SÃO versionados:
!monitoring/monitoring_history.jsonl
!monitoring/monitoring_summary.csv
!monitoring/drift_history.jsonl



# ============================
# TEMPORÁRIOS
# ============================
*.tmp
*.temp
*.bak
*~
.DS_Store
Thumbs.db

# ============================
# MANTER ESTRUTURA
# ============================
!predictions/.gitkeep
!monitoring/.gitkeep
!.gitkeep
"""
    
    gitignore_path = Path(ARTIFACTS_DIR) / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    print(f"✅ .gitignore criado: {gitignore_path}")


def create_main_readme():
    """Cria README principal da pasta artifacts"""
    print("\n📄 Criando README principal...")
    
    readme_content = """# 📦 Artifacts - Histórico de Predições e Monitoramento

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
"""
    
    readme_path = Path(ARTIFACTS_DIR) / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ README principal criado: {readme_path}")


def verify_structure():
    """Verifica se a estrutura foi criada corretamente"""
    print("\n🔍 Verificando estrutura...")
    
    artifacts_path = Path(ARTIFACTS_DIR)
    all_good = True
    
    # Verificar diretórios
    for directory in DIRECTORIES:
        dir_path = artifacts_path / directory
        if dir_path.exists():
            print(f"✅ {directory}/ existe")
        else:
            print(f"❌ {directory}/ NÃO existe")
            all_good = False
    
    # Verificar .gitignore
    gitignore_path = artifacts_path / ".gitignore"
    if gitignore_path.exists():
        print(f"✅ .gitignore existe")
    else:
        print(f"❌ .gitignore NÃO existe")
        all_good = False
    
    if all_good:
        print("\n✅ Estrutura verificada e completa!")
    else:
        print("\n⚠️ Alguns arquivos/diretórios estão faltando")
    
    return all_good


def show_next_steps():
    """Mostra próximos passos"""
    print("\n" + "=" * 70)
    print("📋 PRÓXIMOS PASSOS")
    print("=" * 70)
    print("""
1. Execute o pipeline de predição:
   airflow dags trigger ml_prediction_monitoring_pipeline_v2

2. Verifique os arquivos gerados:
   ls -lh data/artifacts/predictions/
   ls -lh data/artifacts/monitoring/

3. Após algumas execuções, analise o histórico:
   python history_analysis.py

4. Visualize os resultados:
   ls data/artifacts/analysis/

5. Adicione ao Git (se desejado):
   git add data/artifacts/
   git commit -m "Add artifacts structure"

📚 Documentação completa em: data/artifacts/README.md
""")


def main():
    """Função principal"""
    create_directory_structure()
    create_gitignore()
    create_main_readme()
    
    if verify_structure():
        show_next_steps()


if __name__ == "__main__":
    main()