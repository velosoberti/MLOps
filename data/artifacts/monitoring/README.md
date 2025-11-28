# Monitoring Directory

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
