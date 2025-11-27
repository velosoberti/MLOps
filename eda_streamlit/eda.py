import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configuração do estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def show_basic_info(df):
    """Exibe informações básicas do dataset"""
    st.subheader("📊 Informações Básicas")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", df.shape[0])
    with col2:
        st.metric("Total de Colunas", df.shape[1])
    with col3:
        st.metric("Memória Utilizada", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.write("**Tipos de Dados:**")
    st.dataframe(pd.DataFrame({
        'Coluna': df.dtypes.index,
        'Tipo': df.dtypes.values,
        'Não-Nulos': df.count().values,
        'Nulos': df.isnull().sum().values
    }), use_container_width=True)

def show_missing_values(df):
    """Analisa e visualiza valores ausentes"""
    st.subheader("🔍 Análise de Valores Ausentes")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    if missing.sum() == 0:
        st.success("✅ Não há valores ausentes no dataset!")
    else:
        missing_df = pd.DataFrame({
            'Coluna': missing.index,
            'Valores Ausentes': missing.values,
            'Percentual (%)': missing_pct.values
        }).sort_values('Valores Ausentes', ascending=False)
        
        missing_df = missing_df[missing_df['Valores Ausentes'] > 0]
        st.dataframe(missing_df, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(missing_df['Coluna'], missing_df['Percentual (%)'])
        ax.set_xlabel('Percentual de Valores Ausentes (%)')
        ax.set_title('Distribuição de Valores Ausentes por Coluna')
        st.pyplot(fig)
        plt.close()

def show_numerical_stats(df, exclude_cols=None):
    """Exibe estatísticas descritivas das variáveis numéricas"""
    st.subheader("📈 Estatísticas Descritivas - Variáveis Numéricas")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().T
        stats['missing'] = df[numeric_cols].isnull().sum()
        stats['missing_pct'] = (stats['missing'] / len(df)) * 100
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
    else:
        st.warning("Não há variáveis numéricas para exibir.")

def plot_distributions(df, exclude_cols=None, bins=30):
    """Plota distribuições das variáveis numéricas"""
    st.subheader("📊 Distribuições das Variáveis Numéricas")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) == 0:
        st.warning("Não há variáveis numéricas para plotar.")
        return
    
    cols_per_row = 2
    n_rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(numeric_cols) == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        df[col].hist(bins=bins, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribuição: {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequência')
        ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Média: {df[col].mean():.2f}')
        ax.axvline(df[col].median(), color='green', linestyle='--', label=f'Mediana: {df[col].median():.2f}')
        ax.legend()
    
    # Remove eixos vazios
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_boxplots(df, exclude_cols=None):
    """Plota boxplots para detecção de outliers"""
    st.subheader("📦 Boxplots - Detecção de Outliers")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) == 0:
        st.warning("Não há variáveis numéricas para plotar.")
        return
    
    cols_per_row = 2
    n_rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(numeric_cols) == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        df.boxplot(column=col, ax=ax)
        ax.set_title(f'Boxplot: {col}')
        ax.set_ylabel(col)
    
    # Remove eixos vazios
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_correlation_matrix(df, exclude_cols=None):
    """Plota matriz de correlação"""
    st.subheader("🔗 Matriz de Correlação")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) < 2:
        st.warning("É necessário pelo menos 2 variáveis numéricas para calcular correlação.")
        return
    
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Matriz de Correlação')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Exibe correlações mais fortes
    st.write("**Correlações mais fortes (em módulo):**")
    corr_flat = corr.abs().unstack()
    corr_flat = corr_flat[corr_flat < 1].sort_values(ascending=False)
    top_corr = corr_flat.head(10)
    
    for idx, value in top_corr.items():
        st.write(f"- {idx[0]} ↔ {idx[1]}: {corr.loc[idx[0], idx[1]]:.3f}")

def analyze_target_variable(df, target_col):
    """Analisa a variável alvo"""
    st.subheader(f"🎯 Análise da Variável Alvo: {target_col}")
    
    if target_col not in df.columns:
        st.error(f"Coluna '{target_col}' não encontrada no dataset!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        value_counts = df[target_col].value_counts()
        st.write("**Distribuição:**")
        st.dataframe(pd.DataFrame({
            'Valor': value_counts.index,
            'Contagem': value_counts.values,
            'Percentual (%)': (value_counts.values / len(df) * 100).round(2)
        }), use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        value_counts.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
        ax.set_title(f'Distribuição de {target_col}')
        ax.set_xlabel(target_col)
        ax.set_ylabel('Contagem')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + max(value_counts.values)*0.01, str(v), 
                   ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def compare_features_by_target(df, target_col, exclude_cols=None):
    """Compara features numéricas agrupadas pela variável alvo"""
    st.subheader(f"⚖️ Comparação de Features por {target_col}")
    
    if target_col not in df.columns:
        st.error(f"Coluna '{target_col}' não encontrada no dataset!")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) == 0:
        st.warning("Não há variáveis numéricas para comparar.")
        return
    
    cols_per_row = 2
    n_rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(numeric_cols) == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        df.boxplot(column=col, by=target_col, ax=ax)
        ax.set_title(f'{col} por {target_col}')
        ax.set_xlabel(target_col)
        ax.set_ylabel(col)
        plt.suptitle('')
    
    # Remove eixos vazios
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def run_complete_eda(df, target_col=None, exclude_cols=None):
    """Executa análise exploratória completa"""
    st.title("🔬 Análise Exploratória de Dados (EDA)")
    
    # Informações básicas
    show_basic_info(df)
    st.divider()
    
    # Valores ausentes
    show_missing_values(df)
    st.divider()
    
    # Estatísticas descritivas
    show_numerical_stats(df, exclude_cols)
    st.divider()
    
    # Distribuições
    plot_distributions(df, exclude_cols)
    st.divider()
    
    # Boxplots
    plot_boxplots(df, exclude_cols)
    st.divider()
    
    # Correlação
    plot_correlation_matrix(df, exclude_cols)
    st.divider()
    
    # Análise da variável alvo
    if target_col:
        analyze_target_variable(df, target_col)
        st.divider()
        compare_features_by_target(df, target_col, exclude_cols)

# Exemplo de uso no Streamlit:
# run_complete_eda(training_data, target_col='Outcome', exclude_cols=['patient_id', 'event_timestamp'])

import streamlit as st
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="/home/luisveloso/MLOps_projects/feature_store/feature_repo")
training_data = store.get_saved_dataset(name="my_training_dataset").to_df()

run_complete_eda(
    training_data, 
    target_col='Outcome',
    exclude_cols=['patient_id', 'event_timestamp']
)
#python eda.py
#streamlit run eda.py