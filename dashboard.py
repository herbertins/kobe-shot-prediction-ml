import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monitoramento do Modelo - Kobe Shot Prediction", layout="wide")

st.title("Dashboard de Monitoramento do Modelo")

# Carrega os dados de produção e treino
@st.cache_data
def carregar_dados():
    df_producao = pd.read_parquet("data/01_raw/dataset_kobe_prod.parquet")
    df_treino = pd.read_parquet("data/03_primary/base_train.parquet")
    return df_producao, df_treino

df_producao, df_treino = carregar_dados()

st.markdown("###Dimensão dos dados")
col1, col2 = st.columns(2)
with col1:
    st.metric("Dados de Produção", f"{df_producao.shape[0]} linhas")
with col2:
    st.metric("Dados de Treinamento", f"{df_treino.shape[0]} linhas")

st.markdown("---")
st.markdown("###Distribuição de Previsões (Produção)")
col1, col2 = st.columns(2)
with col1:
    pred_dist = df_producao["shot_made_flag"].value_counts().sort_index()
    st.bar_chart(pred_dist.rename({0: "Erro", 1: "Acerto"}))

with col2:
    if "shot_made_flag" in df_producao.columns:
        acuracia = (df_producao["shot_made_flag"] == df_producao["shot_made_flag"]).mean()
        st.metric("Acurácia (Produção)", f"{acuracia * 100:.2f}%")
    else:
        st.info("Variável de resposta não disponível. Exibindo apenas as predições.")

st.markdown("---")
st.markdown("###Distribuição Espacial (lat x lon)")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Treinamento**")
    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=df_treino, x="lon", y="lat", fill=True, cmap="Blues", ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("**Produção**")
    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=df_producao, x="lon", y="lat", fill=True, cmap="Oranges", ax=ax2)
    st.pyplot(fig2)

st.markdown("---")
st.markdown("###Comparação de Distribuição: Produção x Treinamento")

variavel = st.selectbox("Selecione a variável:", ["lat", "lon", "minutes_remaining", "shot_distance"])

fig3, ax3 = plt.subplots()
sns.kdeplot(df_treino[variavel], label="Treinamento", ax=ax3)
sns.kdeplot(df_producao[variavel], label="Produção", ax=ax3)
ax3.set_title(f"Distribuição da variável: {variavel}")
ax3.legend()
st.pyplot(fig3)