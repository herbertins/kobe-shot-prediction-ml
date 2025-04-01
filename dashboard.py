import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título
st.title("📊 Kobe Shot Predictor")

# Carrega os dados preditos na produção
try:
    df = pd.read_parquet("data/07_model_output/predictions_prod.parquet")

    # Métricas gerais
    total = len(df)
    acertos = df["prediction_label"].sum()
    erros = total - acertos
    taxa_acerto = acertos / total * 100

    st.metric("Total de Previsões", total)
    st.metric("Acertos (Previsões de Cesta)", acertos)
    st.metric("Erros (Previsões de Erro)", erros)
    st.metric("Taxa de Acerto", f"{taxa_acerto:.2f}%")

    # Gráfico
    st.subheader("Distribuição das Previsões")
    fig, ax = plt.subplots()
    df["prediction_label"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["Erros", "Acertos"], rotation=0)
    ax.set_ylabel("Quantidade")
    st.pyplot(fig)

    # Visualização da base
    st.subheader("Amostra das Previsões")
    st.dataframe(df.sample(10))

except FileNotFoundError:
    st.warning("Arquivo de previsões não encontrado. Rode a pipeline de aplicação primeiro.")