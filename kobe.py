import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Kobe Shot Prediction", layout="centered")
st.title("🏀 Kobe Shot Prediction")
st.markdown("Preencha os dados abaixo para prever se o arremesso será convertido ou não.")

# Carrega a base de treino para limites de latitude e longitude
@st.cache_data
def carregar_limites():
    df = pd.read_parquet("data/03_primary/base_train.parquet")
    lat_min, lat_max = round(df["lat"].min(), 4), round(df["lat"].max(), 4)
    lon_min, lon_max = round(df["lon"].min(), 4), round(df["lon"].max(), 4)
    return lat_min, lat_max, lon_min, lon_max

lat_min, lat_max, lon_min, lon_max = carregar_limites()

# Carrega o modelo treinado com pipeline do PyCaret
@st.cache_resource
def carregar_modelo():
    return load_model("data/06_models/final_model")

modelo = carregar_modelo()

# Layout em duas colunas
col1, col2 = st.columns(2)

with col1:
    lat = st.slider("Latitude", min_value=lat_min, max_value=lat_max, value=(lat_min + lat_max)/2, step=0.0001)
    minutes_remaining = st.slider("Minutos restantes no período", min_value=0, max_value=12, value=6)
    playoffs = st.selectbox("É jogo de playoff?", [0, 1])

with col2:
    lon = st.slider("Longitude", min_value=lon_min, max_value=lon_max, value=(lon_min + lon_max)/2, step=0.0001)
    shot_distance = st.slider("Distância do arremesso (ft)", min_value=0, max_value=40, value=15)
    period = st.selectbox("Período do jogo", [1, 2, 3, 4])
    
# Predição
if st.button("Prever"):
    entrada = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "minutes_remaining": minutes_remaining,
        "period": period,
        "playoffs": playoffs,
        "shot_distance": shot_distance
    }])

    resultado = predict_model(modelo, data=entrada)
    predicao = resultado["prediction_label"][0]
    score = round(resultado["prediction_score"][0] * 100, 2)

    st.subheader("Resultado da Predição:")
    if predicao == 1:
        st.success(f"✅ Cesta provável! ({score}%)")
    else:
        st.error(f"❌ Alta chance de erro no arremesso ({score}%)")