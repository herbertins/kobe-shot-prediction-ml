
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel
from typing import List

# Criação da app FastAPI
app = FastAPI(title="Kobe Shot Prediction API")

# Modelo de entrada
class InputData(BaseModel):
    lat: float
    lng: float
    minutes_remaining: int
    period: int
    playoffs: int
    shot_distance: float

# Carrega o modelo final salvo no MLflow (ajuste o caminho conforme necessário)
model = mlflow.sklearn.load_model("final_model")

@app.post("/predict")
def predict(data: List[InputData]):
    df = pd.DataFrame([d.dict() for d in data])
    prediction = model.predict(df)
    return {"predictions": prediction.tolist()}
