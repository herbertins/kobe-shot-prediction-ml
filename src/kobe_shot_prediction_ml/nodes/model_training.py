from typing import Tuple, Any
from pycaret.classification import *
from sklearn.metrics import log_loss, f1_score
import pandas as pd
import mlflow

def train_logistic_model(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:    
    setup(df_train, target='shot_made_flag', session_id=42, preprocess=True, html=False, n_jobs=-1)
    model = create_model('lr')
    results = predict_model(model, data=df_test)
    return model, pd.DataFrame(results)


def train_decision_tree(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
    setup(df_train, target='shot_made_flag', session_id=42, preprocess=True, html=False, n_jobs=-1)
    model = create_model('dt')
    results = predict_model(model, data=df_test)
    return model, pd.DataFrame(results)
    
def select_and_log_best_model(lr_model: Any, lr_predictions: pd.DataFrame, dt_model: Any, dt_predictions: pd.DataFrame) -> Tuple[str, Any]:
    
    # Calcula métricas das predições do modelo logístico
    lr_logloss = log_loss(lr_predictions["shot_made_flag"], lr_predictions["prediction_label"])
    lr_f1 = f1_score(lr_predictions["shot_made_flag"], lr_predictions["prediction_label"])

    # Calcula métricas das predições da árvore de decisão
    dt_logloss = log_loss(dt_predictions["shot_made_flag"], dt_predictions["prediction_label"])
    dt_f1 = f1_score(dt_predictions["shot_made_flag"], dt_predictions["prediction_label"])

    # Cria um DataFrame de comparação
    comparison_df = pd.DataFrame([
        {
            "model": "logistic_regression",
            "log_loss": lr_logloss,
            "f1_score": lr_f1
        },
        {
            "model": "decision_tree",
            "log_loss": dt_logloss,
            "f1_score": dt_f1
        }
    ])

    # Seleciona o melhor modelo (maior f1_score)
    best_row = comparison_df.loc[comparison_df["f1_score"].idxmax()]
    best_model_name = best_row["model"]
    best_model = lr_model if best_model_name == "logistic_regression" else dt_model
    
    save_model(best_model, "data/06_models/final_model")
    
    return best_model


def save_metrics (lr_predictions: pd.DataFrame, dt_predictions: pd.DataFrame): 
    # Salva as metricas
    with mlflow.start_run(run_name="Treinamento", nested=True):
        with mlflow.start_run(run_name="logistic_regression_train", nested=True):    
            logloss = log_loss(lr_predictions["shot_made_flag"], lr_predictions["prediction_label"])
            f1 = f1_score(lr_predictions["shot_made_flag"], lr_predictions["prediction_label"])

            mlflow.log_metric("log_loss_lr", logloss)
            mlflow.log_metric("f1_score_lr", f1)
            
        with mlflow.start_run(run_name="decision_tree_train", nested=True): 
            logloss = log_loss(dt_predictions["shot_made_flag"], dt_predictions["prediction_label"])
            f1 = f1_score(dt_predictions["shot_made_flag"], dt_predictions["prediction_label"])
            
            mlflow.log_metric("log_loss_lr", logloss)
            mlflow.log_metric("f1_score_lr", f1)