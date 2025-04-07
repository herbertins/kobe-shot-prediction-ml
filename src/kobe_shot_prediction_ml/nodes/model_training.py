from typing import Tuple,Dict, Any
from pycaret.classification import *
from sklearn.metrics import log_loss, f1_score
import pandas as pd
import mlflow

def train_logistic_model(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
    
    mlflow.set_experiment("Treinamento")
   
    setup(df_train, target='shot_made_flag', session_id=42, preprocess=True, html=False, n_jobs=-1)
    model = create_model('lr')
    results = predict_model(model, data=df_test)

    logloss = log_loss(results["shot_made_flag"], results["prediction_label"])
    f1 = f1_score(results["shot_made_flag"], results["prediction_label"])

    mlflow.log_metric("log_loss_lr", logloss)
    mlflow.log_metric("f1_score_lr", f1)
    
    report = {
        "model": "logistic_regression",
        "log_loss": logloss,
        "f1_score": f1,
        "prediction_sample": results.head(5).to_dict(orient="records")
    }
    
    # save_model(model, "models/model_lr")
    
    return model, pd.DataFrame(report)


def train_decision_tree(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
    
    # mlflow.set_experiment("Treinamento")
    
    setup(df_train, target='shot_made_flag', session_id=42, preprocess=True, html=False, n_jobs=-1)
    model = create_model('dt')
    results = predict_model(model, data=df_test)

    logloss = log_loss(results["shot_made_flag"], results["prediction_label"])
    f1 = f1_score(results["shot_made_flag"], results["prediction_label"])

    mlflow.log_metric("log_loss_lr", logloss)
    mlflow.log_metric("f1_score_lr", f1)

    report = {
        "model": "decision_tree",
        "log_loss": logloss,
        "f1_score": f1,
        "prediction_sample": results.head(5).to_dict(orient="records")
    }
    
    # save_model(model, "models/model_tree")
    
    return model, pd.DataFrame(report)
    
# def select_and_log_best_model(lr_results: dict,dt_results: dict,) -> str:

#         mlflow.sklearn.log_model(dt_model, artifact_path="final_model")
#         mlflow.log_param("selected_model", dt_model)

#     return dt_model