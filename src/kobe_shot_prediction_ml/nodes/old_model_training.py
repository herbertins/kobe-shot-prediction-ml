from pycaret.classification import *
from sklearn.metrics import log_loss, f1_score
import pandas as pd
import mlflow

def train_models(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    
    # Inicia experimento no MLflow
    mlflow.set_experiment("Treinamento")
    
    # Treinar logistic regression
    with mlflow.start_run(run_name="logistic_model"):

        s = setup(
                data=df_train,
                target='shot_made_flag',
                session_id=42,
                preprocess=True,
                html=False,
                n_jobs=-1
            )
        
        model_lr = create_model('lr')
        lr_results = predict_model(model_lr, data=df_test)
        
        mlflow.log_metric("log_loss_lr", log_loss(lr_results['shot_made_flag'], lr_results['prediction_label']))
        mlflow.log_metric("f1_score_lr", f1_score(lr_results["shot_made_flag"], lr_results["prediction_label"]))

        # save_model(model_lr, "models/model_logistic")
    
    # Treinar arvore de decisão
    with mlflow.start_run(run_name="decision_tree"):
        
        s = setup(
                data=df_train,
                target='shot_made_flag',
                session_id=42,
                preprocess=True,
                html=False,
                n_jobs=-1
            )
        
        model_dt = create_model('dt')
        dt_results = predict_model(model_dt, data=df_test)

        mlflow.log_metric("log_loss_lr", log_loss(dt_results['shot_made_flag'], lr_results['prediction_label']))
        mlflow.log_metric("f1_score_lr", f1_score(dt_results["shot_made_flag"], lr_results["prediction_label"]))
        
        # save_model(model_dt, "models/model_tree")

    save_model(model_dt, "final_model")

    # Decide qual será o modelo final com base na métrica
    return {"message": "Modelos treinados e registrados no MLflow."}