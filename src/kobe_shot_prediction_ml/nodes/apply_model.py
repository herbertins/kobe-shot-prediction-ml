from pycaret.classification import load_model, predict_model
from sklearn.metrics import log_loss, f1_score
import pandas as pd
import mlflow

def apply_model_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_experiment("PipelineAplicacao")

    if "shot_made_flag" not in df.columns:
        print("Atenção: coluna shot_made_flag não está presente nos dados de produção. Métricas supervisionadas não serão computadas.")
        df_valid = None
    else:
         df_valid = df.dropna(subset=["shot_made_flag"])

    with mlflow.start_run(run_name="predict_prod"):
        # Carrega o modelo
        model = load_model("final_model")

        # Aplica na base com target presente
        pred_df = predict_model(model, data=df_valid)

        # Calcula e loga métricas
        y_true = pred_df['shot_made_flag']
        y_pred = pred_df['prediction_label']

        f1 = f1_score(y_true, y_pred)
        ll = log_loss(y_true, y_pred)

        pred_df.to_parquet("data/07_model_output/predictions_prod.parquet", index=False)
        mlflow.log_artifact("data/07_model_output/predictions_prod.parquet")

        mlflow.log_metric("f1_score_prod", f1)
        mlflow.log_metric("log_loss_prod", ll)

        return pred_df