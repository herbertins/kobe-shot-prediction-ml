from kedro.pipeline import Pipeline, node, pipeline
from kobe_shot_prediction_ml.nodes.model_training import (
    train_logistic_model,
    train_decision_tree,
    select_and_log_best_model
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logistic_model,
            inputs=["kobe_train_data", "kobe_test_data"],
            outputs=["kobe_model_lr", "kobe_predictions_model_lr"],
            name="train_logistic_model_node"
        ),
        node(
            func=train_decision_tree,
            inputs=["kobe_train_data", "kobe_test_data"],
            outputs=["kobe_model_dt", "kobe_predictions_model_dt"],
            name="train_decision_tree_node"
        ),
        node(
            func=select_and_log_best_model,
            inputs=["kobe_model_lr", "kobe_predictions_model_lr", "kobe_model_dt", "kobe_predictions_model_dt"],
            outputs="kobe_best_model_node",
            name="select_and_log_best_model_node"
        )
    ])