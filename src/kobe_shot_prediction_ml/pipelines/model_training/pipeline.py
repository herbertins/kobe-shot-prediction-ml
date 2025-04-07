from kedro.pipeline import Pipeline, node, pipeline
from kobe_shot_prediction_ml.nodes.model_training import (
    train_logistic_model,
    train_decision_tree
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logistic_model,
            inputs=["kobe_train_data", "kobe_test_data"],
            outputs=["model_lr", "logistic_model_report"],
            name="train_logistic_model_node"
        ),
        node(
            func=train_decision_tree,
            inputs=["kobe_train_data", "kobe_test_data"],
            outputs=["model_dt", "decision_tree_model_report"],
            name="train_decision_tree_node"
        )
        # node(
        #     func=select_and_log_best_model,
        #     inputs=["model_lr", "model_dt", "params:selected_model_name"],
        #     outputs="final_model",
        #     name="select_and_log_best_model_node"
        # )
    ])