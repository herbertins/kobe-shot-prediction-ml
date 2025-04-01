from kedro.pipeline import Pipeline, node, pipeline
from kobe_shot_prediction_ml.nodes.model_training import train_models

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_models,
            inputs=["kobe_train_data", "kobe_test_data"],
            outputs="training_report",
            name="train_models_node"
        )
    ])