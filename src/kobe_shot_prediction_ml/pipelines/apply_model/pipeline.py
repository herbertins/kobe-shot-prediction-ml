from kedro.pipeline import Pipeline, node, pipeline
from kobe_shot_prediction_ml.nodes.apply_model import apply_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=apply_model,
            inputs=["kobe_best_model_node", "kobe_prod_data"],
            outputs="kobe_predictions",
            name="apply_model_node"
        )
    ])