from kedro.pipeline import Pipeline, node, pipeline
from kobe_shot_prediction_ml.nodes.apply_model import apply_model_pipeline

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=apply_model_pipeline,
            inputs="kobe_prod_data",
            outputs="kobe_predictions",
            name="apply_model_pipeline_node"
        )
    ])