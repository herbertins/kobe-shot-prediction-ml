from kedro.pipeline import Pipeline, node, pipeline
from kobe_shot_prediction_ml.nodes.data_processing import filter_and_clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=filter_and_clean_data,
            inputs="kobe_dev_data",
            outputs="kobe_filtered_data",
            name="filter_and_clean_data_node"
        )
    ])