from kedro.pipeline import Pipeline, node, pipeline
from kobe_shot_prediction_ml.nodes.data_processing import split_train_test

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_train_test,
            inputs="kobe_filtered_data",
            outputs=["kobe_train_data", "kobe_test_data"],
            name="split_train_test_node"
        )
    ])