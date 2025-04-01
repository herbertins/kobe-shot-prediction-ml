from typing import Dict
from kedro.pipeline import Pipeline

from kobe_shot_prediction_ml.pipelines.data_processing import pipeline as dp_pipeline
from kobe_shot_prediction_ml.pipelines.data_split import pipeline as ds_pipeline
from kobe_shot_prediction_ml.pipelines.model_training import pipeline as mt_pipeline
from kobe_shot_prediction_ml.pipelines.apply_model import pipeline as ap_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    data_processing_pipeline = dp_pipeline.create_pipeline()
    data_split_pipeline = ds_pipeline.create_pipeline()
    model_training_pipeline = mt_pipeline.create_pipeline()
    apply_model_pipeline = ap_pipeline.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_split_pipeline + model_training_pipeline + apply_model_pipeline,
        "dp": data_processing_pipeline,
        "ds": data_split_pipeline,
        "mt": model_training_pipeline,
        "ap": apply_model_pipeline
    }