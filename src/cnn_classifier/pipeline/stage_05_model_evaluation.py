import os
import json

from src.cnn_classifier.components.data_transformation_components import DataTransformation
from src.cnn_classifier.components.model_evaluation import (
    ModelEvaluation,
    ModelEvaluationConfig
)
from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.models.model_factory import get_model
from src.cnn_classifier.utils import logger

class_names_path = '/Users/Soham/Documents/Projects/kaggle-flower-recognition-image-classification/artifacts/data_transformation/class_names.json'

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        class_names = []
        with open(class_names_path, 'r') as file : 
            class_names = json.load(file)
        
        config = ConfigurationManager()

        # load data transformation config + artifacts
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train_loader, val_loader, test_loader = data_transformation.initiate_data_transformation()

        summary_path = os.path.join("artifacts", "model_training", "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"summary.json not found at: {summary_path}")

        with open(summary_path, "r") as f:
            summary = json.load(f)

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=os.path.join("artifacts", "model_evaluation"),
            model_training_dir=os.path.join("artifacts", "model_training"),
            model_name="",  # not needed anymore
            num_classes=5,
            device="mps"
        )

        model_evaluation = ModelEvaluation(
            config=model_evaluation_config,
            model_factory=get_model,
            test_loader=test_loader,
            class_names=class_names
        )

        evaluation_results = model_evaluation.evaluate_all_models()
        logger.info("All models evaluated successfully")
        return evaluation_results