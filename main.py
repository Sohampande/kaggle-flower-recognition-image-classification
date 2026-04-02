from src.cnn_classifier.components.model_preparation import ModelPreparation
from src.cnn_classifier.pipeline.stage_03_model_preparation import ModelPreparationTrainingPipeline
from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.utils import logger

STAGE_NAME = "MODEL PREPARATION STAGE"


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage started <<<<<< : {STAGE_NAME}")
        obj = ModelPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage completed <<<<<< : {STAGE_NAME}")
    except Exception as e:
        logger.exception(e)
        raise e