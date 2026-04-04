from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.components.model_preparation import ModelPreparation
from src.cnn_classifier.utils import logger

STAGE_NAME = 'Model Preparation Stage'

class ModelPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_preparation_config = config.get_model_preparation_config()

            model_preparation = ModelPreparation(config=model_preparation_config)
            artifact = model_preparation.initiate_model_preparation()

            logger.info("Model Preparation Completed Successfully!")

            logger.info(f"Model: {artifact[0].__class__.__name__}")
            logger.info(f"Criterion: {artifact[1]}")
            logger.info(f"Optimizer: {artifact[2]}")
            logger.info(f"Scheduler: {artifact[3]}")
            logger.info(f"Device: {artifact[4]}")

        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == "__main__":
    try:
        print(f">>>>>> stage started <<<<<< : {STAGE_NAME}")
        obj = ModelPreparationTrainingPipeline()
        obj.main()
        print(f">>>>>> stage completed <<<<<< : {STAGE_NAME}")
    except Exception as e:
        print(e)
        raise e