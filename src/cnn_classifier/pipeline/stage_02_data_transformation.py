from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.components.data_transformation_components import DataTransformation
from src.cnn_classifier.utils import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        artifact = data_transformation.initiate_data_transformation()

        logger.info(f"Class names: {artifact.class_names}")
        logger.info(f"Dataset sizes: {artifact.dataset_sizes}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e