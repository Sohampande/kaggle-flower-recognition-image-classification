from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.components.data_transformation_components import DataTransformation
from src.cnn_classifier.utils import logger
import json

STAGE_NAME = "Data Transformation Stage"
class_names_path = '/Users/Soham/Documents/Projects/kaggle-flower-recognition-image-classification/artifacts/data_transformation/class_names.json'
dataset_sizes_path = '/Users/Soham/Documents/Projects/kaggle-flower-recognition-image-classification/artifacts/data_transformation/dataset_sizes.json'
class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        class_names = []
        dataset_sizes = []
        with open(class_names_path, 'r') as file : 
            class_names = json.load(file)
        
        with open(dataset_sizes_path, 'r') as file : 
            dataset_sizes = json.load(file)
        
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        artifact = data_transformation.initiate_data_transformation()

        logger.info(f"Class names: {class_names}")
        logger.info(f"Dataset sizes: {dataset_sizes}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e