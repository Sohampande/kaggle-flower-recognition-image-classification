from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.data_ingestion_components import DataIngestion
from cnn_classifier.utils import logger

STAGE_NAME = '1st Stage : Data Ingestion'

class DataIngestionTrainingPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()



if __name__ == 'main':
    try:
        logger.info(f'>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<')
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<')
    except Exception as e:
        raise e