from src.cnn_classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnn_classifier.utils import logger

STAGE_NAME = 'Data Ingestion Step'

try:
    logger.info(f'>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<')
except Exception as e:
    raise e
