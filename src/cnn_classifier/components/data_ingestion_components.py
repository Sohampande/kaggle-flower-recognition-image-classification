from src.cnn_classifier.utils import logger
import kagglehub
import kaggle
import os
from src.cnn_classifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            root_dataset_name = self.config.root_dataset_name
            root_dir = self.config.root_dir
            os.makedirs('artifacts/data_ingestion', exist_ok=True)
            logger.info(f'Downloading the dataset {root_dataset_name} into directory {root_dir}')

            kaggle.api.dataset_download_files(root_dataset_name, path=root_dir, unzip=True)
            logger.info(f'Dataset Downloaded into {root_dir}')
        except Exception as e:
            raise e
