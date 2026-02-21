from src.cnn_classifier.constants import *
from src.cnn_classifier.utils.common import read_yaml, create_directories
from src.cnn_classifier.entity.config_entity import DataIngestionConfig

class ConfigurationManager():
    def __init__(
            self,
            config_path = CONFIG_FILE_PATH,
            params_path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            root_dataset_name = config.root_dataset_name,
            local_data_file = config.local_data_file
        )

        return data_ingestion_config
