from src.cnn_classifier.constants import *
from src.cnn_classifier.utils.common import read_yaml, create_directories
from src.cnn_classifier.entity.config_entity import DataIngestionConfig
from src.cnnClassifier.entity.config_entity import DataTransformationConfig

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
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        os.makedirs(config.root_dir, exist_ok=True)

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=self.config.data_ingestion.unzip_dir,
            class_names_path=config.class_names_path,
            image_size=tuple(config.image_size),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            mean=list(config.mean),
            std=list(config.std),
            seed=config.seed
        )
