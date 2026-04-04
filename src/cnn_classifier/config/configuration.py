from src.cnn_classifier.constants import *
from src.cnn_classifier.utils.common import read_yaml, create_directories
from src.cnn_classifier.entity.config_entity import DataIngestionConfig
from src.cnn_classifier.entity.config_entity import DataTransformationConfig
from src.cnn_classifier.entity.config_entity import ModelPreparationConfig
from src.cnn_classifier.entity.config_entity import ModelTrainingConfig
from src.cnn_classifier.entity.config_entity import ModelEvaluationConfig
from src.cnn_classifier.entity.config_entity import ModelSelectionConfig

import os

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
    
    def get_model_preparation_config(self) -> ModelPreparationConfig:
        config = self.config.model_preparation

        return ModelPreparationConfig(
            model_name = config.model_name,
            num_classes = config.num_classes,
            learning_rate = config.learning_rate,
            weight_decay = config.weight_decay,
            optimizer_name = config.optimizer_name,
            scheduler_name = config.scheduler_name,
            step_size = config.step_size,
            gamma = config.gamma,
            device = config.device,
            model_dir = Path("artifacts/model_preparation")
        )
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params.training

        create_directories([config.root_dir])

        return ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            num_epochs=params.num_epochs,
            device=params.device
        )
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config

        return ModelEvaluationConfig(
            root_dir=os.path.join(config.artifacts_root, "model_evaluation"),
            model_training_dir=os.path.join(config.artifacts_root, "model_training"),
            model_name="",  # filled later after reading summary.json
            num_classes=0,  # filled later after loading class_names
            device="mps"
        )
    
    def get_model_selection_config(self) -> ModelSelectionConfig:
        create_directories([MODEL_SELECTION_DIR])

        config = ModelSelectionConfig(
            model_training_dir=MODEL_TRAINING_DIR,
            model_selection_dir=MODEL_SELECTION_DIR,
            selection_metric="val_macro_f1",   # or "val_accuracy"
            max_allowed_gap=0.08
        )

        return config
