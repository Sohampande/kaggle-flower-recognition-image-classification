from src.cnn_classifier.utils import logger
from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.components.model_training import ModelTraining
from src.cnn_classifier.components.data_transformation_components import DataTransformation
from src.cnn_classifier.components.model_preparation import ModelPreparation
import json
from pathlib import Path

STAGE_NAME = "Model Training All Models Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        # data loaders: same for all models
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train_loader, val_loader, test_loader = data_transformation.initiate_data_transformation()

        # preparation config
        model_preparation_config = config.get_model_preparation_config()
        model_preparation = ModelPreparation(config=model_preparation_config)

        # training config
        model_training_config = config.get_model_training_config()

        # all model names
        model_names = [
            "baseline",
            "dropout",
            "batchnorm",
            "depth",
            "residual"
        ]

        all_results = []

        for model_name in model_names:
            logger.info(f"\n{'=' * 20} Training {model_name} {'=' * 20}")

            model, criterion, optimizer, scheduler = model_preparation.prepare_model(model_name=model_name)

            trainer = ModelTraining(
                config=model_training_config,
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            metrics = trainer.train()
            all_results.append(metrics)

        # pick overall best model across architectures
        best_overall = max(all_results, key=lambda x: x["best_val_accuracy"])

        summary = {
            "all_results": all_results,
            "best_overall_model": best_overall["model_name"],
            "best_overall_val_accuracy": best_overall["best_val_accuracy"],
            "best_overall_epoch": best_overall["best_epoch"],
            "best_overall_model_path": best_overall["best_model_path"]
        }

        summary_path = Path(model_training_config.root_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"All model training completed.")
        logger.info(f"Best overall model: {best_overall['model_name']}")
        logger.info(f"Summary saved at: {summary_path}")