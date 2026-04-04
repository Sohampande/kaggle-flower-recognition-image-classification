from src.cnn_classifier.config.configuration import ConfigurationManager
from src.cnn_classifier.components.model_selection import ModelSelection
from src.cnn_classifier.utils import logger

STAGE_NAME = "Model Selection Stage"

class ModelSelectionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_selection_config = config.get_model_selection_config()

        model_selection = ModelSelection(config=model_selection_config)
        model_selection_artifact = model_selection.initiate_model_selection()

        logger.info(f"Model selection completed: {model_selection_artifact}")
        return model_selection_artifact