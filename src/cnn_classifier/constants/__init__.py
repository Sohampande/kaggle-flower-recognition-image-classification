from pathlib import Path
import os

ARTIFACTS_DIR = "artifacts"

CONFIG_FILE_PATH = Path('config/config.yaml')
PARAMS_FILE_PATH = Path('params.yaml')

MODEL_TRAINING_DIR = os.path.join(ARTIFACTS_DIR, "model_training")
MODEL_SELECTION_DIR = os.path.join(ARTIFACTS_DIR, "model_selection")