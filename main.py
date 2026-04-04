from src.cnn_classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnn_classifier.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from src.cnn_classifier.pipeline.stage_03_model_preparation import ModelPreparationTrainingPipeline
from src.cnn_classifier.pipeline.stage_04_model_training import ModelTrainingPipeline
from src.cnn_classifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from src.cnn_classifier.pipeline.stage_06_model_selection import ModelSelectionPipeline
from src.cnn_classifier.utils import logger
import pandas as pd
import matplotlib.pyplot as plt

summary_file_location = '/Users/Soham/Documents/Projects/kaggle-flower-recognition-image-classification/artifacts/model_selection/summary.csv'

# Running all the stages from 1 - 6, in the desired order. 
stage_names = ("DATA INGESTION",
              'DATA TRANSFORMATION',
              'MODEL PREPARATION',
              'MODEL TRAINING',
              'MODEL EVALUATION',
              'MODEL SELECTION'
            )
stage_classes = (
    DataIngestionTrainingPipeline,
    DataTransformationTrainingPipeline,
    ModelPreparationTrainingPipeline,
    ModelTrainingPipeline,
    ModelEvaluationPipeline,
    ModelSelectionPipeline
        )

for (stage_name, stage_class) in zip(stage_names, stage_classes):
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        obj = stage_class()
        obj.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

# Objective is to read the file and the draw an historgram using the information from the file.
df = pd.read_csv(summary_file_location)
if df.empty:
    raise ValueError("summary.csv is empty, cannot create plot")

df = df.sort_values(by="selection_score", ascending=False)

x = range(len(df))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

ax.bar(
    [i - width for i in x],
    df["final_train_accuracy"],
    width=width,
    label="Train Accuracy"
)
ax.bar(
    x,
    df["final_val_accuracy"],
    width=width,
    label="Validation Accuracy"
)
ax.bar(
    [i + width for i in x],
    df["selection_score"],
    width=width,
    label="Selection Score"
)

ax.set_title("Model Comparison")
ax.set_xlabel("Model Name")
ax.set_ylabel("Score")
ax.set_xticks(list(x))
ax.set_xticklabels(df["model_name"], rotation=30, ha="right")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plot_path = '/Users/Soham/Documents/Projects/kaggle-flower-recognition-image-classification/artifacts/model_selection/model_comparison'
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

logger.info(f"Model comparison plot saved to: {plot_path}")

# the histogram is now finished and stored. 
# Now, main.py has completed all of its functionality and we can move onto deployment