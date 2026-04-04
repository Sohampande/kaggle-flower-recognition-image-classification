from dataclasses import dataclass

@dataclass
class ModelEvaluationArtifact:
    accuracy: float
    loss: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list
    metrics_file_path: str
    confusion_matrix_file_path: str

@dataclass
class ModelSelectionArtifact:
    best_model_name: str
    best_model_path: str
    summary_csv_path: str
    summary_json_path: str
    plot_path: str
    best_model_info_path: str