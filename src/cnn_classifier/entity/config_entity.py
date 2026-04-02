from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    root_dataset_name : str 
    local_data_file : Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    class_names_path: Path
    image_size: tuple[int, int]
    batch_size: int
    num_workers: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    mean: list[float]
    std: list[float]
    seed: int
@dataclass
class ModelPreparationConfig:
    model_name: str
    num_classes: int
    learning_rate: float
    weight_decay: float
    optimizer_name: str
    scheduler_name: str
    step_size: int
    gamma: float
    device: str
    model_dir: Path