from src.cnn_classifier.models.models import (
    BaselineCNN,
    WiderCNN,
    DropoutCNN,
    BatchNormCNN,
    DeeperCNN,
    ResidualCNN,
)

def get_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    elif model_name == "width":
        return WiderCNN(num_classes=num_classes)
    elif model_name == "dropout":
        return DropoutCNN(num_classes=num_classes)
    elif model_name == "batchnorm":
        return BatchNormCNN(num_classes=num_classes)
    elif model_name == "depth":
        return DeeperCNN(num_classes=num_classes)
    elif model_name == "residual":
        return ResidualCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")