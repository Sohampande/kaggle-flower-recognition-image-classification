import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
) 
from src.cnn_classifier.utils import logger
from src.cnn_classifier.entity.config_entity import ModelEvaluationConfig
from cnn_classifier.entity.artifact_entity import ModelEvaluationArtifact

class ModelEvaluation:
    def __init__(self, config, model_factory, test_loader, class_names):
        """
        Args:
            config: ModelEvaluationConfig
            model_factory: function like get_model(model_name, num_classes)
            test_loader: DataLoader for evaluation
            class_names: list of class names
        """
        self.config = config
        self.model_factory = model_factory
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = "mps"

    def _load_model(self, model_name, model_path):
        logger.info(f"Loading model: {model_name}")

        model = self.model_factory(
            model_name=model_name,
            num_classes=self.config.num_classes
        )

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        return model

    def evaluate_all_models(self):
        logger.info("Starting evaluation for ALL models")

        os.makedirs(self.config.root_dir, exist_ok=True)

        all_results = []

        model_dirs = [
            d for d in os.listdir(self.config.model_training_dir)
            if os.path.isdir(os.path.join(self.config.model_training_dir, d))
        ]

        for model_name in model_dirs:
            logger.info(f"Evaluating model: {model_name}")

            model_path = os.path.join(
                self.config.model_training_dir,
                model_name,
                "best_model.pth"
            )

            if not os.path.exists(model_path):
                logger.warning(f"Skipping {model_name} (no best_model.pth)")
                continue

            model = self._load_model(model_name, model_path)
            criterion = nn.CrossEntropyLoss()

            all_preds = []
            all_labels = []
            running_loss = 0.0
            total_samples = 0

            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, dim=1)

                    batch_size = labels.size(0)
                    running_loss += loss.item() * batch_size
                    total_samples += batch_size

                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            avg_loss = running_loss / total_samples
            acc = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)

            # save per-model
            model_eval_dir = os.path.join(self.config.root_dir, model_name)
            os.makedirs(model_eval_dir, exist_ok=True)

            metrics = {
                "model_name": model_name,
                "loss": avg_loss,
                "accuracy": acc,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1
            }

            with open(os.path.join(model_eval_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            with open(os.path.join(model_eval_dir, "confusion_matrix.json"), "w") as f:
                json.dump({
                    "class_names": self.class_names,
                    "confusion_matrix": cm.tolist()
                }, f, indent=4)

            all_results.append(metrics)

            logger.info(f"{model_name} → Acc: {acc:.4f}, F1: {f1:.4f}")

        # save combined results
        summary_path = os.path.join(self.config.root_dir, "all_models_evaluation.json")

        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=4)

        logger.info(f"All model evaluation saved to {summary_path}")

        return all_results