import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.cnn_classifier.utils import logger


class ModelTraining:
    def __init__(
        self,
        config,
        model_name: str,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
    ):
        self.config = config
        self.model_name = model_name
        self.device = "mps"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # model-specific folder
        self.model_dir = Path(self.config.root_dir) / self.model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.model_dir / "best_model.pth"
        self.last_model_path = self.model_dir / "last_model.pth"
        self.metrics_file_path = self.model_dir / "metrics.json"
        self.history_file_path = self.model_dir / "history.json"

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        logger.info(f"Training model: {self.model_name}")
        logger.info(f"Saving outputs to: {self.model_dir}")

    def _train_one_epoch(self):
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc=f"Training {self.model_name}", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def _validate_one_epoch(self):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Validating {self.model_name}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def _save_checkpoint(self, path: Path, epoch: int, best_val_accuracy: float):
        checkpoint = {
            "model_name": self.model_name,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "history": self.history
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def _save_json(self, data, path: Path):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def train(self):
        logger.info(f"Starting training for model: {self.model_name}")

        best_val_accuracy = 0.0
        best_epoch = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"{self.model_name} - Epoch [{epoch + 1}/{self.config.num_epochs}]")

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate_one_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            logger.info(
                f"{self.model_name} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if self.scheduler is not None:
                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch + 1
                self._save_checkpoint(self.best_model_path, epoch + 1, best_val_accuracy)

            self._save_checkpoint(self.last_model_path, epoch + 1, best_val_accuracy)

        metrics = {
            "model_name": self.model_name,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "final_train_loss": self.history["train_loss"][-1],
            "final_train_accuracy": self.history["train_accuracy"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "final_val_accuracy": self.history["val_accuracy"][-1],
            "epochs": self.config.num_epochs,
            "best_model_path": str(self.best_model_path),
            "last_model_path": str(self.last_model_path)
        }

        self._save_json(metrics, self.metrics_file_path)
        self._save_json(self.history, self.history_file_path)

        logger.info(f"Training completed for model: {self.model_name}")

        return metrics