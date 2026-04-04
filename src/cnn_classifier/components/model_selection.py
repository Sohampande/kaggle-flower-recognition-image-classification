import json
from pathlib import Path

from src.cnn_classifier.utils import logger
from src.cnn_classifier.entity.artifact_entity import ModelSelectionArtifact


class ModelSelection:
    def __init__(self, config):
        self.config = config
        self.model_training_dir = Path(config.model_training_dir)
        self.model_selection_dir = Path(config.model_selection_dir)
        self.model_selection_dir.mkdir(parents=True, exist_ok=True)

    def _load_model_metrics(self):
        """
        Read metrics.json for each trained model folder inside artifacts/model_training.
        """
        logger.info("Loading training metrics for all models")

        model_rows = []

        if not self.model_training_dir.exists():
            raise FileNotFoundError(
                f"Model training directory not found: {self.model_training_dir}"
            )

        for item in self.model_training_dir.iterdir():
            if not item.is_dir():
                continue

            metrics_path = item / "metrics.json"
            best_model_path = item / "best_model.pth"

            if not metrics_path.exists():
                logger.warning(f"Skipping {item.name}: metrics.json not found")
                continue

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            row = {
                "model_name": metrics.get("model_name", item.name),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_accuracy": metrics.get("best_val_accuracy"),
                "final_train_accuracy": metrics.get("final_train_accuracy"),
                "final_val_accuracy": metrics.get("final_val_accuracy"),
                "final_train_loss": metrics.get("final_train_loss"),
                "final_val_loss": metrics.get("final_val_loss"),
                "epochs": metrics.get("epochs"),
                "best_model_path": metrics.get("best_model_path", str(best_model_path)),
                "val_macro_f1": metrics.get("val_macro_f1"),   # future-compatible
                "train_val_gap": None,
                "gap_ok": True,
                "selection_score": None,
                "selected": False,
            }

            train_acc = row["final_train_accuracy"]
            val_acc = row["final_val_accuracy"]

            if train_acc is not None and val_acc is not None:
                gap = abs(train_acc - val_acc)
                row["train_val_gap"] = gap
                row["gap_ok"] = gap <= self.config.max_allowed_gap

            model_rows.append(row)

        if not model_rows:
            raise ValueError("No valid model metrics found for selection")

        return model_rows

    def _compute_selection_scores(self, model_rows):
        """
        Compute a comparable score for each model.
        Priority:
        1) val_macro_f1 if present
        2) otherwise best_val_accuracy

        Models with excessive train/val gap get penalized.
        """
        logger.info("Computing model selection scores")

        for row in model_rows:
            if self.config.selection_metric == "val_macro_f1" and row["val_macro_f1"] is not None:
                base_score = row["val_macro_f1"]
            else:
                base_score = row["best_val_accuracy"]

            if base_score is None:
                row["selection_score"] = float("-inf")
                continue

            gap = row["train_val_gap"]
            gap_ok = row["gap_ok"]

            # Hard penalty if gap too large
            if gap is not None and not gap_ok:
                penalty = max(0.0, gap - self.config.max_allowed_gap)
                row["selection_score"] = base_score - penalty
            else:
                row["selection_score"] = base_score

        return model_rows

    def _choose_best_model(self, model_rows):
        """
        Choose the best model by highest selection_score.
        Tiebreakers:
        1) smaller train_val_gap
        2) higher best_val_accuracy
        """
        logger.info("Selecting best model")

        def sort_key(row):
            gap = row["train_val_gap"]
            if gap is None:
                gap = float("inf")

            best_val_acc = row["best_val_accuracy"]
            if best_val_acc is None:
                best_val_acc = float("-inf")

            return (
                row["selection_score"],
                -gap,
                best_val_acc,
            )

        best_row = max(model_rows, key=sort_key)

        for row in model_rows:
            row["selected"] = row["model_name"] == best_row["model_name"]

        return best_row, model_rows

    def _save_outputs(self, best_row, model_rows):
        """
        Save:
        - summary.json
        - comparison.csv
        - best_model_info.json
        """
        summary_json_path = self.model_selection_dir / "summary.json"
        summary_csv_path = self.model_selection_dir / "summary.csv"
        best_model_info_path = self.model_selection_dir / "best_model_info.json"

        ranked_rows = sorted(
            model_rows,
            key=lambda x: x["selection_score"] if x["selection_score"] is not None else float("-inf"),
            reverse=True
        )

        summary_payload = {
            "selection_metric": self.config.selection_metric,
            "max_allowed_gap": self.config.max_allowed_gap,
            "best_overall_model": best_row["model_name"],
            "best_model_path": best_row["best_model_path"],
            "best_model_score": best_row["selection_score"],
            "models": ranked_rows
        }

        best_model_info = {
            "model_name": best_row["model_name"],
            "best_model_path": best_row["best_model_path"],
            "selection_score": best_row["selection_score"],
            "best_val_accuracy": best_row["best_val_accuracy"],
            "val_macro_f1": best_row["val_macro_f1"],
            "train_val_gap": best_row["train_val_gap"],
            "gap_ok": best_row["gap_ok"],
            "best_epoch": best_row["best_epoch"]
        }

        with open(summary_json_path, "w") as f:
            json.dump(summary_payload, f, indent=4)

        with open(best_model_info_path, "w") as f:
            json.dump(best_model_info, f, indent=4)

        # save CSV manually
        headers = [
            "model_name",
            "best_epoch",
            "best_val_accuracy",
            "val_macro_f1",
            "final_train_accuracy",
            "final_val_accuracy",
            "train_val_gap",
            "gap_ok",
            "selection_score",
            "selected",
            "best_model_path"
        ]

        with open(summary_csv_path, "w") as f:
            f.write(",".join(headers) + "\n")
            for row in ranked_rows:
                values = [row.get(h) for h in headers]
                values = ["" if v is None else str(v) for v in values]
                f.write(",".join(values) + "\n")

        logger.info(f"Model selection summary saved to: {summary_json_path}")
        logger.info(f"Model selection CSV saved to: {summary_csv_path}")
        logger.info(f"Best model info saved to: {best_model_info_path}")

        return str(summary_csv_path), str(summary_json_path), str(best_model_info_path)

    def initiate_model_selection(self):
        logger.info("Starting model selection")

        model_rows = self._load_model_metrics()
        model_rows = self._compute_selection_scores(model_rows)
        best_row, model_rows = self._choose_best_model(model_rows)
        summary_csv_path, summary_json_path, best_model_info_path = self._save_outputs(
            best_row, model_rows
        )

        logger.info(f"Best selected model: {best_row['model_name']}")
        logger.info(f"Best model checkpoint: {best_row['best_model_path']}")

        return ModelSelectionArtifact(
            best_model_name=best_row["model_name"],
            best_model_path=best_row["best_model_path"],
            summary_csv_path=summary_csv_path,
            summary_json_path=summary_json_path,
            plot_path="",  # optional for later if you add charts
            best_model_info_path=best_model_info_path
        )