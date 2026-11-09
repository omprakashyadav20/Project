"""Final held-out prediction metrics and charts, without fitting or tuning."""

from numbers import Real
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score, root_mean_squared_error,
)
from sklearn.utils.multiclass import type_of_target, unique_labels
import seaborn as sns


class EvaluationAgent:
    """Evaluate already selected models using held-out labels and predictions."""

    def __init__(self, output_dir: str | Path = "outputs/evaluation") -> None:
        self.output_dir = Path(output_dir)

    def evaluate_predictions(
        self,
        y_true: Any,
        y_pred: Any,
        task_type: Literal["classification", "regression"],
        *,
        average: Literal["auto", "binary", "macro", "weighted"] = "auto",
        positive_label: Any = None,
        y_proba: Any = None,
        decision_scores: Any = None,
        classes: Any = None,
    ) -> dict[str, Any]:
        """Return metrics, averaging details, chart paths, and warnings.

        Pass ModelingAgent's ``y_test`` and ``predictions`` directly. This method
        never fits a model, chooses thresholds, or performs model selection.
        Inputs must be nonempty aligned 1-D vectors without missing values.
        Two pandas Series must also have identical indices in identical order.

        Classification defaults to binary averaging for two classes, otherwise
        macro. Binary positive_label defaults to the last class (sorted unless
        classes is supplied); the chosen label is reported. Macro treats classes
        equally; weighted uses true-label support. Undefined components with
        nonzero averaging weight make the aggregate None, rather than silently
        assigning zero or excluding a class. Per-class scores are also returned.

        For optional ROC-AUC, provide exactly one of y_proba/decision_scores and
        classes in the estimator's classes_ order. Matrix columns follow that
        order; a binary vector scores classes[1], regardless of positive_label.
        Binary probabilities may be an n-vector or n-by-2 matrix. Multiclass
        scores must be n-by-k; AUC uses one-vs-rest macro (or weighted if average
        is weighted), supporting both probabilities and decision scores.
        Ambiguous mappings, malformed scores, or missing test classes leave AUC
        unavailable with a reason, while retaining other metrics.

        Regression R-squared is None for fewer than two observations or a
        constant true target. Non-finite computed metrics are also None.
        Chart failures return warnings; incomplete files are removed and all
        figures are closed. Saved PNG names are unique and contain no labels.

        Raises:
            ValueError: For invalid task/configuration, labels, vector lengths,
                missing values, index misalignment, or non-finite regression data.
        """
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'.")
        if average not in ("auto", "binary", "macro", "weighted"):
            raise ValueError("average must be 'auto', 'binary', 'macro', or 'weighted'.")
        if y_proba is not None and decision_scores is not None:
            raise ValueError("Provide only one of y_proba or decision_scores.")
        if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series) and not y_true.index.equals(y_pred.index):
            raise ValueError("y_true and y_pred Series indices must match in the same order.")
        actual = self._vector(y_true, "y_true")
        predicted = self._vector(y_pred, "y_pred")
        if len(actual) != len(predicted):
            raise ValueError("y_true and y_pred must have equal lengths.")
        messages: list[str] = []
        result: dict[str, Any] = {
            "task_type": task_type, "sample_count": len(actual), "metrics": {},
            "charts": {}, "warnings": messages,
        }
        if task_type == "regression":
            if any(value is not None for value in (y_proba, decision_scores, classes, positive_label)) or average != "auto":
                raise ValueError("Classification options are not applicable to regression.")
            actual = self._real_values(actual, "y_true")
            predicted = self._real_values(predicted, "y_pred")
            metrics = {
                "mae": float(mean_absolute_error(actual, predicted)),
                "mse": float(mean_squared_error(actual, predicted)),
                "rmse": float(root_mean_squared_error(actual, predicted)),
                "r2": None,
            }
            if len(actual) < 2 or np.all(actual == actual[0]):
                messages.append("R-squared is undefined for fewer than two samples or a constant actual target.")
            else:
                metrics["r2"] = float(r2_score(actual, predicted, force_finite=False))
            for name, value in metrics.items():
                if value is not None and not np.isfinite(value):
                    metrics[name] = None
                    messages.append(f"{name} is non-finite and is reported as unavailable.")
            result["metrics"] = metrics

            def actual_plot(axis: Any) -> None:
                axis.scatter(actual, predicted, alpha=0.7)
                low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
                axis.plot([low, high], [low, high], "--", color="gray")
                axis.set(title="Actual vs predicted", xlabel="Actual", ylabel="Predicted")

            def residual_plot(axis: Any) -> None:
                axis.scatter(predicted, actual - predicted, alpha=0.7)
                axis.axhline(0, linestyle="--", color="gray")
                axis.set(title="Residuals vs predicted", xlabel="Predicted", ylabel="Residual (actual - predicted)")

            result["charts"] = {
                "actual_vs_predicted": self._save_chart("actual_vs_predicted", actual_plot, messages),
                "residuals": self._save_chart("residuals", residual_plot, messages),
            }
            return result

        try:
            if type_of_target(actual) not in ("binary", "multiclass") or type_of_target(predicted) not in ("binary", "multiclass"):
                raise ValueError("Labels must be discrete scalar class labels.")
            observed = unique_labels(actual, predicted)
            labels = observed if classes is None else self._vector(classes, "classes")
            if len(unique_labels(labels)) != len(labels) or not set(observed).issubset(set(labels)):
                raise ValueError("classes must be unique and include all actual and predicted labels.")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid classification labels: {exc}") from exc
        method = ("binary" if len(labels) == 2 else "macro") if average == "auto" else average
        if method == "binary" and len(labels) != 2:
            raise ValueError("Binary averaging requires exactly two class labels; provide classes or use macro/weighted.")
        if positive_label is not None and method != "binary":
            raise ValueError("positive_label applies only to binary averaging.")
        positive = labels[-1] if positive_label is None and method == "binary" else positive_label
        if method == "binary" and positive not in labels:
            raise ValueError("positive_label must be one of the class labels.")
        matrix = confusion_matrix(actual, predicted, labels=labels)
        support = matrix.sum(axis=1)
        predicted_count = matrix.sum(axis=0)
        true_positive = matrix.diagonal()
        precision = [float(tp / total) if total else None for tp, total in zip(true_positive, predicted_count)]
        recall = [float(tp / total) if total else None for tp, total in zip(true_positive, support)]
        f1 = [float(2 * tp / total) if total else None for tp, total in zip(true_positive, support + predicted_count)]
        weights = (np.array([int(label == positive) for label in labels]) if method == "binary"
                   else support if method == "weighted" else np.ones(len(labels)))

        def aggregate(name: str, values: list[float | None]) -> float | None:
            active = [(value, weight) for value, weight in zip(values, weights) if weight > 0]
            if any(value is None for value, _ in active):
                messages.append(f"{name} ({method}) is undefined for at least one class with nonzero averaging weight.")
                return None
            return float(sum(value * weight for value, weight in active) / sum(weight for _, weight in active))

        result["metrics"] = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": aggregate("Precision", precision), "recall": aggregate("Recall", recall),
            "f1": aggregate("F1", f1), "roc_auc": None,
        }
        result.update(
            averaging_method=method, positive_label=self._scalar(positive),
            class_labels=[self._scalar(label) for label in labels],
            confusion_matrix=matrix.tolist(),
            per_class_metrics=[{
                "label": self._scalar(label), "support": int(support[i]),
                "precision": precision[i], "recall": recall[i], "f1": f1[i],
            } for i, label in enumerate(labels)],
        )
        if any(value is None for values in (precision, recall, f1) for value in values):
            messages.append("Some per-class metrics are undefined because a class has no actual or predicted samples; see per_class_metrics.")
        if len(unique_labels(actual)) < 2:
            messages.append("Held-out labels contain a single class; classification metrics do not measure discrimination between classes.")
        auc, reason, auc_method = self._auc(actual, y_proba, decision_scores, labels, classes is not None, positive, method)
        result["metrics"]["roc_auc"] = auc
        result.update(roc_auc_reason=reason, roc_auc_method=auc_method)
        if reason:
            messages.append(f"ROC-AUC unavailable: {reason}")

        def matrix_plot(axis: Any) -> None:
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axis,
                        xticklabels=[str(label) for label in labels], yticklabels=[str(label) for label in labels])
            axis.set(title="Held-out confusion matrix", xlabel="Predicted label", ylabel="Actual label")

        result["charts"]["confusion_matrix"] = self._save_chart("confusion_matrix", matrix_plot, messages)
        return result

    @staticmethod
    def _vector(values: Any, name: str) -> np.ndarray:
        """Validate a required 1-D vector without dropping or imputing rows."""
        try:
            array = np.asarray(values)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
            if pd.isna(array).any():
                raise ValueError(f"{name} contains missing values; evaluation never imputes or drops rows.")
            if array.dtype == object:
                array = np.asarray(array.tolist())
            return array
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name}: {exc}") from exc

    @staticmethod
    def _real_values(values: np.ndarray, name: str) -> np.ndarray:
        if np.iscomplexobj(values) or any(not isinstance(value, (Real, np.bool_)) for value in values):
            raise ValueError(f"{name} must contain finite real numbers.")
        array = values.astype(float)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite real numbers.")
        return array

    @staticmethod
    def _scalar(value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    @staticmethod
    def _auc(
        actual: np.ndarray, probabilities: Any, decisions: Any, labels: np.ndarray,
        explicit_classes: bool, positive: Any, method: str,
    ) -> tuple[float | None, str | None, str | None]:
        """Validate score semantics and calculate binary or one-vs-rest AUC."""
        if probabilities is None and decisions is None:
            return None, "No prediction probabilities or decision scores were provided.", None
        if len(unique_labels(actual)) < 2:
            return None, "At least two actual test classes are required.", None
        if not explicit_classes:
            return None, "Provide classes in estimator.classes_ order to identify score columns and orientation.", None
        try:
            raw = probabilities if probabilities is not None else decisions
            if np.iscomplexobj(raw):
                raise ValueError("Scores must be real numbers.")
            scores = np.asarray(raw, dtype=float)
            if scores.ndim not in (1, 2) or scores.shape[0] != len(actual):
                raise ValueError("Scores must have one row per held-out label.")
            if not np.isfinite(scores).all():
                raise ValueError("Scores contain missing or non-finite values.")
            if probabilities is not None:
                if ((scores < 0) | (scores > 1)).any():
                    raise ValueError("Probabilities must be between 0 and 1.")
                if scores.ndim == 2 and not np.allclose(scores.sum(axis=1), 1, atol=1e-6, rtol=0):
                    raise ValueError("Probability matrix rows must sum to 1.")
            if len(labels) == 2:
                if scores.ndim == 2:
                    if scores.shape[1] != 2:
                        raise ValueError("Binary score matrices must have exactly two columns.")
                    score = scores[:, 1]
                else:
                    score = scores
                scored_label = labels[1]
                if method == "binary" and positive == labels[0]:
                    score = -score
                    scored_label = labels[0]
                return float(roc_auc_score(actual == scored_label, score)), None, "binary"
            if scores.ndim != 2 or scores.shape[1] != len(labels):
                raise ValueError("Multiclass scores must have one column per supplied class (one-vs-rest, not one-vs-one).")
            support = np.array([(actual == label).sum() for label in labels])
            if (support == 0).any():
                raise ValueError("Every supplied class must occur in held-out labels for multiclass ROC-AUC.")
            aucs = [roc_auc_score(actual == label, scores[:, i]) for i, label in enumerate(labels)]
            weighted = method == "weighted"
            return float(np.average(aucs, weights=support if weighted else None)), None, "ovr_weighted" if weighted else "ovr_macro"
        except (ValueError, TypeError) as exc:
            return None, str(exc), None

    def _save_chart(self, kind: str, draw: Callable[[Any], None], messages: list[str]) -> str | None:
        """Save one headless chart exclusively, cleaning up on success or failure."""
        figure = Figure(figsize=(8, 6))
        FigureCanvasAgg(figure)
        path = self.output_dir / f"{kind}_{uuid4().hex}.png"
        created = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            draw(figure.subplots())
            figure.tight_layout()
            with path.open("xb") as output:
                created = True
                figure.savefig(output, format="png", dpi=120)
            return str(path.resolve())
        except Exception as exc:
            messages.append(f"Could not save {kind} chart: {exc}")
            if created:
                try:
                    path.unlink()
                except OSError as cleanup_error:
                    messages.append(f"Could not remove incomplete chart {path}: {cleanup_error}")
            return None
        finally:
            figure.clear()
            plt.close(figure)
