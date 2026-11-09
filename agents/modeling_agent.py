"""Compare complete model pipelines with training-only cross-validation."""

from numbers import Integral
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from agents.preprocessing_agent import PreprocessingAgent


class ModelingAgent:
    """Train the best of three baseline models using fold-local preprocessing."""

    def train_models(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"],
        *,
        test_size: float | int = 0.2,
        random_state: int = 42,
        scale_numeric: bool = True,
        cv: int = 5,
    ) -> dict[str, Any]:
        """Split, compare, refit, and evaluate a complete supervised pipeline.

        Uses PreprocessingAgent for validation and reproducible splitting. Its
        precomputed transformed arrays are deliberately not used: sklearn.clone
        removes fitted preprocessing state and each CV fold fits from raw rows.
        Classification maximizes mean stratified-CV accuracy; regression
        minimizes mean CV RMSE (reported as positive values). Ties follow model
        declaration order. Folds are capped by training size and, for
        classification, the smallest observed training-class count.

        A candidate failing any fold is excluded and its error is returned.
        If a candidate fails refitting on all training rows, the next CV-ranked
        candidate is tried. Test data is used only after selection, with one
        predict call and one held-out metric. No test metric drives selection.

        Returns:
            A dictionary containing trained_pipeline (accepting raw features),
            model_comparison, selected_model_name, best_cv_score, predictions,
            y_test, held_out_metrics, feature_names, metadata, and warnings.

        Raises:
            TypeError/ValueError: For invalid inputs or insufficient training
                data to perform at least two valid CV folds.
            RuntimeError: If every candidate fails, refitting fails for every
                ranked candidate, or the selected pipeline cannot predict.
        """
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'.")
        if isinstance(cv, bool) or not isinstance(cv, Integral) or cv < 2:
            raise ValueError("cv must be an integer of at least 2.")
        prepared = PreprocessingAgent().preprocess_dataset(
            df, target_column, task_type, test_size=test_size,
            random_state=random_state, scale_numeric=scale_numeric,
        )
        X_train, X_test = prepared["X_train_raw"], prepared["X_test_raw"]
        y_train, y_test = prepared["y_train"], prepared["y_test"]
        messages = list(prepared["metadata"]["warnings"])
        if task_type == "classification":
            counts = y_train.value_counts()
            counts = counts[counts > 0]
            if len(counts) < 2 or counts.min() < 2:
                raise ValueError("Stratified cross-validation requires at least two training classes with at least two rows per class; provide more labeled data or adjust test_size.")
            folds = min(int(cv), int(counts.min()))
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            scoring, metric = "accuracy", "accuracy"
        else:
            folds = min(int(cv), len(y_train))
            if folds < 2:
                raise ValueError("Regression cross-validation requires at least two training rows; provide more data or adjust test_size.")
            splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
            scoring, metric = "neg_root_mean_squared_error", "rmse"
        if folds < cv:
            messages.append(f"Reduced cross-validation folds from {cv} to {folds} to match available training data.")

        # Freeze identical folds across candidates. No held-out rows participate.
        splits = list(splitter.split(X_train, y_train))
        configuration = clone(prepared["preprocessing_pipeline"])
        comparisons: list[dict[str, Any]] = []
        pipelines: dict[str, Pipeline] = {}
        for name, estimator in self._build_models(task_type, random_state).items():
            row: dict[str, Any] = {
                "model_name": name, "metric": metric, "fold_scores": [],
                "cv_mean": None, "cv_std": None, "status": "failed", "error": None,
                "warnings": [],
            }
            try:
                compatible = is_classifier(estimator) if task_type == "classification" else is_regressor(estimator)
                if not compatible:
                    raise ValueError(f"Estimator is incompatible with task_type={task_type!r}.")
                pipeline = Pipeline([
                    ("preprocessing", clone(configuration)), ("model", clone(estimator)),
                ])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    scores = cross_val_score(
                        pipeline, X_train, y_train, cv=splits, scoring=scoring,
                        error_score="raise", n_jobs=1,
                    )
                row["warnings"] = list(dict.fromkeys(str(item.message) for item in caught))
                if not np.isfinite(scores).all():
                    raise ValueError("Cross-validation produced non-finite scores.")
                scores = -scores if task_type == "regression" else scores
                row.update(fold_scores=scores.tolist(), cv_mean=float(scores.mean()),
                           cv_std=float(scores.std()), status="success")
                pipelines[name] = pipeline
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                messages.append(f"Excluded {name}: {row['error']}")
            comparisons.append(row)

        ranked = sorted(
            (row for row in comparisons if row["status"] == "success"),
            key=lambda row: row["cv_mean"], reverse=task_type == "classification",
        )
        if not ranked:
            details = "; ".join(f"{row['model_name']}: {row['error']}" for row in comparisons)
            raise RuntimeError(f"All candidate models failed cross-validation. {details}")
        selected: dict[str, Any] | None = None
        trained: Pipeline | None = None
        for row in ranked:
            try:
                trained = clone(pipelines[row["model_name"]])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    trained.fit(X_train, y_train)
                messages.extend(f"Final fit: {item.message}" for item in caught)
                selected = row
                break
            except Exception as exc:
                row.update(status="refit_failed", error=f"{type(exc).__name__}: {exc}")
                messages.append(f"Could not refit {row['model_name']}: {exc}")
        if selected is None or trained is None:
            raise RuntimeError("All CV-ranked candidates failed fitting on the full training data. " + "; ".join(messages))
        try:
            predictions = trained.predict(X_test)
            score = (accuracy_score(y_test, predictions) if task_type == "classification"
                     else root_mean_squared_error(y_test, predictions))
        except Exception as exc:
            raise RuntimeError(f"Selected model failed held-out evaluation: {exc}") from exc
        return {
            "trained_pipeline": trained, "model_comparison": comparisons,
            "selected_model_name": selected["model_name"], "best_cv_score": selected["cv_mean"],
            "predictions": predictions, "y_test": y_test,
            "held_out_metrics": {metric: float(score)},
            "feature_names": trained.named_steps["preprocessing"].get_feature_names_out().tolist(),
            "metadata": {**prepared["metadata"], "cv_folds": folds, "selection_metric": metric,
                         "selection_direction": "maximize" if task_type == "classification" else "minimize"},
            "warnings": messages,
        }

    @staticmethod
    def _build_models(task_type: str, random_state: int) -> dict[str, Any]:
        """Construct unfitted, deterministic baseline candidates."""
        if task_type == "classification":
            return {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
                "RandomForestClassifier": RandomForestClassifier(random_state=random_state, n_jobs=1),
            }
        return {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_state),
            "RandomForestRegressor": RandomForestRegressor(random_state=random_state, n_jobs=1),
        }
