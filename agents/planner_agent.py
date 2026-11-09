"""Inspect a dataset and recommend the stages of a data science workflow."""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class PlannerAgent:
    """Produce dataset metadata and a plan without modifying the input data."""

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"] | None = None,
    ) -> dict[str, Any]:
        """Inspect ``df`` and return metadata, task selection, warnings, and a plan.

        Boolean and categorical targets imply classification. Finite numeric
        targets with fractional values and more than two distinct values imply
        regression. Integer-valued or binary numeric targets are ambiguous:
        their task remains unset unless ``task_type`` is supplied. An explicit
        override takes precedence over inference, which is reported separately.
        Missing targets are excluded from inference and flagged for removal.

        Raises:
            TypeError: If df is not a DataFrame or target_column is not a string.
            ValueError: If the data is empty, columns are duplicated, the target
                is absent, or task_type is unsupported.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not isinstance(target_column, str):
            raise TypeError("target_column must be a string column name.")
        if df.empty:
            raise ValueError("Dataset must not be empty; provide rows and columns.")
        if not df.columns.is_unique:
            raise ValueError("Dataset column names must be unique.")
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column!r} does not exist in the dataset.")
        if task_type not in (None, "classification", "regression"):
            raise ValueError("task_type must be 'classification', 'regression', or None.")

        target = df[target_column]
        observed = target.dropna()
        unique_count = int(observed.nunique())
        warnings: list[str] = []
        inferred: str | None = None
        numeric = is_numeric_dtype(target.dtype) and not is_bool_dtype(target.dtype)
        numerical_columns = [
            column for column in df.columns
            if is_numeric_dtype(df[column].dtype) and not is_bool_dtype(df[column].dtype)
        ]
        categorical_columns = list(df.select_dtypes(include=["object", "string", "category", "bool"]).columns)
        missing_values = {column: int(count) for column, count in df.isna().sum().items()}
        if any(missing_values.values()):
            warnings.append("Dataset contains missing values; handle them during preprocessing.")
        if target.isna().any():
            warnings.append("Target contains missing values; remove unlabeled rows before training.")

        if observed.empty:
            warnings.append("Target has no observed values; task type cannot be inferred.")
        elif is_bool_dtype(target.dtype) or target_column in categorical_columns:
            inferred = "classification"
        elif numeric:
            values = observed.to_numpy()
            if np.iscomplexobj(values):
                warnings.append("Complex numerical targets require transformation before modeling.")
            else:
                values = observed.to_numpy(dtype=float)
                finite = bool(np.isfinite(values).all())
                integer_valued = finite and bool(np.equal(values, np.trunc(values)).all())
                if not finite:
                    warnings.append("Target contains infinite values; clean them before modeling.")
                elif integer_valued or unique_count <= 2:
                    warnings.append(
                        "Ambiguous numerical target: it could represent class labels or quantities. "
                        "Set task_type='classification' or task_type='regression' explicitly."
                    )
                else:
                    inferred = "regression"
        else:
            warnings.append("Unsupported target dtype; transform the target or provide task_type explicitly.")
        if unique_count == 1:
            warnings.append("Target has only one distinct value; meaningful model training requires variation.")

        selected = task_type or inferred
        if task_type is not None and inferred is not None and task_type != inferred:
            warnings.append(f"Explicit task_type overrides inferred task type {inferred!r}.")

        modeling = {
            "classification": "Train and compare classification models using an appropriate validation split.",
            "regression": "Train and compare regression models using an appropriate validation split.",
        }.get(selected, "Resolve task_type before choosing models or splitting the data.")
        metrics = {
            "classification": ["accuracy", "precision", "recall", "f1"],
            "regression": ["mae", "rmse", "r2"],
        }.get(selected, [])
        return {
            "dataset_info": {
                "shape": df.shape,
                "numerical_columns": numerical_columns,
                "categorical_columns": categorical_columns,
                "missing_values": missing_values,
                "target": {
                    "column": target_column,
                    "dtype": str(target.dtype),
                    "unique_count": unique_count,
                    "missing_count": int(target.isna().sum()),
                    "unique_ratio": unique_count / len(observed) if len(observed) else None,
                },
            },
            "inferred_task_type": inferred,
            "task_type": selected,
            "warnings": warnings,
            "recommended_pipeline": [
                {"stage": "EDA", "recommendation": "Inspect distributions, relationships, missingness, and target balance."},
                {"stage": "preprocessing", "recommendation": "Separate the target, split data, then fit imputation, encoding, and any scaling on training data only."},
                {"stage": "modeling", "recommendation": modeling},
                {"stage": "evaluation", "recommendation": "Evaluate on held-out data after resolving the task type.", "metrics": metrics},
                {"stage": "reporting", "recommendation": "Summarize findings, model performance, limitations, and next steps."},
            ],
        }
