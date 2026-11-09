"""Reproducible, leakage-safe feature preprocessing for supervised learning."""

from math import ceil
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target


def _categorical_strings(values: pd.DataFrame) -> pd.DataFrame:
    """Normalize pandas nullable/category/bool columns for sklearn imputation.

    Missing entries remain np.nan; observed categories become string labels.
    This function lives in the fitted pipeline so future inputs use the same
    normalization. Target labels never pass through this transformation.
    """
    return values.astype(object).map(lambda value: np.nan if pd.isna(value) else str(value))


class PreprocessingAgent:
    """Split labeled data and fit a reusable feature transformation pipeline."""

    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"],
        *,
        test_size: float | int = 0.2,
        random_state: int = 42,
        scale_numeric: bool = True,
    ) -> dict[str, Any]:
        """Return transformed splits, original labels, pipeline, and metadata.

        Numerical features receive training-median imputation and optional
        standard scaling. Categories receive training-mode imputation and
        one-hot encoding; unseen categories produce all-zero indicator groups.
        Entirely missing *training* columns are dropped with a returned warning.
        Classification uses stratification whenever both partition sizes and
        class counts permit; otherwise an explicit metadata warning describes
        the unstratified fallback. Check metadata['warnings'] before modeling.

        The returned pipeline accepts raw feature DataFrames. Transformed arrays
        may be dense or scipy sparse, depending on the ColumnTransformer density.
        Feature names are aligned with their transformed columns. Split indices
        are retained in y and also reported in metadata. The input is unchanged.

        Raises:
            TypeError: For invalid input/configuration types.
            ValueError: For empty data, missing labels, invalid task/split,
                unsupported features, non-finite numeric values, or no usable
                training features. Targets are never imputed or encoded.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Dataset must not be empty.")
        if not isinstance(target_column, str):
            raise TypeError("target_column must be a string column name.")
        if not df.columns.is_unique or not all(isinstance(name, str) for name in df.columns):
            raise ValueError("Dataset must have unique string column names.")
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column!r} does not exist in the dataset.")
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'.")
        if not isinstance(scale_numeric, bool):
            raise TypeError("scale_numeric must be a boolean.")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral) or not 0 <= random_state <= 2**32 - 1:
            raise ValueError("random_state must be an integer between 0 and 2**32 - 1.")
        if isinstance(test_size, bool) or not isinstance(test_size, Real):
            raise ValueError("test_size must be a fraction between 0 and 1 or a positive integer row count.")
        if isinstance(test_size, Integral):
            if not 0 < test_size < len(df):
                raise ValueError("Integer test_size must leave at least one training and one test row.")
            n_test = int(test_size)
            test_size = int(test_size)
        else:
            if not 0 < test_size < 1:
                raise ValueError("Fractional test_size must be strictly between 0 and 1.")
            n_test = ceil(len(df) * test_size)
            test_size = float(test_size)
        if n_test >= len(df):
            raise ValueError("Dataset and test_size must leave at least one training row.")

        X = df.drop(columns=[target_column])
        y = df[target_column].copy()
        if X.shape[1] == 0:
            raise ValueError("Dataset must contain at least one input feature besides the target.")
        if y.isna().any():
            raise ValueError("Target contains missing labels; remove or label those rows before preprocessing.")
        if task_type == "regression":
            if not is_numeric_dtype(y.dtype) or is_bool_dtype(y.dtype) or is_complex_dtype(y.dtype):
                raise ValueError("Regression target must contain finite real numerical values.")
            if not np.isfinite(y.to_numpy(dtype=float)).all():
                raise ValueError("Regression target must contain finite real numerical values.")
        else:
            try:
                target_kind = type_of_target(y)
            except (TypeError, ValueError) as exc:
                raise ValueError("Classification target must contain consistent discrete scalar labels.") from exc
            if target_kind not in ("binary", "multiclass"):
                raise ValueError("Classification target must contain discrete class labels.")

        numeric: list[str] = []
        categorical: list[str] = []
        for name in X.columns:
            series = X[name]
            if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
                if is_complex_dtype(series.dtype):
                    raise ValueError(f"Unsupported complex feature {name!r}; convert it to real features explicitly.")
                if np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).any():
                    raise ValueError(f"Numerical feature {name!r} contains infinite values; clean them first.")
                numeric.append(name)
            elif is_bool_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
                if any(not isinstance(value, (str, bool, Real, np.bool_)) for value in series.dropna()):
                    raise ValueError(f"Unsupported values in categorical feature {name!r}; use scalar strings, booleans, or numbers.")
                categorical.append(name)
            else:
                raise ValueError(f"Unsupported feature type {series.dtype} in {name!r}; convert it explicitly before preprocessing.")

        warnings: list[str] = []
        stratified = False
        stratify = None
        if task_type == "classification":
            counts = y.value_counts()
            counts = counts[counts > 0]
            if counts.min() >= 2 and n_test >= len(counts) and len(df) - n_test >= len(counts):
                stratify = y
                stratified = True
            else:
                warnings.append("Stratified splitting is not possible with these class counts and split sizes; used an unstratified split. Some classes may be absent from a partition.")
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=stratify,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unable to split dataset: {exc}") from exc

        dropped = [name for name in X.columns if X_train_raw[name].isna().all()]
        if dropped:
            warnings.append(f"Dropped entirely missing training feature columns: {dropped!r}.")
        numeric = [name for name in numeric if name not in dropped]
        categorical = [name for name in categorical if name not in dropped]
        if not numeric and not categorical:
            raise ValueError("No usable features remain: all feature columns are entirely missing in training data.")
        transformers = []
        if numeric:
            steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numeric:
                steps.append(("scaler", StandardScaler()))
            transformers.append(("numerical", Pipeline(steps), numeric))
        if categorical:
            transformers.append(("categorical", Pipeline([
                ("normalize", FunctionTransformer(_categorical_strings, feature_names_out="one-to-one")),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical))
        pipeline = Pipeline([("preprocessor", ColumnTransformer(transformers, remainder="drop"))])
        try:
            X_train = pipeline.fit_transform(X_train_raw)
            X_test = pipeline.transform(X_test_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unable to preprocess features: {exc}") from exc
        return {
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "X_train_raw": X_train_raw.copy(), "X_test_raw": X_test_raw.copy(),
            "preprocessing_pipeline": pipeline,
            "feature_names": pipeline.get_feature_names_out().tolist(),
            "metadata": {
                "target_column": target_column, "task_type": task_type,
                "test_size": test_size, "random_state": int(random_state),
                "scale_numeric": scale_numeric, "stratified": stratified,
                "numerical_columns": numeric, "categorical_columns": categorical,
                "dropped_columns": dropped, "warnings": warnings,
                "train_rows": len(y_train), "test_rows": len(y_test),
                "train_indices": y_train.index.tolist(), "test_indices": y_test.index.tolist(),
                "transformed_feature_count": X_train.shape[1],
            },
        }
