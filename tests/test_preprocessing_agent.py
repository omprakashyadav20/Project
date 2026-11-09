"""Preprocessing behavior and train/test leakage regression tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from agents.preprocessing_agent import PreprocessingAgent


def dense(values):
    return values.toarray() if hasattr(values, "toarray") else values


@pytest.fixture
def frame():
    return pd.DataFrame({"number": [1, 2, None, 4, 5, 6, 7, None, 9, 10], "category": ["a", "b", None, "a", "b", "a", "b", "a", "b", "a"], "target": ["yes", "no"] * 5})


def test_missing_values_encoding_and_labels(frame):
    original = frame.copy(deep=True)
    result = PreprocessingAgent().preprocess_dataset(frame, "target", "classification")
    assert np.isfinite(dense(result["X_train"])).all()
    assert np.isfinite(dense(result["X_test"])).all()
    assert result["X_train"].shape == (8, 3)
    assert len(result["feature_names"]) == 3
    assert result["metadata"]["stratified"]
    pd.testing.assert_series_equal(result["y_train"], frame.loc[result["y_train"].index, "target"])
    pd.testing.assert_frame_equal(frame, original)


def test_scaling_optional_and_reproducible(frame):
    agent = PreprocessingAgent()
    first = agent.preprocess_dataset(frame, "target", "classification")
    second = agent.preprocess_dataset(frame, "target", "classification")
    np.testing.assert_array_equal(dense(first["X_train"]), dense(second["X_train"]))
    pd.testing.assert_series_equal(first["y_test"], second["y_test"])
    assert dense(first["X_train"])[:, 0].mean() == pytest.approx(0, abs=1e-12)
    assert dense(first["X_train"])[:, 0].std() == pytest.approx(1)
    unscaled = agent.preprocess_dataset(frame, "target", "classification", scale_numeric=False)
    numeric = unscaled["preprocessing_pipeline"].named_steps["preprocessor"].named_transformers_["numerical"]
    assert "scaler" not in numeric.named_steps
    assert dense(unscaled["X_train"])[:, 0].mean() > 0


def test_training_only_imputation_scaling_and_unseen_categories():
    train, test = train_test_split(np.arange(10), test_size=0.2, random_state=42)
    df = pd.DataFrame({"number": np.arange(10, dtype=float), "category": ["known"] * 10, "target": np.arange(10, dtype=float)})
    df.loc[test, "number"] = 10000
    df.loc[test, "category"] = "unseen"
    df.loc[train[0], "number"] = np.nan
    result = PreprocessingAgent().preprocess_dataset(df, "target", "regression")
    fitted = result["preprocessing_pipeline"].named_steps["preprocessor"]
    numeric = fitted.named_transformers_["numerical"]
    expected = df.loc[train, "number"].median()
    assert numeric.named_steps["imputer"].statistics_[0] == expected
    expected_mean = df.loc[train, "number"].fillna(expected).mean()
    assert numeric.named_steps["scaler"].mean_[0] == pytest.approx(expected_mean)
    assert fitted.named_transformers_["categorical"].named_steps["encoder"].categories_[0].tolist() == ["known"]
    assert (dense(result["X_test"])[:, 1:] == 0).all()
    assert result["X_train"].shape[1] == result["X_test"].shape[1]
    future = result["preprocessing_pipeline"].transform(df.drop(columns="target"))
    assert future.shape[0] == 10


def test_categorical_imputation_uses_training_mode():
    train, test = train_test_split(np.arange(10), test_size=0.2, random_state=42)
    df = pd.DataFrame({"category": ["training"] * 10, "target": np.arange(10, dtype=float)})
    df.loc[test, "category"] = "held_out"
    df.loc[train[0], "category"] = None
    result = PreprocessingAgent().preprocess_dataset(df, "target", "regression")
    fitted = result["preprocessing_pipeline"].named_steps["preprocessor"].named_transformers_["categorical"]
    assert fitted.named_steps["imputer"].statistics_.tolist() == ["training"]
    assert dense(result["X_train"]).min() == 1


def test_all_missing_training_columns_are_dropped():
    train, test = train_test_split(np.arange(10), test_size=0.2, random_state=42)
    df = pd.DataFrame({"empty": [np.nan] * 10, "usable": range(10), "target": range(10)})
    df.loc[test, "empty"] = 99
    result = PreprocessingAgent().preprocess_dataset(df, "target", "regression")
    assert result["metadata"]["dropped_columns"] == ["empty"]
    assert result["metadata"]["warnings"]
    assert result["feature_names"] == ["numerical__usable"]
    with pytest.raises(ValueError, match="No usable features"):
        PreprocessingAgent().preprocess_dataset(df.drop(columns="usable"), "target", "regression")


def test_nullable_boolean_category_and_numeric_features():
    df = pd.DataFrame({"num": pd.Series([1, None, 3, 4, 5, 6], dtype="Int64"), "bool": pd.Series([True, None, False, True, False, True], dtype="boolean"), "cat": pd.Categorical(["a", None, "b", "a", "a", "b"]), "target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    result = PreprocessingAgent().preprocess_dataset(df, "target", "regression")
    assert np.isfinite(dense(result["X_train"])).all()
    assert result["metadata"]["categorical_columns"] == ["bool", "cat"]


def test_stratification_fallback():
    df = pd.DataFrame({"x": range(10), "target": ["rare"] + ["common"] * 9})
    result = PreprocessingAgent().preprocess_dataset(df, "target", "classification")
    assert not result["metadata"]["stratified"]
    assert any("unstratified" in warning for warning in result["metadata"]["warnings"])


@pytest.mark.parametrize("df,target,task,message", [
    (pd.DataFrame(), "target", "regression", "empty"),
    (pd.DataFrame({"x": [1, 2]}), "target", "regression", "does not exist"),
    (pd.DataFrame({"x": [1, 2], "target": [1, None]}), "target", "regression", "missing labels"),
    (pd.DataFrame({"x": [1, 2], "target": [1, 2]}), "target", "unknown", "task_type"),
    (pd.DataFrame({"target": [1, 2]}), "target", "regression", "input feature"),
    (pd.DataFrame({"x": pd.date_range("2025-01-01", periods=2), "target": [1, 2]}), "target", "regression", "Unsupported feature"),
    (pd.DataFrame({"x": [[1], [2]], "target": [1, 2]}), "target", "regression", "Unsupported values"),
    (pd.DataFrame({"x": [1, np.inf], "target": [1, 2]}), "target", "regression", "infinite"),
    (pd.DataFrame({"x": [1, 2], "target": ["a", "b"]}), "target", "regression", "Regression target"),
    (pd.DataFrame({"x": [1, 2], "target": [0.1, 0.2]}), "target", "classification", "discrete"),
])
def test_invalid_data(df, target, task, message):
    with pytest.raises(ValueError, match=message):
        PreprocessingAgent().preprocess_dataset(df, target, task)


@pytest.mark.parametrize("config", [{"test_size": 0}, {"test_size": 1.0}, {"test_size": 99}, {"test_size": True}, {"random_state": -1}, {"scale_numeric": "yes"}])
def test_invalid_configuration(frame, config):
    with pytest.raises((ValueError, TypeError)):
        PreprocessingAgent().preprocess_dataset(frame, "target", "classification", **config)


def test_invalid_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        PreprocessingAgent().preprocess_dataset([], "target", "regression")
