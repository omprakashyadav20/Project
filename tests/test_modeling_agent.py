"""Model comparison, reproducibility, failures, and fold-level leakage checks."""

from collections import Counter
from functools import wraps

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from agents.modeling_agent import ModelingAgent


@pytest.fixture
def regression():
    x = np.arange(40, dtype=float)
    return pd.DataFrame({"x": x, "category": ["a", "b"] * 20, "target": 3 * x + 2})


@pytest.fixture
def classification():
    return pd.DataFrame({"x": np.arange(40, dtype=float), "category": ["a", "b"] * 20, "target": ["no"] * 20 + ["yes"] * 20})


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_models_selection_and_reproducibility(task, classification, regression):
    df = classification if task == "classification" else regression
    original = df.copy(deep=True)
    agent = ModelingAgent()
    first = agent.train_models(df, "target", task, cv=3)
    second = agent.train_models(df, "target", task, cv=3)
    expected = ({"LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"}
                if task == "classification" else {"LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"})
    assert {row["model_name"] for row in first["model_comparison"]} == expected
    assert all(row["status"] == "success" for row in first["model_comparison"])
    scores = [row["cv_mean"] for row in first["model_comparison"]]
    assert first["best_cv_score"] == (max(scores) if task == "classification" else min(scores))
    assert first["selected_model_name"] == second["selected_model_name"]
    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    assert first["model_comparison"] == second["model_comparison"]
    assert len(first["predictions"]) == len(first["y_test"]) == 8
    assert isinstance(first["trained_pipeline"], Pipeline)
    assert first["feature_names"]
    metric = "accuracy" if task == "classification" else "rmse"
    assert first["held_out_metrics"][metric] >= 0
    for row in first["model_comparison"]:
        assert len(row["fold_scores"]) == 3
        assert all(score >= 0 for score in row["fold_scores"])
    raw_test = df.drop(columns="target").loc[first["y_test"].index]
    np.testing.assert_array_equal(first["trained_pipeline"].predict(raw_test), first["predictions"])
    pd.testing.assert_frame_equal(df, original)


def test_fold_local_preprocessing_and_single_test_prediction(regression, monkeypatch):
    # Observe the actual rows passed to imputer.fit, including the initial
    # preprocessing call, every candidate's CV fits, and the final selected fit.
    fits = []
    original_fit = SimpleImputer.fit
    original_predict = Pipeline.predict
    predicted_indices = []

    def record_fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame) and list(X.columns) == ["x"]:
            fits.append(tuple(X.index))
        return original_fit(self, X, y, **kwargs)

    @wraps(original_predict)
    def record_predict(self, X, **kwargs):
        if isinstance(X, pd.DataFrame):
            predicted_indices.append(tuple(X.index))
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(SimpleImputer, "fit", record_fit)
    monkeypatch.setattr(Pipeline, "predict", record_predict)
    result = ModelingAgent().train_models(regression, "target", "regression", cv=3)
    train_indices = result["metadata"]["train_indices"]
    test_indices = result["metadata"]["test_indices"]
    expected = Counter({tuple(train_indices): 2})
    for train, _ in KFold(3, shuffle=True, random_state=42).split(train_indices):
        expected[tuple(np.array(train_indices)[train])] += 3
    assert Counter(fits) == expected
    assert all(not set(indices).intersection(test_indices) for indices in fits)
    # Nested preprocessing pipelines transform but do not predict: this is the
    # one final complete-pipeline test prediction, after CV selection.
    assert predicted_indices.count(tuple(test_indices)) == 1


def test_held_out_targets_do_not_affect_selection(regression):
    train, test = train_test_split(np.arange(len(regression)), random_state=42, test_size=0.2)
    first = ModelingAgent().train_models(regression, "target", "regression", cv=2)
    modified = regression.copy()
    modified.loc[test, "target"] = 1e6
    second = ModelingAgent().train_models(modified, "target", "regression", cv=2)
    assert first["model_comparison"] == second["model_comparison"]
    assert first["selected_model_name"] == second["selected_model_name"]
    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    assert first["held_out_metrics"] != second["held_out_metrics"]


def test_missing_values_and_cv_only_categories(regression):
    regression.loc[[0, 5, 9, 12], "x"] = np.nan
    regression["category"] = [f"unique_{i}" for i in range(len(regression))]
    result = ModelingAgent().train_models(regression, "target", "regression", cv=3)
    assert np.isfinite(result["predictions"]).all()
    assert all(row["status"] == "success" for row in result["model_comparison"])


def test_adaptive_folds_and_small_regression():
    df = pd.DataFrame({"x": range(6), "target": ["a", "b"] * 3})
    result = ModelingAgent().train_models(df, "target", "classification", test_size=2)
    assert result["metadata"]["cv_folds"] == 2
    assert any("Reduced" in warning for warning in result["warnings"])
    small = pd.DataFrame({"x": [1, 2, 3], "target": [2, 4, 6]})
    result = ModelingAgent().train_models(small, "target", "regression", test_size=1)
    assert result["metadata"]["cv_folds"] == 2


def test_insufficient_classification_data():
    df = pd.DataFrame({"x": range(4), "target": ["a", "b"] * 2})
    with pytest.raises(ValueError, match="at least two rows per class"):
        ModelingAgent().train_models(df, "target", "classification", test_size=2)


def test_failed_and_incompatible_candidates(regression, monkeypatch):
    monkeypatch.setattr(ModelingAgent, "_build_models", staticmethod(lambda task, seed: {
        "incompatible": LogisticRegression(),
        "broken": DecisionTreeRegressor(max_depth=0),
        "working": LinearRegression(),
    }))
    result = ModelingAgent().train_models(regression, "target", "regression", cv=2)
    assert result["selected_model_name"] == "working"
    assert [row["status"] for row in result["model_comparison"]] == ["failed", "failed", "success"]
    assert "incompatible" in result["model_comparison"][0]["error"]
    monkeypatch.setattr(ModelingAgent, "_build_models", staticmethod(lambda task, seed: {
        "broken": DecisionTreeRegressor(max_depth=0),
    }))
    with pytest.raises(RuntimeError, match="All candidate models failed"):
        ModelingAgent().train_models(regression, "target", "regression", cv=2)


def test_refit_failure_uses_next_cv_ranked_model(regression, monkeypatch):
    from agents import modeling_agent

    # Controlled CV ranking allows a deterministic full-training fit failure.
    monkeypatch.setattr(ModelingAgent, "_build_models", staticmethod(lambda task, seed: {
        "bad_refit": DecisionTreeRegressor(max_depth=0), "working": LinearRegression(),
    }))
    scores = iter([np.array([-0.1, -0.1]), np.array([-0.2, -0.2])])
    monkeypatch.setattr(modeling_agent, "cross_val_score", lambda *args, **kwargs: next(scores))
    result = ModelingAgent().train_models(regression, "target", "regression", cv=2)
    assert result["selected_model_name"] == "working"
    assert result["model_comparison"][0]["status"] == "refit_failed"


@pytest.mark.parametrize("config", [{"task_type": "other"}, {"cv": 1}, {"cv": True}, {"cv": 2.5}])
def test_invalid_configuration(regression, config):
    options = {"task_type": "regression", **config}
    with pytest.raises(ValueError):
        ModelingAgent().train_models(regression, "target", **options)


def test_single_training_row():
    with pytest.raises(ValueError, match="at least two training rows"):
        ModelingAgent().train_models(pd.DataFrame({"x": [1, 2], "target": [2, 4]}), "target", "regression", test_size=1)
