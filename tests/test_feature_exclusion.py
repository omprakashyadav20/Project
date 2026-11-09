"""Identifier exclusions preserve original EDA while isolating every model fit."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from utils.workflow import AnalysisError, analysis_key, default_excluded_features, run_analysis


def test_iris_id_excluded_from_all_fits_and_predictions(tmp_path, monkeypatch):
    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    measurements = list(df.columns)
    df.insert(0, "Id", np.arange(1, len(df) + 1))
    df["Species"] = [iris.target_names[label] for label in iris.target]
    original = df.copy(deep=True)
    fits = []
    original_fit = SimpleImputer.fit

    def observe_fit(self, X, y=None, **kwargs):
        assert list(X.columns) == measurements
        fits.append(tuple(X.index))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(SimpleImputer, "fit", observe_fit)
    settings = dict(cv=3, output_dir=tmp_path / "outputs", report_dir=tmp_path / "reports")
    first = run_analysis(df, "Species", "classification", **settings)
    prep, model = first["preprocessing"], first["modeling"]
    assert list(prep["X_train_raw"].columns) == measurements
    assert list(prep["X_test_raw"].columns) == measurements
    assert model["trained_pipeline"].feature_names_in_.tolist() == measurements
    assert all("Id" not in name for name in model["feature_names"])
    for result in (prep, model):
        assert result["metadata"]["excluded_features"] == ["Id"]
        assert result["metadata"]["included_features"] == measurements
    # Explicit preprocessing, Modeling's initial preprocessing, and final refit.
    expected = Counter({tuple(prep["y_train"].index): 3})
    for train, _ in StratifiedKFold(3, shuffle=True, random_state=42).split(prep["X_train_raw"], prep["y_train"]):
        expected[tuple(prep["y_train"].index[train])] += 3
    assert Counter(fits) == expected
    assert all(not set(indices).intersection(prep["y_test"].index) for indices in fits)
    assert first["eda"]["dataset_info"]["shape"] == [150, 6]
    assert "Id" in first["eda"]["numerical_statistics"]
    assert first["planner"]["dataset_info"]["shape"] == (150, 6)
    assert first["evaluation"]["metrics"]["roc_auc"] is not None
    text = first["reporting"]["report_text"]
    assert "| Excluded model features | Id |" in text
    assert "| Included model features | sepal length" in text
    assert "150 rows" in text
    assert Path(first["reporting"]["report_path"]).is_file()
    pd.testing.assert_frame_equal(df, original)
    # Altering an excluded identifier cannot alter CV selection or final scores.
    df["Id"] = -np.arange(len(df)) * 100000
    second = run_analysis(df, "Species", "classification", **settings)
    assert first["modeling"]["model_comparison"] == second["modeling"]["model_comparison"]
    assert first["modeling"]["selected_model_name"] == second["modeling"]["selected_model_name"]
    np.testing.assert_array_equal(first["modeling"]["predictions"], second["modeling"]["predictions"])
    assert first["evaluation"]["metrics"] == second["evaluation"]["metrics"]


def test_custom_exclusions_and_explicit_opt_out(tmp_path):
    df = pd.DataFrame({"Id": range(20), "x": np.arange(20) * 0.3, "extra": range(20), "y": np.arange(20) * 1.5})
    settings = dict(cv=2, output_dir=tmp_path / "outputs", report_dir=tmp_path / "reports")
    result = run_analysis(df, "y", "regression", excluded_columns=["Id", "extra"], **settings)
    assert result["modeling"]["metadata"]["included_features"] == ["x"]
    assert result["modeling"]["trained_pipeline"].feature_names_in_.tolist() == ["x"]
    assert result["evaluation"]["metrics"]["rmse"] is not None
    assert "extra" in result["eda"]["dataset_info"]["column_names"]
    opted_in = run_analysis(df, "y", "regression", excluded_columns=[], **settings)
    assert opted_in["modeling"]["metadata"]["excluded_features"] == []
    assert "Id" in opted_in["modeling"]["trained_pipeline"].feature_names_in_


@pytest.mark.parametrize("excluded,message", [
    (["y"], "target column cannot"), (["missing"], "must exist"),
    (["Id", "x"], "At least one"), ("Id", "must be a list"),
])
def test_invalid_exclusions_stop_before_modeling(tmp_path, excluded, message):
    df = pd.DataFrame({"Id": range(6), "x": range(6), "y": np.arange(6) * 0.5})
    with pytest.raises(AnalysisError, match=message) as caught:
        run_analysis(df, "y", "regression", excluded_columns=excluded, output_dir=tmp_path / "outputs")
    assert caught.value.stage == "Preprocessing"
    assert "modeling" not in caught.value.partial_results


def test_default_identifier_selection_and_cache_key():
    frame = pd.DataFrame(columns=["Id", "ID", " id ", "other", "target"])
    assert default_excluded_features(frame, "target") == ["Id", "ID", " id "]
    assert "Id" not in default_excluded_features(frame, "Id")
    assert analysis_key(b"csv", "target", "classification", excluded_columns=["Id"]) != analysis_key(b"csv", "target", "classification", excluded_columns=[])
