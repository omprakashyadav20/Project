"""Behavioral tests for dataset inspection and task selection."""

import pandas as pd
import pytest

from agents.planner_agent import PlannerAgent


@pytest.mark.parametrize("target", [["yes", "no", "yes"], [True, False, True], pd.Categorical(["a", "b", "a"])])
def test_classification(target):
    result = PlannerAgent().analyze_dataset(pd.DataFrame({"target": target}), "target")
    assert result["task_type"] == "classification"
    assert result["inferred_task_type"] == "classification"
    assert [step["stage"] for step in result["recommended_pipeline"]] == [
        "EDA", "preprocessing", "modeling", "evaluation", "reporting"
    ]


def test_regression_and_metadata_without_mutation():
    df = pd.DataFrame({"feature": [1, 2, 3], "group": ["a", "b", "c"], "target": [1.2, 2.8, 4.1]})
    original = df.copy(deep=True)
    result = PlannerAgent().analyze_dataset(df, "target")
    assert result["task_type"] == "regression"
    assert result["dataset_info"]["shape"] == (3, 3)
    assert result["dataset_info"]["numerical_columns"] == ["feature", "target"]
    assert result["dataset_info"]["categorical_columns"] == ["group"]
    assert result["dataset_info"]["target"]["unique_count"] == 3
    assert "rmse" in result["recommended_pipeline"][3]["metrics"]
    pd.testing.assert_frame_equal(df, original)


def test_missing_values():
    df = pd.DataFrame({"feature": [1, None, 3], "target": ["a", None, "b"]})
    result = PlannerAgent().analyze_dataset(df, "target")
    assert result["dataset_info"]["missing_values"] == {"feature": 1, "target": 1}
    assert result["dataset_info"]["target"]["missing_count"] == 1
    assert any("unlabeled" in warning for warning in result["warnings"])


def test_invalid_target():
    with pytest.raises(ValueError, match="does not exist"):
        PlannerAgent().analyze_dataset(pd.DataFrame({"x": [1]}), "missing")


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame(columns=["target"])])
def test_empty_dataset(df):
    with pytest.raises(ValueError, match="must not be empty"):
        PlannerAgent().analyze_dataset(df, "target")


@pytest.mark.parametrize("target", [[0, 1, 0], [10, 20, 30], [1.0, 2.0, 3.0], [0.1, 0.2, 0.1]])
def test_ambiguous_numerical_targets(target):
    df = pd.DataFrame({"target": target})
    result = PlannerAgent().analyze_dataset(df, "target")
    assert result["task_type"] is None
    assert any("Ambiguous" in warning for warning in result["warnings"])
    for override in ("classification", "regression"):
        explicit = PlannerAgent().analyze_dataset(df, "target", task_type=override)
        assert explicit["task_type"] == override
        assert explicit["inferred_task_type"] is None


def test_invalid_override():
    with pytest.raises(ValueError, match="task_type must be"):
        PlannerAgent().analyze_dataset(pd.DataFrame({"target": [1]}), "target", task_type="other")


def test_all_missing_target():
    result = PlannerAgent().analyze_dataset(pd.DataFrame({"target": [None, None]}), "target")
    assert result["task_type"] is None
    assert result["dataset_info"]["target"]["unique_ratio"] is None
    assert any("no observed values" in warning for warning in result["warnings"])


def test_invalid_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        PlannerAgent().analyze_dataset([], "target")
