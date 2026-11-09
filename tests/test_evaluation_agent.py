"""Held-out metrics, score mappings, chart artifacts, and undefined cases."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from agents.evaluation_agent import EvaluationAgent
from agents.modeling_agent import ModelingAgent


@pytest.fixture
def agent(tmp_path):
    return EvaluationAgent(tmp_path / "evaluation")


def assert_charts(result):
    for path in result["charts"].values():
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    json.dumps(result, allow_nan=False)


def test_binary_classification_and_probability_matrix(agent):
    actual = ["no", "no", "yes", "yes"]
    predicted = ["no", "yes", "yes", "yes"]
    probabilities = [[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
    before = plt.get_fignums()
    result = agent.evaluate_predictions(actual, predicted, "classification", y_proba=probabilities, classes=["no", "yes"])
    assert result["metrics"] == {"accuracy": 0.75, "precision": pytest.approx(2 / 3), "recall": 1, "f1": 0.8, "roc_auc": 1}
    assert result["averaging_method"] == "binary"
    assert result["positive_label"] == "yes"
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["class_labels"] == ["no", "yes"]
    assert result["roc_auc_reason"] is None
    assert plt.get_fignums() == before
    assert_charts(result)


def test_binary_decision_orientation_and_reversed_classes(agent):
    actual = ["a", "b", "a", "b"]
    # Scores refer to classes[1]='a', not lexicographically greater 'b'.
    result = agent.evaluate_predictions(actual, actual, "classification", decision_scores=[2, -2, 3, -3], classes=["b", "a"], positive_label="b")
    assert result["metrics"]["roc_auc"] == 1
    assert result["positive_label"] == "b"
    assert result["class_labels"] == ["b", "a"]


@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_multiclass_classification(agent, average):
    actual = ["a", "a", "b", "c", "c"]
    predicted = ["a", "b", "b", "a", "c"]
    probabilities = np.array([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.1, 0.8, 0.1], [0.5, 0.1, 0.4], [0.1, 0.1, 0.8]])
    result = agent.evaluate_predictions(actual, predicted, "classification", y_proba=probabilities, classes=["a", "b", "c"], average=average)
    assert result["metrics"]["accuracy"] == pytest.approx(0.6)
    for name, func in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
        assert result["metrics"][name] == pytest.approx(func(actual, predicted, average=average))
    assert result["metrics"]["roc_auc"] == pytest.approx(roc_auc_score(actual, probabilities, labels=["a", "b", "c"], multi_class="ovr", average=average))
    assert result["averaging_method"] == average
    assert result["roc_auc_method"] == f"ovr_{average}"
    assert_charts(result)


def test_multiclass_decision_scores_and_auto_average(agent):
    scores = [[5, -1, -2], [-3, 4, 1], [1, -1, 8]]
    result = agent.evaluate_predictions([0, 1, 2], [0, 1, 2], "classification", decision_scores=scores, classes=[0, 1, 2])
    assert result["averaging_method"] == "macro"
    assert result["metrics"]["roc_auc"] == 1


def test_regression_metrics_and_plots(agent):
    result = agent.evaluate_predictions([1, 2, 3], [1, 2, 4], "regression")
    assert result["metrics"] == {"mae": pytest.approx(1 / 3), "mse": pytest.approx(1 / 3), "rmse": pytest.approx(np.sqrt(1 / 3)), "r2": 0.5}
    assert set(result["charts"]) == {"actual_vs_predicted", "residuals"}
    assert result["warnings"] == []
    assert_charts(result)


@pytest.mark.parametrize("actual,predicted", [([1], [2]), ([2, 2], [2, 2]), ([2, 2], [1, 3])])
def test_undefined_r_squared(agent, actual, predicted):
    result = agent.evaluate_predictions(actual, predicted, "regression")
    assert result["metrics"]["r2"] is None
    assert any("R-squared is undefined" in message for message in result["warnings"])
    assert_charts(result)


def test_single_class_and_undefined_precision(agent):
    result = agent.evaluate_predictions([0, 0, 0], [0, 0, 0], "classification", y_proba=[0.1, 0.2, 0.1], classes=[0, 1])
    assert result["metrics"]["accuracy"] == 1
    assert result["metrics"]["precision"] is None
    assert result["metrics"]["recall"] is None
    assert result["metrics"]["f1"] is None
    assert result["metrics"]["roc_auc"] is None
    assert "At least two" in result["roc_auc_reason"]
    assert any("single class" in message for message in result["warnings"])
    json.dumps(result, allow_nan=False)


def test_no_positive_predictions_keeps_defined_f1(agent):
    result = agent.evaluate_predictions([0, 1], [0, 0], "classification")
    assert result["metrics"]["precision"] is None
    assert result["metrics"]["recall"] == 0
    assert result["metrics"]["f1"] == 0
    assert "No prediction" in result["roc_auc_reason"]


def test_macro_undefined_components_are_not_silently_omitted(agent):
    result = agent.evaluate_predictions([0, 1, 2], [0, 0, 0], "classification", average="macro")
    assert result["metrics"]["precision"] is None
    assert result["metrics"]["recall"] == pytest.approx(1 / 3)
    assert result["metrics"]["f1"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("options,reason", [
    ({"y_proba": [0.1, 0.9]}, "Provide classes"),
    ({"y_proba": [0.1], "classes": [0, 1]}, "one row"),
    ({"y_proba": [0.1, np.nan], "classes": [0, 1]}, "non-finite"),
    ({"y_proba": [0.1, 1.1], "classes": [0, 1]}, "between 0 and 1"),
    ({"y_proba": [[0.8, 0.8], [0.2, 0.8]], "classes": [0, 1]}, "sum to 1"),
    ({"decision_scores": [[1], [2]], "classes": [0, 1]}, "two columns"),
])
def test_invalid_optional_scores_leave_other_metrics_available(agent, options, reason):
    result = agent.evaluate_predictions([0, 1], [0, 1], "classification", **options)
    assert result["metrics"]["accuracy"] == 1
    assert result["metrics"]["roc_auc"] is None
    assert reason in result["roc_auc_reason"]


def test_multiclass_missing_test_class_auc_unavailable(agent):
    result = agent.evaluate_predictions([0, 1], [0, 1], "classification", y_proba=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], classes=[0, 1, 2])
    assert result["metrics"]["roc_auc"] is None
    assert "Every supplied class" in result["roc_auc_reason"]


@pytest.mark.parametrize("actual,predicted,task,message", [
    ([], [], "regression", "nonempty"),
    ([1, 2], [1], "regression", "equal lengths"),
    ([1, None], [1, 2], "regression", "missing values"),
    ([1, 2], [1, np.nan], "regression", "missing values"),
    ([[1], [2]], [1, 2], "regression", "one-dimensional"),
    ([1, np.inf], [1, 2], "regression", "finite real"),
    ([1, 2], [1, 2j], "regression", "finite real"),
    (["one", "two"], [1, 2], "regression", "finite real"),
    ([1, 2], [1, 2], "invalid", "task_type"),
    ([0.1, 0.2], [0.1, 0.2], "classification", "discrete"),
])
def test_invalid_inputs(agent, actual, predicted, task, message):
    with pytest.raises(ValueError, match=message):
        agent.evaluate_predictions(actual, predicted, task)


def test_invalid_configuration_and_index_alignment(agent):
    for options in [
        {"average": "unknown"}, {"positive_label": "missing"},
        {"y_proba": [0, 1], "decision_scores": [0, 1]},
        {"classes": [0, 0]}, {"classes": [0, 2]},
    ]:
        with pytest.raises(ValueError):
            agent.evaluate_predictions([0, 1], [0, 1], "classification", **options)
    with pytest.raises(ValueError, match="indices must match"):
        agent.evaluate_predictions(pd.Series([0, 1], index=[1, 2]), pd.Series([0, 1], index=[2, 1]), "classification")
    with pytest.raises(ValueError, match="not applicable"):
        agent.evaluate_predictions([0, 1], [0, 1], "regression", average="macro")


def test_nullable_labels(agent):
    result = agent.evaluate_predictions(pd.Series([True, False], dtype="boolean"), [True, False], "classification")
    assert result["metrics"]["accuracy"] == 1
    assert result["positive_label"] is True


def test_unique_chart_paths_and_failure_cleanup(agent, monkeypatch):
    first = agent.evaluate_predictions([1, 2], [1, 2], "regression")
    second = agent.evaluate_predictions([1, 2], [1, 2], "regression")
    assert set(first["charts"].values()).isdisjoint(second["charts"].values())
    for path in [*first["charts"].values(), *second["charts"].values()]:
        assert Path(path).parent == agent.output_dir.resolve()
    before_files = set(agent.output_dir.iterdir())
    before_figures = plt.get_fignums()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", fail)
    result = agent.evaluate_predictions([1, 2], [1, 2], "regression")
    assert all(path is None for path in result["charts"].values())
    assert any("disk full" in message for message in result["warnings"])
    assert set(agent.output_dir.iterdir()) == before_files
    assert plt.get_fignums() == before_figures
    assert result["metrics"]["r2"] == 1


def test_modeling_agent_result_integration(agent, monkeypatch):
    df = pd.DataFrame({"x": range(30), "target": ["a", "b"] * 15})
    modeling = ModelingAgent().train_models(df, "target", "classification", cv=2)
    pipeline = modeling["trained_pipeline"]
    X_test = df.drop(columns="target").loc[modeling["y_test"].index]
    probabilities = pipeline.predict_proba(X_test)

    def no_fit(*args, **kwargs):
        raise AssertionError("Evaluation must not fit or tune models")

    monkeypatch.setattr(type(pipeline), "fit", no_fit)
    result = agent.evaluate_predictions(modeling["y_test"], modeling["predictions"], "classification", y_proba=probabilities, classes=pipeline.classes_)
    assert result["metrics"]["accuracy"] == modeling["held_out_metrics"]["accuracy"]
    assert result["metrics"]["roc_auc"] is not None
