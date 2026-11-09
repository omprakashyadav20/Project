"""Report content, real-agent integration, partial inputs, and safe artifacts."""

from copy import deepcopy
from pathlib import Path
import re
from urllib.parse import unquote

import pandas as pd
import pytest

from agents.eda_agent import EDAAgent
from agents.evaluation_agent import EvaluationAgent
from agents.modeling_agent import ModelingAgent
from agents.planner_agent import PlannerAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.reporting_agent import ReportingAgent


@pytest.fixture
def agent(tmp_path):
    return ReportingAgent(tmp_path / "reports")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_complete_report_from_real_agent_outputs(agent, tmp_path, task):
    df = pd.DataFrame({"number": list(range(20)), "category": ["a", "b"] * 10,
                       "target": ["yes", "no"] * 10 if task == "classification" else [1.5 * i + 0.1 for i in range(20)]})
    df.loc[0, "number"] = None
    planner = PlannerAgent().analyze_dataset(df, "target", task_type=task)
    eda = EDAAgent(tmp_path / "charts with spaces").analyze_dataset(df)
    prep = PreprocessingAgent().preprocess_dataset(df, "target", task)
    model = ModelingAgent().train_models(df, "target", task, cv=2)
    evaluation = EvaluationAgent(tmp_path / "evaluation").evaluate_predictions(model["y_test"], model["predictions"], task)
    result = agent.generate_report(planner, eda, prep, model, evaluation, title=f"Sample {task} analysis")
    text = result["report_text"]
    path = Path(result["report_path"])
    assert path.parent == agent.output_dir.resolve()
    assert path.suffix == ".md"
    assert path.read_text(encoding="utf-8") == text
    assert f"# Sample {task} analysis" in text
    assert "+05:30 (Asia/Kolkata)" in text
    for heading in ["Executive summary", "Dataset and data quality", "Exploratory data analysis", "Preprocessing and leakage safeguards", "Model comparison and selection", "Final held-out evaluation", "Limitations and warnings", "Conclusion"]:
        assert f"## {heading}" in text
    assert "20 rows × 3 columns" in text
    assert "target" in text and task in text
    assert "training-median imputation" in text
    assert "one-hot encoding" in text
    assert "training-only CV fold" in text
    assert model["selected_model_name"] in text
    assert "| Model | CV metric | Mean | Std | Fold scores | Status | Error |" in text
    assert "| number | 1 | 5 |" in text
    assert "Duplicate rows (excluding first occurrences): 0" in text
    if task == "classification":
        assert "| ACCURACY |" in text
        assert "| F1 |" in text
        assert "Precision/recall/F1 averaging: binary" in text
        assert "No prediction probabilities" in text
    else:
        for metric in ["MAE", "MSE", "RMSE", "R2"]:
            assert f"| {metric} |" in text
        assert "actual\\_vs\\_predicted" in text
    links = re.findall(r"!\[.*?\]\((.*?)\)", text)
    assert links
    for link in links:
        assert (path.parent / unquote(link)).resolve().is_file()
    assert any("%20" in link for link in links)


def test_missing_optional_data(agent):
    planner = {"dataset_info": {"shape": (10, 2), "target": {"column": "outcome"}, "missing_values": {"feature": 2}}, "task_type": "regression"}
    original = deepcopy(planner)
    result = agent.generate_report(planner_result=planner)
    assert "10 rows × 2 columns" in result["report_text"]
    assert "| feature | 2 | Unavailable |" in result["report_text"]
    assert "No selected model was supplied" in result["report_text"]
    assert "Final predictive performance cannot be assessed" in result["report_text"]
    assert any("EDA results" in warning for warning in result["warnings"])
    assert planner == original


def test_all_results_missing_and_malformed_nested_data(agent):
    result = agent.generate_report()
    assert Path(result["report_path"]).is_file()
    assert "Unavailable" in result["report_text"]
    assert "a selected model was not available" in result["report_text"].lower()
    result = agent.generate_report(
        planner_result={"dataset_info": "wrong"}, eda_result={"numerical_statistics": {"x": None}},
        preprocessing_result={"metadata": []}, modeling_result={"model_comparison": "wrong"},
        evaluation_result={"metrics": [], "charts": "wrong"},
    )
    assert any("malformed" in warning for warning in result["warnings"])
    assert Path(result["report_path"]).is_file()


def test_modeling_metric_fallback_and_no_invented_scores(agent):
    result = agent.generate_report(modeling_result={
        "metadata": {"task_type": "regression", "train_rows": 8, "test_rows": 2},
        "selected_model_name": "LinearRegression", "best_cv_score": 1.5,
        "held_out_metrics": {"rmse": 2.25},
    })
    text = result["report_text"]
    assert "Modeling Agent fallback (limited metrics)" in text
    assert "| RMSE | 2.25 |" in text
    assert "| MAE | Unavailable |" in text
    assert "| R2 | Unavailable |" in text
    assert "selected model, with reported CV score 1.5" in text
    assert "excellent" not in text.lower() and "highly accurate" not in text.lower()


def test_undefined_metrics_warnings_and_failed_models(agent):
    result = agent.generate_report(
        modeling_result={"model_comparison": [{"model_name": "failed model", "status": "failed", "error": "fit failed", "warnings": ["convergence issue"]}]},
        evaluation_result={"task_type": "classification", "metrics": {"accuracy": 1, "precision": None, "recall": float("nan"), "roc_auc": None}, "averaging_method": "binary", "roc_auc_reason": "Single actual class", "warnings": ["single class test set"]},
    )
    text = result["report_text"]
    assert "| PRECISION | Unavailable |" in text
    assert "| RECALL | Unavailable |" in text
    assert "Single actual class" in text
    assert "fit failed" in text
    assert any("convergence issue" in message for message in result["warnings"])
    assert any("single class test set" in message for message in result["warnings"])
    assert "nan" not in text.lower()


def test_conflicting_task_and_target_metadata_is_flagged(agent):
    result = agent.generate_report(
        planner_result={"task_type": "classification", "dataset_info": {"target": {"column": "first"}}},
        modeling_result={"metadata": {"task_type": "classification", "target_column": "second"}},
        evaluation_result={"task_type": "regression", "metrics": {"rmse": 0.5}},
    )
    assert any("Conflicting task types" in message for message in result["warnings"])
    assert any("Conflicting target columns" in message for message in result["warnings"])
    assert "| Task type | regression |" in result["report_text"]


def test_unique_sanitized_filenames_and_content(agent):
    title = "../../CON <script>alert(1)</script>\n# injected | [click](javascript:alert(1))"
    first = agent.generate_report(title=title, planner_result={"warnings": ["<img src=x onerror=alert(1)>\n## fake | column"]})
    original = Path(first["report_path"]).read_bytes()
    second = agent.generate_report(title=title)
    assert first["report_path"] != second["report_path"]
    for result in [first, second]:
        path = Path(result["report_path"])
        assert path.parent == agent.output_dir.resolve()
        assert re.fullmatch(r"[A-Za-z0-9_-]+\.md", path.name)
        assert "<script>" not in result["report_text"]
        assert "\n# injected" not in result["report_text"]
        assert "[click](javascript:" not in result["report_text"]
        assert "&lt;script&gt;" in result["report_text"]
    assert "<img" not in first["report_text"]
    assert Path(first["report_path"]).read_bytes() == original


def test_missing_and_unsafe_chart_references(agent, tmp_path):
    result = agent.generate_report(eda_result={"charts": {
        "histograms": {"missing": str(tmp_path / "missing.png"), "injection": "javascript:alert(1)", "bad": 123},
        "correlation_heatmap": None,
    }})
    assert "![" not in result["report_text"]
    assert any("does not exist" in message for message in result["warnings"])
    assert any("only local" in message for message in result["warnings"])
    assert any("Invalid EDA chart path" in message for message in result["warnings"])


@pytest.mark.parametrize("argument", ["planner_result", "eda_result", "preprocessing_result", "modeling_result", "evaluation_result"])
def test_invalid_top_level_inputs(agent, argument):
    with pytest.raises(TypeError, match="mapping or None"):
        agent.generate_report(**{argument: [1, 2]})


def test_invalid_titles_and_write_failure(agent, tmp_path):
    with pytest.raises(TypeError, match="title must be a string"):
        agent.generate_report(title=123)
    with pytest.raises(ValueError, match="must not be blank"):
        agent.generate_report(title="  ")
    blocker = tmp_path / "file_instead_of_directory"
    blocker.write_text("preserve", encoding="utf-8")
    with pytest.raises(OSError, match="Unable to save report"):
        ReportingAgent(blocker).generate_report()
    assert blocker.read_text(encoding="utf-8") == "preserve"
