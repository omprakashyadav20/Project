"""End-to-end tests run all real agents, including real CV and artifact writes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.workflow import AnalysisError, STAGES, analysis_key, parse_csv, run_analysis


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_real_end_to_end_analysis(tmp_path, task):
    df = pd.DataFrame({"feature": np.arange(30, dtype=float), "group": ["a", "b", None] * 10,
                       "all_null": [np.nan] * 30,
                       "target": ["yes", "no"] * 15 if task == "classification" else np.arange(30) * 2.3 + 0.1})
    df.loc[3, "feature"] = np.nan
    frame = parse_csv(df.to_csv(index=False).encode())
    original = frame.copy(deep=True)
    stages = []
    result = run_analysis(frame, "target", task, cv=3, output_dir=tmp_path / "outputs", report_dir=tmp_path / "reports",
                          progress=lambda index, stage: stages.append((index, stage)))
    assert stages == list(enumerate(STAGES))
    assert set(result) == {stage.lower() for stage in STAGES}
    prep, model, evaluation = result["preprocessing"], result["modeling"], result["evaluation"]
    assert prep["metadata"]["test_indices"] == model["metadata"]["test_indices"]
    assert set(model["metadata"]["train_indices"]).isdisjoint(model["metadata"]["test_indices"])
    assert prep["metadata"]["dropped_columns"] == ["all_null"]
    np.testing.assert_array_equal(model["trained_pipeline"].predict(prep["X_test_raw"]), model["predictions"])
    assert len(model["predictions"]) == evaluation["sample_count"] == 6
    assert len(model["model_comparison"]) == 3
    assert all(row["status"] == "success" for row in model["model_comparison"])
    metric = "accuracy" if task == "classification" else "rmse"
    assert evaluation["metrics"][metric] == pytest.approx(model["held_out_metrics"][metric])
    if task == "classification":
        assert evaluation["metrics"]["roc_auc"] is not None
        assert evaluation["roc_auc_reason"] is None
    report = result["reporting"]
    assert Path(report["report_path"]).read_text(encoding="utf-8") == report["report_text"]
    assert model["selected_model_name"] in report["report_text"]
    assert list((tmp_path / "outputs" / "eda").glob("*.png"))
    assert all(Path(path).is_file() for path in evaluation["charts"].values())
    pd.testing.assert_frame_equal(frame, original)


def test_multiclass_probability_mapping_and_duplicate_indices(tmp_path):
    df = pd.DataFrame({"x": range(30), "target": ["alpha", "beta", "gamma"] * 10}, index=[0] * 30)
    result = run_analysis(df, "target", "classification", cv=2, output_dir=tmp_path / "outputs", report_dir=tmp_path / "reports")
    assert result["evaluation"]["roc_auc_method"] == "ovr_macro"
    assert result["evaluation"]["metrics"]["roc_auc"] is not None
    assert result["preprocessing"]["y_train"].index.is_unique
    assert list(df.index) == [0] * 30


@pytest.mark.parametrize("data,message", [
    (b"", "empty"), (b"a,b\n", "no data"), (b"a,a\n1,2\n", "unique"),
    (b"a,\n1,2\n", "nonblank"), (b"a,b\n1,2,3\n", "fields"),
    (b"a,b\n1\n", "fields"), (b'a,b\n"unterminated,2', "parse"),
    (b"a;b\n1;2\n", "comma-separated"), (b"a,b\n\xff,2", "UTF-8"),
    (b"a,b\n1,\x00", "null bytes"),
])
def test_invalid_csv(data, message):
    with pytest.raises(ValueError, match=message):
        parse_csv(data)


def test_csv_quoted_fields_missing_values_and_bom():
    df = parse_csv('\ufeffname,value\n"a,b",1\nother,\n'.encode("utf-8"))
    assert df.loc[0, "name"] == "a,b"
    assert pd.isna(df.loc[1, "value"])


def test_result_identity_includes_data_target_task_and_settings():
    settings = dict(cv=3, test_size=0.2, random_state=42, scale_numeric=True)
    key = analysis_key(b"a,b\n1,2", "b", "regression", **settings)
    assert key == analysis_key(b"a,b\n1,2", "b", "regression", **settings)
    assert key != analysis_key(b"a,b\n1,3", "b", "regression", **settings)
    assert key != analysis_key(b"a,b\n1,2", "a", "regression", **settings)
    assert key != analysis_key(b"a,b\n1,2", "b", "classification", **settings)
    for name, value in dict(cv=2, test_size=0.3, random_state=7, scale_numeric=False).items():
        assert key != analysis_key(b"a,b\n1,2", "b", "regression", **{**settings, name: value})


@pytest.mark.parametrize("kind,stage,message", [("missing_target", "Preprocessing", "missing labels"), ("small", "Modeling", "at least two")])
def test_failure_retains_real_completed_stages(tmp_path, kind, stage, message):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "target": ["a", "b", "a", "b"]})
    if kind == "missing_target":
        df.loc[0, "target"] = None
    with pytest.raises(AnalysisError, match=message) as caught:
        run_analysis(df, "target", "classification", output_dir=tmp_path / "outputs", report_dir=tmp_path / "reports")
    assert caught.value.stage == stage
    assert "eda" in caught.value.partial_results
    assert "reporting" not in caught.value.partial_results
    assert not (tmp_path / "reports").exists()


def test_held_out_values_do_not_select_the_model(tmp_path):
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame({"x": np.arange(25, dtype=float), "target": np.arange(25) * 1.7 + 0.1})
    _, test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    first = run_analysis(df, "target", "regression", cv=2, output_dir=tmp_path / "out", report_dir=tmp_path / "reports")
    df.loc[test, "target"] = 1e6
    second = run_analysis(df, "target", "regression", cv=2, output_dir=tmp_path / "out", report_dir=tmp_path / "reports")
    assert first["modeling"]["model_comparison"] == second["modeling"]["model_comparison"]
    assert first["modeling"]["selected_model_name"] == second["modeling"]["selected_model_name"]
    assert first["evaluation"]["metrics"] != second["evaluation"]["metrics"]
    assert first["reporting"]["report_path"] != second["reporting"]["report_path"]
