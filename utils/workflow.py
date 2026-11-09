"""CSV validation and the dashboard's testable, framework-independent workflow."""

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from agents.planner_agent import PlannerAgent
from agents.eda_agent import EDAAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.modeling_agent import ModelingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reporting_agent import ReportingAgent


STAGES = ("Planner", "EDA", "Preprocessing", "Modeling", "Evaluation", "Reporting")


class AnalysisError(RuntimeError):
    """A failed stage together with results completed before that failure."""

    def __init__(self, stage: str, cause: Exception, partial_results: dict[str, Any]) -> None:
        super().__init__(f"{stage} could not complete: {cause}")
        self.stage = stage
        self.partial_results = partial_results


def parse_csv(data: bytes) -> pd.DataFrame:
    """Read a UTF-8 CSV, rejecting malformed rows and ambiguous column names.

    Headers are checked before pandas can silently rename duplicate/blank names.
    Row widths are checked before pandas can infer an unintended index. Missing
    feature values remain missing; no target values are dropped or imputed.
    """
    if not isinstance(data, bytes):
        raise TypeError("CSV input must be bytes.")
    if not data.strip():
        raise ValueError("The CSV is empty. Upload a file with a header and data rows.")
    try:
        text = data.decode("utf-8-sig")
        if "\x00" in text:
            raise ValueError("The CSV contains null bytes. Export it as a UTF-8 CSV.")
        reader = csv.reader(StringIO(text, newline=""), strict=True)
        header = next((row for row in reader if row and any(cell.strip() for cell in row)), None)
        if header is None or len(header) < 2:
            raise ValueError("The CSV must have at least two comma-separated columns (features and target).")
        if any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise ValueError("CSV column names must be nonblank and unique.")
        for row in reader:
            if not row:  # Match pandas' handling of blank lines.
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV row ending at line {reader.line_num} has {len(row)} fields; expected {len(header)}.")
        df = pd.read_csv(BytesIO(data), encoding="utf-8-sig", on_bad_lines="error")
    except UnicodeDecodeError as exc:
        raise ValueError("Cannot decode CSV. Save the file with UTF-8 encoding and upload again.") from exc
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot parse CSV: {exc}") from exc
    if df.empty:
        raise ValueError("The CSV has no data rows.")
    if list(df.columns) != header:
        raise ValueError("CSV headers could not be read unambiguously; check blank lines and column names.")
    return df


def analysis_key(data: bytes, target: str, task: str | None, **settings: Any) -> str:
    """Identify an analysis by complete file contents and all run settings."""
    config = json.dumps({"target": target, "task": task, **settings}, sort_keys=True)
    return sha256(data + b"\x00" + config.encode("utf-8")).hexdigest()


def default_excluded_features(df: pd.DataFrame, target_column: str) -> list[str]:
    """Suggest conventional Id columns without inspecting target relationships."""
    return [name for name in df.columns
            if name != target_column and isinstance(name, str) and name.strip().casefold() == "id"]


def run_analysis(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = True,
    cv: int = 5,
    excluded_columns: list[str] | None = None,
    output_dir: str | Path = "outputs",
    report_dir: str | Path = "reports",
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Execute all six real agents in order, retaining partial results on failure.

    ModelingAgent owns CV and model selection; its transformed preprocessing
    inputs are never reused in CV. Both preprocessing calls use identical split
    settings. A private RangeIndex makes split reconstruction unambiguous even
    for caller DataFrames with duplicate indices. The caller's data is untouched.
    No model fitting occurs unless this function is explicitly called.
    Planner and EDA retain the original columns. Feature exclusions are applied
    before either preprocessing call, so CV and held-out prediction never see
    those columns. None defaults to conventional Id columns; [] opts out.
    """
    results: dict[str, Any] = {}
    frame = df.reset_index(drop=True).copy() if isinstance(df, pd.DataFrame) else df
    output_dir = Path(output_dir)
    settings = dict(test_size=test_size, random_state=random_state, scale_numeric=scale_numeric)
    for index, stage in enumerate(STAGES):
        if progress:
            progress(index, stage)
        try:
            if stage == "Planner":
                results["planner"] = PlannerAgent().analyze_dataset(frame, target_column, task_type=task_type)
            elif stage == "EDA":
                results["eda"] = EDAAgent(output_dir / "eda").analyze_dataset(frame)
            elif stage == "Preprocessing":
                excluded = default_excluded_features(frame, target_column) if excluded_columns is None else excluded_columns
                if not isinstance(excluded, list) or not all(isinstance(name, str) for name in excluded):
                    raise ValueError("excluded_columns must be a list of feature column names or None.")
                if target_column in excluded:
                    raise ValueError("The target column cannot be excluded as a feature.")
                if any(name not in frame.columns for name in excluded):
                    raise ValueError("Excluded feature columns must exist in the dataset.")
                excluded = [name for name in frame.columns if name in excluded]
                included = [name for name in frame.columns if name != target_column and name not in excluded]
                if not included:
                    raise ValueError("At least one input feature must remain after excluding columns.")
                model_frame = frame.drop(columns=excluded)
                results["preprocessing"] = PreprocessingAgent().preprocess_dataset(model_frame, target_column, task_type, **settings)
                results["preprocessing"]["metadata"].update(included_features=included, excluded_features=excluded)
            elif stage == "Modeling":
                modeled = ModelingAgent().train_models(model_frame, target_column, task_type, cv=cv, **settings)
                modeled["metadata"].update(included_features=included, excluded_features=excluded)
                prepared = results["preprocessing"]
                if (modeled["metadata"]["train_indices"] != prepared["metadata"]["train_indices"]
                        or modeled["metadata"]["test_indices"] != prepared["metadata"]["test_indices"]):
                    raise ValueError("Modeling and preprocessing splits differ; evaluation was stopped.")
                results["modeling"] = modeled
            elif stage == "Evaluation":
                modeled = results["modeling"]
                score_options: dict[str, Any] = {}
                score_warning = None
                if task_type == "classification":
                    pipeline = modeled["trained_pipeline"]
                    X_test = results["preprocessing"]["X_test_raw"]
                    try:
                        score_options["classes"] = pipeline.classes_
                        if hasattr(pipeline, "predict_proba"):
                            score_options["y_proba"] = pipeline.predict_proba(X_test)
                        elif hasattr(pipeline, "decision_function"):
                            score_options["decision_scores"] = pipeline.decision_function(X_test)
                    except Exception as exc:
                        score_options = {}
                        score_warning = f"Optional ROC-AUC scores could not be generated: {exc}"
                evaluated = EvaluationAgent(output_dir / "evaluation").evaluate_predictions(
                    modeled["y_test"], modeled["predictions"], task_type, **score_options,
                )
                if score_warning:
                    evaluated["warnings"].append(score_warning)
                results["evaluation"] = evaluated
            else:
                results["reporting"] = ReportingAgent(report_dir).generate_report(
                    results["planner"], results["eda"], results["preprocessing"],
                    results["modeling"], results["evaluation"],
                )
        except Exception as exc:
            raise AnalysisError(stage, exc, results) from exc
    return results
