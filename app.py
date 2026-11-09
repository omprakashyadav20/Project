"""Streamlit dashboard. Launch with: python -m streamlit run app.py."""

from hashlib import sha256
import logging
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote

import pandas as pd
import streamlit as st

from agents.planner_agent import PlannerAgent
from utils.workflow import AnalysisError, STAGES, analysis_key, default_excluded_features, parse_csv, run_analysis


LOGGER = logging.getLogger(__name__)


def show_charts(charts: dict[str, Any]) -> None:
    """Display saved agent charts and gracefully handle unavailable files."""
    for kind, entry in charts.items():
        entries = entry.items() if isinstance(entry, dict) else [(kind, entry)]
        for name, path in entries:
            if path and Path(path).is_file():
                st.image(str(path), caption=f"{kind.replace('_', ' ').title()} · {name}")
            elif path:
                st.warning(f"Chart file is no longer available: {name}")


def show_report(report: dict[str, Any]) -> None:
    """Render Markdown with local chart paths resolved through st.image."""
    text = report["report_text"]
    st.download_button("Download Markdown report", text.encode("utf-8"),
                       file_name=Path(report["report_path"]).name, mime="text/markdown",
                       key="download_report")
    st.caption(f"Saved to {report['report_path']}")
    position = 0
    for match in re.finditer(r"!\[((?:\\.|[^\]])*)\]\(([^)\n]+)\)", text):
        st.markdown(text[position:match.start()])
        image = Path(report["report_path"]).parent / unquote(match.group(2))
        if image.is_file():
            st.image(str(image), caption=re.sub(r"\\(.)", r"\1", match.group(1)))
        else:
            st.caption("A chart referenced by this report is no longer available.")
        position = match.end()
    st.markdown(text[position:])


def show_results(results: dict[str, Any]) -> None:
    """Render available stages, including partial results from a failed run."""
    tabs = st.tabs(["Plan", "EDA", "Preprocessing", "Models", "Evaluation", "Report"])
    with tabs[0]:
        planner = results.get("planner")
        if planner:
            st.subheader("Analysis plan")
            st.dataframe(pd.DataFrame(planner["recommended_pipeline"]), hide_index=True)
            for warning in planner["warnings"]:
                st.warning(warning)
        else:
            st.info("No planning results available.")
    with tabs[1]:
        eda = results.get("eda")
        if eda:
            st.subheader("Data quality")
            left, right = st.columns(2)
            left.metric("Duplicate rows", eda["duplicate_rows"]["count"])
            right.metric("Missing cells", sum(item["count"] for item in eda["missing_values"].values()))
            st.dataframe(pd.DataFrame.from_dict(eda["missing_values"], orient="index"))
            st.subheader("Numerical statistics")
            st.dataframe(pd.DataFrame.from_dict(eda["numerical_statistics"], orient="index"))
            st.subheader("Categorical summaries")
            st.dataframe(pd.DataFrame.from_dict(eda["categorical_summaries"], orient="index"))
            with st.expander("EDA visualizations", expanded=True):
                show_charts(eda["charts"])
            for warning in eda["warnings"]:
                st.warning(warning)
        else:
            st.info("EDA has not completed.")
    with tabs[2]:
        prep = results.get("preprocessing")
        if prep:
            metadata = prep["metadata"]
            st.subheader("Training and test split")
            left, right = st.columns(2)
            left.metric("Training rows", metadata["train_rows"])
            right.metric("Test rows", metadata["test_rows"])
            st.json({key: value for key, value in metadata.items() if key not in ("train_indices", "test_indices", "warnings")})
            st.write("Numerical features use median imputation and optional scaling. Categorical features use mode imputation and one-hot encoding. All transformations are learned from training rows only.")
            st.caption("Cross-validation fits a fresh preprocessing pipeline inside each fold.")
            for warning in metadata["warnings"]:
                st.warning(warning)
        else:
            st.info("Preprocessing has not completed.")
    with tabs[3]:
        model = results.get("modeling")
        if model:
            st.subheader("Cross-validation comparison")
            st.write(f"Selected model: **{model['selected_model_name']}**")
            st.caption(f"{model['metadata']['cv_folds']} folds · {model['metadata']['selection_metric']} · {model['metadata']['selection_direction']} · selected without using test performance")
            st.dataframe(pd.DataFrame(model["model_comparison"]), hide_index=True)
            for warning in model["warnings"]:
                st.warning(warning)
        else:
            st.info("Model comparison has not completed.")
    with tabs[4]:
        evaluation = results.get("evaluation")
        if evaluation:
            st.subheader("Final held-out evaluation")
            for column, (name, value) in zip(st.columns(len(evaluation["metrics"])), evaluation["metrics"].items()):
                column.metric(name.upper().replace("_", "-"), "Unavailable" if value is None else f"{value:.4f}")
            if evaluation["task_type"] == "classification":
                st.caption(f"Averaging: {evaluation['averaging_method']} · Positive label: {evaluation['positive_label']} · ROC-AUC method: {evaluation['roc_auc_method'] or 'unavailable'}")
                st.dataframe(pd.DataFrame(evaluation["per_class_metrics"]), hide_index=True)
            show_charts(evaluation["charts"])
            for warning in evaluation["warnings"]:
                st.warning(warning)
        else:
            st.info("Held-out evaluation has not completed.")
    with tabs[5]:
        if results.get("reporting"):
            show_report(results["reporting"])
        else:
            st.info("A report becomes available after all analysis stages complete.")


def main() -> None:
    """Handle uploads and settings; train exclusively on an explicit run click."""
    st.set_page_config(page_title="Autonomous Data Scientist", page_icon="📊", layout="wide")
    st.title("Autonomous Data Scientist")
    st.caption("From dataset to evidence: explore, compare models, and share a complete report.")
    with st.sidebar:
        st.header("Analysis settings")
        test_size = st.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05)
        cv = st.slider("Maximum CV folds", 2, 10, 5)
        random_state = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42, step=1)
        scale_numeric = st.checkbox("Scale numerical features", value=True)
        st.caption("Results stay in this browser session. Change a setting, then run again to create matching results.")
    upload = st.file_uploader("Upload a CSV dataset", type=["csv"], accept_multiple_files=False)
    if upload is None:
        st.info("Upload a UTF-8 CSV to preview your data and configure an analysis.")
        return
    data = upload.getvalue()
    digest = sha256(data).hexdigest()
    try:
        if st.session_state.get("csv_digest") != digest:
            frame = parse_csv(data)
            st.session_state["csv_frame"] = frame
            st.session_state["csv_digest"] = digest
        df = st.session_state["csv_frame"]
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
        return
    st.subheader("Dataset preview")
    left, middle, right = st.columns(3)
    left.metric("Rows", len(df))
    middle.metric("Columns", len(df.columns))
    right.metric("Missing cells", int(df.isna().sum().sum()))
    st.dataframe(df.head(50), hide_index=True)
    with st.expander("Column names and data types"):
        st.dataframe(pd.DataFrame({"Column": df.columns, "Data type": [str(dtype) for dtype in df.dtypes]}), hide_index=True)
    target = st.selectbox("Target column", list(df.columns), index=len(df.columns) - 1, key=f"target_{digest}")
    features = [name for name in df.columns if name != target]
    excluded = st.multiselect(
        "Exclude feature columns", features, default=default_excluded_features(df, target),
        key=f"excluded_{digest}_{target}",
        help="Id is excluded by default. Excluded columns remain in the data-quality analysis but are removed before preprocessing, cross-validation, and held-out prediction.",
    )
    included = [name for name in features if name not in excluded]
    st.caption("Included model features: " + (", ".join(included) if included else "None"))
    if not included:
        st.error("Keep at least one feature column to run analysis.")
    try:
        suggestion = PlannerAgent().analyze_dataset(df, target)
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
        return
    inferred = suggestion["inferred_task_type"]
    options = ["classification", "regression"]
    task = st.selectbox("Task type", options, index=options.index(inferred) if inferred in options else None,
                        placeholder="Choose classification or regression", key=f"task_{digest}_{target}")
    if inferred:
        st.caption(f"Planner suggestion: {inferred}. You can override it above.")
    else:
        st.info("The Planner cannot determine the task unambiguously. Choose a task type explicitly.")
    for warning in suggestion["warnings"]:
        st.warning(warning)
    missing_labels = bool(df[target].isna().any())
    if missing_labels:
        st.error("The selected target contains missing labels. Complete or remove those rows in the CSV before running; target values are never imputed.")
    settings = dict(test_size=test_size, cv=cv, random_state=int(random_state), scale_numeric=scale_numeric,
                    excluded_columns=[name for name in features if name in excluded])
    key = analysis_key(data, target, task, **settings)
    clicked = st.button("Run Autonomous Analysis", type="primary", disabled=task is None or missing_labels or not included)
    if clicked:
        bar = st.progress(0, text="Starting analysis…")
        def progress(index: int, stage: str) -> None:
            bar.progress(index / len(STAGES), text=f"{index + 1}/{len(STAGES)} · {stage}")
        try:
            result = run_analysis(df, target, task, progress=progress, **settings)
            st.session_state["analysis_entry"] = {"key": key, "results": result, "error": None}
            bar.progress(1.0, text="Analysis complete")
        except AnalysisError as exc:
            st.session_state["analysis_entry"] = {"key": key, "results": exc.partial_results, "error": str(exc)}
            bar.empty()
        except Exception:
            LOGGER.exception("Unexpected dashboard analysis failure")
            st.session_state["analysis_entry"] = {"key": key, "results": {}, "error": "An unexpected error interrupted analysis. Check the terminal log for details."}
            bar.empty()
    entry = st.session_state.get("analysis_entry")
    if entry and entry["key"] == key:
        if entry["error"]:
            st.error(entry["error"])
            st.info("Review the message and completed stages below. For small datasets, consider a smaller test fraction or more labeled rows.")
        else:
            st.success("Analysis complete. Explore the results below or download your report.")
        show_results(entry["results"])
    elif entry:
        st.info("The dataset or analysis settings have changed. Run analysis to see results for the current selection.")


if __name__ == "__main__":
    main()
