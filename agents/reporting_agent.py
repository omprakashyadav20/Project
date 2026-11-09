"""Generate evidence-based Markdown reports from agent result dictionaries."""

from collections.abc import Mapping
from datetime import datetime
import html
import math
from numbers import Real
import os
from pathlib import Path
import re
from typing import Any
import unicodedata
from urllib.parse import quote
from uuid import uuid4
from zoneinfo import ZoneInfo


class ReportingAgent:
    """Combine available results without fitting models or inventing findings."""

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self.output_dir = Path(output_dir)

    def generate_report(
        self,
        planner_result: Mapping[str, Any] | None = None,
        eda_result: Mapping[str, Any] | None = None,
        preprocessing_result: Mapping[str, Any] | None = None,
        modeling_result: Mapping[str, Any] | None = None,
        evaluation_result: Mapping[str, Any] | None = None,
        *,
        title: str = "Autonomous Data Scientist Report",
    ) -> dict[str, Any]:
        """Save and return a Markdown report assembled from existing agent outputs.

        Each argument accepts the corresponding agent's original dictionary;
        missing/partial dictionaries produce explicitly unavailable sections.
        Evaluation metrics take precedence over ModelingAgent's limited held-out
        metric fallback. Task type prefers evaluation, modeling, preprocessing,
        then planner; inconsistent sources generate warnings. No metrics are
        recomputed and no input objects are modified or serialized wholesale.

        Dynamic text is escaped for Markdown/HTML. Existing local raster chart
        files are linked relative to the saved report, with URL-encoded paths.
        Missing/unsafe chart references are documented instead of embedded.
        Filenames use a sanitized title, timestamp, and UUID with exclusive
        creation. Timestamps use Asia/Kolkata, independent of the host timezone.

        Returns:
            report_path (absolute string), report_text, and warnings (strings).

        Raises:
            TypeError: If a top-level result is neither a mapping nor None, or
                title is not a string.
            ValueError: If title is blank.
            OSError: If the report directory or file cannot be written.
        """
        if not isinstance(title, str):
            raise TypeError("title must be a string.")
        if not title.strip():
            raise ValueError("title must not be blank.")
        messages: list[str] = []
        sources = {}
        for name, result in (
            ("Planner", planner_result), ("EDA", eda_result),
            ("Preprocessing", preprocessing_result), ("Modeling", modeling_result),
            ("Evaluation", evaluation_result),
        ):
            if result is not None and not isinstance(result, Mapping):
                raise TypeError(f"{name} result must be a mapping or None.")
            sources[name] = result if result is not None else {}
            if not sources[name]:
                messages.append(f"{name} results were not provided or are empty.")

        def mapping(value: Any, name: str) -> Mapping:
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                messages.append(f"{name} must be a mapping; omitted malformed data.")
                return {}
            return value

        def records(value: Any, name: str) -> list:
            if value is None:
                return []
            if not isinstance(value, (list, tuple)):
                messages.append(f"{name} must be a list; omitted malformed data.")
                return []
            return list(value)

        def collect_warnings(value: Any, source: str) -> None:
            if value is None:
                return
            entries = [value] if isinstance(value, str) else records(value, f"{source} warnings")
            messages.extend(f"{source}: {entry}" for entry in entries if isinstance(entry, str) and entry.strip())

        planner, eda, prep, model, evaluation = (sources[name] for name in ("Planner", "EDA", "Preprocessing", "Modeling", "Evaluation"))
        pinfo = mapping(planner.get("dataset_info"), "Planner dataset_info")
        einfo = mapping(eda.get("dataset_info"), "EDA dataset_info")
        pmeta = mapping(prep.get("metadata"), "Preprocessing metadata")
        mmeta = mapping(model.get("metadata"), "Modeling metadata")
        target_info = mapping(pinfo.get("target"), "Planner target")
        for name, source in sources.items():
            collect_warnings(source.get("warnings"), name)
        collect_warnings(pmeta.get("warnings"), "Preprocessing")
        collect_warnings(mmeta.get("warnings"), "Modeling metadata")

        tasks = [(name, value) for name, value in (
            ("Evaluation", evaluation.get("task_type")), ("Modeling", mmeta.get("task_type")),
            ("Preprocessing", pmeta.get("task_type")), ("Planner", planner.get("task_type")),
        ) if value is not None]
        valid_tasks = []
        for name, value in tasks:
            if isinstance(value, str) and value in ("classification", "regression"):
                valid_tasks.append((name, value))
            else:
                messages.append(f"{name} task type is invalid or unresolved.")
        task = valid_tasks[0][1] if valid_tasks else None
        if len({value for _, value in valid_tasks}) > 1:
            messages.append("Conflicting task types were supplied; sections retain source-specific results and must be reconciled before interpretation.")
        if task is None:
            messages.append("Task type is unavailable; task-specific performance conclusions cannot be made.")
        targets = [value for value in (pmeta.get("target_column"), mmeta.get("target_column"), target_info.get("column")) if isinstance(value, str)]
        target = targets[0] if targets else None
        if len(set(targets)) > 1:
            messages.append("Conflicting target columns were supplied; results may describe different analyses.")
        info = {**pinfo, **einfo}
        shape = info.get("shape")
        shape_text = (f"{self._text(shape[0])} rows × {self._text(shape[1])} columns"
                      if isinstance(shape, (list, tuple)) and len(shape) == 2 else "Unavailable")
        selected = model.get("selected_model_name")
        selected = selected if isinstance(selected, str) and selected.strip() else None
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_- ")[:60] or "report"
        filename = f"report_{slug}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}.md"
        path = self.output_dir / filename
        lines = [f"# {self._text(title)}", "", f"Generated: {now.isoformat(timespec='seconds')} (Asia/Kolkata)", ""]

        def section(name: str) -> None:
            lines.extend([f"## {name}", ""])

        def table(headers: list[str], rows: list[list[Any]]) -> None:
            if not rows:
                lines.extend(["Unavailable in supplied results.", ""])
                return
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            lines.extend("| " + " | ".join(self._text(value) for value in row) + " |" for row in rows)
            lines.append("")

        section("Executive summary")
        lines.extend([
            f"Dataset: {shape_text}. Target: {self._text(target)}. Task: {self._text(task)}.", "",
            f"The Modeling Agent reports the selected model as {self._text(selected)}." if selected else "No selected model was supplied.", "",
        ])
        metrics = mapping(evaluation.get("metrics"), "Evaluation metrics")
        metric_source = "Evaluation Agent"
        if not metrics:
            metrics = mapping(model.get("held_out_metrics"), "Modeling held_out_metrics")
            metric_source = "Modeling Agent fallback (limited metrics)"
            if metrics:
                messages.append("Evaluation metrics were unavailable; using only the Modeling Agent's supplied held-out metrics.")
        allowed_metrics = {"classification": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                           "regression": ["mae", "mse", "rmse", "r2"]}
        metric_names = allowed_metrics.get(task, [])
        available_metrics = {name: metrics[name] for name in metric_names if name in metrics and self._finite(metrics[name])}
        if available_metrics:
            summary = "; ".join(f"{name.upper()} = {self._text(value)}" for name, value in available_metrics.items())
            lines.extend([f"Supplied held-out results ({metric_source}): {summary}.", ""])
        else:
            lines.extend(["No finite task-appropriate held-out metrics were supplied.", ""])
            messages.append("No finite task-appropriate held-out metrics are available.")

        section("Dataset and data quality")
        table(["Property", "Value"], [
            ["Shape", shape_text], ["Target column", target], ["Task type", task],
            ["Numerical columns (including target when numerical)", info.get("numerical_columns")],
            ["Categorical columns (including target when categorical)", info.get("categorical_columns")],
            ["Other column types", info.get("other_columns")],
            ["Target dtype", target_info.get("dtype")], ["Observed target unique values", target_info.get("unique_count")],
        ])
        dtypes = mapping(einfo.get("data_types"), "EDA data_types")
        if dtypes:
            table(["Column", "Data type"], [[name, dtype] for name, dtype in dtypes.items()])
        lines.extend(["### Missing values", ""])
        missing = mapping(eda.get("missing_values"), "EDA missing_values")
        if not missing:
            missing = mapping(pinfo.get("missing_values"), "Planner missing_values")
        missing_rows = []
        for name, value in missing.items():
            if isinstance(value, Mapping):
                missing_rows.append([name, value.get("count"), value.get("percentage")])
            else:
                missing_rows.append([name, value, None])
        table(["Column", "Missing count", "Missing (%)"], missing_rows)
        duplicates = mapping(eda.get("duplicate_rows"), "EDA duplicate_rows")
        lines.extend([f"Duplicate rows (excluding first occurrences): {self._text(duplicates.get('count'))}; percentage: {self._text(duplicates.get('percentage'))}.", ""])

        section("Exploratory data analysis")
        stats = mapping(eda.get("numerical_statistics"), "EDA numerical_statistics")
        rows = []
        for name, value in stats.items():
            item = mapping(value, f"Statistics for {name}")
            rows.append([name, *[item.get(key) for key in ("count", "mean", "median", "std", "min", "max")]])
        table(["Column", "Count", "Mean", "Median", "Sample std", "Minimum", "Maximum"], rows)
        cats = mapping(eda.get("categorical_summaries"), "EDA categorical_summaries")
        rows = []
        for name, value in cats.items():
            item = mapping(value, f"Categorical summary for {name}")
            rows.append([name, item.get("unique_count"), item.get("most_frequent_values"), item.get("most_frequent_count")])
        table(["Column", "Unique values", "Most frequent values", "Frequency"], rows)

        def charts(value: Any, label: str) -> None:
            chart_map = mapping(value, f"{label} charts")
            references = []
            for kind, entry in chart_map.items():
                entries = entry.items() if isinstance(entry, Mapping) else [(kind, entry)]
                for name, chart in entries:
                    if chart is None:
                        continue
                    caption = self._text(f"{label}: {kind} / {name}")
                    if not isinstance(chart, (str, Path)):
                        messages.append(f"Invalid {label} chart path for {name}; reference omitted.")
                        continue
                    try:
                        if re.match(r"^(?:[a-z]+://|javascript:|data:|file:)", str(chart), re.I):
                            raise ValueError("only local raster image paths are supported")
                        image = Path(chart).resolve()
                        if image.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                            raise ValueError("unsupported chart image format")
                        if not image.is_file():
                            raise ValueError("chart file does not exist")
                        relative = os.path.relpath(image, path.parent.resolve()).replace(os.sep, "/")
                        references.append(f"![{caption}]({quote(relative, safe='/')})")
                    except (OSError, ValueError) as exc:
                        messages.append(f"{label} chart {name}: {exc}.")
                        references.append(f"{caption}: reference unavailable.")
            lines.extend(["\n\n".join(references) if references else "No available chart references were supplied.", ""])

        charts(eda.get("charts"), "EDA")

        section("Preprocessing and leakage safeguards")
        # Modeling includes the same split metadata, useful in partial reports.
        metadata = {**mmeta, **pmeta}
        table(["Setting", "Value"], [[label, metadata.get(key)] for label, key in (
            ("Included model features", "included_features"),
            ("Excluded model features", "excluded_features"),
            ("Training rows", "train_rows"), ("Testing rows", "test_rows"),
            ("Requested test size (fraction or row count)", "test_size"),
            ("Random state", "random_state"), ("Stratified split", "stratified"),
            ("Numerical scaling", "scale_numeric"), ("Dropped all-null training columns", "dropped_columns"),
            ("Transformed feature count", "transformed_feature_count"),
        )])
        if "excluded_features" in metadata:
            lines.extend(["Data-quality and EDA summaries describe the original dataset, including excluded columns. Included model features are the inputs selected before any all-null training columns are dropped. Excluded features are removed before preprocessing, model selection, cross-validation, and held-out prediction.", ""])
        if metadata.get("numerical_columns"):
            lines.extend(["Numerical processing: training-median imputation. Standard scaling: " + self._text(metadata.get("scale_numeric")) + ".", ""])
        if metadata.get("categorical_columns"):
            lines.extend(["Categorical processing: normalize observed labels, training-mode imputation, and one-hot encoding with unknown categories ignored.", ""])
        if prep or mmeta:
            lines.extend(["PreprocessingAgent method: split before fitting transformations; learn imputation, scaling, and encoding from training rows only. Target labels are retained without silent imputation or encoding.", ""])
        else:
            lines.extend(["No preprocessing details were supplied; preprocessing safeguards cannot be confirmed from this report.", ""])
        if model:
            lines.extend(["ModelingAgent method: clone an unfitted preprocessing configuration inside each training-only CV fold, select by CV score, then refit the selected complete pipeline on all training data. Held-out results are used for final evaluation only. This report summarizes the supplied outputs and does not independently audit the run.", ""])

        section("Model comparison and selection")
        comparisons = records(model.get("model_comparison"), "Model comparison")
        rows = []
        for value in comparisons:
            item = mapping(value, "Model comparison row")
            rows.append([item.get("model_name"), item.get("metric"), item.get("cv_mean"), item.get("cv_std"), item.get("fold_scores"), item.get("status"), item.get("error")])
            collect_warnings(item.get("warnings"), f"Model {item.get('model_name', 'unavailable')}")
            if item.get("error"):
                messages.append(f"Model {item.get('model_name', 'unavailable')}: {item['error']}")
        table(["Model", "CV metric", "Mean", "Std", "Fold scores", "Status", "Error"], rows)
        lines.extend([
            f"Selected model (as reported): {self._text(selected)}. Best CV score: {self._text(model.get('best_cv_score'))}.", "",
            f"CV folds: {self._text(mmeta.get('cv_folds'))}. Selection metric: {self._text(mmeta.get('selection_metric'))}. Direction: {self._text(mmeta.get('selection_direction'))}.", "",
        ])

        section("Final held-out evaluation")
        lines.extend([f"Metric source: {metric_source}.", ""])
        table(["Metric", "Value"], [[name.upper(), metrics.get(name) if self._finite(metrics.get(name)) else None] for name in metric_names])
        unavailable = [name for name in metric_names if name not in available_metrics]
        if unavailable:
            messages.append("Held-out metrics unavailable or undefined: " + ", ".join(unavailable) + ".")
        if task == "classification":
            lines.extend([
                f"Precision/recall/F1 averaging: {self._text(evaluation.get('averaging_method'))}. Positive label: {self._text(evaluation.get('positive_label'))}.", "",
                f"ROC-AUC method: {self._text(evaluation.get('roc_auc_method'))}. Unavailability reason: {self._text(evaluation.get('roc_auc_reason'))}.", "",
            ])
        charts(evaluation.get("charts"), "Evaluation")

        section("Limitations and warnings")
        messages = list(dict.fromkeys(messages))
        lines.extend(["- " + self._text(message) for message in messages] or ["No warnings were supplied by the included results."])
        lines.extend(["", "This report describes the supplied run only. It does not establish performance on future data, causality, fairness, or deployment readiness.", ""])
        section("Conclusion")
        if selected:
            lines.extend([f"The supplied modeling result identifies {self._text(selected)} as the selected model, with reported CV score {self._text(model.get('best_cv_score'))}.", ""])
        else:
            lines.extend(["A selected model was not available; no model recommendation can be concluded.", ""])
        if available_metrics:
            lines.extend(["The available held-out measurements are " + "; ".join(f"{name.upper()} = {self._text(value)}" for name, value in available_metrics.items()) + ". Interpret these alongside the warnings and evaluation scope above.", ""])
        else:
            lines.extend(["Final predictive performance cannot be assessed from the available metrics.", ""])
        text = "\n".join(lines)
        created = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as output:
                created = True
                output.write(text)
        except OSError as exc:
            if created:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise OSError(f"Unable to save report in {self.output_dir}: {exc}") from exc
        return {"report_path": str(path.resolve()), "report_text": text, "warnings": messages}

    @staticmethod
    def _finite(value: Any) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    @staticmethod
    def _text(value: Any) -> str:
        """Render scalar/list values safely without dumping estimators or data."""
        if value is None:
            return "Unavailable"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, Real):
            return format(value, ".6g") if math.isfinite(value) else "Unavailable"
        if isinstance(value, (list, tuple)):
            return ", ".join(ReportingAgent._text(item) for item in value) if value else "None listed"
        if not isinstance(value, str):
            return "Unavailable"
        clean = " ".join("".join(char if not unicodedata.category(char).startswith("C") else " " for char in value).split())
        escaped = html.escape(clean, quote=True)
        return re.sub(r"([\\`*_\[\]{}|#])", r"\\\1", escaped)
