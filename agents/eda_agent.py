"""Dataset summaries and headless charts for dashboards and reports."""

from pathlib import Path
import re
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import seaborn as sns


class EDAAgent:
    """Analyze a dataset without modifying it or opening plotting windows.

    Charts default to outputs/eda relative to the current working directory.
    ``output_dir`` can redirect them for tests or other application contexts.
    """

    def __init__(self, output_dir: str | Path = "outputs/eda") -> None:
        self.output_dir = Path(output_dir)

    def analyze_dataset(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return JSON-compatible metadata, summaries, warnings, and PNG paths.

        Missing values are excluded from statistics and frequency counts.
        Standard deviation uses the sample definition (ddof=1). Non-finite
        numerical values are excluded from statistics and plots with a warning.
        Frequency plots show at most the 20 most common observed values.
        Constant columns get histograms but are excluded from correlations.
        Duplicate counts exclude the first occurrence of each row.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: For empty data or ambiguous column labels.
            OSError: If the chart output directory cannot be created.

        Individual chart failures are recorded in warnings so other analysis
        remains available. Figures are cleared in finally blocks; using the
        object-oriented Agg canvas avoids pyplot's global figure registry.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Dataset must not be empty; provide rows and columns.")
        names = [str(column) for column in df.columns]
        if not df.columns.is_unique or len(set(names)) != len(names):
            raise ValueError("Column names must be unique, including their string representations.")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Cannot create EDA output directory {self.output_dir}: {exc}") from exc

        warnings: list[str] = []
        numerical: list[str] = []
        categorical: list[str] = []
        other: list[str] = []
        statistics: dict[str, Any] = {}
        categories: dict[str, Any] = {}
        clean_numeric: dict[str, pd.Series] = {}
        charts: dict[str, Any] = {"histograms": {}, "categorical_frequencies": {}, "correlation_heatmap": None}

        def save_chart(kind: str, name: str, draw: Callable[[Any], Any]) -> str | None:
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:60] or "column"
            path = self.output_dir / f"{kind}_{slug}_{uuid4().hex}.png"
            figure = Figure(figsize=(9, 6))
            FigureCanvasAgg(figure)
            created = False
            try:
                axis = figure.subplots()
                draw(axis)
                figure.tight_layout()
                # Exclusive creation prevents overwriting even if a name collides.
                with path.open("xb") as output:
                    created = True
                    figure.savefig(output, format="png", dpi=120)
                return str(path.resolve())
            except Exception as exc:
                warnings.append(f"Could not generate {kind} chart for {name!r}: {exc}")
                if created:
                    try:
                        path.unlink()
                    except OSError as cleanup_error:
                        warnings.append(f"Could not remove incomplete chart {path}: {cleanup_error}")
                return None
            finally:
                figure.clear()

        for column, name in zip(df.columns, names):
            series = df[column]
            if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
                numerical.append(name)
                if pd.api.types.is_complex_dtype(series.dtype):
                    warnings.append(f"Complex numerical column {name!r} is not supported for statistics or plots.")
                    statistics[name] = dict.fromkeys(["count", "mean", "median", "std", "min", "max"])
                    continue
                values = series.astype(float)
                finite = values.where(np.isfinite(values)).dropna()
                if int(np.isinf(values).sum()):
                    warnings.append(f"Excluded infinite values from numerical analysis of {name!r}.")
                clean_numeric[name] = values.where(np.isfinite(values))
                statistics[name] = {
                    "count": int(finite.size),
                    **{key: self._number(value) for key, value in {
                        "mean": finite.mean(), "median": finite.median(),
                        "std": finite.std(), "min": finite.min(), "max": finite.max(),
                    }.items()},
                }
                if finite.empty:
                    warnings.append(f"No finite observed values in numerical column {name!r}; histogram skipped.")
                else:
                    def histogram(axis: Any, values: pd.Series = finite, label: str = name) -> None:
                        sns.histplot(x=values.to_numpy(), ax=axis, bins="auto")
                        axis.set(title=f"Distribution of {label}", xlabel=label, ylabel="Count")
                    path = save_chart("histogram", name, histogram)
                    if path:
                        charts["histograms"][name] = path
            elif is_bool_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
                categorical.append(name)
                # Count original values before converting display labels to strings.
                counts = series.value_counts(dropna=True)
                counts = counts[counts > 0]  # Ignore unused categorical levels.
                top = counts.head(20)
                categories[name] = {
                    "unique_count": int(series.nunique(dropna=True)),
                    "most_frequent_values": [str(value) for value in counts[counts == counts.max()].index] if not counts.empty else [],
                    "most_frequent_count": int(counts.iloc[0]) if not counts.empty else 0,
                    "top_values": [{"value": str(value), "count": int(count)} for value, count in top.items()],
                }
                if top.empty:
                    warnings.append(f"No observed values in categorical column {name!r}; frequency plot skipped.")
                else:
                    def frequencies(axis: Any, counts: pd.Series = top, label: str = name) -> None:
                        axis.barh(range(len(counts)), counts.to_numpy())
                        axis.set_yticks(range(len(counts)), labels=[str(value) for value in counts.index])
                        axis.invert_yaxis()
                        axis.set(title=f"Most frequent values of {label} (up to 20)", xlabel="Count")
                    path = save_chart("frequency", name, frequencies)
                    if path:
                        charts["categorical_frequencies"][name] = path
            else:
                other.append(name)

        candidates = {name: values for name, values in clean_numeric.items() if values.nunique() > 1}
        correlation: dict[str, Any] = {}
        if len(candidates) >= 2:
            matrix = pd.DataFrame(candidates).corr(min_periods=2)
            off_diagonal = matrix.to_numpy().copy()
            np.fill_diagonal(off_diagonal, np.nan)
            if np.isfinite(off_diagonal).any():
                correlation = {name: {other_name: self._number(value) for other_name, value in row.items()} for name, row in matrix.to_dict().items()}
                def heatmap(axis: Any) -> None:
                    sns.heatmap(matrix, ax=axis, mask=matrix.isna(), vmin=-1, vmax=1, cmap="coolwarm", annot=len(matrix) <= 12)
                    axis.set_title("Numerical correlation (Pearson)")
                charts["correlation_heatmap"] = save_chart("correlation", "heatmap", heatmap)

        return {
            "dataset_info": {
                "shape": [int(df.shape[0]), int(df.shape[1])], "column_names": names,
                "data_types": {name: str(df[column].dtype) for column, name in zip(df.columns, names)},
                "numerical_columns": numerical, "categorical_columns": categorical, "other_columns": other,
            },
            "missing_values": {name: {"count": int(df[column].isna().sum()), "percentage": float(df[column].isna().mean() * 100)} for column, name in zip(df.columns, names)},
            "duplicate_rows": {"count": int(df.duplicated().sum()), "percentage": float(df.duplicated().mean() * 100)},
            "numerical_statistics": statistics, "categorical_summaries": categories,
            "correlation": correlation, "charts": charts, "warnings": warnings,
        }

    @staticmethod
    def _number(value: Any) -> float | None:
        """Convert missing and non-finite statistics into JSON null values."""
        return float(value) if pd.notna(value) and np.isfinite(value) else None
