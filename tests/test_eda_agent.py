"""EDA summaries, edge cases, chart artifacts, and input preservation."""

import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from agents.eda_agent import EDAAgent


@pytest.fixture
def agent(tmp_path):
    return EDAAgent(tmp_path / "eda")


def test_numerical_statistics_and_charts(agent):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    original = df.copy(deep=True)
    before = plt.get_fignums()
    result = agent.analyze_dataset(df)
    assert result["dataset_info"]["shape"] == [3, 2]
    assert result["numerical_statistics"]["a"] == {"count": 3, "mean": 2, "median": 2, "std": 1, "min": 1, "max": 3}
    assert result["correlation"]["a"]["b"] == pytest.approx(1)
    paths = [*result["charts"]["histograms"].values(), result["charts"]["correlation_heatmap"]]
    for path in paths:
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == before
    pd.testing.assert_frame_equal(df, original)
    json.dumps(result, allow_nan=False)


def test_categorical_data(agent):
    result = agent.analyze_dataset(pd.DataFrame({"label": ["a", "a", "b"], "flag": [True, False, True]}))
    assert result["dataset_info"]["categorical_columns"] == ["label", "flag"]
    assert result["categorical_summaries"]["label"]["unique_count"] == 2
    assert result["categorical_summaries"]["label"]["most_frequent_values"] == ["a"]
    assert result["categorical_summaries"]["label"]["most_frequent_count"] == 2
    assert len(result["charts"]["categorical_frequencies"]) == 2
    assert result["charts"]["correlation_heatmap"] is None


def test_missing_values_and_duplicates(agent):
    result = agent.analyze_dataset(pd.DataFrame({"a": [1, 1, None, 3], "b": ["x", "x", None, "y"]}))
    assert result["missing_values"] == {"a": {"count": 1, "percentage": 25}, "b": {"count": 1, "percentage": 25}}
    assert result["duplicate_rows"] == {"count": 1, "percentage": 25}


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame(columns=["a"]), pd.DataFrame(index=[0])])
def test_empty_data(agent, df):
    with pytest.raises(ValueError, match="must not be empty"):
        agent.analyze_dataset(df)


def test_invalid_input(agent):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        agent.analyze_dataset([1, 2])
    with pytest.raises(ValueError, match="unique"):
        agent.analyze_dataset(pd.DataFrame([[1, 2]], columns=["a", "a"]))


def test_null_constant_nullable_and_infinite_columns(agent):
    df = pd.DataFrame({"null_numeric": pd.Series([None, None, None], dtype="Float64"), "null_category": pd.Series([None, None, None], dtype="string"), "constant": [7, 7, 7], "infinite": [float("inf"), 1, 2]})
    result = agent.analyze_dataset(df)
    assert result["numerical_statistics"]["null_numeric"]["mean"] is None
    assert result["categorical_summaries"]["null_category"]["unique_count"] == 0
    assert result["numerical_statistics"]["constant"]["std"] == 0
    assert result["numerical_statistics"]["infinite"]["mean"] == 1.5
    assert result["charts"]["correlation_heatmap"] is None
    assert len(result["charts"]["histograms"]) == 2
    assert result["warnings"]
    json.dumps(result, allow_nan=False)


def test_safe_unique_paths_and_single_value(agent):
    df = pd.DataFrame({"../../unsafe/name": [1]})
    first = agent.analyze_dataset(df)
    second = agent.analyze_dataset(df)
    first_path = Path(first["charts"]["histograms"]["../../unsafe/name"])
    second_path = Path(second["charts"]["histograms"]["../../unsafe/name"])
    assert first_path.parent == agent.output_dir.resolve()
    assert first_path != second_path
    assert first_path.exists() and second_path.exists()
    assert first["numerical_statistics"]["../../unsafe/name"]["std"] is None


def test_unused_categories_and_tied_modes(agent):
    df = pd.DataFrame({
        "empty": pd.Categorical([None, None], categories=["unused"]),
        "ties": pd.Categorical(["a", "b"], categories=["a", "b", "unused"]),
    })
    result = agent.analyze_dataset(df)
    assert result["categorical_summaries"]["empty"]["most_frequent_values"] == []
    assert result["categorical_summaries"]["empty"]["top_values"] == []
    assert set(result["categorical_summaries"]["ties"]["most_frequent_values"]) == {"a", "b"}
    assert "empty" not in result["charts"]["categorical_frequencies"]


def test_plot_failure_is_reported_and_figures_released(agent, monkeypatch):
    from matplotlib.figure import Figure

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", fail)
    before = plt.get_fignums()
    result = agent.analyze_dataset(pd.DataFrame({"x": [1, 2, 3]}))
    assert result["charts"]["histograms"] == {}
    assert any("disk full" in warning for warning in result["warnings"])
    assert plt.get_fignums() == before
    assert list(agent.output_dir.iterdir()) == []
