"""Streamlit rendering and rerun behavior with real analysis behind upload input."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from utils import workflow


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def dashboard(monkeypatch, tmp_path):
    df = pd.DataFrame({"x": range(20), "group": ["a", "b"] * 10, "target": ["yes", "no"] * 10})
    uploaded = {"data": df.to_csv(index=False).encode()}
    calls = []
    original_run = workflow.run_analysis

    def run(*args, **kwargs):
        calls.append(True)
        return original_run(*args, **kwargs, output_dir=tmp_path / "outputs", report_dir=tmp_path / "reports")

    # Only the browser upload boundary is supplied by the test; all six agents,
    # CV, metrics, charts, report generation, and rendering run for real.
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: SimpleNamespace(getvalue=lambda: uploaded["data"]))
    monkeypatch.setattr(workflow, "run_analysis", run)
    return AppTest.from_file(str(APP_PATH), default_timeout=60), uploaded, calls


def test_initial_dashboard_without_upload(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: None)
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    assert "Autonomous Data Scientist" in app.title[0].value
    assert not app.button


def test_real_dashboard_run_persistence_and_selection_isolation(dashboard):
    app, uploaded, calls = dashboard
    app.run()
    assert not app.exception
    assert not calls
    assert app.selectbox[1].value == "classification"
    app.button[0].click().run()
    assert not app.exception
    assert len(calls) == 1
    report_path = app.session_state["analysis_entry"]["results"]["reporting"]["report_path"]
    assert Path(report_path).is_file()
    assert len(app.get("download_button")) == 1
    app.run()
    assert not app.exception and len(calls) == 1
    assert app.session_state["analysis_entry"]["results"]["reporting"]["report_path"] == report_path
    app.selectbox[1].select("regression").run()
    assert not app.exception and len(calls) == 1
    assert not app.get("download_button")
    assert any("settings have changed" in item.value for item in app.info)
    app.selectbox[1].select("classification").run()
    assert len(calls) == 1 and len(app.get("download_button")) == 1
    app.slider[0].set_value(0.3).run()
    assert len(calls) == 1 and not app.get("download_button")
    app.selectbox[0].select("group").run()
    assert len(calls) == 1 and not app.get("download_button")
    uploaded["data"] = b"x,target\n1,a\n2,b\n3,a\n4,b\n"
    app.run()
    assert not app.exception and len(calls) == 1
    assert not app.get("download_button")


def test_ambiguous_target_requires_explicit_task(dashboard):
    app, uploaded, calls = dashboard
    uploaded["data"] = b"x,target\n1,10\n2,20\n3,30\n4,40\n"
    app.run()
    assert not app.exception
    assert app.selectbox[1].value is None
    assert app.button[0].disabled
    assert not calls


def test_default_id_exclusion_and_feature_selection_reruns(dashboard):
    app, uploaded, calls = dashboard
    df = pd.DataFrame({"Id": range(20), "measurement": range(20), "Species": ["a", "b"] * 10})
    uploaded["data"] = df.to_csv(index=False).encode()
    app.run()
    assert not app.exception
    assert app.multiselect[0].value == ["Id"]
    assert not calls
    app.button[0].click().run()
    assert not app.exception and len(calls) == 1
    results = app.session_state["analysis_entry"]["results"]
    assert results["modeling"]["metadata"]["excluded_features"] == ["Id"]
    assert results["modeling"]["trained_pipeline"].feature_names_in_.tolist() == ["measurement"]
    assert "Id" in results["eda"]["dataset_info"]["column_names"]
    app.multiselect[0].set_value([]).run()
    assert not app.exception and len(calls) == 1
    assert not app.get("download_button")
    app.multiselect[0].set_value(["Id"]).run()
    assert len(calls) == 1 and app.get("download_button")
    app.multiselect[0].set_value(["Id", "measurement"]).run()
    assert not app.exception and app.button[0].disabled
    assert len(calls) == 1


def test_parsing_missing_target_and_small_data_errors(dashboard):
    app, uploaded, calls = dashboard
    uploaded["data"] = b"x,target\n1,a,extra\n"
    app.run()
    assert not app.exception
    assert any("fields" in item.value for item in app.error)
    uploaded["data"] = b"x,target\n1,a\n2,\n3,a\n4,b\n"
    app.run()
    assert not app.exception and app.button[0].disabled
    assert not calls
    uploaded["data"] = b"x,target\n1,a\n2,b\n3,a\n4,b\n"
    app.run()
    app.button[0].click().run()
    assert not app.exception
    assert len(calls) == 1
    assert any("Modeling could not complete" in item.value for item in app.error)
    assert "eda" in app.session_state["analysis_entry"]["results"]
    assert not app.get("download_button")
