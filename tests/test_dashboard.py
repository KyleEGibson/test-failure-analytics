"""Headless tests of the Streamlit dashboard (streamlit.testing AppTest)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_dashboard_renders_both_tabs(app):
    assert [t.label for t in app.tabs] == ["EOS-100 production", "SECOM semiconductor line"]
    assert metric(app, "Units (first pass)") == "1,247"
    assert len(app.get("plotly_chart")) >= 4


def test_drill_down_isolates_station_for_voltage(app):
    mode = next(s for s in app.selectbox if s.label == "Failure mode")
    mode.select("bus voltage").run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "station **BENCH-02**" in text


def test_filters_rescope_metrics(app):
    stations = next(m for m in app.multiselect if m.label == "Station")
    stations.select("BENCH-01").run()
    assert not app.exception
    assert int(metric(app, "Units (first pass)").replace(",", "")) < 1247


def test_drill_down_explains_away_operator(app):
    mode = next(s for s in app.selectbox if s.label == "Failure mode")
    mode.select("cooldown time").run()
    captions = " ".join(c.value for c in app.caption)
    assert "operator was explained away by shift" in captions
