"""Report generation: narrative accuracy, chart theming and the built page."""

import re

import pandas as pd
import plotly.graph_objects as go
import pytest

from failure_analytics import charts, report, spc


def chart_from(rates, n=100):
    weeks = pd.date_range("2026-01-05", periods=len(rates), freq="W-MON")
    defectives = pd.Series([round(r * n) for r in rates], index=weeks)
    return spc.p_chart(defectives, pd.Series(n, index=weeks))


def test_narrative_for_stable_process():
    chart = chart_from([0.05, 0.06, 0.04, 0.05, 0.06, 0.05, 0.04, 0.06])
    text = report.describe_p_chart(chart, spc.western_electric(chart["p"], chart["center"], chart["sigma"]))
    assert "in statistical control" in text


def test_narrative_names_the_week_that_broke_the_limit():
    chart = chart_from([0.05, 0.05, 0.25, 0.05, 0.05, 0.04, 0.05, 0.06])
    text = report.describe_p_chart(chart, spc.western_electric(chart["p"], chart["center"], chart["sigma"]))
    assert text.startswith(f"The week of {chart.index[2].day} {chart.index[2]:%b} broke the upper control limit")


def test_theme_resolves_roles():
    fig = go.Figure()
    fig.add_scatter(y=[1, 2], meta="series-2")
    fig.add_bar(y=[1, 2], meta="critical")
    charts._hline(fig, 1.5, "limit", role="critical")
    charts.apply_theme(fig, charts.DARK)
    assert fig.data[0].line.color == charts.DARK["series-2"]
    assert fig.data[1].marker.color == charts.DARK["critical"]
    assert fig.data[1].marker.line.color == charts.DARK["surface"]  # 2px gap between bars
    assert fig.layout.shapes[0].line.color == charts.DARK["critical"]


def test_open_markers_keep_their_outline():
    fig = go.Figure(go.Scatter(y=[1], meta="critical", marker={"symbol": "diamond-open"}))
    charts.apply_theme(fig, charts.LIGHT)
    assert fig.data[0].marker.line.color == charts.LIGHT["critical"]


def test_pareto_uses_a_single_axis():
    frame = pd.DataFrame({"failure_mode": ["a", "b"], "failures": [3, 1],
                          "share_pct": [75.0, 25.0], "cumulative_pct": [75.0, 100.0]})
    fig = charts.pareto(frame)
    assert all(getattr(trace, "yaxis", None) in (None, "y") for trace in fig.data)
    assert "yaxis2" not in fig.layout.to_plotly_json()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    return report.build(out, eos_db=None).read_text(encoding="utf-8")


def test_report_builds_with_every_chart(built):
    assert built.count('class="plotly-graph-div"') == 8
    assert built.count("Table view") == 8


def test_report_has_no_unrendered_values(built):
    text = re.sub(r"<script.*?</script>", "", built, flags=re.S)
    for bad in ("nan%", ">nan<", "None%", "{{", "}}"):
        assert bad not in text, bad


def test_plotly_loads_before_charts(built):
    assert built.index("plotly.min.js") < built.index('class="plotly-graph-div"')


def test_report_states_known_answers(built):
    for expected in ("station <strong>BENCH-02</strong>", "gimbal lot <strong>G-2603</strong>", "shift <strong>swing</strong>"):
        assert expected in built
