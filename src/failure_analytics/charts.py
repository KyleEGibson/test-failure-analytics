"""Plotly figure builders for the report and dashboard.

Traces/shapes carry a color *role* (series-1, limit, critical...) in meta/name
instead of a hex value, so the report's JS can swap the palette for dark mode.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import plotly.graph_objects as go

LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink-2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "series-1": "#2a78d6", "series-2": "#eb6834", "series-3": "#1baf7a",
    "other": "#b4b2a9", "limit": "#52514e", "critical": "#d03b3b", "band": "rgba(42,120,214,0.08)",
}
DARK = {
    "surface": "#1a1a19", "ink": "#f0efec", "ink-2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
    "series-1": "#3987e5", "series-2": "#d95926", "series-3": "#199e70",
    "other": "#5f5e59", "limit": "#c3c2b7", "critical": "#e66767", "band": "rgba(57,135,229,0.12)",
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
SERIES = ["series-1", "series-2", "series-3"]


def _base(fig: go.Figure, title: str, y_title: str, x_title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        height=height,
        margin={"l": 64, "r": 24, "t": 64, "b": 56},
        font={"family": FONT, "size": 12},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom", "title": None},
        hoverlabel={"font": {"family": FONT}},
        bargap=0.25,
    )
    fig.update_xaxes(title_text=x_title, showline=True, zeroline=False, ticks="outside", ticklen=4)
    fig.update_yaxes(title_text=y_title, showline=True, zeroline=False, ticks="outside", ticklen=4)
    return fig


def _hline(fig: go.Figure, y: float, label: str, role: str = "limit", dash: str = "dash") -> None:
    """Dashed threshold line with a label on the right."""
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=y, y1=y, name=role,
                  line={"width": 1.5, "dash": dash})
    fig.add_annotation(xref="paper", x=1, y=y, text=label, showarrow=False, xanchor="right",
                       yanchor="bottom", name=role, font={"size": 11})


def _vline(fig: go.Figure, x, label: str, role: str = "limit") -> None:
    fig.add_shape(type="line", yref="paper", y0=0, y1=1, x0=x, x1=x, name=role, line={"width": 1.5, "dash": "dot"})
    fig.add_annotation(yref="paper", y=1, x=x, text=label, showarrow=False, xanchor="left", yanchor="top",
                       xshift=4, name=role, font={"size": 11})


def _stroked(symbol) -> bool:
    """Open and x symbols are drawn by their outline, so the outline carries the color."""
    return isinstance(symbol, str) and (symbol.startswith("x") or symbol.endswith("-open"))


def apply_theme(fig: go.Figure, theme: dict[str, str]) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=theme["surface"], plot_bgcolor=theme["surface"],
        font_color=theme["ink-2"], title_font_color=theme["ink"],
        legend_font_color=theme["ink-2"],
        hoverlabel={"bgcolor": theme["surface"], "font_color": theme["ink"], "bordercolor": theme["axis"]},
    )
    fig.update_xaxes(gridcolor=theme["grid"], linecolor=theme["axis"], tickcolor=theme["axis"],
                     tickfont_color=theme["muted"], title_font_color=theme["ink-2"])
    fig.update_yaxes(gridcolor=theme["grid"], linecolor=theme["axis"], tickcolor=theme["axis"],
                     tickfont_color=theme["muted"], title_font_color=theme["ink-2"])
    for trace in fig.data:
        color = theme.get(trace.meta or "series-1")
        if trace.type == "bar":
            trace.marker.color = color
            trace.marker.line = {"color": theme["surface"], "width": 2}  # 2px surface gap between fills
        elif trace.type == "box":
            trace.marker.color = color
            trace.line.color = color
            trace.fillcolor = theme["band"]
        else:
            trace.line.color = color
            trace.marker.color = color
            if _stroked(trace.marker.symbol):
                trace.marker.line = {"color": color, "width": 2}
            else:
                trace.marker.line = {"color": theme["surface"], "width": 2}  # ring on overlapping marks
    for shape in fig.layout.shapes:
        if shape.name == "band":
            shape.fillcolor = theme["band"]
            shape.line.width = 0
        else:
            shape.line.color = theme.get(shape.name, theme["limit"])
    for note in fig.layout.annotations:
        note.font.color = theme["critical"] if note.name == "critical" else theme["ink-2"]
    return fig


def weekly_failures_by_mode(first_pass: pd.DataFrame, failures: pd.DataFrame, top_modes: list[str]) -> tuple[go.Figure, pd.DataFrame]:
    """Stacked bars: first-pass failing units per week, split by failure mode (top 3 + other)."""
    merged = failures.merge(first_pass[["run_id", "week"]], on="run_id")
    merged["mode"] = merged["failure_mode"].where(merged["failure_mode"].isin(top_modes), "Other")
    table = merged.pivot_table(index="week", columns="mode", values="run_id", aggfunc="nunique", fill_value=0)
    table = table.reindex(index=sorted(first_pass["week"].unique()), fill_value=0)
    order = [m for m in top_modes if m in table.columns] + (["Other"] if "Other" in table.columns else [])
    table = table[order]
    fig = go.Figure()
    for i, mode in enumerate(order):
        fig.add_bar(x=table.index, y=table[mode], name=mode, meta=SERIES[i] if mode != "Other" else "other",
                    hovertemplate=f"{mode}<br>week %{{x}}: %{{y}} units<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(dtick=1)
    return _base(fig, "First-pass failures per week, by failure mode", "Failing units", "Production week"), table


def pareto(pareto_frame: pd.DataFrame) -> go.Figure:
    """Pareto on ONE percent axis: bars are each mode's share, the line is the cumulative share."""
    labels = ["<br>".join(textwrap.wrap(mode, 14)) for mode in pareto_frame["failure_mode"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=pareto_frame["share_pct"], name="Share of failures", meta="series-1",
                customdata=pareto_frame[["failures", "failure_mode"]],
                hovertemplate="%{customdata[1]}<br>%{y:.1f}% (%{customdata[0]} failures)<extra></extra>")
    fig.update_xaxes(tickangle=0)
    fig.add_scatter(x=labels, y=pareto_frame["cumulative_pct"], name="Cumulative share",
                    mode="lines+markers", meta="series-2", line={"width": 2}, marker={"size": 8},
                    hovertemplate="cumulative %{y:.1f}%<extra></extra>")
    _hline(fig, 80, "80%", dash="dot")
    fig.update_yaxes(range=[0, 105], ticksuffix="%")
    return _base(fig, "Failure Pareto (first-pass)", "Share of first-pass failures")


def station_drift(history: pd.DataFrame, weeks: pd.Series, chart: pd.DataFrame, usl: float,
                  signal_week: int | None, failure_week: int | None) -> tuple[go.Figure, pd.DataFrame]:
    """Weekly mean bus voltage per station against the drifting station's X-bar limits and the spec."""
    frame = history.assign(week=weeks.to_numpy())
    table = frame.groupby(["week", "station"])["value"].mean().unstack("station").round(3)
    fig = go.Figure()
    fig.add_scatter(x=chart.index, y=chart["ucl"], mode="lines", line={"width": 0}, meta="limit",
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=chart.index, y=chart["lcl"], mode="lines", line={"width": 0}, meta="limit", fill="tonexty",
                    fillcolor="rgba(137,135,129,0.12)", name="BENCH-02 control limits (weeks 1-3)", hoverinfo="skip")
    for i, station in enumerate(table.columns):
        fig.add_scatter(x=table.index, y=table[station], name=station, mode="lines+markers", meta=SERIES[i],
                        line={"width": 2}, marker={"size": 8},
                        hovertemplate=f"{station}<br>week %{{x}}: %{{y:.3f}} V<extra></extra>")
    _hline(fig, usl, f"Upper spec {usl:g} V", role="critical")
    if signal_week:
        _vline(fig, signal_week - 0.5, f"SPC signal (week {signal_week})")
    if failure_week:
        _vline(fig, failure_week - 0.5, f"First failures (week {failure_week})", role="critical")
    fig.update_xaxes(dtick=1)
    return _base(fig, "Bus voltage weekly mean by test station", "Mean reading (V)", "Production week"), table


def by_level_distribution(values: pd.DataFrame, level: str, measurement: str, usl: float, units: str) -> tuple[go.Figure, pd.DataFrame]:
    """One box per level of a factor (one color: levels are nominal categories)."""
    fig = go.Figure()
    for lvl, group in values.groupby(level):
        fig.add_box(y=group["value"], name=str(lvl), meta="series-1", boxpoints="outliers", showlegend=False,
                    hovertemplate=f"{lvl}<br>%{{y:.2f}} {units}<extra></extra>")
    _hline(fig, usl, f"Upper spec {usl:g} {units}", role="critical")
    table = values.groupby(level)["value"].describe()[["count", "mean", "50%", "max"]].round(3)
    table["over spec"] = values.assign(over=values["value"] > usl).groupby(level)["over"].sum()
    return _base(fig, f"{measurement} by {level.replace('_', ' ')}", f"{measurement} ({units})"), table


def weekly_mean_by_level(values: pd.DataFrame, level: str, measurement: str, usl: float, units: str) -> tuple[go.Figure, pd.DataFrame]:
    table = values.groupby(["week", level])["value"].mean().unstack(level).round(2)
    fig = go.Figure()
    for i, lvl in enumerate(table.columns):
        fig.add_scatter(x=table.index, y=table[lvl], name=f"{lvl} shift" if level == "shift" else str(lvl),
                        mode="lines+markers", meta=SERIES[i], line={"width": 2}, marker={"size": 8},
                        hovertemplate=f"{lvl}<br>week %{{x}}: %{{y:.2f}} {units}<extra></extra>")
    _hline(fig, usl, f"Upper spec {usl:g} {units}", role="critical")
    fig.update_xaxes(dtick=1)
    return _base(fig, f"{measurement}: weekly mean by {level}", f"Mean {measurement} ({units})", "Production week"), table


def p_chart(chart: pd.DataFrame, violations: pd.DataFrame, title: str) -> tuple[go.Figure, pd.DataFrame]:
    x = chart.index
    fig = go.Figure()
    fig.add_scatter(x=x, y=chart["ucl"], name="Upper control limit", mode="lines", meta="limit",
                    line={"width": 1.5, "dash": "dash", "shape": "hvh"}, hovertemplate="UCL %{y:.1%}<extra></extra>")
    fig.add_scatter(x=x, y=chart["center"], name="Center (average)", mode="lines", meta="muted",
                    line={"width": 1.5}, hovertemplate="center %{y:.1%}<extra></extra>")
    fig.add_scatter(x=x, y=chart["p"], name="Fail rate", mode="lines+markers", meta="series-1",
                    line={"width": 2}, marker={"size": 8}, customdata=chart[["defectives", "n"]],
                    hovertemplate="%{x|%d %b}: %{y:.1%} (%{customdata[0]:.0f} of %{customdata[1]:.0f})<extra></extra>")
    if len(violations):
        flagged = violations.groupby("position")["description"].apply("; ".join)
        points = chart.iloc[flagged.index]
        fig.add_scatter(x=points.index, y=points["p"], name="Out of control", mode="markers", meta="critical",
                        marker={"size": 13, "symbol": "diamond-open", "line": {"width": 2}},
                        customdata=flagged.values,
                        hovertemplate="%{x|%d %b}: %{customdata}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    table = chart[["n", "defectives", "p", "ucl"]].rename(columns={"defectives": "failures", "p": "fail rate"}).copy()
    table.index = table.index.strftime("%Y-%m-%d")
    table["rule violations"] = ""
    if len(violations):
        table.iloc[flagged.index, table.columns.get_loc("rule violations")] = flagged.values
    return _base(fig, title, "Fail rate", "Week starting"), table


def stability_dumbbell(stability: pd.DataFrame) -> go.Figure:
    """Separation of each top signal in the earlier vs later period."""
    frame = stability.iloc[::-1]  # strongest at the top
    fig = go.Figure()
    for _, row in frame.iterrows():
        fig.add_scatter(x=[row["separation_earlier"], row["separation_later"]], y=[row["signal"]] * 2, mode="lines",
                        meta="muted", line={"width": 2}, showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=frame["separation_earlier"], y=frame["signal"], mode="markers", name="Earlier 70% of units",
                    meta="series-1", marker={"size": 10}, hovertemplate="%{y}: %{x:.2f} earlier<extra></extra>")
    fig.add_scatter(x=frame["separation_later"], y=frame["signal"], mode="markers", name="Later 30% of units",
                    meta="series-2", marker={"size": 10, "symbol": "diamond"},
                    hovertemplate="%{y}: %{x:.2f} later<extra></extra>")
    fig.update_xaxes(range=[0, max(0.6, float(frame[["separation_earlier", "separation_later"]].max().max()) + 0.05)])
    return _base(fig, "Do the strongest signals still separate failures later?",
                 "", "Separation (0 = none, 1 = perfect)", height=420)


def era_drift(raw: pd.DataFrame, signal: str) -> go.Figure:
    frame = raw[["timestamp", "failed", signal]].dropna()
    fig = go.Figure()
    passed, failed = frame[~frame["failed"]], frame[frame["failed"]]
    fig.add_scatter(x=passed["timestamp"], y=passed[signal], mode="markers", name="Pass", meta="other",
                    marker={"size": 5, "opacity": 0.6}, hovertemplate="%{x|%d %b}: %{y:.2f}<extra>pass</extra>")
    fig.add_scatter(x=failed["timestamp"], y=failed[signal], mode="markers", name="Fail", meta="critical",
                    marker={"size": 9, "symbol": "x-thin"}, hovertemplate="%{x|%d %b}: %{y:.2f}<extra>fail</extra>")
    weekly = frame.set_index("timestamp")[signal].resample("W").median()
    fig.add_scatter(x=weekly.index, y=weekly.values, mode="lines", name="Weekly median", meta="series-1",
                    line={"width": 2}, hovertemplate="median %{y:.2f}<extra></extra>")
    return _base(fig, f"Signal {signal} over time", f"{signal} reading", "")
