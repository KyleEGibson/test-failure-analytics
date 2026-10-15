"""Build the static HTML analytics report.

    python -m failure_analytics report --out site
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly
from jinja2 import Environment, PackageLoader, select_autoescape

from . import __version__, charts, eos, secom, signals, spc
from .eos_generator import generate

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EOS_DB = ROOT / "data" / "generated" / "eos_production.db"
REPO_URL = "https://github.com/KyleEGibson/test-failure-analytics"

_env = Environment(loader=PackageLoader("failure_analytics", "templates"), autoescape=select_autoescape(["j2"]))


def _figure(fig, table: pd.DataFrame | None = None, caption: str = "", index: bool = True) -> dict:
    charts.apply_theme(fig, charts.LIGHT)
    html = fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})
    table_html = None
    if table is not None:
        table_html = table.to_html(classes="data", border=0, index=index, float_format=lambda v: f"{v:,.3f}".rstrip("0").rstrip("."))
    return {"html": html, "table": table_html, "caption": caption}


def _eos_section(db_path: Path) -> dict:
    db = eos.EOSDatabase(db_path)
    first_pass = db.first_pass()
    failures = db.failures()
    pareto = db.pareto()
    findings = eos.analyze(db)
    by_mode = {f.failure_mode: f for f in findings}
    weekly = eos.weekly_yield(first_pass)

    top_modes = pareto["failure_mode"].head(3).tolist()
    fig_weekly, weekly_table = charts.weekly_failures_by_mode(first_pass, failures, top_modes)
    fig_pareto = charts.pareto(pareto)

    def readings(measurement: str) -> pd.DataFrame:
        history = db.history(measurement)
        return history.merge(first_pass[["run_id", "week", "shift", "gimbal_lot"]], on="run_id")

    voltage = by_mode["bus voltage"]
    v_hist = db.history("bus voltage")
    v_weeks = eos.week_of(first_pass, v_hist)
    drifting = v_hist["station"] == voltage.level
    v_chart = eos.weekly_xbar(v_hist[drifting], v_weeks[drifting])
    fig_voltage, voltage_table = charts.station_drift(v_hist, v_weeks, v_chart, 29.0,
                                                      voltage.spc_signal_week, voltage.onset_week)

    slew = by_mode["90 deg azimuth slew time"]
    fig_slew, slew_table = charts.by_level_distribution(readings("90 deg azimuth slew time"), "gimbal_lot",
                                                        "90 deg slew time", 2.0, "s")
    slew_conf = eos.confounder_check(first_pass, eos.mode_flags(first_pass, failures, slew.failure_mode),
                                     "gimbal_lot", "firmware")

    cool = by_mode["cooldown time"]
    fig_cool, cool_table = charts.weekly_mean_by_level(readings("cooldown time"), "shift", "cooldown time", 15.0, "s")

    retests = db.retests()
    return {
        "units": len(first_pass),
        "fpy": 1 - first_pass["failed"].mean(),
        "weeks": int(first_pass["week"].max()),
        "fpy_first": weekly["fpy"].iloc[:3].mean(),
        "fpy_last": weekly["fpy"].iloc[-2:].mean(),
        "findings": findings,
        "by_mode": by_mode,
        "naive_voltage": voltage.screen_all.iloc[0],
        "naive_cooldown": cool.screen_all.iloc[0],
        "slew_conf": slew_conf.set_index(["firmware", "gimbal_lot"]),
        "pareto_top": pareto.iloc[0],
        "retests": retests,
        "fig_weekly": _figure(fig_weekly, weekly_table, "Failing units per week (a unit failing two checks counts under both)."),
        "fig_pareto": _figure(fig_pareto, pareto.set_index("failure_mode"), "Pareto of first-pass failures."),
        "fig_voltage": _figure(fig_voltage, voltage_table, "Weekly mean bus voltage reading per station."),
        "fig_slew": _figure(fig_slew, slew_table, "Slew time by gimbal motor lot."),
        "fig_cool": _figure(fig_cool, cool_table, "Weekly mean cooldown time by shift."),
    }


def describe_p_chart(chart: pd.DataFrame, violations: pd.DataFrame) -> str:
    """Plain-language summary of a p-chart, written from the rule violations."""
    def week(ts) -> str:
        return f"{ts.day} {ts:%b}"

    z = (chart["p"] - chart["center"]) / chart["sigma"]
    flagged = chart.index[sorted(violations["position"].unique())] if len(violations) else []
    high = [w for w in flagged if z[w] > 0]
    low = [w for w in flagged if z[w] < 0]
    beyond = chart.index[chart["p"] > chart["ucl"]]
    if not len(flagged):
        return "The weekly fail rate stays within its control limits with no run-rule violations: the process is in statistical control."
    parts = []
    if len(beyond):
        parts.append(f"The week of {', '.join(week(w) for w in beyond)} broke the upper control limit")
    if high:
        span = f"the week of {week(high[0])}" if len(high) == 1 else f"the weeks of {week(high[0])} to {week(high[-1])}"
        parts.append(f"run rules flag a sustained high fail rate from {span}")
        later = z[chart.index > high[-1]]
        if len(later):
            text = f"After that the rate drops: {(later < 0).sum()} of the next {len(later)} weeks sit below the center line"
            if low:
                text += f", and run rules flag the low level by the week of {week(low[0])}"
            parts.append(text)
    elif low:
        parts.append(f"run rules flag an unusually low fail rate by the week of {week(low[0])}")
    sentence = "; ".join(parts[:2]) + "."
    if len(parts) > 2:
        sentence += " " + parts[2] + "."
    return (sentence[0].upper() + sentence[1:] +
            " The process was not stable over this period, which matters for every analysis below.")


def _secom_section() -> dict:
    raw = secom.load_raw()
    cleaned, cleaning = secom.clean(raw)
    sensors = secom.sensor_columns(cleaned)

    weekly = secom.daily_yield(cleaned, "W")
    chart = spc.p_chart(weekly["failures"], weekly["units"])
    violations = spc.western_electric(chart["p"], chart["center"], chart["sigma"])
    fig_p, p_table = charts.p_chart(chart, violations, "SECOM weekly fail rate (p-chart)")

    ranked_all = signals.rank_signals(cleaned[sensors], cleaned["failed"])
    train, test = secom.time_split(cleaned)
    ranked_train = signals.rank_signals(train[sensors], train["failed"])
    stability = signals.stability(train, test, ranked_train)
    gap = signals.cross_validation_gap(cleaned, sensors)
    top_signal = ranked_all["signal"].iloc[0]
    monthly = raw.groupby(raw["timestamp"].dt.to_period("M")).agg(
        units=("failed", "size"), fail_rate=("failed", "mean"), median=(top_signal, "median"))
    monthly.index = monthly.index.strftime("%b %Y")

    stab_table = stability.set_index("signal").round(3)
    holds = stability.loc[stability["holds_up"], "signal"].tolist()
    return {
        "units": len(raw),
        "failures": int(raw["failed"].sum()),
        "fail_rate": raw["failed"].mean(),
        "start": raw["timestamp"].min(),
        "end": raw["timestamp"].max(),
        "cleaning": cleaning.summary(),
        "significant": int((ranked_all["q_value"] < 0.05).sum()),
        "gap": gap,
        "top_signal": top_signal,
        "monthly": monthly,
        "holds": holds,
        "trend_text": describe_p_chart(chart, violations),
        "fig_p": _figure(fig_p, p_table, "Weekly SECOM fail rate with 3-sigma limits for each week's sample size."),
        "fig_stability": _figure(charts.stability_dumbbell(stability), stab_table,
                                 "Separation of the 10 strongest signals (ranked on the earlier period)."),
        "fig_drift": _figure(charts.era_drift(raw, top_signal), monthly.round(3),
                             f"{top_signal} readings by unit, with monthly summary in the table."),
        "ranked": ranked_all.head(10).set_index("signal")[["direction", "auc", "q_value"]],
    }


def build(out_dir: str | Path, eos_db: str | Path | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eos_path = Path(eos_db) if eos_db else generate(DEFAULT_EOS_DB)
    html = _env.get_template("report.html.j2").render(
        eos=_eos_section(eos_path),
        secom=_secom_section(),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=__version__,
        repo=REPO_URL,
        plotly_version=plotly.offline.get_plotlyjs_version(),
        themes=json.dumps({"light": charts.LIGHT, "dark": charts.DARK}),
        classes={
            eos.CLASS_TEST_EQUIPMENT: "c-equipment", eos.CLASS_COMPONENT: "c-component",
            eos.CLASS_PROCESS: "c-process", eos.CLASS_RANDOM: "c-random",
        },
    )
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
