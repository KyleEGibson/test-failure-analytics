"""Interactive failure analytics dashboard.

    streamlit run app/dashboard.py

EOS-100 production history (or upload a station results.db) and SECOM.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from failure_analytics import charts, eos, secom, signals, spc
from failure_analytics.eos_generator import generate
from failure_analytics.report import DEFAULT_EOS_DB, describe_p_chart

st.set_page_config(page_title="Test Failure Analytics", layout="wide")


def theme() -> dict[str, str]:
    try:
        return charts.DARK if st.context.theme.type == "dark" else charts.LIGHT
    except AttributeError:
        return charts.LIGHT


def show(fig, table: pd.DataFrame | None = None) -> None:
    st.plotly_chart(charts.apply_theme(fig, theme()), theme=None, use_container_width=True,
                    config={"displayModeBar": False})
    if table is not None:
        with st.expander("Table view"):
            st.dataframe(table, use_container_width=True)


@st.cache_resource
def default_eos_db() -> str:
    return str(DEFAULT_EOS_DB if DEFAULT_EOS_DB.exists() else generate(DEFAULT_EOS_DB))


@st.cache_data
def load_eos(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    db = eos.EOSDatabase(path)
    return db.first_pass(), db.failures()


@st.cache_data
def load_secom() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = secom.load_raw()
    cleaned, report = secom.clean(raw)
    return raw, cleaned, report.summary()


@st.cache_data
def secom_ranking(_cleaned: pd.DataFrame) -> pd.DataFrame:
    sensors = secom.sensor_columns(_cleaned)
    return signals.rank_signals(_cleaned[sensors], _cleaned["failed"])


st.title("Test Failure Analytics")
eos_tab, secom_tab = st.tabs(["EOS-100 production", "SECOM semiconductor line"])

# =========================================================================== EOS
with eos_tab:
    upload = st.file_uploader("Analyze your own test station results.db (optional)", type=["db", "sqlite"])
    if upload is not None:
        tmp = Path(tempfile.mkdtemp()) / "uploaded.db"
        tmp.write_bytes(upload.getvalue())
        db_path = str(tmp)
        st.caption(f"Analyzing **{upload.name}**")
    else:
        db_path = default_eos_db()
        st.caption("Synthetic EOS-100 production history with three planted root causes "
                   "(same schema as the EOS-100 test station).")

    first_pass, failures = load_eos(db_path)
    if first_pass.empty:
        st.warning("No test runs found in this database.")
        st.stop()

    weeks = (int(first_pass["week"].min()), int(first_pass["week"].max()))
    f1, f2, f3 = st.columns([2, 1, 1])
    week_range = f1.slider("Production weeks", *weeks, value=weeks) if weeks[0] < weeks[1] else weeks
    stations = f2.multiselect("Station", sorted(first_pass["station"].unique()))
    shifts = f3.multiselect("Shift", sorted(first_pass["shift"].unique()))

    mask = first_pass["week"].between(*week_range)
    if stations:
        mask &= first_pass["station"].isin(stations)
    if shifts:
        mask &= first_pass["shift"].isin(shifts)
    view = first_pass[mask].reset_index(drop=True)
    view_failures = failures[failures["run_id"].isin(view["run_id"])]

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Units (first pass)", f"{len(view):,}")
    t2.metric("First-pass yield", f"{1 - view['failed'].mean():.1%}" if len(view) else "-")
    t3.metric("Failing units", f"{int(view['failed'].sum()):,}")
    t4.metric("Failure modes seen", view_failures["failure_mode"].nunique())

    if view_failures.empty:
        st.success("No first-pass failures in this selection.")
        st.stop()

    counts = view_failures["failure_mode"].value_counts()
    pareto = pd.DataFrame({"failure_mode": counts.index, "failures": counts.values})
    pareto["share_pct"] = (100 * pareto["failures"] / pareto["failures"].sum()).round(1)
    pareto["cumulative_pct"] = pareto["share_pct"].cumsum().round(1)

    left, right = st.columns(2)
    with left:
        fig, table = charts.weekly_failures_by_mode(view, view_failures, pareto["failure_mode"].head(3).tolist())
        show(fig, table)
    with right:
        show(charts.pareto(pareto), pareto.set_index("failure_mode"))

    st.subheader("Root cause drill-down")
    mode = st.selectbox("Failure mode", pareto["failure_mode"].tolist())
    flags = eos.mode_flags(view, view_failures, mode)
    onset = eos.onset_week(view, flags) if view["week"].nunique() > 3 else None
    window = view["week"] >= onset if onset else pd.Series(True, index=view.index)
    screen, explained = eos.isolate(view[window], flags[window])

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Factor screen** {'from week %d (onset)' % onset if onset else 'over the selection'}")
        shown = screen.assign(
            p_value=screen["p_value"].map(lambda p: f"{p:.1e}"),
            worst_rate=screen["worst_rate"].map("{:.1%}".format),
            rest_rate=screen["rest_rate"].map("{:.1%}".format),
            risk_ratio=screen["risk_ratio"].map(lambda r: "∞" if r == float("inf") else f"{r:.2f}"),
        )
        st.dataframe(shown, use_container_width=True, hide_index=True)
    with c2:
        survivors = screen[screen["significant"] & screen["explained_by"].isna()]
        if onset and len(survivors):
            top = survivors.iloc[0]
            st.markdown(f"**Isolated factor:** {top['factor'].replace('_', ' ')} **{top['worst_level']}** "
                        f"({top['worst_rate']:.1%} vs {top['rest_rate']:.1%} elsewhere)")
        else:
            st.markdown("**No assignable cause** in this selection: failures look like a stable background rate.")
        for factor, by in explained.items():
            st.caption(f"{factor} was explained away by {by}: it stops separating failures once {by} is held fixed.")

    factor = st.selectbox("Compare readings by", eos.FACTORS, index=eos.FACTORS.index("station"))
    history = (eos.EOSDatabase(db_path).history(mode)
               .drop(columns="station")  # comes back with the factors below
               .merge(view[["run_id", "week", *eos.FACTORS]], on="run_id"))
    if not history.empty:
        limit = history["high"].dropna()
        usl = float(limit.iloc[0]) if len(limit) else float(history["value"].max())
        units = history["units"].iloc[0] or ""
        fig, table = charts.by_level_distribution(history, factor, mode, usl, units)
        show(fig, table)

# ========================================================================= SECOM
with secom_tab:
    raw, cleaned, cleaning = load_secom()
    st.caption("McCann & Johnston (2008), SECOM, UCI Machine Learning Repository, CC BY 4.0. "
               f"{len(raw):,} units, 590 process sensors, {int(raw['failed'].sum())} failures.")

    freq = st.radio("Control chart period", ["Weekly", "Daily"], horizontal=True)
    yield_frame = secom.daily_yield(cleaned, "W" if freq == "Weekly" else "D")
    chart = spc.p_chart(yield_frame["failures"], yield_frame["units"])
    violations = spc.western_electric(chart["p"], chart["center"], chart["sigma"])
    fig, table = charts.p_chart(chart, violations, f"SECOM {freq.lower()} fail rate (p-chart)")
    show(fig, table)
    st.write(describe_p_chart(chart, violations))

    st.subheader("Signal explorer")
    ranked = secom_ranking(cleaned)
    s1, s2 = st.columns([1, 3])
    top_n = s1.slider("Signals to list", 5, 50, 15)
    s1.dataframe(ranked.head(top_n)[["signal", "direction", "auc", "q_value"]].assign(
        q_value=lambda d: d["q_value"].map(lambda q: f"{q:.1e}")), hide_index=True, use_container_width=True)
    with s2:
        chosen = st.selectbox("Signal", ranked["signal"].head(top_n).tolist())
        show(charts.era_drift(raw, chosen))

    with st.expander("Data cleaning summary"):
        st.table(pd.Series(cleaning, name="count"))
