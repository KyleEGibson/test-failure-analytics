"""Failure analysis of an EOS-100 test station results database.

Works on any database with the station schema (runs, test_results,
measurements) plus a ``units`` table of manufacturing records.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency

from . import spc

FACTORS = ["station", "shift", "operator", "firmware", "gimbal_lot", "cooler_lot", "detector_lot"]
SIGNIFICANCE = 0.05


def _sql(name: str) -> str:
    return files("failure_analytics").joinpath("sql", f"{name}.sql").read_text(encoding="utf-8")


class EOSDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def query(self, name: str, **params) -> pd.DataFrame:
        conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        try:
            has_units = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'units'").fetchone()
            if not has_units:
                # A plain station results.db has no manufacturing records; analyze
                # it with every lot 'unknown'. TEMP tables never touch the file.
                conn.execute("CREATE TEMP TABLE units (serial TEXT, build_date TEXT, gimbal_lot TEXT,"
                             " cooler_lot TEXT, detector_lot TEXT)")
            return pd.read_sql_query(_sql(name), conn, params=params or None)
        finally:
            conn.close()

    def first_pass(self) -> pd.DataFrame:
        frame = self.query("first_pass_runs")
        frame["started_at"] = pd.to_datetime(frame["started_at"])
        frame["failed"] = frame["failed"].astype(bool)
        return frame

    def pareto(self) -> pd.DataFrame:
        return self.query("failure_pareto")

    def failures(self) -> pd.DataFrame:
        return self.query("first_pass_failures")

    def retests(self) -> pd.DataFrame:
        return self.query("retest_outcomes")

    def history(self, measurement: str) -> pd.DataFrame:
        frame = self.query("measurement_history", measurement=measurement)
        frame["started_at"] = pd.to_datetime(frame["started_at"])
        return frame


def mode_flags(first_pass: pd.DataFrame, failures: pd.DataFrame, mode: str) -> pd.Series:
    """Boolean per first-pass run: did it fail this particular measurement?"""
    failing_runs = set(failures.loc[failures["failure_mode"] == mode, "run_id"])
    return first_pass["run_id"].isin(failing_runs)


def weekly_yield(first_pass: pd.DataFrame) -> pd.DataFrame:
    grouped = first_pass.groupby("week")["failed"]
    weekly = pd.DataFrame({"units": grouped.size(), "failures": grouped.sum().astype(int)})
    weekly["fpy"] = 1 - weekly["failures"] / weekly["units"]
    return weekly


def factor_screen(first_pass: pd.DataFrame, flags: pd.Series) -> pd.DataFrame:
    """Chi-square test of each factor against one failure mode, strongest first.

    ``worst_level`` is the level with the highest failure rate; ``risk_ratio``
    compares it with every other level combined.
    """
    rows = []
    for factor in FACTORS:
        table = pd.crosstab(first_pass[factor], flags)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        _, p, _, _ = chi2_contingency(table)
        rates = flags.groupby(first_pass[factor]).mean()
        worst = rates.idxmax()
        rest = flags[first_pass[factor] != worst].mean()
        rows.append({
            "factor": factor,
            "p_value": float(p),
            "worst_level": worst,
            "worst_rate": float(rates[worst]),
            "rest_rate": float(rest),
            "risk_ratio": float(rates[worst] / rest) if rest > 0 else float("inf"),
        })
    screen = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    # Bonferroni across the factors screened for this mode
    screen["significant"] = screen["p_value"] < SIGNIFICANCE / len(screen)
    return screen


def onset_week(first_pass: pd.DataFrame, flags: pd.Series, baseline_weeks: int = 3) -> int | None:
    """First week whose failure rate breaks the upper p-chart limit of the baseline weeks."""
    weekly = pd.DataFrame({"n": first_pass.groupby("week").size(), "d": flags.groupby(first_pass["week"]).sum()})
    base = weekly.loc[weekly.index <= baseline_weeks]
    p_bar = max(base["d"].sum() / base["n"].sum(), 0.5 / base["n"].sum())  # floor for a zero-defect baseline
    ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / weekly["n"])
    breaking = weekly.index[(weekly["d"] / weekly["n"] > ucl) & (weekly.index > baseline_weeks)]
    return int(breaking[0]) if len(breaking) else None


def confounder_check(first_pass: pd.DataFrame, flags: pd.Series, suspect: str, alternative: str) -> pd.DataFrame:
    """Failure rate by ``suspect`` within each level of ``alternative``.

    If the suspect still separates failures when the alternative is held fixed,
    the alternative does not explain the failures.
    """
    table = (
        pd.DataFrame({suspect: first_pass[suspect], alternative: first_pass[alternative], "fail": flags})
        .groupby([alternative, suspect])["fail"]
        .agg(units="size", failures="sum")
        .reset_index()
    )
    table["rate"] = table["failures"] / table["units"]
    return table


def conditional_p(first_pass: pd.DataFrame, flags: pd.Series, factor: str, given: str) -> float | None:
    """p-value for ``factor`` vs failure with ``given`` held fixed (stratified chi-square).

    Chi-square statistics from each stratum of ``given`` are summed with their
    degrees of freedom. Returns None when no stratum contains two levels of
    ``factor``, meaning the two factors are aliased and cannot be separated.
    """
    stat_total, dof_total = 0.0, 0
    for _, index in first_pass.groupby(given).groups.items():
        table = pd.crosstab(first_pass.loc[index, factor], flags.loc[index])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        stat, _, dof, _ = chi2_contingency(table, correction=False)
        stat_total += stat
        dof_total += dof
    if dof_total == 0:
        return None
    return float(chi2.sf(stat_total, dof_total))


def isolate(first_pass: pd.DataFrame, flags: pd.Series) -> tuple[pd.DataFrame, dict[str, str]]:
    """Screen factors, then drop any factor explained away by another.

    Returns the screen and ``explained``: factor -> the factor that explains it.
    A factor survives only if it still separates failures with every other
    significant factor held fixed.
    """
    screen = factor_screen(first_pass, flags)
    candidates = screen.loc[screen["significant"], "factor"].tolist()
    explained: dict[str, str] = {}
    for factor in candidates:
        for other in candidates:
            if other == factor:
                continue
            p = conditional_p(first_pass, flags, factor, other)
            if p is not None and p >= SIGNIFICANCE:
                explained[factor] = other
                break
    screen["explained_by"] = screen["factor"].map(explained)
    return screen, explained


def weekly_xbar(history: pd.DataFrame, week: pd.Series, baseline_weeks: int = 3) -> pd.DataFrame:
    """Weekly mean chart of one measurement, limits from the baseline weeks.

    Sigma is estimated from the moving range of the baseline readings, and each
    week's limits scale with sqrt(n) for that week.
    """
    values = history["value"].to_numpy()
    base = values[(week <= baseline_weeks).to_numpy()]
    limits, _ = spc.imr_limits(base)
    grouped = pd.DataFrame({"week": week.to_numpy(), "value": values}).groupby("week")["value"]
    chart = pd.DataFrame({"mean": grouped.mean(), "n": grouped.size()})
    chart["center"] = limits.center
    chart["sigma"] = limits.sigma / np.sqrt(chart["n"])
    chart["ucl"] = chart["center"] + 3 * chart["sigma"]
    chart["lcl"] = chart["center"] - 3 * chart["sigma"]
    return chart


def first_signal_week(chart: pd.DataFrame, baseline_weeks: int = 3) -> int | None:
    violations = spc.western_electric(chart["mean"], chart["center"], chart["sigma"])
    weeks = chart.index.to_numpy()[violations["position"].to_numpy()] if len(violations) else []
    later = [int(w) for w in weeks if w > baseline_weeks]
    return min(later) if later else None


CLASS_TEST_EQUIPMENT = "Test equipment (false failures)"
CLASS_COMPONENT = "Component / supplier lot"
CLASS_PROCESS = "Process / environment"
CLASS_RANDOM = "No assignable cause (background)"


@dataclass
class Finding:
    failure_mode: str
    failures: int
    classification: str
    factor: str | None
    level: str | None
    level_rate: float
    rest_rate: float
    onset_week: int | None
    spc_signal_week: int | None
    no_fault_found_pct: float
    screen_all: pd.DataFrame
    screen_window: pd.DataFrame
    explained: dict[str, str]


def _classify(factor: str | None, nff_pct: float) -> str:
    if factor is None:
        return CLASS_RANDOM
    if factor == "station" and nff_pct >= 50:
        return CLASS_TEST_EQUIPMENT
    if factor.endswith("_lot"):
        return CLASS_COMPONENT
    if factor in ("shift", "operator"):
        return CLASS_PROCESS
    return f"Associated with {factor}"


def week_of(first_pass: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    day0 = first_pass["started_at"].min().normalize()
    return ((history["started_at"].dt.normalize() - day0).dt.days // 7 + 1).astype(int)


def analyze(db: EOSDatabase, min_failures: int = 5) -> list[Finding]:
    """One finding per failure mode with at least ``min_failures`` first-pass failures.

    1. Onset: first week the mode's failure rate breaks the baseline p-chart.
    2. Screen every factor within the window from onset onward, so factors
       that merely changed at some other time drop out.
    3. Remove factors explained away by another (stratified chi-square).
    4. Use retest outcomes to tell test-equipment problems from real defects.
    5. Find the first week an X-bar chart of the raw measurement signalled,
       which can precede the first hard failure.
    """
    first_pass = db.first_pass()
    failures = db.failures()
    retests = db.retests().groupby("failure_mode")[["failed_units", "passed_retest"]].sum()
    findings = []
    for mode, count in db.pareto().set_index("failure_mode")["failures"].items():
        if count < min_failures:
            continue
        flags = mode_flags(first_pass, failures, mode)
        onset = onset_week(first_pass, flags)
        window = first_pass["week"] >= onset if onset else pd.Series(True, index=first_pass.index)
        screen_window, explained = isolate(first_pass[window], flags[window])
        survivors = screen_window[screen_window["significant"] & screen_window["explained_by"].isna()]
        top = survivors.iloc[0] if onset and len(survivors) else None
        nff = 100.0 * retests.loc[mode, "passed_retest"] / retests.loc[mode, "failed_units"] if mode in retests.index else 0.0

        history = db.history(mode)
        if top is not None and top["factor"] == "station":
            history = history[history["station"] == top["worst_level"]]
        chart = weekly_xbar(history, week_of(first_pass, history))

        findings.append(Finding(
            failure_mode=mode,
            failures=int(count),
            classification=_classify(None if top is None else top["factor"], nff),
            factor=None if top is None else top["factor"],
            level=None if top is None else str(top["worst_level"]),
            level_rate=0.0 if top is None else float(top["worst_rate"]),
            rest_rate=0.0 if top is None else float(top["rest_rate"]),
            onset_week=onset,
            spc_signal_week=first_signal_week(chart) if onset else None,
            no_fault_found_pct=float(nff),
            screen_all=factor_screen(first_pass, flags),
            screen_window=screen_window,
            explained=explained,
        ))
    return findings


def measurement_control(history: pd.DataFrame, baseline: int = 100) -> tuple[spc.Limits, pd.DataFrame]:
    """I-MR limits from the first ``baseline`` readings, then rule violations over all readings."""
    limits, _ = spc.imr_limits(history["value"].iloc[:baseline])
    violations = spc.western_electric(history["value"], limits.center, limits.sigma)
    return limits, violations
