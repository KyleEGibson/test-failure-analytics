"""EOS-100 production analysis: SQL queries, confounder handling and known answers."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from failure_analytics import eos
from failure_analytics.eos_generator import MEASUREMENTS, generate


@pytest.fixture(scope="session")
def db(tmp_path_factory):
    return eos.EOSDatabase(generate(tmp_path_factory.mktemp("eos") / "eos.db"))


@pytest.fixture(scope="session")
def findings(db):
    return {f.failure_mode: f for f in eos.analyze(db)}


# ------------------------------------------------------------ data and SQL


def test_generator_is_deterministic(tmp_path):
    def snapshot(path):
        with sqlite3.connect(path) as conn:
            return pd.read_sql("SELECT dut_serial, outcome, started_at FROM runs ORDER BY started_at", conn)
    pd.testing.assert_frame_equal(snapshot(generate(tmp_path / "a.db")), snapshot(generate(tmp_path / "b.db")))


def test_schema_matches_test_station(db):
    """The generated data uses the same tables and columns as the EOS-100 station's results.db."""
    with sqlite3.connect(db.path) as conn:
        columns = {table: [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
                   for table in ("runs", "test_results", "measurements")}
    assert columns["measurements"] == ["run_id", "test_id", "name", "value", "low", "high", "units", "passed"]
    assert columns["test_results"][-1] == "comm_retries"
    assert "dut_firmware" in columns["runs"]


def test_first_pass_has_one_row_per_unit(db):
    first_pass = db.first_pass()
    assert first_pass["serial"].is_unique
    with sqlite3.connect(db.path) as conn:
        units = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    assert len(first_pass) == units


def test_pareto_shares_add_up(db):
    pareto = db.pareto()
    assert pareto["failures"].is_monotonic_decreasing
    assert pareto["share_pct"].sum() == pytest.approx(100, abs=0.5)
    assert pareto["cumulative_pct"].iloc[-1] == pytest.approx(100)
    assert set(pareto["failure_mode"]) <= {m[2] for m in MEASUREMENTS}


def test_no_fault_found_uses_retested_units(db, findings):
    """Regression: the rate was divided by all failed units, including ones never retested,
    so the report and the SQL disagreed (82% vs 86%)."""
    retests = db.retests().groupby("failure_mode")[["failed_units", "retested_units", "passed_retest"]].sum()
    row = retests.loc["bus voltage"]
    assert row["retested_units"] < row["failed_units"]  # some failures were never retested
    assert findings["bus voltage"].no_fault_found_pct == pytest.approx(100 * row["passed_retest"] / row["retested_units"])


def test_pareto_sql_matches_pandas(db):
    """Cross-check the SQL window-function query with an independent pandas count."""
    counts = db.failures()["failure_mode"].value_counts()
    sql = db.pareto().set_index("failure_mode")["failures"]
    pd.testing.assert_series_equal(sql.sort_index(), counts.sort_index(), check_names=False)


def test_plain_station_database_without_units_table(db, tmp_path):
    """A results.db straight from the test station has no units table; it must still
    analyze (lots 'unknown') and the file must not be modified."""
    plain = tmp_path / "station.db"
    with sqlite3.connect(db.path) as src, sqlite3.connect(plain) as dst:
        src.backup(dst)
        dst.execute("DROP TABLE units")
    before = plain.read_bytes()
    station_db = eos.EOSDatabase(plain)
    assert set(station_db.first_pass()["gimbal_lot"]) == {"unknown"}
    findings = {f.failure_mode: f for f in eos.analyze(station_db)}
    assert findings["bus voltage"].level == "BENCH-02"  # still found without lot data
    assert plain.read_bytes() == before


# --------------------------------------------------------- statistics


def test_conditional_p_explains_away_nested_factor():
    """Operators nested in shifts: operator stops mattering once shift is held fixed."""
    rng = np.random.default_rng(3)
    n = 800
    shift = np.where(np.arange(n) % 2, "swing", "day")
    operator = np.where(shift == "swing", rng.choice(["c", "d"], n), rng.choice(["a", "b"], n))
    failed = pd.Series(rng.random(n) < np.where(shift == "swing", 0.15, 0.01))
    frame = pd.DataFrame({"shift": shift, "operator": operator})
    assert eos.conditional_p(frame, failed, "operator", "shift") > 0.05
    assert eos.conditional_p(frame, failed, "shift", "operator") is None  # aliased: cannot separate


def test_onset_week_finds_step_change():
    weeks = np.repeat(np.arange(1, 9), 100)
    rng = np.random.default_rng(4)
    failed = pd.Series(rng.random(800) < np.where(weeks >= 6, 0.12, 0.01))
    assert eos.onset_week(pd.DataFrame({"week": weeks}), failed) == 6


def test_onset_week_none_for_stable_process():
    weeks = np.repeat(np.arange(1, 9), 100)
    failed = pd.Series(np.random.default_rng(5).random(800) < 0.02)
    assert eos.onset_week(pd.DataFrame({"week": weeks}), failed) is None


# ------------------------------------------------- known answers (planted)


def test_finds_supplier_lot(findings):
    f = findings["90 deg azimuth slew time"]
    assert f.classification == eos.CLASS_COMPONENT
    assert (f.factor, f.level) == ("gimbal_lot", "G-2603")
    assert f.onset_week == 7


def test_finds_test_equipment_drift_with_early_warning(findings):
    f = findings["bus voltage"]
    assert f.classification == eos.CLASS_TEST_EQUIPMENT
    assert (f.factor, f.level) == ("station", "BENCH-02")
    assert f.no_fault_found_pct > 50
    assert f.spc_signal_week == 9 and f.spc_signal_week < f.onset_week


def test_finds_environment_and_rejects_operator(findings):
    f = findings["cooldown time"]
    assert f.classification == eos.CLASS_PROCESS
    assert (f.factor, f.level) == ("shift", "swing")
    assert f.explained.get("operator") == "shift"


def test_background_failures_get_no_cause(findings):
    assert findings["worst-case dead pixels"].classification == eos.CLASS_RANDOM


def test_naive_screen_would_have_been_wrong(findings):
    """Documents why the windowed, stratified method exists: the naive whole-history
    screen blames the wrong factor for two of the three planted causes."""
    naive_voltage = findings["bus voltage"].screen_all.iloc[0]["factor"]
    naive_cooldown = findings["cooldown time"].screen_all.iloc[0]["factor"]
    assert naive_voltage != "station"
    assert naive_cooldown != "shift"


def test_firmware_red_herring_cleared(db):
    first_pass = db.first_pass()
    flags = eos.mode_flags(first_pass, db.failures(), "90 deg azimuth slew time")
    table = eos.confounder_check(first_pass, flags, "gimbal_lot", "firmware").set_index(["firmware", "gimbal_lot"])
    assert table.loc[("2.4.2", "G-2602"), "failures"] == 0      # new firmware, old lot: fine
    assert table.loc[("2.4.2", "G-2603"), "failures"] > 20      # new firmware, new lot: failing
