"""Generate a synthetic EOS-100 production test history with known root causes.

The output uses the same schema as the EOS-100 test station's results.db
(runs, test_results, measurements), plus a ``units`` table that stands in for
manufacturing records (component lots, build date).

PLANTED ROOT CAUSES — the answer key. The analysis code never reads this
module; tests/test_known_answers.py checks that the analysis rediscovers them.

1. Component lot (real defect): gimbal motor lot G-2603, used from week 7,
   slews slowly -> "90 deg azimuth slew time" failures. Failures repeat on
   retest because the unit really is out of spec.
2. Test equipment (false failures): station BENCH-02's voltage measurement
   drifts upward from week 9 -> "bus voltage" failures on BENCH-02 only.
   Units pass on retest at BENCH-01 (no fault found).
3. Environment: from week 6 the lab HVAC cannot keep up on the swing shift ->
   slower cooldown -> "cooldown time" failures concentrated on swing shift.

Red herring: firmware 2.4.2 is released in week 5, before lot G-2603 appears,
so a naive before/after comparison blames the firmware.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

START = datetime(2026, 6, 1, 6, 0)  # a Monday
WEEKS = 12
UNITS_PER_SHIFT = (9, 12)  # inclusive range
SHIFTS = {"day": 6, "swing": 14}
OPERATORS = {"day": ["amorales", "jchen"], "swing": ["dpatel", "kgibson"]}
STATIONS = ["BENCH-01", "BENCH-02"]
STATION_VERSION = "1.0.0"
STATION_COMMIT = "ae6ab4f"

# (test id, requirement, measurement name, low, high, units)
MEASUREMENTS = [
    ("tests/component/test_power.py::test_boot_time", "SYS-002", "time to READY", None, 3.0, "s"),
    ("tests/component/test_power.py::test_bus_voltage", "SYS-005", "bus voltage", 27.0, 29.0, "V"),
    ("tests/component/test_power.py::test_peak_current_during_cooldown_and_slew", "SYS-006",
     "peak input current", None, 4.0, "A"),
    ("tests/component/test_thermal.py::test_cooldown_time_and_setpoint", "SYS-010", "cooldown time", None, 15.0, "s"),
    ("tests/component/test_thermal.py::test_temperature_stability", "SYS-011", "maximum temperature", -41.0, -39.0, "C"),
    ("tests/component/test_gimbal.py::test_pointing_accuracy[az45_el10]", "SYS-020",
     "azimuth error", -0.1, 0.1, "deg"),
    ("tests/component/test_gimbal.py::test_slew_time_90_degrees", "SYS-022",
     "90 deg azimuth slew time", None, 2.0, "s"),
    ("tests/component/test_imager.py::test_dead_pixel_count", "SYS-031",
     "worst-case dead pixels", None, 5, "px"),
    ("tests/component/test_imager.py::test_exposure_linearity", "SYS-032", "linearity R^2", 0.99, None, ""),
]

SCHEMA = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT, station TEXT, operator TEXT,
    station_version TEXT, git_commit TEXT, dut_model TEXT, dut_serial TEXT, dut_firmware TEXT,
    injected_faults TEXT, comm_retries INTEGER, outcome TEXT
);
CREATE TABLE test_results (
    run_id TEXT NOT NULL REFERENCES runs(run_id), test_id TEXT NOT NULL, requirements TEXT,
    outcome TEXT NOT NULL, duration_s REAL, message TEXT, comm_retries INTEGER
);
CREATE TABLE measurements (
    run_id TEXT NOT NULL REFERENCES runs(run_id), test_id TEXT NOT NULL, name TEXT NOT NULL,
    value REAL, low REAL, high REAL, units TEXT, passed INTEGER NOT NULL
);
CREATE TABLE units (
    serial TEXT PRIMARY KEY, build_date TEXT, gimbal_lot TEXT, cooler_lot TEXT, detector_lot TEXT
);
"""


def _gimbal_lot(week: int, rng: np.random.Generator) -> str:
    if week < 3:
        return "G-2601"
    if week < 6:
        return "G-2602"
    if week < 8:  # two-week transition: both lots on the line
        return "G-2603" if rng.random() < 0.5 else "G-2602"
    return "G-2603"


class _Unit:
    """True physical properties of one built unit."""

    def __init__(self, serial: str, build: datetime, week: int, rng: np.random.Generator) -> None:
        self.serial = serial
        self.build = build
        self.gimbal_lot = _gimbal_lot(week, rng)
        self.cooler_lot = rng.choice(["C-11", "C-12"])
        self.detector_lot = "D-A1" if week < 6 else "D-A2"
        slow_lot = self.gimbal_lot == "G-2603"
        self.slew_mean = rng.normal(1.90 if slow_lot else 1.56, 0.07 if slow_lot else 0.04)
        self.voltage = rng.normal(28.0, 0.06)
        self.cooler_tau = rng.normal(1.0, 0.04)
        self.dead_pixels = rng.poisson(1.0) + (rng.integers(8, 40) if rng.random() < 0.008 else 0)
        self.regulation_fault = rng.random() < 0.005
        self.linearity = min(0.9999, rng.normal(0.9978, 0.0006)) if rng.random() > 0.003 else 0.975


def _measure(unit: _Unit, station: str, shift: str, when: datetime, rng: np.random.Generator) -> dict[str, float]:
    week = (when - START).days // 7
    # Planted cause 2: BENCH-02 voltage calibration drifts upward from week 9.
    station_offset = 0.0
    if station == "BENCH-02" and week >= 8:
        station_offset = 0.35 * ((when - (START + timedelta(weeks=8))).days / 7 + 1)
    # Planted cause 3: hot lab on swing shift from week 6 slows cooldown.
    heat = 3.9 if (shift == "swing" and week >= 5) else 0.0
    return {
        "time to READY": rng.normal(2.03, 0.04),
        "bus voltage": unit.voltage + station_offset + rng.normal(0, 0.03),
        "peak input current": rng.normal(3.42, 0.11),
        "cooldown time": 9.9 * unit.cooler_tau + heat + rng.normal(0, 0.7),
        "maximum temperature": (-38.4 if unit.regulation_fault else -39.72) + rng.normal(0, 0.08),
        "azimuth error": rng.normal(0, 0.025),
        "90 deg azimuth slew time": unit.slew_mean + rng.normal(0, 0.03),
        "worst-case dead pixels": float(unit.dead_pixels),
        "linearity R^2": unit.linearity,
    }


def generate(path: str | Path, seed: int = 2026) -> Path:
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    serial_no = 100000
    retest_queue: list[_Unit] = []
    for day in range(WEEKS * 7):
        date = START + timedelta(days=day)
        if date.weekday() >= 5:
            continue
        week = day // 7
        firmware = "2.4.1" if week < 4 else "2.4.2"
        for shift, hour in SHIFTS.items():
            clock = date.replace(hour=hour) + timedelta(minutes=int(rng.integers(0, 20)))
            jobs = [(u, True) for u in retest_queue]
            retest_queue = []
            for _ in range(int(rng.integers(UNITS_PER_SHIFT[0], UNITS_PER_SHIFT[1] + 1))):
                serial_no += 1
                unit = _Unit(f"SN{serial_no}", clock, week, rng)
                conn.execute("INSERT INTO units VALUES (?,?,?,?,?)",
                             (unit.serial, clock.date().isoformat(), unit.gimbal_lot, unit.cooler_lot, unit.detector_lot))
                jobs.append((unit, False))
            for unit, is_retest in jobs:
                station = STATIONS[int(rng.integers(0, 2))]
                if is_retest:
                    station = "BENCH-01"  # failed units are retested on the reference bench
                operator = OPERATORS[shift][int(rng.integers(0, 2))]
                started = clock
                clock += timedelta(minutes=float(rng.normal(14, 2)))
                values = _measure(unit, station, shift, started, rng)
                run_id = f"{started:%Y%m%d-%H%M%S}-{uuid.UUID(int=int(rng.integers(0, 2**63))).hex[:4]}"
                run_failed = False
                for test_id, req, name, low, high, units in MEASUREMENTS:
                    value = values[name]
                    passed = (low is None or value >= low) and (high is None or value <= high)
                    run_failed |= not passed
                    conn.execute("INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?)",
                                 (run_id, test_id, name, round(value, 4), low, high, units, int(passed)))
                    conn.execute("INSERT INTO test_results VALUES (?,?,?,?,?,?,?)",
                                 (run_id, test_id, req, "passed" if passed else "failed",
                                  round(abs(rng.normal(1.2, 0.3)), 2),
                                  "" if passed else f"{name} = {value:.3f} {units} outside limits", 0))
                conn.execute(
                    "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, started.isoformat(timespec="seconds"), clock.isoformat(timespec="seconds"),
                     station, operator, STATION_VERSION, STATION_COMMIT, "EOS-100", unit.serial, firmware,
                     "", 0, "FAIL" if run_failed else "PASS"),
                )
                if run_failed and not is_retest:
                    retest_queue.append(unit)
    conn.commit()
    conn.close()
    return path
