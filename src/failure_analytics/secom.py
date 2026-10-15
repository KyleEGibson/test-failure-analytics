"""Load and clean the UCI SECOM semiconductor manufacturing dataset.

Source: McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C54305 (CC BY 4.0).

Each row is one production unit: 590 anonymized process sensor signals, an
in-house line test result (pass/fail) and a timestamp.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data" / "raw" / "secom.zip"
ARCHIVE_SHA256 = "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"


def load_raw(archive: Path = ARCHIVE) -> pd.DataFrame:
    """Return the dataset sorted by time: ``timestamp``, ``failed`` (bool), ``S000``..``S589``."""
    data = archive.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"{archive} checksum mismatch ({digest}); the source data has changed")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        features = pd.read_csv(zf.open("secom.data"), sep=" ", header=None)
        labels = pd.read_csv(zf.open("secom_labels.data"), sep=" ", header=None, names=["label", "timestamp"])
    features.columns = [f"S{i:03d}" for i in range(features.shape[1])]
    frame = pd.concat(
        [
            pd.DataFrame({
                "timestamp": pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S"),
                "failed": labels["label"] == 1,
            }),
            features,
        ],
        axis=1,
    )
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


@dataclass
class CleaningReport:
    raw_features: int
    dropped_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    dropped_duplicate: list[str] = field(default_factory=list)
    imputed_values: int = 0
    kept: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "raw sensor signals": self.raw_features,
            "dropped: more than 50% missing": len(self.dropped_missing),
            "dropped: constant (no information)": len(self.dropped_constant),
            "dropped: exact duplicate of another signal": len(self.dropped_duplicate),
            "signals kept for analysis": len(self.kept),
            "missing values imputed (median)": self.imputed_values,
        }


def sensor_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("S")]


def clean(frame: pd.DataFrame, max_missing: float = 0.5) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop uninformative signals and median-impute the rest.

    Returns a new frame (timestamp, failed and the kept signals) and a report of
    every decision, so the cleaning is auditable.
    """
    sensors = sensor_columns(frame)
    report = CleaningReport(raw_features=len(sensors))
    X = frame[sensors]

    missing = X.isna().mean()
    report.dropped_missing = missing[missing > max_missing].index.tolist()
    X = X.drop(columns=report.dropped_missing)

    constant = X.nunique(dropna=True) <= 1
    report.dropped_constant = constant[constant].index.tolist()
    X = X.drop(columns=report.dropped_constant)

    duplicated = X.T.duplicated()
    report.dropped_duplicate = duplicated[duplicated].index.tolist()
    X = X.drop(columns=report.dropped_duplicate)

    report.imputed_values = int(X.isna().sum().sum())
    X = X.fillna(X.median())
    report.kept = X.columns.tolist()

    cleaned = pd.concat([frame[["timestamp", "failed"]], X], axis=1)
    return cleaned, report


def daily_yield(frame: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Units tested and failures per period, for days with production."""
    grouped = frame.groupby(frame["timestamp"].dt.to_period(freq))["failed"]
    out = pd.DataFrame({"units": grouped.size(), "failures": grouped.sum().astype(int)})
    out = out[out["units"] > 0]
    out.index = out.index.to_timestamp()
    out.index.name = "period"
    return out


def time_split(frame: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: train on the earlier units, evaluate on the later ones."""
    cut = int(np.floor(len(frame) * train_fraction))
    return frame.iloc[:cut], frame.iloc[cut:]
