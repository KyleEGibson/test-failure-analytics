"""Signal ranking, FDR correction and the SECOM data pipeline."""

import numpy as np
import pandas as pd
import pytest

from failure_analytics import secom, signals


def test_benjamini_hochberg_known_values():
    q = signals.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    # sorted p: .01 .03 .04 .20 -> p*n/rank: .04 .06 .0533 .20 -> monotone: .04 .0533 .0533 .20
    assert q == pytest.approx([0.04, 0.05333, 0.05333, 0.20], abs=1e-4)


def test_benjamini_hochberg_ignores_nan():
    """Regression: one untestable signal used to turn every q-value into NaN."""
    q = signals.benjamini_hochberg([0.01, np.nan, 0.04])
    assert np.isnan(q[1])
    assert q[0] == pytest.approx(0.02) and q[2] == pytest.approx(0.04)


def test_rank_signals_finds_planted_signal():
    rng = np.random.default_rng(1)
    failed = np.r_[np.ones(40, bool), np.zeros(400, bool)]
    X = pd.DataFrame({
        "noise_a": rng.normal(size=440),
        "planted": rng.normal(size=440) + failed * 1.5,
        "noise_b": rng.normal(size=440),
        "constant": np.ones(440),
    })
    ranked = signals.rank_signals(X, pd.Series(failed))
    assert ranked["signal"].iloc[0] == "planted"
    assert ranked["direction"].iloc[0] == "higher in failures"
    assert ranked["q_value"].iloc[0] < 1e-6
    assert "constant" not in ranked["signal"].tolist()


def test_stability_flags_signal_that_only_tracks_time():
    """A signal that separates failures early but not later must not 'hold up'."""
    rng = np.random.default_rng(2)
    n = 400
    failed = rng.random(n) < 0.1
    era_proxy = np.where(np.arange(n) < 200, failed * 2.0, 0) + rng.normal(size=n)
    real_cause = failed * 2.0 + rng.normal(size=n)
    frame = pd.DataFrame({"failed": failed, "era_proxy": era_proxy, "real_cause": real_cause})
    train, test = frame.iloc[:200], frame.iloc[200:]
    ranked = pd.DataFrame({"signal": ["era_proxy", "real_cause"]})
    result = signals.stability(train, test, ranked).set_index("signal")
    assert not result.loc["era_proxy", "holds_up"]
    assert result.loc["real_cause", "holds_up"]


@pytest.fixture(scope="module")
def secom_frames():
    raw = secom.load_raw()
    cleaned, report = secom.clean(raw)
    return raw, cleaned, report


def test_secom_shape_and_labels(secom_frames):
    raw, _, _ = secom_frames
    assert raw.shape == (1567, 592)
    assert int(raw["failed"].sum()) == 104
    assert raw["timestamp"].is_monotonic_increasing


def test_secom_cleaning_is_auditable(secom_frames):
    _, cleaned, report = secom_frames
    summary = report.summary()
    assert summary["raw sensor signals"] == 590
    assert len(report.kept) + len(report.dropped_missing) + len(report.dropped_constant) \
        + len(report.dropped_duplicate) == 590
    assert not cleaned[report.kept].isna().any().any()


def test_secom_checksum_enforced(tmp_path):
    bad = tmp_path / "secom.zip"
    bad.write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="checksum mismatch"):
        secom.load_raw(bad)


def test_time_split_is_chronological(secom_frames):
    _, cleaned, _ = secom_frames
    train, test = secom.time_split(cleaned)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) + len(test) == len(cleaned)
