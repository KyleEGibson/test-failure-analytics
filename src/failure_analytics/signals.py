"""Rank process signals by how strongly they separate failing units from passing ones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


def benjamini_hochberg(p_values) -> np.ndarray:
    """False discovery rate adjusted p-values (q-values).

    NaN p-values (untestable signals) stay NaN and do not count as tests.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    n = len(valid)
    if n == 0:
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * n / np.arange(1, n + 1)
    # enforce monotonicity from the largest p downward
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


def rank_signals(X: pd.DataFrame, failed: pd.Series) -> pd.DataFrame:
    """Univariate screen: Mann-Whitney U test per signal, FDR-corrected.

    ``auc`` is the probability that a randomly chosen failing unit reads higher
    than a randomly chosen passing unit (0.5 = no separation). ``separation``
    folds direction away so 0 is none and 1 is perfect.
    """
    failed = np.asarray(failed, dtype=bool)
    rows = []
    for column in X.columns:
        fail_values = X.loc[failed, column]
        pass_values = X.loc[~failed, column]
        if X[column].nunique() <= 1:
            continue  # constant within this subset: nothing to test
        stat, p = mannwhitneyu(fail_values, pass_values, alternative="two-sided")
        auc = stat / (len(fail_values) * len(pass_values))
        rows.append({
            "signal": column,
            "median_pass": float(pass_values.median()),
            "median_fail": float(fail_values.median()),
            "auc": float(auc),
            "separation": float(abs(auc - 0.5) * 2),
            "direction": "higher in failures" if auc > 0.5 else "lower in failures",
            "p_value": float(p),
        })
    ranked = pd.DataFrame(rows)
    ranked["q_value"] = benjamini_hochberg(ranked["p_value"])
    return ranked.sort_values(["p_value", "separation"], ascending=[True, False]).reset_index(drop=True)


@dataclass
class ModelResult:
    test_auc: float
    train_units: int
    test_units: int
    test_failures: int
    importance: pd.DataFrame


def model_importance(train: pd.DataFrame, test: pd.DataFrame, signals: list[str],
                     seed: int = 7, top: int = 15) -> ModelResult:
    """Fit a random forest on earlier units and evaluate on later ones.

    Importance is permutation importance measured on the held-out later units,
    which reflects what the model actually relies on for unseen production
    rather than what it memorized.
    """
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1,
    )
    model.fit(train[signals], train["failed"])
    scores = model.predict_proba(test[signals])[:, 1]
    auc = roc_auc_score(test["failed"], scores)
    perm = permutation_importance(
        model, test[signals], test["failed"], scoring="roc_auc",
        n_repeats=10, random_state=seed, n_jobs=-1,
    )
    importance = (
        pd.DataFrame({"signal": signals, "importance": perm.importances_mean, "std": perm.importances_std})
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
    return ModelResult(float(auc), len(train), len(test), int(test["failed"].sum()), importance)


def stability(train: pd.DataFrame, test: pd.DataFrame, ranked_on_train: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Do the strongest signals from the earlier period still separate failures later?

    A signal whose separation collapses in the later period was tracking
    something that changed over time (a process era), not a cause of failure.
    """
    rows = []
    for signal in ranked_on_train["signal"].head(top):
        auc_train = roc_auc_score(train["failed"], train[signal])
        auc_test = roc_auc_score(test["failed"], test[signal])
        rows.append({
            "signal": signal,
            "separation_earlier": abs(auc_train - 0.5) * 2,
            "separation_later": abs(auc_test - 0.5) * 2,
            # same direction in both periods and still clearly separating
            "holds_up": (auc_train - 0.5) * (auc_test - 0.5) > 0 and abs(auc_test - 0.5) * 2 >= 0.2,
        })
    return pd.DataFrame(rows)


def cross_validation_gap(frame: pd.DataFrame, signals_: list[str], seed: int = 7) -> dict[str, float]:
    """AUC from shuffled 5-fold CV versus a chronological holdout.

    The gap measures how much shuffled CV overstates real-world performance
    when the data drifts over time.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from .secom import time_split

    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1,
    )
    shuffled = cross_val_score(model, frame[signals_], frame["failed"],
                               cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc")
    train, test = time_split(frame)
    model.fit(train[signals_], train["failed"])
    chrono = roc_auc_score(test["failed"], model.predict_proba(test[signals_])[:, 1])
    return {"shuffled_cv_auc": float(shuffled.mean()), "chronological_auc": float(chrono)}
