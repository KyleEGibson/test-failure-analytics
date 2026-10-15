"""SPC math checked against hand-computed values and constructed patterns."""

import numpy as np
import pandas as pd
import pytest

from failure_analytics import spc


def test_imr_limits_hand_computed():
    x = [10, 12, 11, 13, 12]
    # moving ranges 2, 1, 2, 1 -> MR-bar 1.5; sigma = 1.5 / 1.128
    individuals, moving_range = spc.imr_limits(x)
    sigma = 1.5 / 1.128
    assert individuals.center == pytest.approx(11.6)
    assert individuals.ucl == pytest.approx(11.6 + 3 * sigma)
    assert individuals.lcl == pytest.approx(11.6 - 3 * sigma)
    assert moving_range.center == pytest.approx(1.5)
    assert moving_range.ucl == pytest.approx(3.267 * 1.5)


def test_imr_sigma_not_inflated_by_a_shift():
    """A step change must not widen the limits enough to hide itself."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
    individuals, _ = spc.imr_limits(x)
    assert individuals.sigma < 1.5            # MR-based, close to the true 1.0
    assert np.std(x, ddof=1) > 2.5            # the naive estimate would be inflated


def test_p_chart_variable_limits():
    chart = spc.p_chart(pd.Series([2, 10]), pd.Series([100, 100]))
    p_bar = 12 / 200
    sigma = np.sqrt(p_bar * (1 - p_bar) / 100)
    assert chart["center"].iloc[0] == pytest.approx(p_bar)
    assert chart["ucl"].iloc[0] == pytest.approx(p_bar + 3 * sigma)
    assert chart["lcl"].iloc[0] == 0  # clipped, cannot be negative


def test_p_chart_smaller_subgroups_get_wider_limits():
    chart = spc.p_chart(pd.Series([1, 5]), pd.Series([10, 200]))
    assert chart["ucl"].iloc[0] > chart["ucl"].iloc[1]


@pytest.mark.parametrize("values, rule, position", [
    ([0, 0, 0, 3.5, 0], 1, 3),
    ([0, 2.5, 0, 2.5, 0], 2, 3),
    ([0, 1.5, 1.5, 0.2, 1.5, 1.5, 0], 3, 5),
    ([0.1] * 8, 4, 7),
    ([-0.2] * 8, 4, 7),
])
def test_western_electric_rules(values, rule, position):
    violations = spc.western_electric(values, center=0, sigma=1)
    hits = violations[violations["rule"] == rule]
    assert position in hits["position"].tolist()


def test_in_control_series_has_no_violations():
    values = [0.5, -0.5] * 10  # alternating, never 8 on one side
    assert spc.western_electric(values, center=0, sigma=1).empty


def test_rule_2_requires_same_side():
    assert spc.western_electric([2.5, -2.5, 0], center=0, sigma=1).query("rule == 2").empty


def test_cpk():
    x = np.array([9.0, 10.0, 11.0])  # mean 10, sd 1
    assert spc.cpk(x, lsl=7, usl=16) == pytest.approx(1.0)       # lower side limits
    assert spc.cpk(x, usl=13) == pytest.approx(1.0)              # one-sided
    with pytest.raises(ValueError):
        spc.cpk(x)
