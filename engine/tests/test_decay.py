"""Unit tests for decay/reinforcement math. Expected values hand-derived."""

import math

import pytest

from ontomem.decay import (
    DORMANCY_THRESHOLD,
    REINFORCE_BOOST,
    decayed_strength,
    is_dormant,
    reinforce,
)


def test_immutable_never_decays():
    assert decayed_strength(100.0, "immutable", 0) == 100.0
    assert decayed_strength(100.0, "immutable", 10_000) == 100.0


def test_zero_elapsed_is_identity():
    for stability in ("stable", "mutable", "time_bound", "ephemeral"):
        assert decayed_strength(73.0, stability, 0) == 73.0


@pytest.mark.parametrize(
    "stability,lam",
    [("stable", 0.005), ("mutable", 0.020), ("time_bound", 0.050), ("ephemeral", 0.200)],
)
def test_half_life_per_class(stability, lam):
    # at t = ln(2)/λ the strength should halve
    half_life = math.log(2) / lam
    got = decayed_strength(100.0, stability, half_life)
    assert got == pytest.approx(50.0, abs=1e-9)


def test_known_value_ephemeral_one_day():
    # 100 * e^(-0.2 * 1) = 81.873...
    assert decayed_strength(100.0, "ephemeral", 1.0) == pytest.approx(81.8730753, abs=1e-6)


def test_negative_elapsed_raises():
    with pytest.raises(ValueError):
        decayed_strength(100.0, "stable", -1)


def test_unknown_stability_raises():
    with pytest.raises(ValueError):
        decayed_strength(100.0, "permanent", 1)


def test_reinforce_bumps_and_caps():
    assert reinforce(50.0, 15.0) == 65.0
    assert reinforce(95.0, 15.0) == 100.0  # capped
    assert reinforce(100.0, 15.0) == 100.0
    assert reinforce(40.0) == 40.0 + REINFORCE_BOOST  # default boost


def test_is_dormant_boundary():
    assert is_dormant(1.99) is True
    assert is_dormant(DORMANCY_THRESHOLD) is False  # strictly below threshold
    assert is_dormant(2.01) is False
