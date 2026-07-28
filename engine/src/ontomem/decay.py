"""Edge decay and reinforcement math (spec v0.2.1 §6). Pure functions only.

Strength and decay are EDGE-only (nodes never decay). The daily decay job
(a later layer) wires these against each edge's last-reinforcement timestamp.
"""

from __future__ import annotations

import math

from .model import MAX_STRENGTH

# decay rate λ per stability class, per day (spec §6.1)
LAMBDAS = {
    "immutable": 0.000,
    "stable": 0.005,
    "mutable": 0.020,
    "time_bound": 0.050,
    "ephemeral": 0.200,
}

REINFORCE_BOOST = 15.0  # Hebbian bump on traversal (spec §6.2)
DORMANCY_THRESHOLD = 2.0  # below this an edge is flagged dormant, never deleted (§6.3)


def decayed_strength(strength: float, stability: str, elapsed_days: float) -> float:
    """strength(t) = S0 * e^(-λt), with λ chosen by stability class."""
    if elapsed_days < 0:
        raise ValueError(f"elapsed_days must be >= 0, got {elapsed_days!r}")
    if stability not in LAMBDAS:
        raise ValueError(f"unknown stability: {stability!r}")
    return strength * math.exp(-LAMBDAS[stability] * elapsed_days)


def reinforce(strength: float, boost: float = REINFORCE_BOOST) -> float:
    """S0_new = min(strength + boost, MAX_STRENGTH)."""
    return min(strength + boost, MAX_STRENGTH)


def is_dormant(strength: float, threshold: float = DORMANCY_THRESHOLD) -> bool:
    return strength < threshold
