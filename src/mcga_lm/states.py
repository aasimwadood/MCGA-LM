"""Discretisation of the cognitive state vector (paper Sec. 3.5, component 2).

Sec. 3.5 discretises s_cog,t into low_fatigue / moderate_fatigue / high_fatigue
and maps each to explicit generation instructions. The paper does not give the
cut points; equal thirds of the normalised index are used here (ASSUMPTION).
"""

from __future__ import annotations

from typing import Tuple

FATIGUE_LEVELS: Tuple[str, str, str] = ("low", "moderate", "high")

# ASSUMPTION A-32: the paper gives three labels but no cut points; equal thirds
# of the normalised [0, 1] fatigue index are used.
FATIGUE_BOUNDARIES: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)


def fatigue_level(value: float) -> str:
    """Map a fatigue index in [0, 1] to low / moderate / high."""
    low, high = FATIGUE_BOUNDARIES
    if value < low:
        return "low"
    if value < high:
        return "moderate"
    return "high"
