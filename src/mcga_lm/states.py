

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
