"""Lightweight validation shared by acquisition and numerical replay."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


CALIBRATION_BLOCKS = 10
FORMAL_BLOCKS = 100
ENDPOINT_PROTOCOL = "opensky-snapshot-prequential-v2"


def integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def finite_real(
    value: Any, name: str, minimum: float = 0.0, *, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < minimum
        or (positive and result == minimum)
    ):
        raise ValueError(
            f"{name} must be finite and {'greater than' if positive else 'at least'} {minimum}"
        )
    return result


def collection_contract(block: dict[str, Any]) -> dict[str, Any]:
    """Extract the acquisition rule without phase or accepted-attempt metadata."""

    policy = block["acceptance_policy"]
    return {
        "bbox": list(block["bbox"]),
        "snapshots_per_block": block["snapshots_per_block"],
        "interval_seconds": block["interval_seconds"],
        "minimum_eligible_states": policy["minimum_eligible_states"],
        "max_attempts_per_block": policy["max_attempts_per_block"],
        "retry_delay_seconds": policy["retry_delay_seconds"],
    }
