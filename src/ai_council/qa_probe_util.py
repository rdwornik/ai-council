"""Scratch utility for the 2026-07-09 lived QA exercise (NOT merged to main).

A deliberately trivial, self-contained pure function used only to walk the
deployed commit/hook loop on a positive path. The entire scratch branch is
reverted at the end of the exercise; nothing here reaches ``main``.
"""
from __future__ import annotations

from typing import TypeVar

T = TypeVar("T", int, float)


def clamp(value: T, low: T, high: T) -> T:
    """Constrain ``value`` to the inclusive ``[low, high]`` range.

    Args:
        value: The number to constrain.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        ``low`` if ``value < low``, ``high`` if ``value > high``, else ``value``.

    Raises:
        ValueError: If ``low`` is greater than ``high`` (empty range).
    """
    if low > high:
        raise ValueError(f"empty range: low={low!r} > high={high!r}")
    return max(low, min(value, high))
