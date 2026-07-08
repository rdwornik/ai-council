"""Tests for the lived-QA-exercise scratch util (NOT merged to main)."""
import pytest

from ai_council.qa_probe_util import clamp


def test_clamp_within_range() -> None:
    assert clamp(5, 0, 10) == 5


def test_clamp_below_low() -> None:
    assert clamp(-3, 0, 10) == 0


def test_clamp_above_high() -> None:
    assert clamp(99, 0, 10) == 10


def test_clamp_at_bounds() -> None:
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10


def test_clamp_float() -> None:
    assert clamp(1.5, 0.0, 1.0) == 1.0


def test_clamp_inverted_range_raises() -> None:
    with pytest.raises(ValueError):
        clamp(5, 10, 0)
