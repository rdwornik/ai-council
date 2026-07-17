"""Tests for RunPolicy abort predicate and retry budget.

Retry ELIGIBILITY moved to providers/base.classify_error/is_retryable (A3) — see
tests/test_base_provider.py. RunPolicy no longer classifies errors.
"""


from ai_council.policy import RunPolicy

# --- should_abort (uniform condition: zero responses, any round) ---


def test_should_abort_zero_active_round1():
    assert RunPolicy().should_abort(active_count=0, round_number=1) is True


def test_should_abort_zero_active_later_rounds():
    policy = RunPolicy()
    assert policy.should_abort(active_count=0, round_number=2) is True
    assert policy.should_abort(active_count=0, round_number=5) is True


def test_should_not_abort_round1_with_one_survivor():
    """A single round-1 survivor is enough to continue (skips the failed seat)."""
    assert RunPolicy().should_abort(active_count=1, round_number=1) is False


def test_should_not_abort_with_survivors():
    policy = RunPolicy()
    assert policy.should_abort(active_count=2, round_number=1) is False
    assert policy.should_abort(active_count=1, round_number=2) is False


# --- RunPolicy.default / surviving knobs ---


def test_default_factory_returns_standard_values():
    policy = RunPolicy.default()
    assert policy.min_panel_size == 2
    assert policy.max_retries_per_provider == 1
