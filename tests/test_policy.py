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


# --- from_config (B7: load from settings.yaml, code defaults as fallback) ---


def test_from_config_none_returns_defaults():
    assert RunPolicy.from_config(None) == RunPolicy.default()


def test_from_config_empty_dict_returns_defaults():
    assert RunPolicy.from_config({}) == RunPolicy.default()


def test_from_config_overrides_from_mapping():
    policy = RunPolicy.from_config({"min_panel_size": 4, "max_retries_per_provider": 3})
    assert policy.min_panel_size == 4
    assert policy.max_retries_per_provider == 3


def test_from_config_partial_override_keeps_defaults():
    policy = RunPolicy.from_config({"max_retries_per_provider": 5})
    assert policy.max_retries_per_provider == 5
    assert policy.min_panel_size == 2  # untouched field keeps its code default
