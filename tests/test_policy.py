"""Tests for RunPolicy abort/retry predicates."""


from ai_council.policy import RunPolicy

# --- should_abort ---


def test_should_abort_round1_below_threshold():
    policy = RunPolicy()
    assert policy.should_abort(active_count=1, round_number=1) is True


def test_should_abort_round1_at_threshold():
    policy = RunPolicy()
    assert policy.should_abort(active_count=2, round_number=1) is False


def test_should_abort_round1_above_threshold():
    policy = RunPolicy()
    assert policy.should_abort(active_count=3, round_number=1) is False


def test_should_abort_zero_active_any_round():
    policy = RunPolicy()
    assert policy.should_abort(active_count=0, round_number=2) is True
    assert policy.should_abort(active_count=0, round_number=5) is True


def test_should_not_abort_round2_with_one():
    """Round 2+ only aborts at zero — one survivor is enough to continue."""
    policy = RunPolicy()
    assert policy.should_abort(active_count=1, round_number=2) is False


def test_custom_min_panel_size():
    policy = RunPolicy(min_panel_size=3, abort_if_round1_below=3)
    assert policy.should_abort(active_count=2, round_number=1) is True
    assert policy.should_abort(active_count=3, round_number=1) is False


# --- should_retry ---


def test_should_retry_timeout():
    assert RunPolicy().should_retry("Request timed out after 30s") is True


def test_should_retry_timed_out():
    assert RunPolicy().should_retry("Connection timed out") is True


def test_should_retry_rate_limit():
    assert RunPolicy().should_retry("rate_limit exceeded") is True


def test_should_retry_connection_error():
    assert RunPolicy().should_retry("connection reset by peer") is True


def test_should_not_retry_auth_error():
    assert RunPolicy().should_retry("401 Unauthorized") is False


def test_should_not_retry_403():
    assert RunPolicy().should_retry("403 Forbidden") is False


def test_should_not_retry_model_not_found():
    assert RunPolicy().should_retry("model_not_found: gpt-99") is False


def test_should_not_retry_404():
    assert RunPolicy().should_retry("404 Not Found") is False


def test_should_not_retry_content_policy():
    assert RunPolicy().should_retry("content_policy violation detected") is False


def test_should_not_retry_unknown_error():
    assert RunPolicy().should_retry("unexpected server error xyz") is False


def test_non_retryable_takes_priority_over_retryable():
    """If both lists match, non-retryable wins."""
    policy = RunPolicy(
        retryable_errors=["connection"],
        non_retryable_errors=["connection"],
    )
    assert policy.should_retry("connection error") is False


def test_case_insensitive_matching():
    policy = RunPolicy()
    assert policy.should_retry("TIMEOUT: upstream did not respond") is True
    assert policy.should_retry("AUTH: invalid key") is False


# --- RunPolicy.default ---


def test_default_factory_returns_standard_values():
    policy = RunPolicy.default()
    assert policy.min_panel_size == 2
    assert policy.abort_if_round1_below == 2
    assert policy.max_retries_per_provider == 1
    assert "timeout" in policy.retryable_errors
    assert "auth" in policy.non_retryable_errors
