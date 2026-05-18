"""Tests for classify_error() and is_retryable() in src/providers/base.py."""


from ai_council.providers.base import ProviderError, classify_error, is_retryable

# --- classify_error ---


def test_classify_timeout():
    assert classify_error(Exception("Request timed out after 30s")) == "timeout"


def test_classify_timed_out_variant():
    assert classify_error(Exception("Connection timed out")) == "timeout"


def test_classify_rate_limit_429():
    assert classify_error(Exception("429 Too Many Requests")) == "rate_limit"


def test_classify_rate_limit_string():
    assert classify_error(Exception("rate_limit exceeded")) == "rate_limit"


def test_classify_auth_401():
    assert classify_error(Exception("401 Unauthorized")) == "auth"


def test_classify_auth_403():
    assert classify_error(Exception("403 Forbidden")) == "auth"


def test_classify_auth_keyword():
    assert classify_error(Exception("Authentication failed")) == "auth"


def test_classify_auth_api_key():
    assert classify_error(Exception("Invalid api key provided")) == "auth"


def test_classify_model_not_found_404():
    assert classify_error(Exception("404 Not Found")) == "model_not_found"


def test_classify_model_not_found_keyword():
    assert classify_error(Exception("model_not_found: gpt-99")) == "model_not_found"


def test_classify_connection_error():
    assert classify_error(Exception("Connection reset by peer")) == "connection_error"


def test_classify_unreachable():
    assert classify_error(Exception("Host unreachable")) == "connection_error"


def test_classify_server_error_500():
    assert classify_error(Exception("500 Internal Server Error")) == "server_error"


def test_classify_server_error_503():
    assert classify_error(Exception("503 Service Unavailable")) == "server_error"


def test_classify_content_policy():
    assert classify_error(Exception("content_policy violation detected")) == "content_policy"


def test_classify_billing_credit_balance():
    """Anthropic returns HTTP 400 invalid_request_error with 'credit balance is too low'
    when the org has no credits. Must classify as 'billing', not the opaque 'invalid_request'."""
    exc = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'Your credit balance is too low to access the Anthropic API. "
        "Please go to Plans & Billing to upgrade or purchase credits.'}}"
    )
    assert classify_error(exc) == "billing"


def test_classify_billing_insufficient_quota():
    """OpenAI surfaces billing exhaustion as 'insufficient_quota' / 'You exceeded your current quota'."""
    exc = Exception("Error code: 429 - insufficient_quota: You exceeded your current quota, please check your plan and billing details.")
    assert classify_error(exc) == "billing"


def test_classify_unknown():
    assert classify_error(Exception("some weird unforeseen error xyzzy")) == "unknown"


def test_classify_provider_error():
    """ProviderError wraps messages — classify_error should still work."""
    exc = ProviderError("deepseek", "401 Unauthorized - bad key")
    assert classify_error(exc) == "auth"


# --- is_retryable ---


def test_is_retryable_timeout():
    assert is_retryable("timeout") is True


def test_is_retryable_rate_limit():
    assert is_retryable("rate_limit") is True


def test_is_retryable_connection_error():
    assert is_retryable("connection_error") is True


def test_is_retryable_server_error():
    assert is_retryable("server_error") is True


def test_not_retryable_auth():
    assert is_retryable("auth") is False


def test_not_retryable_model_not_found():
    assert is_retryable("model_not_found") is False


def test_not_retryable_content_policy():
    assert is_retryable("content_policy") is False


def test_not_retryable_unknown():
    assert is_retryable("unknown") is False


def test_not_retryable_billing():
    """Billing failures must not retry — they aren't transient."""
    assert is_retryable("billing") is False
