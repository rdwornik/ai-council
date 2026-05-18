"""Abstract base for all AI model providers."""

from abc import ABC, abstractmethod

from ai_council.models import ModelResponse

# Error categories returned by classify_error()
RETRYABLE_ERRORS: frozenset[str] = frozenset(
    {"timeout", "rate_limit", "connection_error", "server_error"}
)
NON_RETRYABLE_ERRORS: frozenset[str] = frozenset(
    {"auth", "model_not_found", "content_policy", "invalid_request", "billing"}
)


def classify_error(exc: Exception) -> str:
    """Map an exception to a canonical error category string.

    Returns one of: timeout, rate_limit, auth, model_not_found, connection_error,
    server_error, content_policy, invalid_request, billing, unknown.
    Used by healthcheck and retry logic to produce specific messages.
    """
    msg = str(exc).lower()
    # Billing exhaustion comes through 400 (Anthropic: "credit balance is too low")
    # or 429 (OpenAI: "insufficient_quota"). Check before generic rate_limit / invalid.
    if (
        "credit balance" in msg
        or "insufficient_quota" in msg
        or "insufficient quota" in msg
        or "exceeded your current quota" in msg
        or "plans & billing" in msg
        or "billing details" in msg
    ):
        return "billing"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return "rate_limit"
    if (
        "401" in msg
        or "403" in msg
        or "unauthorized" in msg
        or "forbidden" in msg
        or "auth" in msg
        or "api key" in msg
        or "api_key" in msg
    ):
        return "auth"
    if "404" in msg or "model_not_found" in msg:
        return "model_not_found"
    if (
        "connection" in msg
        or "unreachable" in msg
        or "network" in msg
        or "connect" in msg
    ):
        return "connection_error"
    if "500" in msg or "502" in msg or "503" in msg or "server error" in msg:
        return "server_error"
    if "content_policy" in msg or "content policy" in msg or "safety" in msg:
        return "content_policy"
    if "invalid" in msg:
        return "invalid_request"
    return "unknown"


def is_retryable(error_type: str) -> bool:
    """True when the error category warrants a single retry attempt."""
    return error_type in RETRYABLE_ERRORS


class ProviderError(Exception):
    """Raised when a provider call fails."""

    def __init__(self, provider_name: str, message: str) -> None:
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] {message}")


class AIProvider(ABC):
    """Abstract base for all AI model providers."""

    @abstractmethod
    def name(self) -> str:
        """Return the short provider name (e.g. 'gemini', 'claude')."""
        ...

    @abstractmethod
    def model_string(self) -> str:
        """Return the actual model identifier string."""
        ...

    @abstractmethod
    async def generate(self, prompt: str, round_number: int) -> ModelResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: The full prompt text to send.
            round_number: The debate round number (1-indexed).

        Returns:
            ModelResponse dataclass with content and metadata.

        Raises:
            ProviderError: On API failure, timeout, or invalid response.
        """
        ...
