"""RunPolicy: debate behavior thresholds and rules. No execution logic."""

from dataclasses import dataclass, field


@dataclass
class RunPolicy:
    """Governs debate behavior. Pure data + predicate methods — no I/O, no side effects."""

    min_panel_size: int = 2
    abort_if_round1_below: int = 2
    max_retries_per_provider: int = 1

    # Error types that warrant a retry
    retryable_errors: list[str] = field(
        default_factory=lambda: ["timeout", "timed out", "rate_limit", "connection"]
    )
    # Error types that fail immediately — no retry
    non_retryable_errors: list[str] = field(
        default_factory=lambda: ["auth", "401", "403", "model_not_found", "404", "content_policy"]
    )

    def should_abort(self, active_count: int, round_number: int) -> bool:
        """True when the debate cannot produce a meaningful result.

        Round 1: abort if fewer than min panel size responded.
        Later rounds: caller decides; policy doesn't abort on round 2+ failures.
        """
        if round_number == 1 and active_count < self.abort_if_round1_below:
            return True
        if active_count == 0:
            return True
        return False

    def should_retry(self, error_message: str) -> bool:
        """True when the error type is transient and worth retrying once."""
        msg_lower = error_message.lower()
        # Non-retryable takes priority
        if any(pattern in msg_lower for pattern in self.non_retryable_errors):
            return False
        return any(pattern in msg_lower for pattern in self.retryable_errors)

    @classmethod
    def default(cls) -> "RunPolicy":
        """Standard production policy."""
        return cls()
