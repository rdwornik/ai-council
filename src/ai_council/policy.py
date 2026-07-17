"""RunPolicy: debate behavior thresholds and rules. No execution logic."""

from dataclasses import dataclass


@dataclass
class RunPolicy:
    """Governs debate behavior. Pure data + predicate methods — no I/O, no side effects.

    Retry eligibility is NOT decided here — the canonical `classify_error`/`is_retryable`
    taxonomy in `providers/base.py` is the single source (A3). This dataclass carries only
    the retry budget, the panel-size floor, and the abort predicate.
    """

    min_panel_size: int = 2
    max_retries_per_provider: int = 1

    def should_abort(self, active_count: int, round_number: int) -> bool:
        """True when a round produced zero usable responses.

        The abort CONDITION is uniform across rounds (no responses at all); the caller
        varies the HANDLING by round — round 1 raises, round 2+ returns a degraded outcome.
        """
        return active_count == 0

    @classmethod
    def default(cls) -> "RunPolicy":
        """Standard production policy."""
        return cls()
