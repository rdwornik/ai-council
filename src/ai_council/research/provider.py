"""Abstract base for research providers."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from ai_council.research.models import ResearchResult


def iso_now() -> str:
    """Single source for the tz-aware ISO-8601 UTC timestamp on ResearchResult.

    All research providers route their ``timestamp`` through here (B2) so the field
    is derived one way. Replaces the deprecated ``datetime.utcnow().isoformat()`` that
    was pasted into every provider.
    """
    return datetime.now(timezone.utc).isoformat()


class ResearchProviderError(Exception):
    """Raised when a research provider call fails."""

    def __init__(self, provider_name: str, message: str) -> None:
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] {message}")


class ResearchProvider(ABC):
    """Abstract base for all research providers."""

    @abstractmethod
    def name(self) -> str:
        """Return the short provider name (e.g. 'perplexity', 'openai_mini')."""
        ...

    @abstractmethod
    def model_string(self) -> str:
        """Return the actual model identifier string."""
        ...

    @abstractmethod
    async def research(self, query: str) -> ResearchResult:
        """Run research for the given query.

        Args:
            query: The research question or topic.

        Returns:
            ResearchResult with content, sources, and metrics.

        Raises:
            ResearchProviderError: On API failure.
        """
        ...
