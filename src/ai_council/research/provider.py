"""Abstract base for research providers."""

from abc import ABC, abstractmethod

from ai_council.research.models import ResearchResult


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
