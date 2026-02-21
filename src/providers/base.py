"""Abstract base for all AI model providers."""

from abc import ABC, abstractmethod

from src.models import ModelResponse


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
