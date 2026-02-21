"""Pure dataclasses for the AI Council debate pipeline. No logic, no deps."""

from dataclasses import dataclass, field


@dataclass
class Question:
    text: str
    source: str  # "cli" or file path


@dataclass
class ModelResponse:
    provider: str          # "gemini", "openai", "claude", "grok", "deepseek"
    model: str             # actual model string used
    round_number: int
    content: str
    latency_sec: float
    token_count: int | None


@dataclass
class Round:
    number: int
    responses: list[ModelResponse] = field(default_factory=list)


@dataclass
class DebateResult:
    question: Question
    rounds: list[Round]
    synthesis: str         # Final markdown synthesis
    synthesizer: str       # Which model did synthesis
    total_duration_sec: float
    panel_mode: str = "default"              # "default", "full", "custom"
    synthesizer_is_participant: bool = False
