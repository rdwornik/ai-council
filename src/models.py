"""Pure dataclasses for the AI Council debate pipeline. No logic, no deps."""

from dataclasses import dataclass, field


@dataclass
class Question:
    text: str
    source: str  # "cli" or file path


@dataclass
class ModelResponse:
    provider: str  # "gemini", "openai", "claude", "grok", "deepseek"
    model: str  # actual model string used
    round_number: int
    content: str
    latency_sec: float
    token_count: int | None  # combined total; kept for display/backward compat
    input_tokens: int | None = None  # prompt tokens (for cost calculation)
    output_tokens: int | None = None  # completion tokens (for cost calculation)


@dataclass
class Round:
    number: int
    responses: list[ModelResponse] = field(default_factory=list)


@dataclass
class ProviderCallMetrics:
    """Cost and performance metrics for a single provider API call."""

    provider: str
    round_number: int  # 0 = synthesis call
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_sec: float
    was_retry: bool = False


@dataclass
class DebateMetrics:
    """Aggregated cost and performance metrics for an entire debate run."""

    calls: list[ProviderCallMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    total_duration_sec: float = 0.0


@dataclass
class DebateResult:
    question: Question
    rounds: list[Round]
    synthesis: str  # Final markdown synthesis
    synthesizer: str  # Which model did synthesis
    total_duration_sec: float
    panel_mode: str = "default"  # "default", "full", "custom"
    synthesizer_is_participant: bool = False
    metrics: DebateMetrics | None = None  # populated after all calls complete
