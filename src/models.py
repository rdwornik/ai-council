"""Pure dataclasses for the AI Council debate pipeline. No logic, no deps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.policy import RunPolicy


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
    was_retry: bool = False  # True when this response came from a retry attempt


@dataclass
class Round:
    number: int
    responses: list[ModelResponse] = field(default_factory=list)


@dataclass
class DebateOutcome:
    """Result of the debate phase, before synthesis. Returned by run_debate()."""

    rounds: list[Round] = field(default_factory=list)
    degraded: bool = False
    degradation_summary: str | None = None
    provider_statuses: dict[str, str] = field(default_factory=dict)  # provider → "ok" | "failed"


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
class RunRequest:
    """Fully resolved parameters for a single debate run. Built by cli.py."""

    question: Question
    panel_names: list[str]
    synthesizer_name: str
    rounds: int
    policy: RunPolicy
    panel_mode: str = "default"  # "default", "full", "custom"
    synthesizer_specified: bool = False  # True if user explicitly chose synthesizer
    slug_override: str | None = None  # inbox file stem → deterministic output filename
    mode: str = "pick"  # "pick", "ideas", "judge"


@dataclass
class DebateResult:
    question: Question
    rounds: list[Round]
    synthesis: str  # Final markdown synthesis
    synthesizer: str  # Which model did synthesis
    total_duration_sec: float
    panel_mode: str = "default"  # "default", "full", "custom"
    mode: str = "pick"  # "pick", "ideas", "judge"
    synthesizer_is_participant: bool = False
    degraded: bool = False
    degradation_summary: str | None = None
    provider_statuses: dict[str, str] = field(default_factory=dict)  # provider → "ok" | "failed"
    metrics: DebateMetrics | None = None  # populated after all calls complete
