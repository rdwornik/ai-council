"""Cost and performance metric computation for debate runs."""

from ai_council.models import DebateMetrics, ModelResponse, ProviderCallMetrics, Round, SeatMetrics
from config.config_loader import ModelConfig


def compute_call_cost(
    input_tokens: int,
    output_tokens: int,
    config: ModelConfig,
) -> float:
    """Return estimated USD cost for a single provider call."""
    return (input_tokens * config.cost_per_1m_input + output_tokens * config.cost_per_1m_output) / 1_000_000


def build_call_metrics(
    response: ModelResponse,
    model_configs: dict[str, ModelConfig],
    *,
    round_number: int,
    was_retry: bool = False,
) -> ProviderCallMetrics:
    """Build a ProviderCallMetrics from a ModelResponse.

    Uses 0 tokens when provider did not return usage data (cost will be $0).
    """
    cfg = model_configs.get(response.provider)
    input_tokens = response.input_tokens or 0
    output_tokens = response.output_tokens or 0
    # CLI (subscription-lane) calls are $0 marginal — never priced at API token rates
    # (L-CLI §2.1). Tokens are still recorded for transparency; only the cost is zeroed.
    if response.backend == "cli":
        cost = 0.0
    else:
        cost = compute_call_cost(input_tokens, output_tokens, cfg) if cfg else 0.0
    return ProviderCallMetrics(
        provider=response.provider,
        round_number=round_number,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        latency_sec=response.latency_sec,
        was_retry=was_retry,
        backend=response.backend,
    )


def build_debate_metrics(
    rounds: list[Round],
    synthesis_call: ProviderCallMetrics,
    model_configs: dict[str, ModelConfig],
    total_duration_sec: float,
    seats: list[SeatMetrics] | None = None,
) -> DebateMetrics:
    """Aggregate metrics across all debate rounds plus synthesis.

    ``seats`` is the L-CLI per-seat backend/identity/fallback telemetry (seats[] sidecar).
    """
    calls: list[ProviderCallMetrics] = []
    for rnd in rounds:
        for response in rnd.responses:
            calls.append(
                build_call_metrics(response, model_configs, round_number=rnd.number)
            )
    calls.append(synthesis_call)

    total_input = sum(c.input_tokens for c in calls)
    total_output = sum(c.output_tokens for c in calls)
    total_cost = sum(c.estimated_cost_usd for c in calls)

    return DebateMetrics(
        calls=calls,
        seats=seats or [],
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_estimated_cost_usd=total_cost,
        total_duration_sec=total_duration_sec,
    )
