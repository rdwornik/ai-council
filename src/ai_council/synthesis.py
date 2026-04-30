"""Final synthesis: build transcript, call synthesizer, return DebateResult."""

import logging
import time

from ai_council.metrics import build_call_metrics, build_debate_metrics
from ai_council.models import DebateResult, ModelResponse, Question, Round
from ai_council.providers.base import AIProvider
from config.config_loader import ModeConfig, ModelConfig, PromptsConfig

logger = logging.getLogger(__name__)


def _format_full_transcript(rounds: list[Round]) -> str:
    """Format all rounds into a single transcript string for synthesis."""
    parts: list[str] = []
    for rnd in rounds:
        parts.append(f"### Round {rnd.number}")
        for resp in rnd.responses:
            parts.append(f"**{resp.provider} ({resp.model})**\n{resp.content}")
        parts.append("")  # blank line between rounds
    return "\n\n".join(parts)


def _build_synthesis_prompt(
    question_text: str,
    transcript: str,
    rounds: list[Round],
    prompts: PromptsConfig,
    mode_config: ModeConfig | None,
) -> str:
    """Build the synthesis prompt, respecting mode."""
    if mode_config is None or mode_config.uses_existing_prompts:
        return prompts.synthesis.format(
            rounds=len(rounds),
            question=question_text,
            full_transcript=transcript,
        )

    # ideas / judge modes — custom synthesis structure
    return (
        "You are an impartial synthesizer. Your job: distill the council's "
        "discussion into a clear, structured output.\n\n"
        f"Question: {question_text}\n\n"
        "Full debate transcript:\n"
        f"{transcript}\n\n"
        f"{mode_config.synthesis_output.strip()}"
    )


async def synthesize(
    question: Question,
    rounds: list[Round],
    synthesizer: AIProvider,
    prompts: PromptsConfig,
    debate_start_time: float,
    panel_mode: str = "default",
    synthesizer_is_participant: bool = False,
    model_configs: dict[str, ModelConfig] | None = None,
    degraded: bool = False,
    degradation_summary: str | None = None,
    provider_statuses: dict[str, str] | None = None,
    mode_config: ModeConfig | None = None,
    debate_mode: str = "pick",
) -> DebateResult:
    """Run synthesis and return the final DebateResult.

    Args:
        question: The original question.
        rounds: All completed debate rounds.
        synthesizer: The AIProvider that will synthesize the debate.
        prompts: Prompt templates from config.
        debate_start_time: monotonic time when the debate started (for duration).
        panel_mode: "default", "full", or "custom".
        synthesizer_is_participant: True if synthesizer was also in the debate panel.
        mode_config: Mode-specific config; None or pick → uses existing prompts.synthesis.
        debate_mode: Canonical mode name to carry into DebateResult.

    Returns:
        DebateResult with synthesis content.

    Raises:
        ProviderError: If synthesizer call fails.
        RuntimeError: If synthesizer returns empty content.
    """
    transcript = _format_full_transcript(rounds)
    synthesis_prompt = _build_synthesis_prompt(
        question.text, transcript, rounds, prompts, mode_config
    )

    logger.info("Running synthesis via %s", synthesizer.name())

    synthesis_response: ModelResponse = await synthesizer.generate(
        synthesis_prompt,
        round_number=len(rounds) + 1,
    )

    if not synthesis_response.content:
        raise RuntimeError(f"Synthesizer {synthesizer.name()} returned empty content")

    total_duration = time.monotonic() - debate_start_time

    metrics = None
    if model_configs is not None:
        synthesis_metrics = build_call_metrics(
            synthesis_response,
            model_configs,
            round_number=0,  # 0 = synthesis call
        )
        metrics = build_debate_metrics(
            rounds, synthesis_metrics, model_configs, total_duration
        )

    return DebateResult(
        question=question,
        rounds=rounds,
        synthesis=synthesis_response.content,
        synthesizer=synthesizer.name(),
        total_duration_sec=total_duration,
        panel_mode=panel_mode,
        mode=debate_mode,
        synthesizer_is_participant=synthesizer_is_participant,
        degraded=degraded,
        degradation_summary=degradation_summary,
        provider_statuses=provider_statuses or {},
        metrics=metrics,
    )
