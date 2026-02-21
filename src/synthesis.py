"""Final synthesis: build transcript, call synthesizer, return DebateResult."""

import logging
import time

from config.config_loader import PromptsConfig
from src.models import DebateResult, ModelResponse, Question, Round
from src.providers.base import AIProvider, ProviderError

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


async def synthesize(
    question: Question,
    rounds: list[Round],
    synthesizer: AIProvider,
    prompts: PromptsConfig,
    debate_start_time: float,
    panel_mode: str = "default",
    synthesizer_is_participant: bool = False,
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

    Returns:
        DebateResult with synthesis content.

    Raises:
        ProviderError: If synthesizer call fails.
        RuntimeError: If synthesizer returns empty content.
    """
    transcript = _format_full_transcript(rounds)
    synthesis_prompt = prompts.synthesis.format(
        rounds=len(rounds),
        question=question.text,
        full_transcript=transcript,
    )

    logger.info("Running synthesis via %s", synthesizer.name())

    synthesis_response: ModelResponse = await synthesizer.generate(
        synthesis_prompt,
        round_number=len(rounds) + 1,
    )

    if not synthesis_response.content:
        raise RuntimeError(f"Synthesizer {synthesizer.name()} returned empty content")

    total_duration = time.monotonic() - debate_start_time

    return DebateResult(
        question=question,
        rounds=rounds,
        synthesis=synthesis_response.content,
        synthesizer=synthesizer.name(),
        total_duration_sec=total_duration,
        panel_mode=panel_mode,
        synthesizer_is_participant=synthesizer_is_participant,
    )
