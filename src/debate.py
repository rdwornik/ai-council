"""Debate orchestration: parallel model calls, critique rounds."""

import asyncio
import logging
from collections.abc import Callable

from config.config_loader import PromptsConfig
from src.models import ModelResponse, Question, Round
from src.providers.base import AIProvider, ProviderError

logger = logging.getLogger(__name__)


def _format_previous_responses(responses: list[ModelResponse]) -> str:
    """Format a list of responses into the critique prompt block."""
    parts: list[str] = []
    for r in responses:
        parts.append(f"--- {r.provider} ({r.model}) ---\n{r.content}")
    return "\n\n".join(parts)


async def _call_provider(
    provider: AIProvider,
    prompt: str,
    round_number: int,
) -> ModelResponse | ProviderError:
    """Call a single provider, returning ProviderError on failure (never raises)."""
    try:
        return await provider.generate(prompt, round_number)
    except ProviderError as exc:
        logger.warning("Provider %s failed in round %d: %s", provider.name(), round_number, exc)
        return exc
    except Exception as exc:
        err = ProviderError(provider.name(), f"Unexpected error: {exc}")
        logger.warning("Provider %s unexpected failure in round %d: %s", provider.name(), round_number, exc)
        return err


async def run_debate(
    question: Question,
    providers: list[AIProvider],
    prompts: PromptsConfig,
    num_rounds: int,
    on_round_complete: Callable[[Round], None] | None = None,
) -> list[Round]:
    """Run the full debate across all rounds.

    Args:
        question: The question being debated.
        providers: List of AIProvider instances to use.
        prompts: Prompt templates from config.
        num_rounds: Total number of debate rounds.
        on_round_complete: Optional callback invoked after each round completes.

    Returns:
        List of Round objects, one per round.

    Raises:
        RuntimeError: If all providers fail in a round.
    """
    rounds: list[Round] = []

    for round_num in range(1, num_rounds + 1):
        if round_num == 1:
            prompts_for_round = {
                p.name(): prompts.initial.format(question=question.text)
                for p in providers
            }
        else:
            previous_responses = rounds[-1].responses
            previous_block = _format_previous_responses(previous_responses)
            prompt_text = prompts.critique.format(
                round=round_num,
                question=question.text,
                previous_responses=previous_block,
            )
            prompts_for_round = {p.name(): prompt_text for p in providers}

        logger.info("Starting round %d with %d providers", round_num, len(providers))

        tasks = [
            _call_provider(p, prompts_for_round[p.name()], round_num)
            for p in providers
        ]
        results = await asyncio.gather(*tasks)

        responses: list[ModelResponse] = []
        for result in results:
            if isinstance(result, ModelResponse):
                responses.append(result)
            # ProviderError already logged in _call_provider

        if not responses:
            raise RuntimeError(f"All providers failed in round {round_num}")

        current_round = Round(number=round_num, responses=responses)
        rounds.append(current_round)

        logger.info(
            "Round %d complete: %d/%d providers succeeded",
            round_num,
            len(responses),
            len(providers),
        )

        if on_round_complete:
            on_round_complete(current_round)

    return rounds
