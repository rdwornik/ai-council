"""Debate orchestration: parallel model calls, critique rounds."""

import asyncio
import logging
import random
from collections.abc import Callable

from config.config_loader import PromptsConfig
from src.models import ModelResponse, Question, Round
from src.providers.base import AIProvider, ProviderError

logger = logging.getLogger(__name__)


def _anonymize_responses(
    responses: list[ModelResponse],
) -> tuple[str, dict[str, str]]:
    """Shuffle responses and label them anonymously.

    Returns:
        (anonymized_block, label→provider_name mapping)
    """
    shuffled = list(responses)
    random.shuffle(shuffled)
    labels = [chr(ord("A") + i) for i in range(len(shuffled))]
    parts = [f"--- Proposal {label} ---\n{r.content}"
             for label, r in zip(labels, shuffled)]
    mapping = {label: r.provider for label, r in zip(labels, shuffled)}
    return "\n\n".join(parts), mapping


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
        prompts: Prompt templates from config (carries personas dict).
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
                p.name(): prompts.initial.format(
                    persona=prompts.personas.get(p.name(), ""),
                    question=question.text,
                )
                for p in providers
            }
        else:
            previous_responses = rounds[-1].responses
            anon_block, label_map = _anonymize_responses(previous_responses)
            logger.debug("Round %d anonymization map: %s", round_num, label_map)
            prompts_for_round = {
                p.name(): prompts.critique.format(
                    persona=prompts.personas.get(p.name(), ""),
                    round=round_num,
                    question=question.text,
                    previous_responses_anonymized=anon_block,
                )
                for p in providers
            }

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
