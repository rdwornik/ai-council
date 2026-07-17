"""Debate orchestration: parallel model calls, critique rounds."""

import asyncio
import logging
import random
from collections.abc import Callable

from ai_council.models import DebateOutcome, ModelResponse, Question, Round
from ai_council.policy import RunPolicy
from ai_council.providers.base import AIProvider, ProviderError, classify_error, is_retryable
from config.config_loader import ModeConfig, PromptsConfig

logger = logging.getLogger(__name__)

# Quality gate: warn when fewer than this many models respond in Round 1
_MIN_QUALITY_RESPONSES = 3


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
    parts = [
        f"--- Proposal {label} ---\n{r.content}" for label, r in zip(labels, shuffled)
    ]
    mapping = {label: r.provider for label, r in zip(labels, shuffled)}
    return "\n\n".join(parts), mapping


async def _call_provider(
    provider: AIProvider,
    prompt: str,
    round_number: int,
    policy: RunPolicy,
) -> ModelResponse | ProviderError:
    """Call a single provider, retrying retryable failures with a growing timeout.

    Total attempts = ``policy.max_retries_per_provider + 1``. Each attempt grows the
    timeout by 1.5x via ``generate(..., timeout=)`` — no provider state is mutated. Retry
    eligibility uses the canonical ``classify_error``/``is_retryable`` taxonomy (A3).
    Never raises — returns the last ProviderError on permanent failure.
    """
    last: ProviderError | None = None
    for attempt in range(policy.max_retries_per_provider + 1):
        timeout = provider.timeout_sec * (1.5**attempt)
        try:
            result = await provider.generate(prompt, round_number, timeout=timeout)
            result.was_retry = attempt > 0
            return result
        except ProviderError as exc:
            last = exc
            if not is_retryable(classify_error(exc)):
                logger.warning(
                    "Provider %s failed in round %d: %s", provider.name(), round_number, exc
                )
                break
            logger.warning(
                "Provider %s attempt %d failed in round %d, retrying: %s",
                provider.name(),
                attempt + 1,
                round_number,
                exc,
            )
        except Exception as exc:
            last = ProviderError(provider.name(), f"Unexpected error: {exc}")
            logger.warning(
                "Provider %s unexpected failure in round %d: %s",
                provider.name(),
                round_number,
                exc,
            )
            break

    assert last is not None  # the loop always runs >= 1 attempt; any non-return path sets last
    return last


def _build_round1_prompt(
    provider_name: str,
    question_text: str,
    prompts: PromptsConfig,
    mode_config: ModeConfig | None,
    persona_directives: dict[str, str],
) -> str:
    """Build the Round 1 prompt for a provider, respecting mode."""
    persona = prompts.personas.get(provider_name, "")
    directive = persona_directives.get(provider_name, "")

    if mode_config is None or mode_config.uses_existing_prompts:
        # pick mode — use existing template unchanged
        return prompts.initial.format(persona=persona, question=question_text)

    # ideas / judge modes — assemble from mode template fields
    parts: list[str] = []
    if directive:
        parts.append(f"CRITICAL INSTRUCTION: {directive}")
        parts.append("")
    if persona:
        parts.append(persona)
        parts.append("")
    if mode_config.round1_header:
        parts.append(mode_config.round1_header)
        parts.append("")
    parts.append(mode_config.round1_instruction.strip())
    if mode_config.round1_structure:
        parts.append("")
        parts.append(mode_config.round1_structure.strip())
    parts.append("")
    parts.append(f"Question: {question_text}")
    return "\n".join(parts)


def _build_round2_prompt(
    provider_name: str,
    round_num: int,
    question_text: str,
    anon_block: str,
    prompts: PromptsConfig,
    mode_config: ModeConfig | None,
    persona_directives: dict[str, str],
) -> str:
    """Build Round 2+ prompts for a provider, respecting mode."""
    persona = prompts.personas.get(provider_name, "")

    if mode_config is None or mode_config.uses_existing_prompts:
        # pick mode — use existing critique template unchanged
        return prompts.critique.format(
            persona=persona,
            round=round_num,
            question=question_text,
            previous_responses_anonymized=anon_block,
        )

    # ideas / judge modes — assemble from mode template fields
    parts: list[str] = []
    if persona:
        parts.append(persona)
        parts.append("")
    parts.append(f"You are participating in a council, round {round_num}.")
    parts.append("")
    parts.append(
        "Below are anonymized contributions from other council members on this question:"
    )
    parts.append("")
    parts.append(f"Question: {question_text}")
    parts.append("")
    parts.append(anon_block)
    parts.append("")
    parts.append(mode_config.round2_instruction.strip())
    return "\n".join(parts)


async def run_debate(
    question: Question,
    providers: list[AIProvider],
    prompts: PromptsConfig,
    num_rounds: int,
    on_round_complete: Callable[[Round], None] | None = None,
    policy: RunPolicy | None = None,
    mode_config: ModeConfig | None = None,
    persona_directives: dict[str, str] | None = None,
) -> DebateOutcome:
    """Run the full debate across all rounds.

    Args:
        question: The question being debated.
        providers: List of AIProvider instances to use.
        prompts: Prompt templates from config (carries personas dict).
        num_rounds: Total number of debate rounds.
        on_round_complete: Optional callback invoked after each round completes.
        policy: Retry/abort policy. Defaults to RunPolicy.default().
        mode_config: Mode-specific prompt templates. None or pick → uses existing prompts.
        persona_directives: Per-provider directive strings for this mode (pre-extracted).

    Returns:
        DebateOutcome with rounds and degradation metadata.

    Raises:
        RuntimeError: If all providers fail in round 1.
    """
    _policy = policy or RunPolicy.default()
    _directives = persona_directives or {}
    rounds: list[Round] = []
    provider_statuses: dict[str, str] = {p.name(): "failed" for p in providers}

    for round_num in range(1, num_rounds + 1):
        if round_num == 1:
            prompts_for_round = {
                p.name(): _build_round1_prompt(
                    p.name(), question.text, prompts, mode_config, _directives
                )
                for p in providers
            }
        else:
            previous_responses = rounds[-1].responses
            anon_block, label_map = _anonymize_responses(previous_responses)
            logger.debug("Round %d anonymization map: %s", round_num, label_map)
            prompts_for_round = {
                p.name(): _build_round2_prompt(
                    p.name(), round_num, question.text, anon_block, prompts, mode_config, _directives
                )
                for p in providers
            }

        logger.info("Starting round %d with %d providers", round_num, len(providers))

        tasks = [
            _call_provider(p, prompts_for_round[p.name()], round_num, _policy)
            for p in providers
        ]
        results = await asyncio.gather(*tasks)

        responses: list[ModelResponse] = []
        for provider, result in zip(providers, results):
            if isinstance(result, ModelResponse):
                responses.append(result)
                provider_statuses[provider.name()] = "ok"
            # ProviderError already logged in _call_provider

        if _policy.should_abort(len(responses), round_num):
            if round_num == 1:
                raise RuntimeError(f"All providers failed in round {round_num}")
            # Round 2+: return partial results rather than aborting entirely
            degradation_summary = (
                f"All providers failed in round {round_num}. "
                f"Returning {len(rounds)} completed round(s)."
            )
            logger.warning("Degraded debate: %s", degradation_summary)
            return DebateOutcome(
                rounds=rounds,
                degraded=True,
                degradation_summary=degradation_summary,
                provider_statuses=provider_statuses,
            )

        # Quality gate: warn when Round 1 has low participation on a large panel
        if (
            round_num == 1
            and len(providers) >= _MIN_QUALITY_RESPONSES
            and len(responses) < _MIN_QUALITY_RESPONSES
        ):
            logger.warning(
                "WARNING: Only %d/%d models responded in Round 1. "
                "Debate quality is degraded. Consider re-running with longer timeouts or fewer models.",
                len(responses),
                len(providers),
            )

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

    return DebateOutcome(rounds=rounds, provider_statuses=provider_statuses)
