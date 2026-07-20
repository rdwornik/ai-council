"""Debate orchestration: parallel model calls, critique rounds."""

import asyncio
import logging
import random
from collections.abc import Callable

from ai_council.models import (
    CruxArtifact,
    CruxChecker,
    CruxStatus,
    DebateOutcome,
    ModelResponse,
    Question,
    Round,
)
from ai_council.policy import RunPolicy
from ai_council.providers.base import AIProvider, ProviderError, classify_error, is_retryable
from ai_council.seat_router import SeatRouter
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
    crux_evidence: str = "",
) -> str:
    """Build Round 2+ prompts for a provider, respecting mode.

    ``crux_evidence`` (#18) is appended as its OWN block, never folded into ``anon_block``:
    that slot is framed as proposals from other council members, so evidence concatenated
    into it would be critiqued, attributed, and voted on as if a panelist had authored it.
    Empty string → the prompt is byte-identical to the pre-#18 prompt.
    """
    persona = prompts.personas.get(provider_name, "")

    if mode_config is None or mode_config.uses_existing_prompts:
        # pick mode — use existing critique template unchanged. The template has no free
        # placeholder for evidence, and adding one would break .format() for anyone on an
        # older settings.yaml, so the block is composed after rendering.
        base = prompts.critique.format(
            persona=persona,
            round=round_num,
            question=question_text,
            previous_responses_anonymized=anon_block,
        )
        return f"{base}\n\n{crux_evidence}" if crux_evidence else base

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
    if crux_evidence:
        # After the anonymized block, before the instruction: the panelist reads the
        # proposals, then the evidence, then what to do with both.
        parts.append(crux_evidence)
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
    seat_router: SeatRouter | None = None,
    crux_check: CruxChecker | None = None,
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
        seat_router: ADR-12 per-seat backend router. None → every seat uses its API leg.
        crux_check: #18 bounded crux service, injected by the orchestrator. Called ONCE,
            between Round 1 and Round 2. None → the step is skipped and Round-2 prompts
            are byte-identical to the pre-#18 prompts.

    Returns:
        DebateOutcome with rounds, degradation metadata, and the crux artifact (if any).

    Raises:
        RuntimeError: If all providers fail in round 1.
    """
    _policy = policy or RunPolicy.default()
    _directives = persona_directives or {}
    rounds: list[Round] = []
    provider_statuses: dict[str, str] = {p.name(): "failed" for p in providers}
    crux_artifact: CruxArtifact | None = None

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

            # #18: ONE bounded crux check, between Round 1 and Round 2 only. Rounds 3+
            # reuse the artifact. The service is handed the ALREADY-ANONYMIZED block, so
            # nothing derived from it can carry panel attribution (ADR-03).
            if round_num == 2 and crux_check is not None:
                try:
                    crux_artifact = await crux_check.check(question.text, anon_block)
                    logger.info("Crux check result: %s", crux_artifact.status.value)
                except Exception as exc:  # noqa: BLE001 - the debate must survive a service bug
                    # Record the failure as retrieval_unavailable rather than None: None is
                    # the "no service injected" state, so collapsing the two would erase the
                    # third outcome from DebateOutcome and the console (terra HIGH-3).
                    logger.warning("Crux check raised, proceeding without evidence: %s", exc)
                    crux_artifact = CruxArtifact(
                        status=CruxStatus.RETRIEVAL_UNAVAILABLE,
                        detail=f"crux service raised: {exc}",
                    )

            crux_evidence = crux_artifact.evidence_block if crux_artifact else ""
            prompts_for_round = {
                p.name(): _build_round2_prompt(
                    p.name(), round_num, question.text, anon_block, prompts, mode_config,
                    _directives, crux_evidence,
                )
                for p in providers
            }

        logger.info("Starting round %d with %d providers", round_num, len(providers))

        async def _run_seat(p: AIProvider) -> ModelResponse | ProviderError:
            """Per-seat: try the CLI backend (admission gate inside the router), else the
            same-seat API leg with the A3 retry contract. The router records seat telemetry."""
            prompt = prompts_for_round[p.name()]
            if seat_router is not None:
                cli_response = await seat_router.try_cli(p.name(), prompt, round_num)
                if cli_response is not None:
                    return cli_response
            result = await _call_provider(p, prompt, round_num, _policy)
            if seat_router is not None:
                seat_router.record_api(p.name(), result)
            return result

        tasks = [_run_seat(p) for p in providers]
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
                seats=seat_router.collect() if seat_router is not None else [],
                crux=crux_artifact,
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

    return DebateOutcome(
        rounds=rounds,
        provider_statuses=provider_statuses,
        seats=seat_router.collect() if seat_router is not None else [],
        crux=crux_artifact,
    )
