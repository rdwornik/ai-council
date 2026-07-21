"""Final synthesis: build transcript, call synthesizer, return DebateResult."""

import logging
import time

from ai_council.metrics import build_call_metrics, build_debate_metrics
from ai_council.models import (
    CruxArtifact,
    DebateResult,
    ModelResponse,
    ProviderCallMetrics,
    Question,
    Round,
    SeatMetrics,
    SynthesisMetrics,
)
from ai_council.providers.base import AIProvider, ProviderError, classify_error
from config.config_loader import ModeConfig, ModelConfig, PromptsConfig

logger = logging.getLogger(__name__)

# Written in place of the synthesis body when the synthesizer never returned one. A reader
# must never mistake a FAILED synthesis for a thin one.
SYNTHESIS_FAILED_MARKER = "SYNTHESIS FAILED"


class EmptySynthesisError(RuntimeError):
    """The synthesizer returned an empty body.

    Subclasses RuntimeError so the documented `Raises:` contract and every existing caller
    keep working. Carries the ModelResponse because this failure happens AFTER a completed —
    and therefore BILLED — provider call: the preservation path books its real usage rather
    than fabricating a zero, which would silently understate the run's spend.
    """

    def __init__(self, message: str, response: ModelResponse) -> None:
        super().__init__(message)
        self.response = response


def build_failed_synthesis_result(
    question: Question,
    rounds: list[Round],
    synthesizer_name: str,
    error: BaseException,
    debate_start_time: float,
    panel_mode: str = "default",
    synthesizer_is_participant: bool = False,
    model_configs: dict[str, ModelConfig] | None = None,
    degradation_summary: str | None = None,
    provider_statuses: dict[str, str] | None = None,
    debate_mode: str = "pick",
    seats: list[SeatMetrics] | None = None,
    crux: CruxArtifact | None = None,
    synth_latency_sec: float = 0.0,
) -> DebateResult:
    """Preserve a completed debate when synthesis fails (P1-9).

    By the time synthesis runs, every panelist across every round has been paid for. Before
    this, `synthesize` raised straight past all of `CouncilRunner.run`'s writers, so a
    synthesizer hiccup — the single most replaceable call in the pipeline — destroyed the
    transcript, the metrics sidecar and the whole round record. That contradicted the
    degradation philosophy the rest of the codebase is built on (debate.py returns partial
    rounds; output.py downgrades a sidecar failure to a note; crux_check never raises).

    The caller writes THIS result and then re-raises, so the run still exits non-zero and no
    verdict package is emitted — there is no verdict without synthesis.
    """
    total_duration = time.monotonic() - debate_start_time
    error_class = classify_error(error) if isinstance(error, ProviderError) else "unknown"
    reason = f"{type(error).__name__}: {error}"

    # No "## Synthesis" heading here — output.py's _build_body already emits one above this.
    synthesis_body = (
        f"**Status:** {SYNTHESIS_FAILED_MARKER} — {reason}\n\n"
        f"The debate completed and every round above is intact, but the synthesizer "
        f"({synthesizer_name}) returned no verdict. The rounds are the authoritative record "
        f"of this run; no verdict package was emitted."
    )

    # An empty-content failure happened AFTER a completed, billed call, so its usage is known
    # and must be booked. A ProviderError means nothing was ever returned — zero tokens there
    # is observed truth, not a fabrication, and the UNKNOWN fields stay None rather than 0.
    #
    # Read the response ONLY off our own exception type. Duck-typing `getattr(error,
    # "response")` here would trust an SDK exception's HTTP response (openai.APIStatusError
    # and httpx.HTTPStatusError both carry that exact attribute name) as a ModelResponse, and
    # the resulting AttributeError would be raised INSIDE the caller's except handler —
    # masking the original error and losing every artifact this function exists to preserve.
    billed_response: ModelResponse | None = (
        error.response if isinstance(error, EmptySynthesisError) else None
    )

    metrics = None
    if model_configs is not None:
        if billed_response is not None:
            failed_synthesis_call = build_call_metrics(
                billed_response, model_configs, round_number=0
            )
        else:
            failed_synthesis_call = ProviderCallMetrics(
                provider=synthesizer_name,
                round_number=0,  # 0 = synthesis call
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
                latency_sec=synth_latency_sec,
            )
        crux_calls = [crux.call_metrics] if crux and crux.call_metrics else None
        metrics = build_debate_metrics(
            rounds, failed_synthesis_call, model_configs, total_duration, seats=seats,
            extra_calls=crux_calls,
        )

    # Keep any pre-existing debate degradation (a dropped seat) alongside the synthesis cause.
    combined_summary = "; ".join(
        part for part in (degradation_summary, f"synthesis failed — {reason}") if part
    )

    return DebateResult(
        question=question,
        rounds=rounds,
        synthesis=synthesis_body,
        synthesizer=synthesizer_name,
        total_duration_sec=total_duration,
        panel_mode=panel_mode,
        mode=debate_mode,
        synthesizer_is_participant=synthesizer_is_participant,
        degraded=True,
        degradation_summary=combined_summary,
        provider_statuses=provider_statuses or {},
        metrics=metrics,
        synthesis_metrics=SynthesisMetrics(
            synthesizer_model=(
                billed_response.model if billed_response is not None else synthesizer_name
            ),
            # None = genuinely unknown (nothing was returned), never an observed zero.
            transcript_size_tokens=(
                billed_response.input_tokens if billed_response is not None else None
            ),
            output_tokens=(
                billed_response.output_tokens if billed_response is not None else None
            ),
            synth_latency_seconds=synth_latency_sec,
            error_class=error_class,
        ),
        crux=crux,
    )


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
    seats: list["SeatMetrics"] | None = None,
    crux: CruxArtifact | None = None,
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

    synth_start = time.monotonic()
    try:
        synthesis_response: ModelResponse = await synthesizer.generate(
            synthesis_prompt,
            round_number=len(rounds) + 1,
        )
    except ProviderError as exc:
        synth_latency = time.monotonic() - synth_start
        error_class = classify_error(exc)
        logger.warning(
            "Synthesis observability: synthesizer=%s latency=%.2fs error_class=%s",
            synthesizer.name(), synth_latency, error_class,
        )
        raise

    if not synthesis_response.content:
        # This call completed and was billed — carry it so the failure path books real usage.
        raise EmptySynthesisError(
            f"Synthesizer {synthesizer.name()} returned empty content", synthesis_response
        )

    synth_latency = time.monotonic() - synth_start
    synthesis_obs = SynthesisMetrics(
        synthesizer_model=synthesis_response.model,
        transcript_size_tokens=synthesis_response.input_tokens,
        output_tokens=synthesis_response.output_tokens,
        synth_latency_seconds=synth_latency,  # wall-clock; provider latency omits client overhead
        error_class="none",
    )

    total_duration = time.monotonic() - debate_start_time

    metrics = None
    if model_configs is not None:
        synthesis_call_metrics = build_call_metrics(
            synthesis_response,
            model_configs,
            round_number=0,  # 0 = synthesis call
        )
        # #18: the crux extraction call belongs to no round (round_number=-1), so it
        # rides in as an extra call rather than being invented into a Round.
        crux_calls = [crux.call_metrics] if crux and crux.call_metrics else None
        metrics = build_debate_metrics(
            rounds, synthesis_call_metrics, model_configs, total_duration, seats=seats,
            extra_calls=crux_calls,
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
        synthesis_metrics=synthesis_obs,
        crux=crux,
    )
