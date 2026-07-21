"""Pure dataclasses for the AI Council debate pipeline. No logic, no deps."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ai_council.policy import RunPolicy


# --- provider_statuses vocabulary (P1-8) -------------------------------------------------
# The CLOSED set of values for DebateOutcome/DebateResult.provider_statuses. Extend HERE,
# never at a call site — this dict feeds the verdict package's panel.dropped and
# degradation.failed_providers, the inter-repo delegation surface at Contract-Version 1.0.
#
# Status is relative to the LAST COMPLETED round: a round that aborted is not a completed
# round. Before P1-8 the dict was monotonic ("ever succeeded in any round") and never flipped
# back, so a seat that answered Round 1 and died in Round 2 read as "ok" and mid-debate seat
# loss was invisible in every field a consumer reads as binding input.
SEAT_STATUS_OK = "ok"  # responded in the last completed round
SEAT_STATUS_LOST = "lost"  # responded in an earlier round, absent from the last completed one
SEAT_STATUS_FAILED = "failed"  # never responded in any round
SEAT_STATUSES: frozenset[str] = frozenset(
    {SEAT_STATUS_OK, SEAT_STATUS_LOST, SEAT_STATUS_FAILED}
)


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
    backend: str = "api"  # "api" | "cli" — the transport that served this response (cost lane)


@dataclass
class FallbackEvent:
    """One CLI-seat degradation event within a run (L-CLI §3.Q5).

    ``cause`` is drawn from the shared 5-token vocabulary CLI_FALLBACK_CAUSES
    (providers/base.py) — the single source shared with the CLI failure classifier.
    """

    round: int
    from_backend: str  # "cli"
    to_backend: str  # "api"
    cause: str  # quota | timeout | parse | identity-unreadable | process-error
    detail: str  # classified error string (never a raw credential)


@dataclass
class SeatMetrics:
    """Per-seat telemetry for the `_metrics.json` seats[] sidecar (L-CLI §3.Q5).

    One entry per seat per run, uniform for API and CLI seats so consumers never branch on
    backend. Model/seat names only — never secret values. ``actual_model`` is null only in a
    degradation record, never on an admitted response (invariant I1).
    """

    seat: str
    requested_backend: str  # "api" | "cli"
    actual_backend: str  # "api" | "cli"
    requested_model: str
    actual_model: str | None
    identity_channel: str  # modelUsage | stderr-banner | session-events | api-echo
    identity_readable: bool
    cli: dict | None = None  # {name, version} when any CLI attempt occurred
    fallback_events: list[FallbackEvent] = field(default_factory=list)


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
    provider_statuses: dict[str, str] = field(default_factory=dict)  # provider → SEAT_STATUSES
    seats: list[SeatMetrics] = field(default_factory=list)  # per-seat backend/identity/fallback telemetry
    crux: CruxArtifact | None = None  # #18 bounded crux check; None when no service was injected


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
    backend: str = "api"  # "api" | "cli"; CLI calls are $0 marginal (subscription lane)


@dataclass
class DebateMetrics:
    """Aggregated cost and performance metrics for an entire debate run."""

    calls: list[ProviderCallMetrics] = field(default_factory=list)
    seats: list[SeatMetrics] = field(default_factory=list)  # L-CLI seats[] sidecar (owned by this lane)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    total_duration_sec: float = 0.0


class CruxStatus(str, Enum):
    """Outcome of the bounded between-rounds crux check (#18).

    ``NO_EMPIRICAL_CRUX`` is a VALID SUCCESS, not an error: Round 1 held no checkable
    empirical claim, so no retrieval was attempted. ``RETRIEVAL_UNAVAILABLE`` means the
    debate DEGRADES (proceeds without the artifact) — it never aborts.

    Inherits ``str`` deliberately: orchestrator.py json-dumps the DebateResult with
    ``default=str``, and a plain Enum would serialize as "CruxStatus.GROUNDED" rather
    than the contract's "grounded".
    """

    GROUNDED = "grounded"
    NO_EMPIRICAL_CRUX = "no_empirical_crux"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"


@dataclass
class CruxArtifact:
    """The ONE canonical evidence artifact injected into every Round-2 prompt (#18).

    Derived from Round-1 content that has ALREADY been anonymized, so it carries zero
    provider/model attribution (ADR-03 blind voting). ``evidence_block`` is empty unless
    ``status is GROUNDED``.
    """

    status: CruxStatus
    crux_claim: str = ""
    evidence_block: str = ""  # the ONE canonical injectable text; "" unless GROUNDED
    sources_count: int = 0
    providers_succeeded: int = 0
    providers_attempted: int = 0
    detail: str | None = None  # why, when not GROUNDED
    call_metrics: ProviderCallMetrics | None = None  # the ONE crux extraction call


class CruxChecker(Protocol):
    """Structural type for the injected crux service (debate.py never imports research/).

    ``check`` takes the ALREADY-ANONYMIZED Round-1 block, never ``list[ModelResponse]``:
    ADR-03 is satisfied by the signature, not by discipline inside the implementation.
    """

    async def check(self, question_text: str, anon_block: str) -> CruxArtifact: ...


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
    target_paths: list[Path] = field(default_factory=list)  # resolved transcript mirror dirs
    return_dir: Path | None = None  # ADR-10 deterministic return dir; None → canonical ./output/ only


@dataclass
class SynthesisMetrics:
    """Per-synthesis run observability data (success path only; failure path logs via WARNING)."""

    synthesizer_model: str
    transcript_size_tokens: int | None  # input tokens sent to synthesizer
    output_tokens: int | None           # tokens in synthesis response
    synth_latency_seconds: float        # wall-clock; excludes client overhead on failure path
    error_class: str                    # "none" on success; classified string in WARNING log on failure


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
    provider_statuses: dict[str, str] = field(default_factory=dict)  # provider → SEAT_STATUSES
    metrics: DebateMetrics | None = None  # populated after all calls complete
    synthesis_metrics: SynthesisMetrics | None = None  # per-synthesis observability
    # #18 Phase A: rides on DebateResult only — deliberately NOT in the verdict package,
    # so contract_version stays "1.0" (see output.py _build_verdict_payload).
    crux: CruxArtifact | None = None
