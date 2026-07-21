"""Seat router — the CLI-seat admission gate + same-seat API fallback (L-CLI IF#2).

Runs BETWEEN transport return and round entry, upstream of critique fan-out: for a
CLI-backed seat it attempts the CLI provider and verifies the served identity (the CLI
adapter raises `identity-unreadable` when the channel can't be read, so any CliOutcome that
returns IS admitted — no unverified content ever enters a round). On ANY CLI failure the seat
falls back to its same-seat API provider (the debate loop's normal A3 retry path), recording a
`fallback_events[]` entry classified into the shared 5-token vocabulary.

Produces one `SeatMetrics` per seat per run — uniform for API and CLI seats (API seats get
`identity_channel="api-echo"`) — which becomes the `seats[]` sidecar. This lane is the first
lander on the `_metrics.json` sidecar, so it defines the extension mechanism (see
`output.py`): namespaced, additive top-level keys, one namespace per lane.

The API leg is intentionally NOT run here — the debate loop owns the A3 retry contract. The
router splits into `try_cli` (attempt+admit or record-fallback) and `record_api` (label the
seat after the loop's API call), so `_call_provider` stays the single retry owner and there is
no import cycle with `debate.py`.
"""

import logging
from dataclasses import dataclass

from ai_council.models import FallbackEvent, ModelResponse, SeatMetrics
from ai_council.providers.base import AIProvider, ProviderError, classify_cli_failure
from ai_council.providers.cli_base import ClaudeCliProvider, CliProvider, CodexCliProvider
from config.config_loader import ModelConfig

logger = logging.getLogger(__name__)

# CLI-subscription adapters (ADR-12 v1 = claude + codex only; grok deferred, deepseek API-only).
CLI_PROVIDER_CLASSES: dict[str, type[CliProvider]] = {
    "claude": ClaudeCliProvider,
    "codex": CodexCliProvider,
}


@dataclass
class SeatSpec:
    """Everything the router needs for one seat: the always-present API provider (backend and
    fallback), and — when backend==cli — the CLI provider driving the subscription lane."""

    api_provider: AIProvider
    requested_backend: str  # "api" | "cli"
    requested_model: str
    cli_provider: CliProvider | None = None
    cli_command: str | None = None


class SeatRouter:
    """Owns per-seat routing state + the accumulating SeatMetrics (one per seat, across rounds)."""

    def __init__(self, specs: dict[str, SeatSpec]) -> None:
        self._specs = specs
        self._metrics: dict[str, SeatMetrics] = {}

    def _seat_metrics(self, name: str) -> SeatMetrics:
        if name not in self._metrics:
            spec = self._specs.get(name)
            self._metrics[name] = SeatMetrics(
                seat=name,
                requested_backend=spec.requested_backend if spec else "api",
                actual_backend="api",
                requested_model=spec.requested_model if spec else "",
                actual_model=None,
                identity_channel="api-echo",
                identity_readable=True,
            )
        return self._metrics[name]

    async def try_cli(self, name: str, prompt: str, round_number: int) -> ModelResponse | None:
        """Attempt the CLI backend for this seat. Returns an admitted ModelResponse on success,
        or None to signal the caller to run the same-seat API leg (backend==api, no CLI seat,
        or a recorded CLI failure)."""
        spec = self._specs.get(name)
        seat = self._seat_metrics(name)
        if spec is None or spec.cli_provider is None or spec.requested_backend != "cli":
            return None

        cli = spec.cli_provider
        seat.cli = {"name": spec.cli_command, "version": cli.version}
        try:
            response = await cli.generate(prompt, round_number)
        # P1-2: this caught only ProviderError, so the module docstring's "On ANY CLI failure the
        # seat falls back" held only for the classified case. A raw AttributeError/ValueError from
        # a parser escaped try_cli entirely — no API leg, no fallback_events[] entry — and via
        # debate.py's gather took every sibling seat down with it. Catching Exception (NOT
        # BaseException) restores the advertised guarantee; CancelledError still propagates, so a
        # shutdown is never mis-booked as a seat failure.
        except Exception as exc:  # noqa: BLE001 - contract: ANY CLI failure degrades ONE seat
            cause = classify_cli_failure(exc)
            # An unclassified failure keeps its type name so the seats[] record stays diagnosable.
            detail = str(exc) if isinstance(exc, ProviderError) else f"{type(exc).__name__}: {exc}"
            seat.fallback_events.append(
                FallbackEvent(
                    round=round_number, from_backend="cli", to_backend="api",
                    cause=cause, detail=detail,
                )
            )
            if cause == "identity-unreadable":
                seat.identity_readable = False
            logger.warning("seat %s: CLI backend degraded (%s) -> retried via API", name, cause)
            return None

        # Admitted: the adapter guarantees a non-null served identity or it raised (invariant I1).
        seat.actual_backend = "cli"
        seat.actual_model = response.model
        seat.identity_channel = cli.identity_channel
        seat.identity_readable = True
        return response

    def record_api(self, name: str, result: ModelResponse | ProviderError) -> None:
        """Label the seat after the debate loop ran its API leg (fallback or backend==api)."""
        seat = self._seat_metrics(name)
        seat.actual_backend = "api"
        seat.identity_channel = "api-echo"
        if isinstance(result, ModelResponse):
            seat.actual_model = result.model
            seat.identity_readable = True
        # On a ProviderError (both lanes failed) actual_model stays None — a degradation record.

    def collect(self) -> list[SeatMetrics]:
        """One SeatMetrics per seat that ran, for the seats[] sidecar."""
        return list(self._metrics.values())


def build_seat_router(
    panel_names: list[str],
    api_providers: dict[str, AIProvider],
    model_configs: dict[str, ModelConfig],
) -> SeatRouter:
    """Assemble a SeatRouter for a panel. A seat with backend==cli gets its CLI adapter built;
    a build failure degrades that seat to API-only (never fatal). The synthesizer is never
    routed here (ADR-12: synthesizer is always API) — it is called directly, not via a seat."""
    specs: dict[str, SeatSpec] = {}
    for name in panel_names:
        cfg = model_configs.get(name)
        api = api_providers.get(name)
        if cfg is None or api is None:
            continue
        cli_provider: CliProvider | None = None
        requested_backend = "api"
        requested_model = cfg.model
        if cfg.backend == "cli" and cfg.cli_command:
            cls = CLI_PROVIDER_CLASSES.get(cfg.cli_command)
            if cls is None:
                logger.warning(
                    "seat %s: unknown cli_command '%s' -> API only", name, cfg.cli_command
                )
            else:
                try:
                    cli_provider = cls(cfg)
                    requested_backend = "cli"
                    requested_model = cfg.cli_model or cfg.model
                except ProviderError as exc:
                    logger.warning("seat %s: CLI provider build failed (%s) -> API only", name, exc)
        specs[name] = SeatSpec(
            api_provider=api,
            requested_backend=requested_backend,
            requested_model=requested_model,
            cli_provider=cli_provider,
            cli_command=cfg.cli_command,
        )
    return SeatRouter(specs)
