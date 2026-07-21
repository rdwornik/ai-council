"""CouncilRunner: coordinates the full debate lifecycle.

Separated from runner.py (utility functions) so the orchestration logic
lives in one place and can be imported / subclassed independently.
"""

import logging
from pathlib import Path

from ai_council.crux_check import build_crux_check_service
from ai_council.debate import run_debate
from ai_council.models import CruxStatus, DebateOutcome, DebateResult, RunRequest
from ai_council.output import (
    RoutingFailure,
    print_cost_summary,
    print_round_summary,
    print_synthesis,
    raise_for_routing_failures,
    save_minority_report,
    save_to_file,
    save_verdict_package,
)
from ai_council.providers.base import AIProvider
from ai_council.runner import exclude_synthesizer_from_panel, pick_synthesizer
from ai_council.seat_router import build_seat_router
from ai_council.synthesis import build_failed_synthesis_result, synthesize
from config.config_loader import AppConfig

logger = logging.getLogger(__name__)


class CouncilRunner:
    """Executes the full debate lifecycle: panel → rounds → synthesis → output."""

    def __init__(self, all_providers: dict[str, AIProvider], config: AppConfig) -> None:
        self._providers = all_providers
        self._config = config

    async def run(self, request: RunRequest, output_dir=None, output_format: str = "text") -> DebateResult:
        """Healthcheck is caller's responsibility. Runs panel selection → debate → synthesis.

        Args:
            request: Fully resolved RunRequest.
            output_dir: Where to save output. Defaults to config.defaults.output_dir.
            output_format: "text" (default) or "json" — json prints result to stdout.

        Returns:
            DebateResult with metrics attached.
        """
        import time

        from rich.console import Console

        console = Console(legacy_windows=False)

        if output_dir is None:
            output_dir = self._config.defaults.output_dir

        secondary_dir: Path | None = None
        if self._config.defaults.secondary_output_enabled:
            secondary_dir = self._config.defaults.secondary_output_dir

        panel_names = request.panel_names
        synthesizer_name = request.synthesizer_name

        panel_names = exclude_synthesizer_from_panel(
            panel_names, synthesizer_name, self._providers
        )
        panel_providers = [self._providers[n] for n in panel_names if n in self._providers]

        if len(panel_providers) < request.policy.min_panel_size:
            raise RuntimeError(
                f"Need at least {request.policy.min_panel_size} providers in panel, "
                f"got {len(panel_providers)}. Check API keys or adjust --models."
            )

        synthesizer, is_participant = pick_synthesizer(
            self._providers, panel_names, synthesizer_name
        )

        # ADR-12 backend routing: CLI-backed seats get their subscription-CLI adapter (+ same-seat
        # API fallback); API seats route through unchanged. The synthesizer is never CLI (it is
        # called directly below, not via a seat). Default backend is api everywhere until the
        # §5 flip (#27), so this is a no-op unless a seat opts in with backend: cli.
        seat_router = build_seat_router(panel_names, self._providers, self._config.models)

        # #18 bounded crux check. Built here (never inside debate.py) so the debate layer
        # stays free of any research/ dependency. None when unconfigured → the debate runs
        # exactly as it did before. The extractor is the synthesizer, a non-participant by
        # default, so no panelist gains an asymmetric role heading into Round 2.
        crux_service = build_crux_check_service(self._config, synthesizer)

        mode_config = self._config.modes.get(request.mode) if self._config.modes else None
        persona_directives = (
            self._config.persona_mode_directives.get(request.mode, {})
            if self._config.persona_mode_directives else {}
        )

        provider_names = [p.name() for p in panel_providers]
        synth_label = synthesizer.name() + (
            " (user-selected)" if request.synthesizer_specified
            else " (participant)" if is_participant
            else " (non-participant)"
        )

        mode_emoji = mode_config.emoji if mode_config else ""
        console.print(
            f"\n[bold cyan]AI Council[/bold cyan] — {len(panel_providers)} models, "
            f"{request.rounds} rounds [{request.panel_mode}] "
            f"{mode_emoji} {request.mode}"
        )
        console.print(f"Panel: {', '.join(provider_names)}")
        console.print(f"Synthesizer: {synth_label}")
        console.print(
            f"Question: [italic]{request.question.text[:80]}"
            f"{'...' if len(request.question.text) > 80 else ''}[/italic]\n"
        )

        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        debate_start = time.monotonic()
        completed_rounds = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:

            def on_round_complete(rnd) -> None:
                completed_rounds.append(rnd)
                progress.print(
                    f"[green]OK[/green] Round {rnd.number} complete ({len(rnd.responses)} responses)"
                )

            debate_task = progress.add_task("Running debate rounds...", total=None)
            outcome: DebateOutcome = await run_debate(
                question=request.question,
                providers=panel_providers,
                prompts=self._config.prompts,
                num_rounds=request.rounds,
                on_round_complete=on_round_complete,
                policy=request.policy,
                mode_config=mode_config,
                persona_directives=persona_directives,
                seat_router=seat_router,
                crux_check=crux_service,
            )
            progress.update(debate_task, description="Running synthesis...")

            # P1-9: a synthesis failure used to raise straight past every writer below, so a
            # synthesizer hiccup destroyed a debate that had already been paid for in full.
            # Preserve the completed rounds, then re-raise after the writes so the run still
            # exits 1 (CONTRACT §4 "hard error; no artifacts guaranteed" — which permits
            # writing some). The verdict package is deliberately NOT emitted below: it
            # hardcodes exit_semantics 0 and would assert a usable verdict that does not exist.
            synthesis_error: BaseException | None = None
            synth_attempt_start = time.monotonic()
            try:
                result = await synthesize(
                    question=request.question,
                    rounds=outcome.rounds,
                    synthesizer=synthesizer,
                    prompts=self._config.prompts,
                    debate_start_time=debate_start,
                    panel_mode=request.panel_mode,
                    synthesizer_is_participant=is_participant,
                    model_configs=self._config.models,
                    degraded=outcome.degraded,
                    degradation_summary=outcome.degradation_summary,
                    provider_statuses=outcome.provider_statuses,
                    mode_config=mode_config,
                    debate_mode=request.mode,
                    seats=outcome.seats,
                    crux=outcome.crux,
                )
            # Catch Exception, not just ProviderError/RuntimeError: a ValueError/TypeError/
            # AttributeError from synthesis or metrics code would otherwise unwind past every
            # writer and recreate the exact loss this guard exists to prevent — the same
            # narrow-except defect class as P1-2. BaseException (CancelledError,
            # KeyboardInterrupt, SystemExit) still propagates: a shutdown writes nothing.
            except Exception as exc:  # noqa: BLE001 - preservation boundary, see above
                logger.error(
                    "Synthesis failed (%s) — preserving %d completed round(s): %s",
                    type(exc).__name__, len(outcome.rounds), exc, exc_info=exc,
                )
                synthesis_error = exc
                result = build_failed_synthesis_result(
                    question=request.question,
                    rounds=outcome.rounds,
                    synthesizer_name=synthesizer.name(),
                    error=exc,
                    debate_start_time=debate_start,
                    panel_mode=request.panel_mode,
                    synthesizer_is_participant=is_participant,
                    model_configs=self._config.models,
                    degradation_summary=outcome.degradation_summary,
                    provider_statuses=outcome.provider_statuses,
                    debate_mode=request.mode,
                    seats=outcome.seats,
                    crux=outcome.crux,
                    synth_latency_sec=time.monotonic() - synth_attempt_start,
                )

        # #18: surface the crux outcome here, not in debate.py (which owns no console).
        # Without this, a retrieval failure would be silent in the transcript — the
        # deliberate cost of NOT setting outcome.degraded (that would leak into the verdict
        # package's degradation block, and contract_version 1.0 is frozen at Phase A).
        if outcome.crux is not None:
            if outcome.crux.status is CruxStatus.GROUNDED:
                console.print(
                    f"[green]OK[/green] Crux check grounded "
                    f"({outcome.crux.sources_count} source(s)): {outcome.crux.crux_claim}"
                )
            elif outcome.crux.status is CruxStatus.NO_EMPIRICAL_CRUX:
                console.print("[dim]Crux check: no empirical crux in Round 1 (no lookup)[/dim]")
            else:
                console.print(
                    f"[yellow]![/yellow] Crux check unavailable — debate continued "
                    f"without evidence ({outcome.crux.detail})"
                )

        for rnd in outcome.rounds:
            print_round_summary(rnd.number, rnd.responses)

        print_synthesis(result)

        if result.metrics:
            print_cost_summary(result.metrics)

        # R4 (#35/#62): every writer below targets the SAME --return-dir, so a fault there is
        # normally common-mode. Accumulate the misses instead of raising at the first one, so
        # each deliverable still reaches the CANONICAL dir, then fail once with the aggregate
        # after the last write. Raising inline would abort this function on the transcript and
        # cost the minority report and verdict package their canonical copies too.
        routing_failures: list[RoutingFailure] = []

        saved_paths = save_to_file(
            result,
            output_dir,
            slug_override=request.slug_override,
            secondary_dir=secondary_dir,
            target_paths=request.target_paths,
            return_dir=request.return_dir,
            routing_failures=routing_failures,
        )
        console.print(f"\n[dim]Saved: {saved_paths[0]}[/dim]")
        if len(saved_paths) > 1:
            for p in saved_paths[1:]:
                console.print(f"[dim]Copied: {p}[/dim]")
        elif secondary_dir is not None and not secondary_dir.exists():
            console.print(
                f"[dim yellow]Secondary output dir not found: {secondary_dir}[/dim yellow]"
            )

        # P1-9: both artifacts below are VERDICT deliverables and there is no verdict without
        # synthesis — dissent detection reads the synthesis text, and the verdict package
        # hardcodes exit_semantics 0, which would contradict the exit 1 this run is heading
        # for. The transcript and its metrics sidecar above are the paid-for content and are
        # already on disk; that is what P1-9 preserves.
        if synthesis_error is None:
            # Rama 4 (#15): emit dissent as a first-class artifact on a non-unanimous verdict,
            # routed to the same destinations as the verdict (incl. any --return-dir). Share the
            # transcript's exact <ts>-<mode>-<slug> so all of a run's artifacts are one matched set.
            run_base = saved_paths[0].stem[len("council-out-"):]
            minority_paths = save_minority_report(
                result,
                output_dir,
                slug_override=request.slug_override,
                secondary_dir=secondary_dir,
                target_paths=request.target_paths,
                return_dir=request.return_dir,
                stem_base=run_base,
                routing_failures=routing_failures,
            )
            if minority_paths:
                console.print(
                    f"[yellow]Minority report (non-unanimous verdict):[/yellow] {minority_paths[0]}"
                )
                for p in minority_paths[1:]:
                    console.print(f"[dim]Minority copied: {p}[/dim]")

            # DRAFT-INT-1 (#26): the transcript-free caller deliverable. Sibling of save_to_file,
            # routed to the same destinations; inherits the transcript's deterministic <ts>.
            written: dict[str, list[Path]] = {"transcript": saved_paths}
            if result.metrics:
                # Never record a path that was not written: the sidecar write can fail and
                # degrade rather than abort (#63), and a manifest advertising a missing file is
                # worse than the original defect — a consumer repo follows it and gets nothing.
                metrics_path = saved_paths[0].with_name(saved_paths[0].stem + "_metrics.json")
                if metrics_path.exists():
                    written["metrics"] = [metrics_path]
            if minority_paths:
                written["minority"] = minority_paths
            verdict_paths = save_verdict_package(
                result,
                output_dir,
                saved_paths[0],
                written=written,
                secondary_dir=secondary_dir,
                target_paths=request.target_paths,
                return_dir=request.return_dir,
                routing_failures=routing_failures,
            )
            console.print(f"[dim]Verdict package: {verdict_paths[0]}[/dim]")
            for p in verdict_paths[1:]:
                console.print(f"[dim]Verdict copied: {p}[/dim]")
        else:
            console.print(
                "[yellow]![/yellow] No verdict package: synthesis failed. The transcript and "
                "metrics for the completed debate were preserved."
            )

        if output_format == "json":
            import dataclasses
            import json
            import sys

            print(json.dumps(dataclasses.asdict(result), indent=2, default=str), file=sys.stdout)

        # R4: every canonical artifact is now on disk and every diagnostic emitted. If any
        # REQUIRED --return-dir write missed, fail here with the full set — deliberately not
        # in a finally, where an exception would mask the original.
        # P1-9: when synthesis already failed, THAT is the primary cause and must reach the
        # operator. Raising the routing aggregate first would replace it with an
        # OutputRoutingError and hide why the run actually failed — so the misses are logged
        # loudly here (never silently dropped: R4 still fails the run, just not the message)
        # and the original cause is re-raised. Exit is 1 either way (CONTRACT §4); exit 3
        # would be wrong, it means "degraded but complete, verdict is usable".
        if synthesis_error is not None:
            for failure in routing_failures:
                logger.error(
                    "Required return-dir write missed for %s -> %s: %s",
                    failure.artifact, failure.destination, failure.cause,
                )
            raise synthesis_error

        # R4: every canonical artifact is now on disk and every diagnostic emitted. If any
        # REQUIRED --return-dir write missed, fail here with the full set — deliberately not
        # in a finally, where an exception would mask the original.
        raise_for_routing_failures(routing_failures)

        return result
