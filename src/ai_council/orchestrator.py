"""CouncilRunner: coordinates the full debate lifecycle.

Separated from runner.py (utility functions) so the orchestration logic
lives in one place and can be imported / subclassed independently.
"""

import logging
from pathlib import Path

from ai_council.debate import run_debate
from ai_council.models import DebateOutcome, DebateResult, RunRequest
from ai_council.output import (
    print_cost_summary,
    print_round_summary,
    print_synthesis,
    save_minority_report,
    save_to_file,
)
from ai_council.providers.base import AIProvider
from ai_council.runner import exclude_synthesizer_from_panel, pick_synthesizer
from ai_council.seat_router import build_seat_router
from ai_council.synthesis import synthesize
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
            )
            progress.update(debate_task, description="Running synthesis...")

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
            )

        for rnd in outcome.rounds:
            print_round_summary(rnd.number, rnd.responses)

        print_synthesis(result)

        if result.metrics:
            print_cost_summary(result.metrics)

        saved_paths = save_to_file(
            result,
            output_dir,
            slug_override=request.slug_override,
            secondary_dir=secondary_dir,
            target_paths=request.target_paths,
            return_dir=request.return_dir,
        )
        console.print(f"\n[dim]Saved: {saved_paths[0]}[/dim]")
        if len(saved_paths) > 1:
            for p in saved_paths[1:]:
                console.print(f"[dim]Copied: {p}[/dim]")
        elif secondary_dir is not None and not secondary_dir.exists():
            console.print(
                f"[dim yellow]Secondary output dir not found: {secondary_dir}[/dim yellow]"
            )

        # Rama 4 (#15): emit dissent as a first-class artifact on a non-unanimous verdict,
        # routed to the same destinations as the verdict (incl. any --return-dir).
        minority_paths = save_minority_report(
            result,
            output_dir,
            slug_override=request.slug_override,
            secondary_dir=secondary_dir,
            target_paths=request.target_paths,
            return_dir=request.return_dir,
        )
        if minority_paths:
            console.print(
                f"[yellow]Minority report (non-unanimous verdict):[/yellow] {minority_paths[0]}"
            )
            for p in minority_paths[1:]:
                console.print(f"[dim]Minority copied: {p}[/dim]")

        if output_format == "json":
            import dataclasses
            import json
            import sys

            print(json.dumps(dataclasses.asdict(result), indent=2, default=str), file=sys.stdout)

        return result
