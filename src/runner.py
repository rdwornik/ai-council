"""CouncilRunner: executes the full debate lifecycle."""

import logging

from config.config_loader import AppConfig, ModelConfig
from src.debate import run_debate
from src.models import DebateResult, RunRequest
from src.output import print_cost_summary, print_round_summary, print_synthesis, save_to_file
from src.policy import RunPolicy
from src.providers.base import AIProvider
from src.synthesis import synthesize

logger = logging.getLogger(__name__)


def build_all_providers(config: AppConfig, provider_classes: dict) -> dict[str, AIProvider]:
    """Instantiate all available providers from config. Returns dict keyed by name."""
    providers: dict[str, AIProvider] = {}
    for name in config.available_providers:
        if name not in provider_classes:
            logger.warning("Provider '%s' unknown, skipping", name)
            continue
        model_cfg = config.models[name]
        try:
            providers[name] = provider_classes[name](model_cfg)
        except Exception as exc:
            logger.warning("Failed to instantiate provider '%s': %s", name, exc)
    return providers


def determine_panel(
    config: AppConfig,
    models_arg: str | None,
    full_flag: bool,
) -> tuple[list[str], str]:
    """Returns (panel_names, panel_mode). --models wins over --full wins over default."""
    if models_arg:
        return [m.strip() for m in models_arg.split(",")], "custom"
    elif full_flag:
        return config.defaults.full_panel, "full"
    else:
        return config.defaults.default_panel, "default"


def exclude_synthesizer_from_panel(
    panel_names: list[str],
    synthesizer_name: str,
    all_providers: dict[str, AIProvider],
) -> list[str]:
    """Remove synthesizer from panel when doing so still leaves >= 2 available debaters."""
    if synthesizer_name not in panel_names:
        return panel_names
    remaining = [n for n in panel_names if n != synthesizer_name]
    available_remaining = [n for n in remaining if n in all_providers]
    if len(available_remaining) >= 2:
        return remaining
    return panel_names


def pick_synthesizer(
    all_providers: dict[str, AIProvider],
    panel_names: list[str],
    preferred: str,
) -> tuple[AIProvider, bool]:
    """Pick synthesizer not in panel. Returns (provider, is_participant).

    is_participant=True only when no non-participant is available.
    """
    not_in_panel = [n for n in all_providers if n not in panel_names]
    if not_in_panel:
        if preferred in not_in_panel:
            return all_providers[preferred], False
        return all_providers[not_in_panel[0]], False
    if preferred in all_providers:
        return all_providers[preferred], True
    return next(iter(all_providers.values())), True


class CouncilRunner:
    """Executes the full debate lifecycle: panel → rounds → synthesis → output."""

    def __init__(self, all_providers: dict[str, AIProvider], config: AppConfig) -> None:
        self._providers = all_providers
        self._config = config

    async def run(self, request: RunRequest, output_dir=None) -> DebateResult:
        """Healthcheck is caller's responsibility. Runs panel selection → debate → synthesis.

        Args:
            request: Fully resolved RunRequest.
            output_dir: Where to save output. Defaults to config.defaults.output_dir.

        Returns:
            DebateResult with metrics attached.
        """
        import time

        from rich.console import Console

        console = Console(legacy_windows=False)

        if output_dir is None:
            output_dir = self._config.defaults.output_dir

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

        provider_names = [p.name() for p in panel_providers]
        synth_label = synthesizer.name() + (
            " (user-selected)" if request.synthesizer_specified
            else " (participant)" if is_participant
            else " (non-participant)"
        )

        console.print(
            f"\n[bold cyan]AI Council[/bold cyan] — {len(panel_providers)} models, "
            f"{request.rounds} rounds [{request.panel_mode}]"
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
            debate_rounds = await run_debate(
                question=request.question,
                providers=panel_providers,
                prompts=self._config.prompts,
                num_rounds=request.rounds,
                on_round_complete=on_round_complete,
            )
            progress.update(debate_task, description="Running synthesis...")

            result = await synthesize(
                question=request.question,
                rounds=debate_rounds,
                synthesizer=synthesizer,
                prompts=self._config.prompts,
                debate_start_time=debate_start,
                panel_mode=request.panel_mode,
                synthesizer_is_participant=is_participant,
                model_configs=self._config.models,
            )

        for rnd in debate_rounds:
            print_round_summary(rnd.number, rnd.responses)

        print_synthesis(result)

        if result.metrics:
            print_cost_summary(result.metrics)

        saved_path = save_to_file(result, output_dir, slug_override=request.slug_override)
        console.print(f"\n[dim]Saved to: {saved_path}[/dim]")

        return result
