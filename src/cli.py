"""Click CLI — orchestrates config loading, provider selection, debate, and output."""

import asyncio
import logging
import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from config.config_loader import AppConfig, load_config
from src.debate import run_debate
from src.models import Question, Round
from src.output import print_round_summary, print_synthesis, save_to_file
from src.providers.anthropic import AnthropicProvider
from src.providers.base import AIProvider
from src.providers.deepseek import DeepSeekProvider
from src.providers.gemini import GeminiProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.xai import XAIProvider
from src.synthesis import synthesize

console = Console()

PROVIDER_CLASSES: dict[str, type[AIProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "claude": AnthropicProvider,
    "grok": XAIProvider,
    "deepseek": DeepSeekProvider,
}


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def _build_all_providers(config: AppConfig) -> dict[str, AIProvider]:
    """Build all available providers. Returns dict keyed by name."""
    providers: dict[str, AIProvider] = {}
    for name in config.available_providers:
        if name not in PROVIDER_CLASSES:
            logging.warning("Provider '%s' unknown, skipping", name)
            continue
        model_cfg = config.models[name]
        try:
            providers[name] = PROVIDER_CLASSES[name](model_cfg)
        except Exception as exc:
            logging.warning("Failed to instantiate provider '%s': %s", name, exc)
    return providers


def _determine_panel(
    config: AppConfig,
    models_arg: str | None,
    full_flag: bool,
) -> tuple[list[str], str]:
    """Returns (panel_names, panel_mode). --models overrides all."""
    if models_arg:
        return [m.strip() for m in models_arg.split(",")], "custom"
    elif full_flag:
        return config.defaults.full_panel, "full"
    else:
        return config.defaults.default_panel, "default"


def _pick_non_participant_synthesizer(
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
    # All available are in panel — use preferred with participant framing
    if preferred in all_providers:
        return all_providers[preferred], True
    return next(iter(all_providers.values())), True


async def _run(
    question_text: str,
    question_source: str,
    config: AppConfig,
    rounds: int,
    models_arg: str | None,
    output_dir: Path,
    synthesizer_name: str,
    use_full_panel: bool,
) -> None:
    all_providers = _build_all_providers(config)

    if not all_providers:
        console.print("[bold red]Error:[/bold red] No providers available. Check API keys in .env.")
        sys.exit(1)

    panel_names, panel_mode = _determine_panel(config, models_arg, use_full_panel)
    panel_providers = [all_providers[n] for n in panel_names if n in all_providers]

    if len(panel_providers) < 2:
        console.print(
            f"[bold red]Error:[/bold red] Need at least 2 providers in panel, got {len(panel_providers)}. "
            "Check API keys in .env or adjust --models."
        )
        sys.exit(1)

    synthesizer, is_participant = _pick_non_participant_synthesizer(
        all_providers, panel_names, synthesizer_name
    )

    question = Question(text=question_text, source=question_source)
    provider_names = [p.name() for p in panel_providers]
    synth_label = synthesizer.name() + (" (participant)" if is_participant else " (non-participant)")

    console.print(f"\n[bold cyan]AI Council[/bold cyan] — {len(panel_providers)} models, {rounds} rounds [{panel_mode}]")
    console.print(f"Panel: {', '.join(provider_names)}")
    console.print(f"Synthesizer: {synth_label}")
    console.print(f"Question: [italic]{question_text[:80]}{'...' if len(question_text) > 80 else ''}[/italic]\n")

    debate_start = time.monotonic()
    completed_rounds: list[Round] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:

        def on_round_complete(rnd: Round) -> None:
            completed_rounds.append(rnd)
            progress.print(f"[green]OK[/green] Round {rnd.number} complete ({len(rnd.responses)} responses)")

        debate_task = progress.add_task("Running debate rounds...", total=None)
        debate_rounds = await run_debate(
            question=question,
            providers=panel_providers,
            prompts=config.prompts,
            num_rounds=rounds,
            on_round_complete=on_round_complete,
        )
        progress.update(debate_task, description="Running synthesis...")

        result = await synthesize(
            question=question,
            rounds=debate_rounds,
            synthesizer=synthesizer,
            prompts=config.prompts,
            debate_start_time=debate_start,
            panel_mode=panel_mode,
            synthesizer_is_participant=is_participant,
        )

    for rnd in debate_rounds:
        print_round_summary(rnd.number, rnd.responses)

    print_synthesis(result)

    saved_path = save_to_file(result, output_dir)
    console.print(f"\n[dim]Saved to: {saved_path}[/dim]")


@click.command()
@click.argument("question", required=False)
@click.option("--file", "question_file", type=click.Path(exists=True), help="Read question from .md file")
@click.option("--rounds", default=None, type=int, help="Number of debate rounds (default: from config)")
@click.option("--models", default=None, help="Comma-separated model list, overrides panel selection")
@click.option("--full", "use_full_panel", is_flag=True,
              help="Use all 5 models. Default uses 3-model panel (claude, gemini, deepseek).")
@click.option("--output", "output_path", default=None, help="Output directory (default: from config)")
@click.option("--synthesizer", default=None, help="Which model synthesizes (default: from config)")
@click.option("--verbose", is_flag=True, help="Enable DEBUG-level logging")
def main(
    question: str | None,
    question_file: str | None,
    rounds: int | None,
    models: str | None,
    use_full_panel: bool,
    output_path: str | None,
    synthesizer: str | None,
    verbose: bool,
) -> None:
    """AI Council -- Multi-model architectural debate tool.

    \b
    Examples:
      python -m src.cli "Should we use REST or GraphQL?" --rounds 1
      python -m src.cli "Monorepo vs polyrepo?" --rounds 1 --full
      python -m src.cli "SQL or NoSQL?" --rounds 1 --models claude,openai
      python -m src.cli --file question.md --rounds 3
    """
    load_dotenv()
    _setup_logging(verbose)

    try:
        config = load_config()
    except FileNotFoundError as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    if question_file:
        question_text = Path(question_file).read_text(encoding="utf-8").strip()
        question_source = question_file
    elif question:
        question_text = question
        question_source = "cli"
    else:
        console.print("[bold red]Error:[/bold red] Provide a QUESTION argument or --file.")
        sys.exit(1)

    effective_rounds = rounds if rounds is not None else config.defaults.rounds
    effective_output = Path(output_path) if output_path else config.defaults.output_dir
    effective_synthesizer = synthesizer if synthesizer else config.defaults.synthesizer

    asyncio.run(
        _run(
            question_text=question_text,
            question_source=question_source,
            config=config,
            rounds=effective_rounds,
            models_arg=models,
            output_dir=effective_output,
            synthesizer_name=effective_synthesizer,
            use_full_panel=use_full_panel,
        )
    )


if __name__ == "__main__":
    main()
