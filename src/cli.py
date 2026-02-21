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
}


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def _build_providers(
    config: AppConfig,
    requested_models: list[str] | None,
) -> list[AIProvider]:
    """Instantiate providers for available + requested models."""
    if requested_models:
        target_names = [m.strip() for m in requested_models]
    else:
        target_names = list(config.available_providers)

    providers: list[AIProvider] = []
    for name in target_names:
        if name not in config.available_providers:
            logging.warning("Provider '%s' skipped: API key not set", name)
            continue
        if name not in PROVIDER_CLASSES:
            logging.warning("Provider '%s' unknown, skipping", name)
            continue
        model_cfg = config.models[name]
        try:
            provider = PROVIDER_CLASSES[name](model_cfg)
            providers.append(provider)
        except Exception as exc:
            logging.warning("Failed to instantiate provider '%s': %s", name, exc)

    return providers


def _pick_synthesizer(
    providers: list[AIProvider],
    preferred: str,
) -> AIProvider:
    """Return the preferred synthesizer, or fall back to first provider."""
    for p in providers:
        if p.name() == preferred:
            return p
    fallback = providers[0]
    logging.warning(
        "Preferred synthesizer '%s' not available, falling back to '%s'",
        preferred,
        fallback.name(),
    )
    return fallback


async def _run(
    question_text: str,
    question_source: str,
    config: AppConfig,
    rounds: int,
    models_arg: str | None,
    output_dir: Path,
    synthesizer_name: str,
) -> None:
    requested_models = models_arg.split(",") if models_arg else None
    providers = _build_providers(config, requested_models)

    if len(providers) < 2:
        console.print(
            f"[bold red]Error:[/bold red] Need at least 2 providers, got {len(providers)}. "
            "Check API keys in .env."
        )
        sys.exit(1)

    question = Question(text=question_text, source=question_source)
    synthesizer = _pick_synthesizer(providers, synthesizer_name)
    provider_names = [p.name() for p in providers]

    console.print(f"\n[bold cyan]AI Council[/bold cyan] — {len(providers)} models, {rounds} rounds")
    console.print(f"Models: {', '.join(provider_names)}")
    console.print(f"Question: [italic]{question_text[:80]}{'…' if len(question_text) > 80 else ''}[/italic]\n")

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
            progress.print(f"[green]✓[/green] Round {rnd.number} complete ({len(rnd.responses)} responses)")

        debate_task = progress.add_task("Running debate rounds…", total=None)
        debate_rounds = await run_debate(
            question=question,
            providers=providers,
            prompts=config.prompts,
            num_rounds=rounds,
            on_round_complete=on_round_complete,
        )
        progress.update(debate_task, description="Running synthesis…")

        result = await synthesize(
            question=question,
            rounds=debate_rounds,
            synthesizer=synthesizer,
            prompts=config.prompts,
            debate_start_time=debate_start,
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
@click.option("--models", default=None, help="Comma-separated model list (default: all available)")
@click.option("--output", "output_path", default=None, help="Output directory (default: from config)")
@click.option("--synthesizer", default=None, help="Which model synthesizes (default: from config)")
@click.option("--verbose", is_flag=True, help="Enable DEBUG-level logging")
def main(
    question: str | None,
    question_file: str | None,
    rounds: int | None,
    models: str | None,
    output_path: str | None,
    synthesizer: str | None,
    verbose: bool,
) -> None:
    """AI Council — Multi-model architectural debate tool.

    \b
    Examples:
      python -m src.cli "Should we use YAML or JSON for config?"
      python -m src.cli --file question.md --rounds 3
      python -m src.cli --models gemini,openai "quick question"
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
        )
    )


if __name__ == "__main__":
    main()
