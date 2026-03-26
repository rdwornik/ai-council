"""Click CLI — parses args, builds RunRequest, delegates to CouncilRunner."""

import asyncio
import logging
import sys
import threading
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from config.config_loader import default_mode, load_config, resolve_mode
from src.healthcheck import run_health_checks
from src.inbox import archive_file, clean_slug, ensure_dirs, parse_file, scan_inbox
from src.mode_detector import detect_mode
from src.models import Question, RunRequest
from src.policy import RunPolicy
from src.providers.anthropic import AnthropicProvider
from src.providers.base import AIProvider
from src.providers.deepseek import DeepSeekProvider
from src.providers.gemini import GeminiProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.xai import XAIProvider
from src.runner import CouncilRunner, build_all_providers, determine_panel

logger = logging.getLogger(__name__)
console = Console(legacy_windows=False)

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


def _check_and_filter_providers(
    all_providers: dict[str, AIProvider],
) -> dict[str, AIProvider]:
    """Run health checks, print results, and ask user what to do on failures."""
    console.print("\n[bold]Checking providers...[/bold]")
    results: dict[str, tuple[bool, str]] = asyncio.run(run_health_checks(all_providers))

    failed_names: list[str] = []
    for name in sorted(results):
        ok, err = results[name]
        if ok:
            console.print(f"  [green]OK  [/green] {name}")
        else:
            short_err = err.splitlines()[0][:120] if err else "unknown error"
            console.print(f"  [red]FAIL[/red] {name}: {short_err}")
            failed_names.append(name)

    if not failed_names:
        console.print()
        return all_providers

    working = {n: p for n, p in all_providers.items() if n not in failed_names}

    if not working:
        console.print("\n[bold red]Error:[/bold red] No providers passed the health check.")
        sys.exit(1)

    console.print(f"\n[yellow]{len(failed_names)} provider(s) failed:[/yellow] {', '.join(failed_names)}")
    console.print(f"Working providers: {', '.join(sorted(working))}")

    if not click.confirm("Continue with working providers only?", default=True):
        sys.exit(0)

    console.print()
    return working


def _interactive_confirm_mode(
    detected_mode: str,
    source_label: str,
    modes: dict,
    timeout_sec: float = 5.0,
) -> str:
    """Show detected mode, let user override with 5s timeout. Returns final mode key."""
    cfg = modes.get(detected_mode)
    emoji = cfg.emoji if cfg else ""
    console.print(
        f"\n[bold]Mode detected:[/bold] {emoji} [cyan]{detected_mode}[/cyan]"
        f" ({source_label})"
    )
    console.print(
        f"  Press Enter to confirm, or type a mode name/alias to override "
        f"[dim]({timeout_sec:.0f}s timeout)[/dim]: ",
        end="",
    )

    result: list[str] = []
    done = threading.Event()

    def _read() -> None:
        try:
            line = sys.stdin.readline().strip()
            result.append(line)
        except Exception:
            result.append("")
        done.set()

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    timed_out = not done.wait(timeout=timeout_sec)

    if timed_out or not result or not result[0]:
        if timed_out:
            console.print(f"[dim](timed out — using {detected_mode})[/dim]")
        else:
            console.print()
        return detected_mode

    user_input = result[0]
    try:
        resolved = resolve_mode(user_input, modes)
        console.print(f"[dim]Using mode: {resolved}[/dim]")
        return resolved
    except ValueError:
        console.print(f"[yellow]Unknown mode '{user_input}', keeping {detected_mode}[/yellow]")
        return detected_mode


@click.command()
@click.argument("question", required=False)
@click.option("--file", "question_file", type=click.Path(exists=True), help="Read question from .md file")
@click.option("--rounds", default=None, type=int, help="Number of debate rounds (default: from config)")
@click.option("--models", default=None, help="Comma-separated model list, overrides panel selection")
@click.option("--full", "use_full_panel", is_flag=True, help="Use all 5 models. Default uses 3-model panel.")
@click.option("--output", "output_path", default=None, help="Output directory (default: from config)")
@click.option(
    "--synthesizer", default=None,
    help="Which model synthesizes: claude, openai, gemini, grok, deepseek (default: claude). "
         "Automatically removed from the debate panel.",
)
@click.option(
    "--mode", "-m", "mode_arg", default=None,
    help="Debate mode: pick (default), ideas, judge — or their aliases. "
         "Skips auto-detection when set.",
)
@click.option("--verbose", is_flag=True, help="Enable DEBUG-level logging")
@click.option("--inbox", "use_inbox", is_flag=True, default=False, help="Process all .md files in inbox folder")
@click.option("--inbox-dir", "inbox_dir_override", default=None, help="Override inbox folder path (default: from config)")
@click.option("--skip-health-check", is_flag=True, default=False, help="Skip the API connectivity check at startup")
def main(
    question: str | None,
    question_file: str | None,
    rounds: int | None,
    models: str | None,
    use_full_panel: bool,
    output_path: str | None,
    synthesizer: str | None,
    mode_arg: str | None,
    verbose: bool,
    use_inbox: bool,
    inbox_dir_override: str | None,
    skip_health_check: bool,
) -> None:
    """AI Council -- Multi-model architectural debate tool.

    \b
    Examples:
      python -m src.cli "Should we use REST or GraphQL?"           # auto-detects mode
      python -m src.cli -m pick "REST vs GraphQL?"                 # force pick mode
      python -m src.cli -m ideas "What features for auth?"         # brainstorm mode
      python -m src.cli -m judge "Is this architecture solid?"     # assessment mode
      python -m src.cli --synthesizer openai "question"            # GPT synthesizes
      python -m src.cli "Monorepo vs polyrepo?" --rounds 1 --full
      python -m src.cli "SQL or NoSQL?" --rounds 1 --models claude,openai
      python -m src.cli --file question.md --rounds 3
      python -m src.cli --inbox
      python -m src.cli --inbox --inbox-dir ./my_queue
    """
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    _global_env = Path.home() / "Documents" / ".secrets" / ".env"
    if _global_env.exists():
        load_dotenv(_global_env, override=False)
    load_dotenv(override=False)
    _setup_logging(verbose)

    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    effective_output = Path(output_path) if output_path else config.defaults.output_dir
    effective_synthesizer = synthesizer if synthesizer else config.defaults.synthesizer

    # Validate --mode arg early so we fail fast before health checks
    if mode_arg is not None and config.modes:
        try:
            resolve_mode(mode_arg, config.modes)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

    all_providers = build_all_providers(config, PROVIDER_CLASSES)
    if not all_providers:
        console.print("[bold red]Error:[/bold red] No providers available. Check API keys in .env.")
        sys.exit(1)

    if not skip_health_check:
        all_providers = _check_and_filter_providers(all_providers)

    runner = CouncilRunner(all_providers, config)
    policy = RunPolicy.default()

    if use_inbox:
        inbox_dir = Path(inbox_dir_override) if inbox_dir_override else config.inbox.dir
        archive_dir = config.inbox.archive_dir
        ensure_dirs(inbox_dir, archive_dir)
        files = scan_inbox(inbox_dir)

        if not files:
            click.echo("No files in inbox.")
            return

        for file_path in files:
            question_text, meta = parse_file(file_path)
            fm_rounds = int(meta["rounds"]) if "rounds" in meta else config.defaults.rounds
            fm_models = str(meta["models"]) if "models" in meta and not use_full_panel else None
            fm_full = use_full_panel or bool(meta.get("full", False))
            fm_synthesizer = (
                synthesizer if synthesizer is not None
                else str(meta["synthesizer"]) if "synthesizer" in meta
                else config.defaults.synthesizer
            )
            # Mode resolution for inbox: CLI --mode > frontmatter mode: > default
            if mode_arg is not None and config.modes:
                fm_mode = resolve_mode(mode_arg, config.modes)
            elif "mode" in meta and config.modes:
                try:
                    fm_mode = resolve_mode(str(meta["mode"]), config.modes)
                except ValueError:
                    logger.warning("Unknown mode '%s' in %s, using default", meta["mode"], file_path.name)
                    fm_mode = default_mode(config.modes) if config.modes else "pick"
            else:
                fm_mode = default_mode(config.modes) if config.modes else "pick"

            mode_cfg = config.modes.get(fm_mode)
            fm_effective_rounds = rounds if rounds is not None else fm_rounds
            if mode_cfg and rounds is None and "rounds" not in meta:
                fm_effective_rounds = mode_cfg.max_rounds

            panel_names, panel_mode = determine_panel(
                config,
                models if models is not None else fm_models,
                fm_full,
            )
            request = RunRequest(
                question=Question(text=question_text, source=str(file_path)),
                panel_names=panel_names,
                synthesizer_name=fm_synthesizer,
                rounds=fm_effective_rounds,
                policy=policy,
                panel_mode=panel_mode,
                synthesizer_specified=synthesizer is not None or "synthesizer" in meta,
                slug_override=clean_slug(file_path.stem),
                mode=fm_mode,
            )
            try:
                asyncio.run(runner.run(request, output_dir=effective_output))
                archived = archive_file(file_path, archive_dir)
                click.echo(f"Archived: {file_path.name} -> {archived.name}")
            except Exception as e:
                logger.error("Failed: %s -- %s", file_path.name, e)
                archive_file(file_path, archive_dir, failed=True)
        return

    if question_file:
        question_text = Path(question_file).read_text(encoding="utf-8").strip()
        question_source = question_file
    elif question:
        question_text = question
        question_source = "cli"
    else:
        console.print("[bold red]Error:[/bold red] Provide a QUESTION argument, --file, or --inbox.")
        sys.exit(1)

    # Mode resolution for interactive: CLI --mode > auto-detect > default
    if mode_arg is not None and config.modes:
        effective_mode = resolve_mode(mode_arg, config.modes)
        mode_source = "user-specified"
    elif config.modes:
        valid_modes = set(config.modes.keys())
        detected, source_label = asyncio.run(
            detect_mode(question_text, all_providers, valid_modes)
        )
        effective_mode = _interactive_confirm_mode(
            detected, source_label, config.modes
        )
        mode_source = source_label
    else:
        effective_mode = "pick"
        mode_source = "default (no modes configured)"

    mode_cfg = config.modes.get(effective_mode)
    effective_rounds = rounds if rounds is not None else (
        mode_cfg.max_rounds if mode_cfg else config.defaults.rounds
    )

    panel_names, panel_mode = determine_panel(config, models, use_full_panel)
    request = RunRequest(
        question=Question(text=question_text, source=question_source),
        panel_names=panel_names,
        synthesizer_name=effective_synthesizer,
        rounds=effective_rounds,
        policy=policy,
        panel_mode=panel_mode,
        synthesizer_specified=synthesizer is not None,
        mode=effective_mode,
    )
    asyncio.run(runner.run(request, output_dir=effective_output))


if __name__ == "__main__":
    main()
