"""Click CLI — parses args, builds RunRequest, delegates to CouncilRunner."""

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from ai_council.research.models import MergedResearchReport

from ai_council.healthcheck import run_health_checks
from ai_council.inbox import archive_file, clean_slug, ensure_dirs, parse_file, scan_downloads_folder, scan_inbox
from ai_council.mode_detector import detect_mode
from ai_council.models import Question, RunRequest
from ai_council.orchestrator import CouncilRunner
from ai_council.output import OutputRoutingError
from ai_council.policy import RunPolicy
from ai_council.providers.anthropic import AnthropicProvider
from ai_council.providers.base import AIProvider
from ai_council.providers.deepseek import DeepSeekProvider
from ai_council.providers.gemini import GeminiProvider
from ai_council.providers.openai_provider import OpenAIProvider
from ai_council.providers.xai import XAIProvider
from ai_council.routing import RoutingError, TargetResolver
from ai_council.runner import build_all_providers, determine_panel
from config.config_loader import (
    AppConfig,
    ModeConfig,
    ResearchConfig,
    default_mode,
    load_config,
    resolve_mode,
)

logger = logging.getLogger(__name__)
console = Console(legacy_windows=False)

PROVIDER_CLASSES: dict[str, type[AIProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "claude": AnthropicProvider,
    "claude-sonnet": AnthropicProvider,
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


def _strip_empty_api_keys(config: AppConfig) -> list[str]:
    """DOC-3 (#30): an API-key env var set to an EMPTY string shadows the real value in the
    `.env` file under `load_dotenv(override=False)`, so a key that IS on disk never loads and the
    provider is silently dropped from ``available_providers``. Detect expected key env vars that
    are present-but-empty, remove them from the environment (so empty reads as ABSENT, never
    silently), and return the names stripped so the caller can warn loudly and reload. Expected
    names are derived from config (debate models + research providers) — never a hardcoded list.
    """
    key_envs = {m.api_key_env for m in config.models.values()}
    if config.research is not None:
        key_envs |= {p.api_key_env for p in config.research.providers.values()}
    stripped = sorted(v for v in key_envs if v in os.environ and not os.environ[v].strip())
    for var in stripped:
        del os.environ[var]
    return stripped


def _run_research_dispatch(
    *,
    query: str,
    config: AppConfig,
    output_dir: Path,
    deep: bool,
    no_cache: bool,
    console: Console,
    output_format: str,
    models_filter: list[str] | None,
    target_paths: list[Path] | None,
    return_dir: Path | None,
) -> "MergedResearchReport | None":
    """Single dispatch point for a research commission — collapses the Lane A / Lane B
    call-sites into one and threads ``--return-dir`` through to ``run_research`` so a
    research run honors the caller's return dir identically to the debate path (#23,
    the A2-narrowing dispatch slice). Callers own their own pre/post handling (inbox
    archive vs interactive exit codes)."""
    from ai_council.research.runner import run_research
    return asyncio.run(
        run_research(
            query=query,
            config=config,
            output_dir=output_dir,
            deep=deep,
            no_cache=no_cache,
            console=console,
            output_format=output_format,
            models_filter=models_filter,
            target_paths=target_paths,
            return_dir=return_dir,
        )
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


def _select_health_check_targets(
    all_providers: dict[str, AIProvider],
    *,
    cli_mode_arg: str | None,
    modes: dict[str, ModeConfig],
    research_cfg: ResearchConfig | None,
) -> tuple[dict[str, AIProvider], bool, str | None]:
    """Decide which providers to health-check and whether the gate is blocking.

    Research mode reaches the debate-provider pool only for the summarizer;
    the retrieval providers are a separate pool and have their own ADR-08
    degradation handling. So for an explicit --mode research / -M r, check
    only the summarizer, and never block (summarizer outage is non-fatal —
    research/merger.py falls back to truncation). All other modes keep the
    pre-existing behaviour: ping the full debate pool with a blocking gate.

    Returns (targets, blocking, missing_summarizer_name). The third element
    is the configured summarizer name when research mode is selected but the
    summarizer provider failed to build (e.g. missing API key) — callers
    surface this as a non-blocking warning at startup.
    """
    if cli_mode_arg is not None and modes and research_cfg is not None:
        try:
            resolved = resolve_mode(cli_mode_arg, modes)
        except ValueError:
            resolved = None
        if resolved == "research":
            name = research_cfg.summary_model
            if name in all_providers:
                return {name: all_providers[name]}, False, None
            return {}, False, name
    return all_providers, True, None


def _check_summarizer_health(
    summarizer_providers: dict[str, AIProvider],
    *,
    missing_name: str | None = None,
) -> None:
    """Run a non-blocking health check on the research summarizer.

    Prints OK/FAIL but never gates: research mode's own merger handles a
    summarizer outage with a truncation fallback. Surfacing the failure
    upfront lets the operator fix the key before minutes of retrieval work.

    If `missing_name` is set, the summarizer failed to build (no API key /
    init error) — warn explicitly instead of silently doing nothing.
    """
    if not summarizer_providers and missing_name is None:
        return
    console.print("\n[bold]Checking research summarizer...[/bold]")
    if missing_name is not None:
        console.print(
            f"  [yellow]WARN[/yellow] {missing_name} (summarizer): "
            f"unavailable (missing API key or provider init failed)"
            f"\n  [dim]Research will fall back to truncation summary.[/dim]"
        )
    if summarizer_providers:
        results = asyncio.run(run_health_checks(summarizer_providers))
        for name in sorted(results):
            ok, err = results[name]
            if ok:
                console.print(f"  [green]OK  [/green] {name} (summarizer)")
            else:
                short_err = err.splitlines()[0][:120] if err else "unknown error"
                console.print(
                    f"  [yellow]WARN[/yellow] {name} (summarizer): {short_err}"
                    f"\n  [dim]Research will fall back to truncation summary.[/dim]"
                )
    console.print()


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


_EPILOG = """\b
MODES  (run --modes for full alias list):
  pick (p/d)      Choose between options -> recommendation  [default]
  ideas (i/e)     Generate ideas -> clusters + wild cards
  judge (j/a)     Evaluate something -> verdict + evidence
  research (r)    Deep web research -> sourced report

  Mode is auto-detected from question text if -M is not specified.

PANEL DEFAULTS:
  Full 5-model panel is used by default (claude, gemini, deepseek, openai, grok).
  Use --lite for the 3-model panel (claude, gemini, openai).
  Use --models to specify a custom panel.

FLAG GROUPS:
  Mode:     -M/--mode, --modes
  Models:   --models, --full, --lite, --synthesizer
  Research: --deep, --no-cache
  Input:    --file, --inbox, --inbox-dir
  Output:   --format, --output, --verbose
  Rounds:   --rounds

EXAMPLES:
  council "Should we use REST or GraphQL?"
  council --lite "Quick question with 3-model panel"
  council -M ideas "What caching strategies should we consider?"
  council -M judge "Is this microservices design production-ready?"
  council -M p "Redis vs Memcached for sessions?" --rounds 1
  council -M research "Best HTAP databases in 2026"
  council -M r "LLM inference hardware comparison" --deep
  council -M r "Redis vs Valkey" --no-cache
  council "Monorepo vs polyrepo?" --synthesizer openai
  council --file question.md --models claude,gemini --rounds 3
  council --inbox
  council --format json "question" > output.json
"""


def _print_modes_callback(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        cfg = load_config()
        modes = cfg.modes
    except Exception:
        modes = {}
    _con = Console(legacy_windows=False)
    if modes:
        _con.print("\n[bold]Debate Modes[/bold]\n")
        for key, mode_cfg in modes.items():
            default_marker = " [dim](default)[/dim]" if mode_cfg.default else ""
            aliases_str = ", ".join(mode_cfg.aliases)
            _con.print(
                f"  {mode_cfg.emoji}  [bold cyan]{key:<7}[/bold cyan]"
                f"  {mode_cfg.description}{default_marker}"
            )
            _con.print(f"           [dim]aliases: {aliases_str}[/dim]\n")
    else:
        _con.print("No modes configured.")
    ctx.exit()


def _report_boundary_failure(exc: BaseException, *, what: str) -> None:
    """Print a clean, type-appropriate message for a failure at the CLI boundary.

    The four run/research boundary sites catch broadly (so a raw traceback never reaches the
    operator) but must NOT collapse every failure into one label. Branching on type:

    * ``OutputRoutingError`` (and anything Lane A1 subclasses from it) means a REQUIRED
      deliverable did not land -- the message names the destination and reads as a routing
      failure, which is what the caller must act on.
    * anything else is an internal defect (a TypeError in synthesis, a KeyError in parsing).
      Calling that a "required-write failure" would mislabel it and, worse, discard the
      traceback needed to debug it -- so the full traceback goes to the log.

    Callers exit non-zero afterwards; this function only reports.
    """
    if isinstance(exc, OutputRoutingError):
        console.print(f"[bold red]Required write failed[/bold red] ({what}): {exc}")
        logger.error("Required output write failed (%s): %s", what, exc)
    else:
        console.print(
            f"[bold red]Unexpected error[/bold red] ({what}): {type(exc).__name__}: {exc}\n"
            "[dim]This is an internal failure, not an output-routing problem. "
            "Re-run with --verbose for the full traceback.[/dim]"
        )
        logger.error("Unexpected failure (%s)", what, exc_info=True)


def _remove_scratch_dir(path: Path) -> None:
    """Remove a ``--no-persist`` scratch dir. NEVER raises (#71).

    Registered via ``ctx.call_on_close``, so this runs inside Click's context teardown --
    which also runs while an exception from the command is propagating. A raising cleanup
    would therefore either turn a successful run red, or chain over the in-flight exception
    and mask the root cause the run was already reporting. On Windows that is not
    hypothetical: ``rmtree`` raises PermissionError whenever a handle is still open.

    #71's harm is a leftover directory; trading it for a crash or a swallowed root cause is
    a bad trade. So a cleanup failure never changes the exit code -- it warns loudly, naming
    the path that survived, and leaves the leak visible.
    """
    try:
        shutil.rmtree(path)
    except OSError as exc:
        console.print(
            f"[yellow]WARNING:[/yellow] could not remove scratch dir {path}: {exc} "
            "-- it is still on disk; remove it manually."
        )
        logger.warning("Scratch dir cleanup failed for %s (non-fatal)", path, exc_info=True)


def _resolve_output_dir(
    ctx: click.Context, config: AppConfig, output_path: str | None, no_persist: bool
) -> Path:
    """Resolve the canonical output dir for ANY command (#39, #65).

    Precedence, highest first: ``--output`` flag > ``--no-persist`` (scratch temp, canonical
    output/ untouched) > ``AICOUNCIL_OUTPUT_DIR`` env override > config default. No routing
    redesign -- this only chooses the canonical dir the writers already write to.

    Single source of truth: every command resolves here, so ``run`` and ``doctor`` cannot
    drift apart (they did -- ``doctor`` honoured none of these controls before #65).

    The ``--no-persist`` scratch dir's removal is registered on ``ctx`` at creation, so
    cleanup fires on success, on ``sys.exit``, and on an unhandled exception (#71).
    """
    env_output = os.environ.get("AICOUNCIL_OUTPUT_DIR")
    if output_path:
        # #74: expand ~ here too. The env branch below always did; --output did not, so
        # `--output ~/foo` created a literal './~/foo' directory instead of resolving to
        # the home dir. Both string-sourced branches now expand symmetrically.
        return Path(output_path).expanduser()
    if no_persist:
        scratch = Path(tempfile.mkdtemp(prefix="aicouncil-scratch-"))
        # Registered immediately after creation: nothing between here and the command body
        # can leave the dir unregistered. Click runs close callbacks from the context's
        # ExitStack in main()'s finally -- success, sys.exit, and exception alike.
        ctx.call_on_close(lambda: _remove_scratch_dir(scratch))
        return scratch
    if env_output:
        return Path(env_output).expanduser()
    return config.defaults.output_dir


class _DefaultGroup(click.Group):
    """Group that falls back to a default subcommand when the first token is not a
    registered command -- preserves the bare ``council "question"`` invocation
    (routed to ``run``) while also exposing ``council run`` / ``council doctor`` /
    ``council --modes``. No new dependency; the whole shim is these two methods.
    """

    default_cmd = "run"

    def _own_opt_names(self) -> set[str]:
        """Option strings the GROUP itself owns (e.g. --modes, --help)."""
        names = {"--help", "-h"}
        for param in self.params:
            names.update(param.opts)
            names.update(param.secondary_opts)
        return names

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # No subcommand token -> run (preserves `council` -> "provide a question" path).
        if not args:
            args = [self.default_cmd]
        # First token is neither a known subcommand nor a group-level option -> it is a
        # positional question or a run-level option; route the whole tail through `run`.
        elif args[0] not in self.commands and args[0] not in self._own_opt_names():
            args = [self.default_cmd, *args]
        return super().parse_args(ctx, args)


@click.group(
    cls=_DefaultGroup,
    context_settings={"max_content_width": 120},
    epilog=_EPILOG,
)
@click.option(
    "--modes", is_flag=True, is_eager=True, expose_value=False,
    callback=_print_modes_callback,
    help="Print all debate modes with aliases and exit.",
)
def main() -> None:
    """AI Council -- multi-model debate and research tool. Use --modes for mode details."""


@main.command(
    "run",
    context_settings={"max_content_width": 120},
    epilog=_EPILOG,
)
@click.argument("question", required=False)
@click.option(
    "--file", "question_file",
    type=click.Path(exists=True),
    help="Read question from a .md file instead of inline argument.",
)
@click.option("--rounds", default=None, type=int, help="Number of debate rounds (default: from mode config).")
@click.option(
    "--models", default=None,
    help="Comma-separated panel override, e.g. claude,openai,grok. Overrides --full and default panel.",
)
@click.option(
    "--full", "use_full_panel",
    is_flag=True,
    help="[No-op] Full panel is now the default. Kept for backward compatibility.",
)
@click.option(
    "--lite", is_flag=True, default=False,
    help="Use the 3-model panel (claude, gemini, openai) instead of the full 5-model default.",
)
@click.option(
    "--output", "output_path",
    default=None,
    help="Output directory for saved transcripts (default: ./output).",
)
@click.option(
    "--no-persist", "no_persist", is_flag=True, default=False,
    help="Write artifacts to a scratch temp dir instead of ./output/, so witness/dev "
         "runs leave the canonical output/ untouched (#39).",
)
@click.option(
    "--return-dir", "return_dir",
    default=None,
    help="ADR-10 deterministic return: also route this run's artifacts to this directory, "
         "in addition to the canonical ./output/ write. That means the debate transcript, "
         "the verdict package JSON, any minority report, and -- in research mode -- the "
         "research report. When unset, output goes to ./output/ only (the hub is never a "
         "default).",
)
@click.option(
    "--synthesizer", default=None,
    help="Model that writes the final verdict: claude, openai, gemini, grok, deepseek. "
         "Defaults to gemini. Automatically excluded from the debate panel.",
)
@click.option(
    "--mode", "-M", "mode_arg", default=None,
    help="Debate mode: pick (default), ideas, or judge - or any alias. "
         "Skips auto-detection when set. Run --modes to see all aliases.",
)
@click.option("--verbose", is_flag=True, help="Enable DEBUG-level logging.")
@click.option("--inbox", "use_inbox", is_flag=True, default=False, help="Process all .md files in the inbox folder.")
@click.option(
    "--inbox-dir", "inbox_dir_override",
    default=None,
    help="Override the inbox folder path (default: from config).",
)
@click.option("--skip-health-check", is_flag=True, default=False, help="Skip the API connectivity check at startup.")
@click.option(
    "--deep", is_flag=True, default=False,
    help="Research mode: include slower deep-research providers (o3-deep-research).",
)
@click.option("--no-cache", "no_cache", is_flag=True, default=False, help="Research mode: skip cache read and write.")
@click.option(
    "--format", "output_format", default="text",
    type=click.Choice(["text", "json"], case_sensitive=False),
    help="Output format: text (default) or json (prints structured result to stdout).",
)
@click.option(
    "--target-project",
    "target_projects_arg",
    multiple=True,
    help=(
        "Target project name(s) for transcript mirroring. Repeat flag for multiple targets. "
        "Must be a name in the config/settings.yaml target_projects list; path resolved under dev_root."
    ),
)
@click.pass_context
def run(
    ctx: click.Context,
    question: str | None,
    question_file: str | None,
    rounds: int | None,
    models: str | None,
    use_full_panel: bool,
    lite: bool,
    output_path: str | None,
    no_persist: bool,
    return_dir: str | None,
    synthesizer: str | None,
    mode_arg: str | None,
    verbose: bool,
    use_inbox: bool,
    inbox_dir_override: str | None,
    skip_health_check: bool,
    deep: bool,
    no_cache: bool,
    output_format: str,
    target_projects_arg: tuple[str, ...],
) -> None:
    """AI Council -- multi-model debate and research tool. Use --modes for mode details."""
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

    # DOC-3 (#30): a present-but-empty API-key env var silently shadows the real .env value under
    # override=False. Strip such keys (treat empty as ABSENT), warn LOUDLY, then reload .env +
    # config so the on-disk value takes effect and available_providers is recomputed.
    _empty_keys = _strip_empty_api_keys(config)
    if _empty_keys:
        console.print(
            "[bold yellow]WARNING:[/bold yellow] API key env var(s) set but empty: "
            f"{', '.join(_empty_keys)} -- treated as absent; reloading .env."
        )
        logger.warning("Empty API key env var(s) treated as absent: %s", ", ".join(_empty_keys))
        if _global_env.exists():
            load_dotenv(_global_env, override=False)
        load_dotenv(override=False)
        config = load_config()

    # Output-dir resolution (#39) -- one resolver shared with `doctor` (#65).
    effective_output = _resolve_output_dir(ctx, config, output_path, no_persist)
    effective_synthesizer = synthesizer if synthesizer else config.defaults.synthesizer

    # ADR-10 deterministic return directory. Precedence (highest first):
    #   1. --return-dir CLI flag (implemented here)
    #   2. RESERVED: ~/.claude global config `council.return_dir` (per ADR-67) — a legal
    #      future setter, deliberately NOT read this pass (ADR-10 defers the reader).
    #      When built, resolve it here as the fallback when --return-dir is unset.
    # Unset → None → canonical ./output/ only; the methodology hub is never a default.
    effective_return_dir = Path(return_dir).expanduser() if return_dir else None

    # Build resolver and validate --target-project args early (before health checks)
    resolver = TargetResolver(config.dev_root or Path("."), config.target_projects)
    try:
        cli_target_paths = resolver.resolve(target_projects_arg)
    except RoutingError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

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
        check_targets, blocking, missing_summarizer = _select_health_check_targets(
            all_providers,
            cli_mode_arg=mode_arg,
            modes=config.modes,
            research_cfg=config.research,
        )
        if blocking:
            all_providers = _check_and_filter_providers(check_targets)
        else:
            _check_summarizer_health(check_targets, missing_name=missing_summarizer)

    runner = CouncilRunner(all_providers, config)
    policy = RunPolicy.from_config(config.policy)

    if use_inbox:
        inbox_dir = Path(inbox_dir_override) if inbox_dir_override else config.inbox.dir
        archive_dir = config.inbox.archive_dir
        ensure_dirs(inbox_dir, archive_dir)

        dl_files: list[Path] = []
        if config.inbox.scan_downloads:
            dl_files = scan_downloads_folder(
                config.inbox.downloads_dir, config.inbox.council_frontmatter_keys
            )
        dl_set = set(dl_files)

        files = scan_inbox(inbox_dir)
        all_files = dl_files + files

        if not all_files:
            click.echo("No files in inbox.")
            return

        # ADR-08: track degraded research runs across the batch; exit 3 at end.
        inbox_any_degraded = False
        inbox_any_failed = False
        for file_path in all_files:
            try:
                question_text, meta = parse_file(file_path, resolver=resolver)
            except RoutingError as exc:
                logger.error("Routing error in %s: %s -- skipping", file_path.name, exc)
                archive_file(file_path, archive_dir, failed=True)
                continue
            # CLI --target-project wins over frontmatter target-project
            fm_target_paths = cli_target_paths if cli_target_paths else meta.get("target_paths", [])
            fm_rounds = int(meta["rounds"]) if "rounds" in meta else config.defaults.rounds
            fm_models = str(meta["models"]) if "models" in meta and not use_full_panel else None
            fm_full = (use_full_panel or not lite) or bool(meta.get("full", False))
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

            if fm_mode == "research":
                if config.research is None:
                    logger.error("No research config in settings.yaml -- skipping %s", file_path.name)
                    archive_file(file_path, archive_dir, failed=True)
                    continue
                try:
                    fm_report = _run_research_dispatch(
                        query=question_text,
                        config=config,
                        output_dir=effective_output,
                        deep=deep,
                        no_cache=no_cache,
                        console=console,
                        output_format=output_format,
                        models_filter=[m.strip() for m in fm_models.split(",")] if fm_models else None,
                        target_paths=fm_target_paths or None,
                        return_dir=effective_return_dir,
                    )
                    if fm_report is not None and fm_report.degraded:
                        inbox_any_degraded = True
                    archived = archive_file(file_path, archive_dir)
                    if file_path in dl_set:
                        click.echo(f"Processed from Downloads: {file_path.name} -> archived")
                    else:
                        click.echo(f"Archived: {file_path.name} -> {archived.name}")
                except Exception as exc:
                    logger.error("Research failed: %s -- %s", file_path.name, exc)
                    _report_boundary_failure(exc, what=f"research: {file_path.name}")
                    archive_file(file_path, archive_dir, failed=True)
                    # Bookkeeping unchanged (still archived as failed, batch continues);
                    # only the batch's final exit code changes -- see below.
                    inbox_any_failed = True
                continue

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
                target_paths=fm_target_paths,
                return_dir=effective_return_dir,
            )
            try:
                asyncio.run(runner.run(request, output_dir=effective_output, output_format=output_format))
                archived = archive_file(file_path, archive_dir)
                if file_path in dl_set:
                    click.echo(f"Processed from Downloads: {file_path.name} -> archived")
                else:
                    click.echo(f"Archived: {file_path.name} -> {archived.name}")
            except Exception as e:
                logger.error("Failed: %s -- %s", file_path.name, e)
                _report_boundary_failure(e, what=f"debate: {file_path.name}")
                archive_file(file_path, archive_dir, failed=True)
                inbox_any_failed = True

        # Exit code computed at the END, after every file has been attempted -- a failure
        # must never abort the batch. Failure DOMINATES degradation: >=1 hard failure is a
        # hard error (1) even if other files merely degraded; degraded-only batches stay 3.
        # Before this, a batch that failed every single file still exited 0.
        if inbox_any_failed:
            sys.exit(1)
        if inbox_any_degraded:
            sys.exit(3)
        return


    file_meta: dict = {}
    if question_file:
        try:
            # #22: route --file through parse_file so YAML frontmatter is stripped
            # (never leaks into the question text) and its overrides are honored below.
            question_text, file_meta = parse_file(Path(question_file), resolver=resolver)
        except RoutingError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)
        question_source = question_file
    elif question:
        question_text = question
        question_source = "cli"
    else:
        console.print("[bold red]Error:[/bold red] Provide a QUESTION argument, --file, or --inbox.")
        sys.exit(1)

    # #22: --file frontmatter precedence -- CLI flag > frontmatter > config default.
    # file_meta is {} for an inline question, so each reduces to the flag/config path.
    effective_synthesizer = (
        synthesizer if synthesizer is not None
        else str(file_meta["synthesizer"]) if "synthesizer" in file_meta
        else config.defaults.synthesizer
    )
    eff_full = (use_full_panel or not lite) or bool(file_meta.get("full", False))
    eff_models = models if models is not None else (
        str(file_meta["models"]) if "models" in file_meta and not eff_full else None
    )
    eff_target_paths = cli_target_paths if cli_target_paths else file_meta.get("target_paths", [])

    # Mode resolution for interactive: CLI --mode > frontmatter mode: > auto-detect > default
    effective_mode: str | None = None
    if mode_arg is not None and config.modes:
        effective_mode = resolve_mode(mode_arg, config.modes)
    elif file_meta.get("mode") is not None and config.modes:
        try:
            effective_mode = resolve_mode(str(file_meta["mode"]), config.modes)
        except ValueError:
            logger.warning(
                "Unknown mode '%s' in %s, falling back to auto-detect",
                file_meta["mode"], question_source,
            )
            effective_mode = None
    if effective_mode is None:
        if config.modes:
            valid_modes = set(config.modes.keys())
            detected, source_label = asyncio.run(
                detect_mode(question_text, all_providers, valid_modes)
            )
            effective_mode = _interactive_confirm_mode(
                detected, source_label, config.modes
            )
        else:
            effective_mode = "pick"

    # Research mode: completely separate code path — no debate rounds
    if effective_mode == "research":
        if config.research is None:
            console.print("[bold red]Error:[/bold red] No research config in settings.yaml.")
            sys.exit(1)
        try:
            report = _run_research_dispatch(
                query=question_text,
                config=config,
                output_dir=effective_output,
                deep=deep,
                no_cache=no_cache,
                console=console,
                output_format=output_format,
                models_filter=[m.strip() for m in eff_models.split(",")] if eff_models else None,
                target_paths=eff_target_paths or None,
                return_dir=effective_return_dir,
            )
        # OutputRoutingError subclasses RuntimeError, so it MUST be caught FIRST -- the
        # pre-existing `except RuntimeError` below would otherwise swallow a required-write
        # failure and mislabel it as a research error.
        except OutputRoutingError as exc:
            _report_boundary_failure(exc, what="research")
            sys.exit(1)
        except RuntimeError as exc:
            # An expected hard error (CONTRACT §4: research RuntimeError -> exit 1), not an
            # internal defect -- keep the original, accurate wording.
            console.print(f"[bold red]Research error:[/bold red] {exc}")
            sys.exit(1)
        except Exception as exc:
            # Previously escaped as a raw Click traceback (e.g. OSError).
            _report_boundary_failure(exc, what="research")
            sys.exit(1)
        # Exit-code convention (ADR-08): 0 ok / 1 hard error / 2 Click usage / 3 degraded.
        if report is not None and report.degraded:
            sys.exit(3)
        return

    mode_cfg = config.modes.get(effective_mode)
    # #22: rounds precedence -- CLI flag > frontmatter > mode default > config default.
    if rounds is not None:
        effective_rounds = rounds
    elif "rounds" in file_meta:
        effective_rounds = int(file_meta["rounds"])
    elif mode_cfg:
        effective_rounds = mode_cfg.max_rounds
    else:
        effective_rounds = config.defaults.rounds

    panel_names, panel_mode = determine_panel(config, eff_models, eff_full)
    request = RunRequest(
        question=Question(text=question_text, source=question_source),
        panel_names=panel_names,
        synthesizer_name=effective_synthesizer,
        rounds=effective_rounds,
        policy=policy,
        panel_mode=panel_mode,
        synthesizer_specified=synthesizer is not None or "synthesizer" in file_meta,
        mode=effective_mode,
        target_paths=eff_target_paths,
        return_dir=effective_return_dir,
    )
    # Boundary: this site had NO handler at all, so every failure -- including a required
    # -write failure -- reached the operator as a raw Click traceback.
    try:
        asyncio.run(runner.run(request, output_dir=effective_output, output_format=output_format))
    except Exception as exc:
        _report_boundary_failure(exc, what="debate")
        sys.exit(1)


@main.command("doctor")
@click.option(
    "--output", "output_path",
    default=None,
    help="Canonical output dir override for the health record (default ./output/).",
)
@click.option(
    "--no-persist", "no_persist",
    is_flag=True,
    default=False,
    help="Write the health record to a scratch temp dir, leaving canonical ./output/ "
         "untouched; the scratch dir is removed when the command exits.",
)
@click.pass_context
def doctor(ctx: click.Context, output_path: str | None, no_persist: bool) -> None:
    """Liveness + config pre-flight: a GREEN/YELLOW/RED truth table over keys, seats, and
    config. Writes a machine-readable record to <output-dir>/health/. Never blocks a run.

    The output dir honours --output / --no-persist / AICOUNCIL_OUTPUT_DIR via the same
    precedence chain as `council run` (#65)."""
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    from ai_council.doctor import run_doctor

    # First load: learn the configured key-env NAMES and snapshot the launching shell's
    # values for them, so a set-but-empty key (env shadowing) can be surfaced below.
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)
    key_envs = {model.api_key_env for model in config.models.values()}
    if config.research is not None:
        key_envs |= {provider.api_key_env for provider in config.research.providers.values()}
    shell_snapshot = {env: os.environ.get(env) for env in key_envs}

    # The doctor measures the REAL GLOBAL credentials regardless of shell state
    # (DRAFT-DOC-3 doctor stance): load ONLY the global secrets file, with override=True so
    # it wins over a poisoned shell (e.g. an empty-but-set key). A repo-local .env is
    # deliberately NOT consulted -- consulting it would let a forbidden repo-local secret
    # (repo rule: global secrets only) mask a genuine global-config gap with a false GREEN.
    # This is the doctor's OWN load -- the shared run-path loader is untouched.
    _global_env = Path.home() / "Documents" / ".secrets" / ".env"
    if _global_env.exists():
        # A diagnostic must run even in a sick environment (L-DOC 2.3): an unreadable or
        # corrupt secrets file warns loudly but does not abort -- the doctor then measures
        # the current environment, and the seat pings report the real reachability.
        try:
            load_dotenv(_global_env, override=True)
        except (OSError, UnicodeDecodeError) as exc:
            console.print(
                f"[yellow]WARNING:[/yellow] could not read global secrets file {_global_env}: "
                f"{exc} -- measuring the current environment only."
            )
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    # #65: resolve through the SAME chain as `run` -- before this, doctor ignored every
    # output control and always wrote to canonical ./output/health/.
    effective_output = _resolve_output_dir(ctx, config, output_path, no_persist)

    exit_code = run_doctor(
        config,
        PROVIDER_CLASSES,
        shell_snapshot=shell_snapshot,
        console=console,
        output_dir=effective_output,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
