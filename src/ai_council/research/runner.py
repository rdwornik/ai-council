"""Research mode orchestrator.

Wires together: provider selection → cache check → parallel research
→ merge → summarize → cache write → output.
"""

import logging
import os
from pathlib import Path

from rich.console import Console

from ai_council.research.cache import cache_get, cache_put
from ai_council.research.display import run_research_with_display
from ai_council.research.merger import make_cache_key, merge_results, summarize_report
from ai_council.research.models import MergedResearchReport
from ai_council.research.output import print_research_summary, save_research_to_file
from ai_council.research.provider import ResearchProvider
from config.config_loader import AppConfig

logger = logging.getLogger(__name__)


def build_research_providers(
    config: AppConfig,
    deep: bool = False,
    models_filter: list[str] | None = None,
) -> list[ResearchProvider]:
    """Instantiate available research providers based on config and --deep flag.

    Skips providers with missing API keys; logs a warning for each.
    If models_filter is provided, only instantiate providers whose names are in the list.
    """
    research_cfg = config.research
    if research_cfg is None:
        logger.warning("No research config in settings.yaml")
        return []

    provider_names = research_cfg.deep_providers if deep else research_cfg.default_providers
    if models_filter:
        provider_names = [n for n in provider_names if n in models_filter]
        logger.debug("Research providers filtered to: %s", provider_names)
    providers: list[ResearchProvider] = []

    for name in provider_names:
        p_cfg = research_cfg.providers.get(name)
        if p_cfg is None:
            logger.warning("Research provider '%s' not found in config", name)
            continue

        api_key = os.environ.get(p_cfg.api_key_env, "").strip()
        if not api_key:
            logger.info(
                "Skipping research provider '%s' — no API key (%s)", name, p_cfg.api_key_env
            )
            continue

        try:
            provider = _instantiate_provider(name, p_cfg, api_key)
            providers.append(provider)
            logger.debug("Research provider ready: %s (%s)", name, p_cfg.model)
        except Exception as exc:
            logger.warning("Failed to instantiate research provider '%s': %s", name, exc)

    return providers


def _instantiate_provider(name: str, p_cfg, api_key: str) -> ResearchProvider:
    """Create the appropriate provider instance by name."""
    if name == "perplexity":
        from ai_council.research.providers.perplexity import PerplexityProvider
        return PerplexityProvider(
            api_key=api_key,
            model=p_cfg.model,
            timeout_sec=p_cfg.timeout_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    elif name == "openai_mini":
        from ai_council.research.providers.openai_mini_research import OpenAIMiniResearchProvider
        return OpenAIMiniResearchProvider(
            api_key=api_key,
            model=p_cfg.model,
            timeout_sec=p_cfg.timeout_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    elif name == "openai_deep":
        from ai_council.research.providers.openai_deep_research import OpenAIDeepResearchProvider
        return OpenAIDeepResearchProvider(
            api_key=api_key,
            model=p_cfg.model,
            timeout_sec=p_cfg.timeout_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    elif name == "gemini":
        from ai_council.research.providers.gemini_research import GeminiResearchProvider
        return GeminiResearchProvider(
            api_key=api_key,
            agent=p_cfg.model,
            timeout_sec=p_cfg.timeout_sec,
            poll_interval_sec=p_cfg.poll_interval_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    elif name == "grok":
        from ai_council.research.providers.grok_research import GrokResearchProvider
        return GrokResearchProvider(
            api_key=api_key,
            model=p_cfg.model,
            base_url=p_cfg.base_url or "https://api.x.ai/v1",
            timeout_sec=p_cfg.timeout_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    else:
        raise ValueError(f"Unknown research provider name: {name}")


def _print_research_paths(console: Console, saved_paths: list[Path], secondary_dir: Path | None) -> None:
    """Print secondary saved path (print_research_summary already prints primary)."""
    if len(saved_paths) > 1:
        for p in saved_paths[1:]:
            console.print(f"[dim]Copied: {p}[/dim]")
    elif secondary_dir is not None and not secondary_dir.exists():
        console.print(
            f"[dim yellow]Secondary output dir not found: {secondary_dir}[/dim yellow]"
        )


async def run_research(
    query: str,
    config: AppConfig,
    output_dir: Path,
    deep: bool = False,
    no_cache: bool = False,
    console: Console | None = None,
    output_format: str = "text",
    models_filter: list[str] | None = None,
) -> MergedResearchReport:
    """Run full research pipeline for a query. Returns merged report."""
    if console is None:
        console = Console(legacy_windows=False)

    research_cfg = config.research
    if research_cfg is None:
        raise RuntimeError("Research config not loaded. Check settings.yaml.")

    secondary_dir: Path | None = None
    if config.defaults.secondary_output_enabled:
        secondary_dir = config.defaults.secondary_output_dir

    cache_key = make_cache_key(query)

    # Cache check
    if not no_cache:
        cached = cache_get(research_cfg.cache_dir, cache_key, research_cfg.cache_ttl_days)
        if cached is not None:
            console.print(f"\n[dim]Research cache hit (key: {cache_key})[/dim]")
            saved_paths = save_research_to_file(cached, output_dir, from_cache=True, secondary_dir=secondary_dir)
            print_research_summary(cached, saved_paths[0], from_cache=True, console=console)
            _print_research_paths(console, saved_paths, secondary_dir)
            if output_format == "json":
                import dataclasses
                import json
                import sys

                print(json.dumps(dataclasses.asdict(cached), indent=2, default=str), file=sys.stdout)
            return cached

    # Build providers
    providers = build_research_providers(config, deep=deep, models_filter=models_filter)
    if not providers:
        raise RuntimeError(
            "No research providers available. Check API keys for "
            "PERPLEXITY_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY."
        )

    # Run in parallel with live display
    results = await run_research_with_display(providers, query, console=console)

    # Merge
    report = merge_results(query, results, cache_key=cache_key)

    # Summarize (async LLM call)
    report = await summarize_report(report, research_cfg, config.models)

    # Cache write
    if not no_cache:
        cache_put(research_cfg.cache_dir, cache_key, report)

    # Output
    saved_paths = save_research_to_file(report, output_dir, from_cache=False, secondary_dir=secondary_dir)
    print_research_summary(report, saved_paths[0], from_cache=False, console=console)
    _print_research_paths(console, saved_paths, secondary_dir)

    if output_format == "json":
        import dataclasses
        import json
        import sys

        print(json.dumps(dataclasses.asdict(report), indent=2, default=str), file=sys.stdout)

    return report
