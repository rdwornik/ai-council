"""Research mode orchestrator.

Wires together: provider selection → cache check → parallel research
→ merge → summarize → cache write → output.
"""

import logging
import os
from pathlib import Path

from rich.console import Console

from config.config_loader import AppConfig, ResearchConfig
from src.research.cache import cache_get, cache_put
from src.research.display import run_research_with_display
from src.research.merger import make_cache_key, merge_results, summarize_report
from src.research.models import MergedResearchReport
from src.research.output import print_research_summary, save_research_to_file
from src.research.provider import ResearchProvider

logger = logging.getLogger(__name__)


def build_research_providers(
    config: AppConfig,
    deep: bool = False,
) -> list[ResearchProvider]:
    """Instantiate available research providers based on config and --deep flag.

    Skips providers with missing API keys; logs a warning for each.
    """
    research_cfg = config.research
    if research_cfg is None:
        logger.warning("No research config in settings.yaml")
        return []

    provider_names = research_cfg.deep_providers if deep else research_cfg.default_providers
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
        from src.research.providers.perplexity import PerplexityProvider
        return PerplexityProvider(
            api_key=api_key,
            model=p_cfg.model,
            timeout_sec=p_cfg.timeout_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    elif name == "openai_mini":
        from src.research.providers.openai_mini_research import OpenAIMiniResearchProvider
        return OpenAIMiniResearchProvider(
            api_key=api_key,
            model=p_cfg.model,
            timeout_sec=p_cfg.timeout_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    elif name == "openai_deep":
        from src.research.providers.openai_deep_research import OpenAIDeepResearchProvider
        return OpenAIDeepResearchProvider(
            api_key=api_key,
            model=p_cfg.model,
            timeout_sec=p_cfg.timeout_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    elif name == "gemini":
        from src.research.providers.gemini_research import GeminiResearchProvider
        return GeminiResearchProvider(
            api_key=api_key,
            agent=p_cfg.model,
            timeout_sec=p_cfg.timeout_sec,
            poll_interval_sec=p_cfg.poll_interval_sec,
            cost_per_1m_input=p_cfg.cost_per_1m_input,
            cost_per_1m_output=p_cfg.cost_per_1m_output,
        )
    else:
        raise ValueError(f"Unknown research provider name: {name}")


async def run_research(
    query: str,
    config: AppConfig,
    output_dir: Path,
    deep: bool = False,
    no_cache: bool = False,
    console: Console | None = None,
    output_format: str = "text",
) -> MergedResearchReport:
    """Run full research pipeline for a query. Returns merged report."""
    if console is None:
        console = Console(legacy_windows=False)

    research_cfg = config.research
    if research_cfg is None:
        raise RuntimeError("Research config not loaded. Check settings.yaml.")

    cache_key = make_cache_key(query)

    # Cache check
    if not no_cache:
        cached = cache_get(research_cfg.cache_dir, cache_key, research_cfg.cache_ttl_days)
        if cached is not None:
            console.print(f"\n[dim]Research cache hit (key: {cache_key})[/dim]")
            file_path = save_research_to_file(cached, output_dir, from_cache=True)
            print_research_summary(cached, file_path, from_cache=True, console=console)
            if output_format == "json":
                import dataclasses
                import json
                import sys

                print(json.dumps(dataclasses.asdict(cached), indent=2, default=str), file=sys.stdout)
            return cached

    # Build providers
    providers = build_research_providers(config, deep=deep)
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
    file_path = save_research_to_file(report, output_dir, from_cache=False)
    print_research_summary(report, file_path, from_cache=False, console=console)

    if output_format == "json":
        import dataclasses
        import json
        import sys

        print(json.dumps(dataclasses.asdict(report), indent=2, default=str), file=sys.stdout)

    return report
