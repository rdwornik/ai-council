"""Headless research retrieval — no console, no cache, no file writes.

``run_research()`` (runner.py) is the CLI's research path and is coupled to a Rich console,
the on-disk cache, the file writers, and stdout JSON. The #18 crux check needs the retrieval
core alone, so this module reuses ``build_research_providers`` / ``ResearchProvider.research``
/ ``merge_results`` behind its own fanout and touches none of that machinery.

Why the fanout is duplicated rather than shared with ``run_research_with_display``: that
function's per-provider coroutine closes over four mutable dicts which its Rich ``Live``
loop reads on every tick, so extracting a display-free core from it means inverting those
closures into callbacks — churn in the live UI path for zero functional gain. The exception
-> ResearchResult conversion IS shared (``_error_result`` is imported, not re-implemented),
so only the two-branch try/except is duplicated. Result: zero lines change in display.py,
runner.py, or cli.py.

ADR-08 note: ``min_successful=1`` here, NOT ``research_cfg.min_successful_providers``. The
exit-code-3 alarm is enforced above ``run_research`` in cli.py, so a headless caller inherits
no exit semantics — correct, because a crux retrieval shortfall must never change a debate's
exit code. Success counts ride on CruxArtifact instead.
"""

import asyncio
import logging

from ai_council.research.display import _error_result
from ai_council.research.merger import merge_results
from ai_council.research.models import MergedResearchReport, ResearchResult
from ai_council.research.provider import ResearchProvider, ResearchProviderError
from ai_council.research.runner import build_research_providers
from config.config_loader import AppConfig

logger = logging.getLogger(__name__)


async def _run_one(provider: ResearchProvider, query: str) -> ResearchResult:
    """Run one provider, converting any failure into an error ResearchResult."""
    try:
        return await provider.research(query)
    except ResearchProviderError as exc:
        logger.warning("Crux research provider %s failed: %s", provider.name(), exc)
        return _error_result(provider, query, str(exc))
    except Exception as exc:  # noqa: BLE001 - a provider must never abort the debate
        logger.warning("Crux research provider %s unexpected error: %s", provider.name(), exc)
        return _error_result(provider, query, str(exc))


async def run_research_headless(
    query: str,
    config: AppConfig,
    *,
    provider_names: list[str] | None = None,
    budget_sec: float | None = None,
) -> MergedResearchReport | None:
    """Retrieve and merge research with no UI, cache, or file I/O.

    Args:
        query: The claim to check.
        config: App config (supplies research provider definitions + API key env names).
        provider_names: Narrow panel subset. None → whatever the default panel builds.
        budget_sec: Hard wall-clock cap. Exceeded → None (caller degrades).

    Returns:
        The merged report, or None when no provider could be built or the budget expired.
        Never raises on provider failure.
    """
    providers = build_research_providers(config, deep=False, models_filter=provider_names)
    if not providers:
        logger.info("Crux check: no research providers available (keys or config missing)")
        return None

    async def _gather() -> list[ResearchResult]:
        return await asyncio.gather(*(_run_one(p, query) for p in providers))

    try:
        results = await asyncio.wait_for(_gather(), timeout=budget_sec)
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning("Crux check: retrieval exceeded budget of %ss", budget_sec)
        return None

    # min_successful=1 per ruling 3 — a one-provider crux panel must not read as degraded.
    return merge_results(
        query,
        results,
        cache_key="",
        selected_panel=[p.name() for p in providers],
        min_successful=1,
    )
