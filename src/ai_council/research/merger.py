"""Merge multiple ResearchResults into a single structured report.

Two-phase process:
1. merge_results() — deduplicate sources, concatenate provider content into
   a structured markdown document.
2. summarize_report() — call cheapest available model to compress merged
   content to ~2500 tokens.
"""

import hashlib
import logging
import os

from openai import AsyncOpenAI

from ai_council.research.models import MergedResearchReport, ResearchResult, Source
from config.config_loader import ResearchConfig

logger = logging.getLogger(__name__)

_SUMMARIZER_PROMPT = """\
You are a research synthesis expert. You have received research reports from multiple
independent sources on the same topic. Your task is to synthesize them into one
comprehensive, well-structured report.

Rules:
- Merge overlapping information; do not repeat the same fact multiple times
- Highlight where sources agree and where they diverge
- Keep all unique insights from each source
- Preserve citations/references where meaningful
- Use clear markdown headings
- Target length: ~{max_tokens} tokens (approximately {target_words} words)

Structure your output as:
## Executive Summary
2-3 sentences covering the core answer.

## Key Findings
Numbered list of the most important findings, with source attribution.

## Detailed Analysis
Organized by subtopic. Merge content from all providers.

## Competing Perspectives
Where sources disagreed or offered different emphases.

## Sources
Deduplicated list of all cited URLs.

---
RESEARCH CONTENT TO SYNTHESIZE:

{content}
"""


def _deduplicate_sources(all_sources: list[Source]) -> list[Source]:
    """Deduplicate sources by URL."""
    seen: set[str] = set()
    unique: list[Source] = []
    for src in all_sources:
        if src.url not in seen:
            seen.add(src.url)
            unique.append(src)
    return unique


def _build_merged_document(results: list[ResearchResult]) -> str:
    """Concatenate provider results into a structured markdown document."""
    parts: list[str] = []
    for result in results:
        if not result.content or result.error:
            continue
        provider_label = result.provider.upper()
        parts.append(f"## Report from {provider_label} ({result.model if hasattr(result, 'model') else result.provider})\n")
        parts.append(result.content.strip())
        if result.sources:
            parts.append("\n### Sources from this provider")
            for src in result.sources:
                parts.append(f"- [{src.title}]({src.url})")
        parts.append("\n---\n")
    return "\n".join(parts)


def merge_results(
    query: str,
    results: list[ResearchResult],
    cache_key: str = "",
    selected_panel: list[str] | None = None,
    min_successful: int | None = None,
) -> MergedResearchReport:
    """Merge research results into a single report (no LLM call).

    selected_panel: provider names selected for this invocation (post --models filter).
    Used as the denominator for degradation: a provider that dropped at build time
    (missing API key) is not in `results` but IS in `selected_panel`, so it counts
    as a failure. See ADR-08.
    """
    successful = [r for r in results if not r.error and r.content]

    all_sources: list[Source] = []
    for r in results:
        all_sources.extend(r.sources)
    unique_sources = _deduplicate_sources(all_sources)

    merged = _build_merged_document(successful)

    total_cost = sum(r.cost_usd for r in results)
    total_duration = max((r.duration_sec for r in results), default=0.0)

    if selected_panel is not None:
        failed_count = max(0, len(selected_panel) - len(successful))
    else:
        failed_count = sum(1 for r in results if r.error or not r.content)

    degraded = (
        min_successful is not None and len(successful) < min_successful
    )

    return MergedResearchReport(
        query=query,
        results=results,
        merged_report=merged,
        summary_2500=merged,  # placeholder; replaced by summarize_report()
        total_sources=len(unique_sources),
        total_cost_usd=total_cost,
        total_duration_sec=total_duration,
        cache_key=cache_key,
        degraded=degraded,
        failed_count=failed_count,
    )


async def summarize_report(
    report: MergedResearchReport,
    research_cfg: ResearchConfig,
    models_cfg: dict,
) -> MergedResearchReport:
    """Compress merged_report to ~2500 tokens using cheapest available model.

    Falls back gracefully: if summarizer model not available or API call fails,
    sets summary_2500 to first 2500 words of merged_report (truncation fallback).
    """
    if not report.merged_report.strip():
        return report

    summary_model_name = research_cfg.summary_model
    model_cfg = models_cfg.get(summary_model_name)
    if model_cfg is None:
        logger.warning("Summarizer model '%s' not in models config — using truncation fallback", summary_model_name)
        report.summary_2500 = _truncation_fallback(report.merged_report, research_cfg.summary_max_tokens)
        return report

    api_key = os.environ.get(model_cfg.api_key_env, "").strip()
    if not api_key:
        logger.warning("No API key for summarizer model '%s' — using truncation fallback", summary_model_name)
        report.summary_2500 = _truncation_fallback(report.merged_report, research_cfg.summary_max_tokens)
        return report

    target_words = int(research_cfg.summary_max_tokens * 0.75)  # rough token→word ratio
    prompt = _SUMMARIZER_PROMPT.format(
        max_tokens=research_cfg.summary_max_tokens,
        target_words=target_words,
        content=report.merged_report,
    )

    try:
        base_url = getattr(model_cfg, "base_url", None)
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = AsyncOpenAI(**client_kwargs)
        response = await client.chat.completions.create(
            model=model_cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=research_cfg.summary_max_tokens,
        )
        summary = response.choices[0].message.content or ""
        if summary.strip():
            report.summary_2500 = summary.strip()
        else:
            report.summary_2500 = _truncation_fallback(report.merged_report, research_cfg.summary_max_tokens)
    except Exception as exc:
        logger.warning("Summarizer API call failed: %s — using truncation fallback", exc)
        report.summary_2500 = _truncation_fallback(report.merged_report, research_cfg.summary_max_tokens)

    return report


def _truncation_fallback(text: str, max_tokens: int) -> str:
    """Return first ~max_tokens*0.75 words of text as a fallback summary."""
    words = text.split()
    limit = int(max_tokens * 0.75)
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + "\n\n*(truncated — summarizer unavailable)*"


def make_cache_key(query: str) -> str:
    """SHA256 hash of normalized query for use as cache filename prefix."""
    normalized = query.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
