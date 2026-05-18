"""File-based research result cache.

Cache location: ~/.ai-council/research_cache/ (configurable via settings.yaml)
Cache key: first 16 chars of SHA256(normalized_query)
TTL: configurable, default 7 days

Files per cache entry:
  {key}_report.md   — merged full report (markdown)
  {key}_summary.txt — 2500-token summary
  {key}_meta.json   — metadata (query, timestamp, providers, cost, ttl)
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_council.research.models import MergedResearchReport, ResearchResult

logger = logging.getLogger(__name__)


def _meta_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}_meta.json"


def _report_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}_report.md"


def _summary_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}_summary.txt"


def cache_get(cache_dir: Path, key: str, ttl_days: int) -> MergedResearchReport | None:
    """Return cached report if it exists and is within TTL, else None."""
    meta_file = _meta_path(cache_dir, key)
    if not meta_file.exists():
        return None

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("cache: failed to read meta for key %s: %s", key, exc)
        return None

    # Check TTL
    try:
        cached_at = datetime.fromisoformat(meta["cached_at"])
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        expires_at = cached_at + timedelta(days=ttl_days)
        now = datetime.now(timezone.utc)
        if now > expires_at:
            logger.debug("cache: key %s expired (cached %s, ttl %dd)", key, cached_at.isoformat(), ttl_days)
            return None
    except (KeyError, ValueError) as exc:
        logger.debug("cache: bad timestamp in meta for key %s: %s", key, exc)
        return None

    report_file = _report_path(cache_dir, key)
    summary_file = _summary_path(cache_dir, key)
    if not report_file.exists():
        return None

    merged_report = report_file.read_text(encoding="utf-8")
    summary = summary_file.read_text(encoding="utf-8") if summary_file.exists() else merged_report

    # Reconstruct minimal ResearchResult list from meta. Per-provider error is
    # preserved so degraded cache hits stay degraded (ADR-08).
    results: list[ResearchResult] = []
    for p in meta.get("providers", []):
        err = p.get("error")
        results.append(ResearchResult(
            provider=p.get("name", ""),
            query=meta.get("query", ""),
            content="" if err else "(loaded from cache)",
            token_count=p.get("token_count", 0),
            cost_usd=p.get("cost_usd", 0.0),
            duration_sec=p.get("duration_sec", 0.0),
            timestamp=meta.get("cached_at", ""),
            error=err,
        ))

    return MergedResearchReport(
        query=meta.get("query", ""),
        results=results,
        merged_report=merged_report,
        summary_2500=summary,
        total_sources=meta.get("total_sources", 0),
        total_cost_usd=meta.get("total_cost_usd", 0.0),
        total_duration_sec=meta.get("total_duration_sec", 0.0),
        cache_key=key,
        degraded=bool(meta.get("degraded", False)),
        failed_count=int(meta.get("failed_count", 0)),
    )


def cache_put(cache_dir: Path, key: str, report: MergedResearchReport) -> None:
    """Write report to cache. Creates cache_dir if needed. Silently skips on error."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        _report_path(cache_dir, key).write_text(report.merged_report, encoding="utf-8")
        _summary_path(cache_dir, key).write_text(report.summary_2500, encoding="utf-8")

        meta: dict = {
            "query": report.query,
            "cache_key": key,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "total_sources": report.total_sources,
            "total_cost_usd": report.total_cost_usd,
            "total_duration_sec": report.total_duration_sec,
            "degraded": report.degraded,
            "failed_count": report.failed_count,
            "providers": [
                {
                    "name": r.provider,
                    "token_count": r.token_count,
                    "cost_usd": r.cost_usd,
                    "duration_sec": r.duration_sec,
                    "error": r.error,
                }
                for r in report.results
            ],
        }
        _meta_path(cache_dir, key).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.debug("cache: saved key %s to %s", key, cache_dir)
    except Exception as exc:
        logger.warning("cache: failed to save key %s: %s", key, exc)


def cache_invalidate(cache_dir: Path, key: str) -> bool:
    """Delete all files for a cache key. Returns True if anything was deleted."""
    deleted = False
    for path in [_meta_path(cache_dir, key), _report_path(cache_dir, key), _summary_path(cache_dir, key)]:
        if path.exists():
            path.unlink()
            deleted = True
    return deleted
