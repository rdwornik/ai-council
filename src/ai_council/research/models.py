"""Pure dataclasses for the research pipeline. No logic, no deps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    title: str
    url: str
    snippet: str | None = None


@dataclass
class ResearchResult:
    provider: str          # "perplexity", "openai_mini", "openai_deep", "gemini"
    query: str
    content: str           # Full research output markdown
    sources: list[Source] = field(default_factory=list)
    token_count: int = 0
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    timestamp: str = ""
    timed_out: bool = False
    error: str | None = None   # Set if provider failed or timed out


@dataclass
class MergedResearchReport:
    query: str
    results: list[ResearchResult]    # Individual provider results (including failed)
    merged_report: str               # Deduplicated, merged markdown
    summary_2500: str                # 2.5K token summary for future debate injection
    total_sources: int = 0
    total_cost_usd: float = 0.0
    total_duration_sec: float = 0.0
    cache_key: str = ""
