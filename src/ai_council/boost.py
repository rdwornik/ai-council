"""Boost stage — the Council's input stage (ADR-11 boost→decide chain, Unit 2 P1).

Takes a raw, template-less, methodology-naive question and emits a well-formed,
type-classified brief that the existing debate stage consumes UNCHANGED
(file-in / file-out, stateless; owner ruled C).

Pipeline: classify → decompose → reformulate → emit.

Gate posture is hybrid (ruled): HARD on deterministic structural checks, ADVISORY
on any LLM judgement. Information gaps become advisory annotations inside the
emitted brief — never a question back to the caller (an interactive clarify-loop
would reopen ADR-11 decision 1; deferred rider).
"""

from dataclasses import dataclass, field
from pathlib import Path

from ai_council.providers.base import AIProvider
from config.config_loader import AppConfig

# Marker for advisory gap annotations inside an emitted brief. The panel reads
# these; the boost NEVER fills a gap with invented content (FR-B5).
GAP_MARKER = "[BOOST-GAP]"

# The boost's coarse commission types (R1: hybrid-as-composition).
CLASSIFICATIONS = frozenset({"decision", "research", "hybrid"})


class BoostError(Exception):
    """Unusable input or a violated structural invariant — exit 1 territory."""


@dataclass
class BoostResult:
    """Outcome of one boost run."""

    briefs: list[Path]
    classification: str  # decision | research | hybrid
    source_label: str    # how the classification was chosen (mirrors detect_mode)
    degraded: bool       # True → degraded-but-complete, ADR-08 exit 3
    advisories: list[str] = field(default_factory=list)


def heuristic_classification(text: str) -> str:
    """Deterministic classifier — R3-as-validation of the LLM guess, and the
    fallback source when the LLM leg is unavailable."""
    raise NotImplementedError("Unit 2 P1: not yet implemented")


async def boost_question(
    raw_text: str,
    *,
    providers: dict[str, AIProvider],
    config: AppConfig,
    out_dir: Path,
    slug: str,
    caller_metadata: dict | None = None,
) -> BoostResult:
    """Boost a raw question into one or more emitted Council briefs."""
    raise NotImplementedError("Unit 2 P1: not yet implemented")
