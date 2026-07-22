"""Boost stage — the Council's input stage (ADR-11 boost→decide chain, Unit 2 P1).

Takes a raw, template-less, methodology-naive question and emits a well-formed,
type-classified brief that the existing debate stage consumes UNCHANGED
(file-in / file-out, stateless; owner ruled C).

Pipeline: classify → decompose → reformulate → emit.

Gate posture is hybrid (ruled): HARD on deterministic structural checks, ADVISORY
on any LLM judgement. Concretely:

- The LLM classifies (decision/research/hybrid) via the cheap-single-call shape
  proven by ``detect_mode``; a deterministic heuristic validates the guess (R3-as-
  validation) and is the fallback source when the LLM leg is unavailable.
- The LLM's hybrid decomposition must pass a HARD verbatim-token gate before any
  of its text enters a brief: a part containing content words the caller did not
  write is rejected outright (the FR-B5 confabulation guard, enforced in code).
- Reformulation is deterministic scaffolding: a brief's body is only ever caller
  text plus the fixed template constants below. Information gaps become advisory
  ``[BOOST-GAP]`` annotations inside the brief — never invented content, and never
  a question back to the caller (an interactive clarify-loop would reopen ADR-11
  decision 1; deferred rider).
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import frontmatter

from ai_council.mode_detector import _pick_cheapest
from ai_council.providers.base import AIProvider
from config.config_loader import AppConfig, BoostConfig, resolve_mode

logger = logging.getLogger(__name__)

# Marker for advisory gap annotations inside an emitted brief. The panel reads
# these; the boost NEVER fills a gap with invented content (FR-B5).
GAP_MARKER = "[BOOST-GAP]"

# The boost's coarse commission types (R1: hybrid-as-composition).
CLASSIFICATIONS = frozenset({"decision", "research", "hybrid"})

# ---------------------------------------------------------------------------
# Deterministic classification / gap heuristics (R3-as-validation; advisory)
# ---------------------------------------------------------------------------

_RESEARCH_SIGNALS = (
    "what does", "what do ", "what are the current", "current approaches",
    "current practice", "state of the art", "survey", "landscape", "literature",
    "what is known", "how do production", "best practices",
)
_DECISION_SIGNALS = (
    "should ", "which ", " vs ", " versus ", "best way", "choose", "decide",
    "is it better", "recommend", "how good", "review ", "assess", "evaluate",
)
_CONSTRAINT_KEYWORDS = (
    "must ", "must not", "cannot", "can't", "budget", "deadline", "within ",
    "no more than", "at most", "at least", "constraint", "require",
)

# Function words excluded from the verbatim gate's content-token comparison, so a
# quoted fragment is not rejected for its glue words alone.
_FUNCTION_WORDS = frozenset(
    "a an and are as at be but by can could do does for from how i in is it its "
    "of on or our should that the their them these they this to we what when "
    "which who why will with you your".split()
)

# ---------------------------------------------------------------------------
# Emitted-brief scaffold — ALL text the boost may add to a brief lives in the
# module-level constants below. That is the confabulation-guard architecture:
# a brief is caller text + these constants, nothing else.
# ---------------------------------------------------------------------------

ADVISORIES_HEADER = "### Boost Advisories"

CLASSIFICATION_NOTE = "classification: {label} ({source})"
HEURISTIC_DISAGREE_NOTE = (
    "advisory: the deterministic heuristic disagrees with the classifier "
    "(llm={llm}, heuristic={heur}); using the classifier's label"
)
DROPPED_KEY_NOTE = "caller frontmatter key '{key}' is not a council key; dropped"
DROPPED_MODE_NOTE = "caller-supplied mode '{mode}' does not resolve; dropped"
DEGRADED_DECOMPOSE_NOTE = (
    "decomposition degraded: {reason} — each leg carries the full raw question; "
    "the panel scopes its own leg"
)
VERBATIM_REJECT_REASON = (
    "the {leg} leg failed the verbatim gate (it contains words the caller did not write)"
)

# Classification source labels (mirror detect_mode's source_label contract).
SOURCE_AUTO = "auto-detected via {provider}"
SOURCE_FALLBACK_NO_PROVIDERS = "heuristic fallback — no providers available"
SOURCE_FALLBACK_UNEXPECTED = "heuristic fallback — unexpected response '{label}' from {provider}"
SOURCE_FALLBACK_TIMEOUT = "heuristic fallback — timeout after {timeout:.0f}s"
SOURCE_FALLBACK_ERROR = "heuristic fallback — {exc_type}"

GAP_NO_OPTIONS = (
    f"{GAP_MARKER} The caller named no options. The panel must enumerate the "
    "options before choosing; do not assume the caller has specific candidates in mind."
)
OPTIONS_INLINE_NOTE = (
    f"{GAP_MARKER} Options appear inline in the caller's raw text above; enumerate "
    "them from Current State verbatim — do not add candidates the caller did not name."
)
GAP_NO_CONSTRAINTS = (
    f"{GAP_MARKER} No constraints were stated. Treat constraints as unknown; do not invent any."
)
CONSTRAINTS_INLINE_NOTE = (
    f"{GAP_MARKER} Constraints appear inline in the caller's raw text above; extract "
    "them from Current State — do not invent others."
)
GAP_NO_FACETS = (
    f"{GAP_MARKER} The caller enumerated no facets. Treat the question as a single "
    "facet; do not invent sub-topics."
)
GAP_NO_SOURCE_RULES = (
    f"{GAP_MARKER} No recency window or source rules were stated. Prefer recent "
    "primary sources; do not assume a domain the caller did not name."
)

CALLER_TEXT_LABEL = "Raw question as supplied by the caller (verbatim):"

DECISION_BODY_TEMPLATE = """## Question: {headline}

### Current State
{caller_text_label}

{raw_text}

### Questions
1. **{item}**
{options_block}

### Constraints
{constraints_block}

{advisories_block}
"""

RESEARCH_BODY_TEMPLATE = """## Question: {headline}

### Background
{caller_text_label}

{raw_text}

### What to find out
1. {item}
{facets_block}

### Source rules
{source_rules_block}

### Output wanted
A survey answering the question above, shaped for the decision it will inform.
Surface unresolved questions and disconfirming evidence explicitly.

{advisories_block}
"""

LINK_RESEARCH_LEG = (
    "> Sub-commission 1 of {total} — research leg of a hybrid boost.\n"
    "> Feed order: the findings of THIS brief feed the decision leg: {counterpart}\n"
)
LINK_DECISION_LEG = (
    "> Sub-commission {index} of {total} — decision leg of a hybrid boost.\n"
    "> Feed order: this brief consumes the findings of the research leg: {counterpart}\n"
)

_DECISION_SECTIONS = ("### Current State", "### Questions", "### Constraints")
_RESEARCH_SECTIONS = ("### Background", "### What to find out", "### Source rules", "### Output wanted")


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


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

def heuristic_classification(text: str) -> str:
    """Deterministic classifier — R3-as-validation of the LLM guess, and the
    fallback source when the LLM leg is unavailable."""
    padded = f" {text.lower()} "
    research_hit = any(signal in padded for signal in _RESEARCH_SIGNALS)
    decision_hit = any(signal in padded for signal in _DECISION_SIGNALS)
    if research_hit and decision_hit:
        return "hybrid"
    if research_hit:
        return "research"
    return "decision"


async def _classify(
    text: str, providers: dict[str, AIProvider], boost_cfg: BoostConfig
) -> tuple[str, str, bool]:
    """Classify via the cheapest provider (detect_mode shape); heuristic fallback.

    Returns (label, source_label, degraded).
    """
    provider = _pick_cheapest(providers)
    if provider is None:
        return heuristic_classification(text), SOURCE_FALLBACK_NO_PROVIDERS, True

    prompt = boost_cfg.classify_prompt.format(question=text)
    try:
        response = await asyncio.wait_for(
            provider.generate(prompt, round_number=1),
            timeout=boost_cfg.timeout_sec,
        )
        label = response.content.strip().lower()
        if label in CLASSIFICATIONS:
            return label, SOURCE_AUTO.format(provider=provider.name()), False
        logger.warning("Boost classifier returned unknown label '%s'; heuristic fallback", label)
        return (
            heuristic_classification(text),
            SOURCE_FALLBACK_UNEXPECTED.format(label=label, provider=provider.name()),
            True,
        )
    except asyncio.TimeoutError:
        logger.warning("Boost classification timed out after %.0fs", boost_cfg.timeout_sec)
        return (
            heuristic_classification(text),
            SOURCE_FALLBACK_TIMEOUT.format(timeout=boost_cfg.timeout_sec),
            True,
        )
    except Exception as exc:
        logger.warning("Boost classification failed (%s); heuristic fallback", exc)
        return (
            heuristic_classification(text),
            SOURCE_FALLBACK_ERROR.format(exc_type=type(exc).__name__),
            True,
        )


# ---------------------------------------------------------------------------
# Decompose (hybrid) — LLM split points behind a HARD verbatim gate
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _is_verbatim(part: str, raw: str) -> bool:
    """HARD gate: every content token of `part` must appear in the caller's raw
    text. A decompose answer that adds facts, names, or options fails here."""
    content = _tokens(part) - _FUNCTION_WORDS
    return bool(content) and content <= _tokens(raw)


def _parse_decompose(content: str) -> tuple[str, str] | None:
    research_m = re.search(r"^\s*RESEARCH:\s*(.+)$", content, re.IGNORECASE | re.MULTILINE)
    decision_m = re.search(r"^\s*DECISION:\s*(.+)$", content, re.IGNORECASE | re.MULTILINE)
    if not research_m or not decision_m:
        return None
    return research_m.group(1).strip(), decision_m.group(1).strip()


async def _decompose(
    text: str, providers: dict[str, AIProvider], boost_cfg: BoostConfig
) -> tuple[tuple[str, str] | None, str | None]:
    """Split a hybrid question into (research_part, decision_part).

    Returns (parts, None) on success or (None, reason) on any failure — the
    caller then falls back to full-text legs (degraded-but-complete, never
    confabulated).
    """
    provider = _pick_cheapest(providers)
    if provider is None:
        return None, SOURCE_FALLBACK_NO_PROVIDERS

    prompt = boost_cfg.decompose_prompt.format(question=text)
    try:
        response = await asyncio.wait_for(
            provider.generate(prompt, round_number=1),
            timeout=boost_cfg.timeout_sec,
        )
    except asyncio.TimeoutError:
        return None, SOURCE_FALLBACK_TIMEOUT.format(timeout=boost_cfg.timeout_sec)
    except Exception as exc:
        return None, SOURCE_FALLBACK_ERROR.format(exc_type=type(exc).__name__)

    parts = _parse_decompose(response.content)
    if parts is None:
        return None, "malformed decompose response"

    for leg, part in (("research", parts[0]), ("decision", parts[1])):
        if not _is_verbatim(part, text):
            logger.warning("Boost decompose %s leg failed the verbatim gate; rejected", leg)
            return None, VERBATIM_REJECT_REASON.format(leg=leg)
    return parts, None


# ---------------------------------------------------------------------------
# Reformulate — deterministic scaffolding only
# ---------------------------------------------------------------------------

def derive_slug(text: str) -> str:
    """Filename slug from the first words of a raw question."""
    words = re.findall(r"[a-z0-9]+", text.lower())[:6]
    return "-".join(words) or "question"


def _headline(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip()
    if len(first_line) > 160:
        return first_line[:157].rstrip() + "..."
    return first_line


def _option_lines(text: str) -> list[str]:
    """Caller-written option-like lines (bullets, numbered, 'A:'), verbatim."""
    return [
        line.strip() for line in text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+[.)]|[A-Z]:)\s+", line)
    ]


def _options_block(text: str) -> str:
    lines = _option_lines(text)
    if lines:
        return "\n".join(f"   {line}" for line in lines)
    padded = f" {text.lower()} "
    if " vs " in padded or " versus " in padded or " or " in padded:
        return f"   {OPTIONS_INLINE_NOTE}"
    return f"   {GAP_NO_OPTIONS}"


def _constraints_block(text: str) -> str:
    padded = f" {text.lower()} "
    if any(keyword in padded for keyword in _CONSTRAINT_KEYWORDS):
        return f"- {CONSTRAINTS_INLINE_NOTE}"
    return f"- {GAP_NO_CONSTRAINTS}"


def _facets_block(text: str) -> str:
    lines = _option_lines(text)
    if lines:
        return "\n".join(f"   {line}" for line in lines)
    return f"   {GAP_NO_FACETS}"


def _source_rules_block(text: str) -> str:
    padded = f" {text.lower()} "
    has_recency = (
        re.search(r"\b20\d\d\b", text) is not None
        or "recent" in padded or " last " in padded or " since " in padded
    )
    if has_recency:
        return f"- {CONSTRAINTS_INLINE_NOTE}"
    return f"- {GAP_NO_SOURCE_RULES}"


def _advisories_block(advisories: list[str]) -> str:
    lines = "\n".join(f"- {advisory}" for advisory in advisories)
    return f"{ADVISORIES_HEADER}\n{lines}"


def _decision_body(item_text: str, raw_text: str, advisories: list[str]) -> str:
    return DECISION_BODY_TEMPLATE.format(
        headline=_headline(item_text),
        caller_text_label=CALLER_TEXT_LABEL,
        raw_text=raw_text,
        item=_headline(item_text),
        options_block=_options_block(item_text),
        constraints_block=_constraints_block(raw_text),
        advisories_block=_advisories_block(advisories),
    )


def _research_body(item_text: str, raw_text: str, advisories: list[str]) -> str:
    return RESEARCH_BODY_TEMPLATE.format(
        headline=_headline(item_text),
        caller_text_label=CALLER_TEXT_LABEL,
        raw_text=raw_text,
        item=_headline(item_text),
        facets_block=_facets_block(item_text),
        source_rules_block=_source_rules_block(raw_text),
        advisories_block=_advisories_block(advisories),
    )


# ---------------------------------------------------------------------------
# Emit — with the boost's own HARD structural self-check
# ---------------------------------------------------------------------------

def _filter_caller_metadata(
    caller_metadata: dict | None, config: AppConfig
) -> tuple[dict, list[str]]:
    """Keep only council frontmatter keys; drop the rest with an advisory."""
    if not caller_metadata:
        return {}, []
    valid = set(config.inbox.council_frontmatter_keys)
    kept = {k: v for k, v in caller_metadata.items() if k in valid}
    advisories = [
        DROPPED_KEY_NOTE.format(key=k) for k in sorted(set(caller_metadata) - valid)
    ]
    if "mode" in kept:
        try:
            resolve_mode(str(kept["mode"]), config.modes)
        except ValueError:
            advisories.append(DROPPED_MODE_NOTE.format(mode=kept.pop("mode")))
    return kept, advisories


def _validate_brief(
    content: str, config: AppConfig, required_sections: tuple[str, ...]
) -> None:
    """The boost's OWN output gate (HARD). Nothing downstream parses the body —
    only frontmatter is read by inbox.parse_file — so a malformed body would
    reach the panel unchecked. Fails loud on any structural violation."""
    post = frontmatter.loads(content)
    if not post.metadata:
        raise BoostError("emitted brief has no frontmatter")
    extra = set(post.metadata) - set(config.inbox.council_frontmatter_keys)
    if extra:
        raise BoostError(f"emitted brief carries non-council frontmatter keys: {sorted(extra)}")
    if "mode" in post.metadata:
        try:
            resolve_mode(str(post.metadata["mode"]), config.modes)
        except ValueError as exc:
            raise BoostError(f"emitted brief mode does not resolve: {exc}") from exc
    if "## Question:" not in post.content:
        raise BoostError("emitted brief is missing its '## Question:' headline")
    for section in required_sections:
        if section not in post.content:
            raise BoostError(f"emitted brief is missing required section {section!r}")


def _emit_brief(
    body: str,
    metadata: dict,
    path: Path,
    config: AppConfig,
    required_sections: tuple[str, ...],
) -> None:
    content = frontmatter.dumps(frontmatter.Post(body, **metadata))
    _validate_brief(content, config, required_sections)
    path.write_text(content + "\n", encoding="utf-8")


def _brief_filename(slug: str, timestamp: str, leg: str | None = None) -> str:
    base = f"council-brief-{timestamp}-{slug}"
    return f"{base}-{leg}.md" if leg else f"{base}.md"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def boost_question(
    raw_text: str,
    *,
    providers: dict[str, AIProvider],
    config: AppConfig,
    out_dir: Path,
    slug: str,
    caller_metadata: dict | None = None,
) -> BoostResult:
    """Boost a raw question into one or more emitted Council briefs.

    Stateless, file-out only. Raises BoostError on unusable input or a violated
    structural invariant (exit 1); a degraded-but-complete run (classifier or
    decompose fell back) is reported via ``BoostResult.degraded`` (exit 3).
    """
    text = (raw_text or "").strip()
    if not text:
        raise BoostError("Empty question — nothing to boost.")
    if config.boost is None:
        raise BoostError("settings.yaml has no `boost:` section — the boost stage is not configured.")

    caller_meta, advisories = _filter_caller_metadata(caller_metadata, config)

    label, source, degraded = await _classify(text, providers, config.boost)
    heuristic_label = heuristic_classification(text)
    if not degraded and heuristic_label != label:
        advisories.append(
            HEURISTIC_DISAGREE_NOTE.format(llm=label, heur=heuristic_label)
        )
    advisories.insert(0, CLASSIFICATION_NOTE.format(label=label, source=source))

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    briefs: list[Path] = []

    if label == "hybrid":
        parts, reason = await _decompose(text, providers, config.boost)
        if parts is None:
            degraded = True
            advisories.append(DEGRADED_DECOMPOSE_NOTE.format(reason=reason))
            research_text = decision_text = text
        else:
            research_text, decision_text = parts

        research_name = _brief_filename(slug, timestamp, "1-research")
        decision_name = _brief_filename(slug, timestamp, "2-decision")

        research_body = (
            LINK_RESEARCH_LEG.format(total=2, counterpart=decision_name)
            + "\n"
            + _research_body(research_text, text, advisories)
        )
        decision_body = (
            LINK_DECISION_LEG.format(index=2, total=2, counterpart=research_name)
            + "\n"
            + _decision_body(decision_text, text, advisories)
        )

        research_meta = {"rounds": config.defaults.rounds, **caller_meta, "mode": "research"}
        decision_meta = {"rounds": config.defaults.rounds, **caller_meta}

        research_path = out_dir / research_name
        decision_path = out_dir / decision_name
        _emit_brief(research_body, research_meta, research_path, config, _RESEARCH_SECTIONS)
        _emit_brief(decision_body, decision_meta, decision_path, config, _DECISION_SECTIONS)
        briefs = [research_path, decision_path]

    elif label == "research":
        metadata = {"rounds": config.defaults.rounds, "mode": "research", **caller_meta}
        path = out_dir / _brief_filename(slug, timestamp)
        _emit_brief(_research_body(text, text, advisories), metadata, path, config, _RESEARCH_SECTIONS)
        briefs = [path]

    else:  # decision
        metadata = {"rounds": config.defaults.rounds, **caller_meta}
        path = out_dir / _brief_filename(slug, timestamp)
        _emit_brief(_decision_body(text, text, advisories), metadata, path, config, _DECISION_SECTIONS)
        briefs = [path]

    logger.info(
        "Boost emitted %d brief(s): classification=%s (%s)%s",
        len(briefs), label, source, " [degraded]" if degraded else "",
    )
    return BoostResult(
        briefs=briefs,
        classification=label,
        source_label=source,
        degraded=degraded,
        advisories=advisories,
    )
