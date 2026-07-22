"""Unit 2 P1 acceptance contract for `council boost` (T1-T8) — frozen BEFORE implementation.

The boost is the Council's input stage (ADR-11 boost→decide chain, owner ruled C):
a raw, template-less, methodology-naive question in → a well-formed, type-classified
brief out, consumed by the existing debate stage UNCHANGED.

T5 is the keystone: the confabulation guard asserts the ABSENCE of invented
specifics in the emitted brief — a flag-presence check alone does not satisfy it.
"""

import re
from unittest.mock import AsyncMock

import frontmatter
import pytest
from click.testing import CliRunner

import ai_council.boost as boost_mod
import ai_council.cli as cli
from ai_council.boost import boost_question, heuristic_classification
from ai_council.inbox import parse_file
from ai_council.mode_detector import CLASSIFICATION_PROMPT
from ai_council.models import ModelResponse
from config.config_loader import load_config, resolve_mode
from tests.conftest import MockProvider

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# A decision-shaped question that deliberately names NO options (T5's gap case).
RAW_DECISION_NO_OPTIONS = "Should we migrate the billing system to a new database this quarter?"

# A research-shaped question: retrieval, not reasoning.
RAW_RESEARCH = "What does current practice say about zero-downtime schema migration in production systems?"

# A compound question whose research part feeds its decision part.
RAW_HYBRID = (
    "What are the current approaches to zero-downtime schema migration in production, "
    "and should we adopt one of them for our billing system?"
)

# A well-behaved decompose response: both parts quoted verbatim from RAW_HYBRID.
GOOD_DECOMPOSE = (
    "RESEARCH: What are the current approaches to zero-downtime schema migration in production\n"
    "DECISION: should we adopt one of them for our billing system"
)

# Structural vocabulary the boost may legitimately emit that is neither caller text
# nor a module-level scaffold constant: provider names, classification labels.
_EXTRA_ALLOWED = {
    "gemini", "openai", "claude", "grok", "deepseek", "perplexity",
    "decision", "research", "hybrid", "classification", "heuristic", "boost",
}


@pytest.fixture(scope="module")
def config():
    return load_config()


def _resp(content: str) -> ModelResponse:
    return ModelResponse(
        provider="gemini", model="mock-model", round_number=1,
        content=content, latency_sec=0.1, token_count=5,
    )


def _mock_llm(*contents: str) -> dict:
    """Providers dict whose single seat answers the given contents in sequence."""
    provider = MockProvider("gemini")
    provider.generate = AsyncMock(side_effect=[_resp(c) for c in contents])
    return {"gemini": provider}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _assert_no_invented_content(body: str, raw: str) -> None:
    """KEYSTONE helper: every content-bearing word of the emitted body must derive
    from the raw question, the boost's fixed module-level scaffold constants, or the
    small structural vocabulary — an LLM-injected prose line fails this."""
    allowed = _words(raw) | _EXTRA_ALLOWED
    for name in dir(boost_mod):
        value = getattr(boost_mod, name)
        if isinstance(value, str):
            allowed |= _words(value)
    invented = {
        w for w in (_words(body) - allowed)
        if len(w) > 3 and not w.isdigit()
    }
    assert not invented, f"invented content in emitted brief: {sorted(invented)}"


# ---------------------------------------------------------------------------
# T1 (FR-B1): raw template-less question → structurally valid brief
# ---------------------------------------------------------------------------

class TestT1StructurallyValidBrief:
    async def test_raw_question_produces_valid_brief(self, config, tmp_path):
        result = await boost_question(
            RAW_DECISION_NO_OPTIONS,
            providers=_mock_llm("decision"),
            config=config, out_dir=tmp_path, slug="t1",
        )
        assert len(result.briefs) == 1
        path = result.briefs[0]
        assert path.exists()
        post = frontmatter.load(str(path))
        assert post.metadata, "frontmatter must be present"
        for section in ("## Question:", "### Current State", "### Questions", "### Constraints"):
            assert section in post.content, f"missing section {section!r}"
        # The caller's raw text survives verbatim — the panel sees the real question.
        assert RAW_DECISION_NO_OPTIONS in post.content

    async def test_no_synthesizer_key_unless_caller_supplied(self, config, tmp_path):
        # The GUIDE's frontmatter table is stale (still names gemini); the boost must
        # not bake in ANY synthesizer default.
        result = await boost_question(
            RAW_DECISION_NO_OPTIONS,
            providers=_mock_llm("decision"),
            config=config, out_dir=tmp_path, slug="t1b",
        )
        post = frontmatter.load(str(result.briefs[0]))
        assert "synthesizer" not in post.metadata

    async def test_caller_frontmatter_passthrough(self, config, tmp_path):
        result = await boost_question(
            RAW_DECISION_NO_OPTIONS,
            providers=_mock_llm("decision"),
            config=config, out_dir=tmp_path, slug="t1c",
            caller_metadata={"synthesizer": "openai", "not-a-council-key": "x"},
        )
        post = frontmatter.load(str(result.briefs[0]))
        assert post.metadata["synthesizer"] == "openai"
        assert "not-a-council-key" not in post.metadata


# ---------------------------------------------------------------------------
# T2 (FR-B2): classification decision / research / hybrid
# ---------------------------------------------------------------------------

class TestT2Classification:
    @pytest.mark.parametrize(
        "label,raw,extra_responses",
        [
            ("decision", RAW_DECISION_NO_OPTIONS, ()),
            ("research", RAW_RESEARCH, ()),
            ("hybrid", RAW_HYBRID, (GOOD_DECOMPOSE,)),
        ],
    )
    async def test_llm_classification(self, config, tmp_path, label, raw, extra_responses):
        result = await boost_question(
            raw,
            providers=_mock_llm(label, *extra_responses),
            config=config, out_dir=tmp_path, slug=f"t2-{label}",
        )
        assert result.classification == label
        assert not result.degraded

    def test_heuristic_decision(self):
        assert heuristic_classification(RAW_DECISION_NO_OPTIONS) == "decision"

    def test_heuristic_research(self):
        assert heuristic_classification(RAW_RESEARCH) == "research"

    def test_heuristic_hybrid(self):
        assert heuristic_classification(RAW_HYBRID) == "hybrid"

    async def test_garbage_llm_response_falls_back_to_heuristic(self, config, tmp_path):
        result = await boost_question(
            RAW_DECISION_NO_OPTIONS,
            providers=_mock_llm("banana"),
            config=config, out_dir=tmp_path, slug="t2-garbage",
        )
        assert result.classification == "decision"  # heuristic verdict
        assert result.degraded
        assert "fallback" in result.source_label


# ---------------------------------------------------------------------------
# T3 (FR-B3, contract): existing entry path accepts the brief UNCHANGED
# ---------------------------------------------------------------------------

class TestT3ExistingEntryPathContract:
    async def test_parse_file_accepts_all_emitted_briefs(self, config, tmp_path):
        valid_keys = set(config.inbox.council_frontmatter_keys)
        results = [
            await boost_question(
                RAW_DECISION_NO_OPTIONS, providers=_mock_llm("decision"),
                config=config, out_dir=tmp_path, slug="t3-d",
            ),
            await boost_question(
                RAW_RESEARCH, providers=_mock_llm("research"),
                config=config, out_dir=tmp_path, slug="t3-r",
            ),
            await boost_question(
                RAW_HYBRID, providers=_mock_llm("hybrid", GOOD_DECOMPOSE),
                config=config, out_dir=tmp_path, slug="t3-h",
            ),
        ]
        briefs = [p for r in results for p in r.briefs]
        assert briefs
        for path in briefs:
            content, metadata = parse_file(path)  # the EXISTING entry path, unchanged
            assert content, f"{path.name}: empty body"
            assert set(metadata) <= valid_keys, (
                f"{path.name}: emitted non-council frontmatter keys {set(metadata) - valid_keys}"
            )
            if "mode" in metadata:
                # Any emitted mode must resolve through the existing resolver.
                resolve_mode(str(metadata["mode"]), config.modes)


# ---------------------------------------------------------------------------
# T4 (FR-B4): research via emitted sub-commission, NOT via detect_mode
# ---------------------------------------------------------------------------

class TestT4ResearchViaFrontmatter:
    async def test_research_mode_forced_in_frontmatter(self, config, tmp_path):
        result = await boost_question(
            RAW_RESEARCH,
            providers=_mock_llm("research"),
            config=config, out_dir=tmp_path, slug="t4",
        )
        post = frontmatter.load(str(result.briefs[0]))
        assert post.metadata.get("mode") == "research"
        assert resolve_mode("research", config.modes) == "research"

    def test_detect_mode_prompt_still_cannot_emit_research(self):
        # Pins that the boost is THE research classification path: detect_mode's
        # prompt must remain pick/ideas/judge-only (FR-B4, no detect_mode changes).
        assert "research" not in CLASSIFICATION_PROMPT


# ---------------------------------------------------------------------------
# T5 (FR-B5, KEYSTONE): confabulation guard — gaps flagged, never filled
# ---------------------------------------------------------------------------

class TestT5ConfabulationGuard:
    async def test_gap_flagged_and_not_filled(self, config, tmp_path):
        # RAW_DECISION_NO_OPTIONS names no candidate options: the brief must flag
        # the gap advisorily AND must not invent a fact to fill it.
        result = await boost_question(
            RAW_DECISION_NO_OPTIONS,
            providers=_mock_llm("decision"),
            config=config, out_dir=tmp_path, slug="t5",
        )
        post = frontmatter.load(str(result.briefs[0]))
        body = post.content
        # (a) the gap IS surfaced, as an advisory annotation inside the brief
        assert boost_mod.GAP_MARKER in body
        # (b) KEYSTONE: absence of invented specifics. The caller named no candidate
        # databases; none may appear in the brief.
        for invented in ("postgres", "mysql", "mongodb", "sqlite", "oracle", "dynamodb"):
            assert invented not in body.lower(), f"brief invented option {invented!r}"
        # (c) structural absence: every content word derives from the raw question
        # or the boost's own fixed scaffold — nothing else may enter the brief.
        _assert_no_invented_content(body, RAW_DECISION_NO_OPTIONS)

    async def test_confabulating_decompose_is_rejected_by_hard_gate(self, config, tmp_path):
        # The decompose LLM answers with invented specifics (products, versions)
        # not present in the raw question. The deterministic verbatim gate must
        # reject them: no invented token may reach any emitted brief.
        confabulated = (
            "RESEARCH: Compare PostgreSQL logical replication against gh-ost cutover defaults\n"
            "DECISION: adopt PostgreSQL 16 with pgroll for the billing system"
        )
        result = await boost_question(
            RAW_HYBRID,
            providers=_mock_llm("hybrid", confabulated),
            config=config, out_dir=tmp_path, slug="t5b",
        )
        assert result.degraded, "rejected decompose must degrade, not silently pass"
        assert result.briefs, "degraded-but-COMPLETE: briefs still emitted"
        for path in result.briefs:
            text = path.read_text(encoding="utf-8").lower()
            for invented in ("postgresql", "gh-ost", "pgroll", "cutover"):
                assert invented not in text, (
                    f"{path.name}: confabulated token {invented!r} leaked through the gate"
                )


# ---------------------------------------------------------------------------
# T6 (FR-B6): hybrid decomposition — ≥2 linked sub-briefs, ≤3, feed order explicit
# ---------------------------------------------------------------------------

class TestT6HybridDecomposition:
    async def test_linked_sub_briefs_with_explicit_feed_order(self, config, tmp_path):
        result = await boost_question(
            RAW_HYBRID,
            providers=_mock_llm("hybrid", GOOD_DECOMPOSE),
            config=config, out_dir=tmp_path, slug="t6",
        )
        assert result.classification == "hybrid"
        assert not result.degraded
        assert 2 <= len(result.briefs) <= 3
        research_path, decision_path = result.briefs[0], result.briefs[-1]
        r_post = frontmatter.load(str(research_path))
        d_post = frontmatter.load(str(decision_path))
        # Research leg is a forced research sub-commission (T4 mechanism).
        assert r_post.metadata.get("mode") == "research"
        # Cross-linked: each leg names its counterpart file.
        assert decision_path.name in r_post.content
        assert research_path.name in d_post.content
        # Feed order explicit: research findings feed the decision leg.
        assert "feed" in (r_post.content + d_post.content).lower()


# ---------------------------------------------------------------------------
# T7 (ADR-08): exit codes 0 / 1 / 3
# ---------------------------------------------------------------------------

class TestT7ExitCodes:
    def test_success_exits_0(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "build_all_providers", lambda config, classes: _mock_llm("decision"))
        result = CliRunner().invoke(
            cli.main, ["boost", RAW_DECISION_NO_OPTIONS, "--out-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert list(tmp_path.glob("*.md")), "no brief emitted"

    def test_unusable_input_exits_1(self):
        result = CliRunner().invoke(cli.main, ["boost"])
        assert result.exit_code == 1
        assert "question" in result.output.lower()

    def test_degraded_but_complete_exits_3(self, tmp_path, monkeypatch):
        # No providers at all → the classifier falls back to the heuristic:
        # degraded-but-complete (ADR-08 exit 3), brief still emitted.
        monkeypatch.setattr(cli, "build_all_providers", lambda config, classes: {})
        result = CliRunner().invoke(
            cli.main, ["boost", RAW_DECISION_NO_OPTIONS, "--out-dir", str(tmp_path)],
        )
        assert result.exit_code == 3, result.output
        assert list(tmp_path.glob("*.md")), "degraded-but-COMPLETE: brief must still be emitted"


# ---------------------------------------------------------------------------
# T8 (#69 spec, P2): --file / --inbox boost parity — pinned, NOT implemented
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="P2 [#69]: boost is not yet wired into --file/--inbox — deferred deliberately "
    "so the inbox/CLI parity blind spot closes as one atomic unit. This pin makes the "
    "parity contract impossible for P2 to forget.",
    strict=True,
)
def test_t8_p2_file_inbox_boost_parity_spec():
    # When P2 wires boost into the run surface, this must become a behavioural test:
    # the same raw input boosted via --file and via --inbox yields identical briefs
    # and identical exit codes (CLAUDE.md §10 inbox-loop-parity anti-pattern).
    run_params = {p.name for p in cli.main.commands["run"].params}
    assert "boost" in run_params, "P2 must expose the boost wiring on the run surface"
