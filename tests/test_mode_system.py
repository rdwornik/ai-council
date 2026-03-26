"""Tests for the mode system: config parsing, alias resolution, prompt assembly, auto-detect."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from config.config_loader import (
    ModeConfig,
    _validate_modes,
    default_mode,
    load_config,
    resolve_mode,
)
from src.debate import _build_round1_prompt, _build_round2_prompt
from src.mode_detector import _pick_cheapest, detect_mode
from src.models import ModelResponse
from src.synthesis import _build_synthesis_prompt


# ---------------------------------------------------------------------------
# ModeConfig helpers
# ---------------------------------------------------------------------------

def _make_pick_mode(**kwargs) -> ModeConfig:
    defaults = dict(
        description="Pick mode",
        emoji="T",
        aliases=["p", "pick", "d", "decide"],
        default=True,
        max_rounds=2,
        token_budget=1500,
    )
    return ModeConfig(**{**defaults, **kwargs})


def _make_ideas_mode(**kwargs) -> ModeConfig:
    defaults = dict(
        description="Ideas mode",
        emoji="L",
        aliases=["i", "ideas"],
        default=False,
        max_rounds=1,
        token_budget=1000,
        round1_header="Brainstorm council.",
        round1_instruction="Generate ideas.",
        round1_structure="## Ideas\nList ideas.",
        round2_instruction="Cross-pollinate.",
        synthesis_output="## Inventory\nAll ideas.",
    )
    return ModeConfig(**{**defaults, **kwargs})


def _make_judge_mode(**kwargs) -> ModeConfig:
    defaults = dict(
        description="Judge mode",
        emoji="S",
        aliases=["j", "judge"],
        default=False,
        max_rounds=2,
        token_budget=1200,
        round1_header="Assessment council.",
        round1_instruction="Assess honestly.",
        round1_structure="## Assessment\nOverall view.",
        round2_instruction="Compare assessments.",
        synthesis_output="## Verdict\nFinal verdict.",
    )
    return ModeConfig(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# resolve_mode / default_mode
# ---------------------------------------------------------------------------

class TestResolveMode:
    def test_canonical_name(self):
        modes = {"pick": _make_pick_mode(), "ideas": _make_ideas_mode()}
        assert resolve_mode("pick", modes) == "pick"

    def test_alias_resolves(self):
        modes = {"pick": _make_pick_mode(), "ideas": _make_ideas_mode()}
        assert resolve_mode("p", modes) == "pick"
        assert resolve_mode("decide", modes) == "pick"
        assert resolve_mode("i", modes) == "ideas"

    def test_unknown_raises(self):
        modes = {"pick": _make_pick_mode()}
        with pytest.raises(ValueError, match="Unknown mode"):
            resolve_mode("nonexistent", modes)

    def test_empty_modes_raises(self):
        with pytest.raises(ValueError):
            resolve_mode("pick", {})


class TestDefaultMode:
    def test_returns_default(self):
        modes = {"pick": _make_pick_mode(), "ideas": _make_ideas_mode()}
        assert default_mode(modes) == "pick"

    def test_no_default_raises(self):
        modes = {"ideas": _make_ideas_mode()}
        with pytest.raises(ValueError, match="No default mode"):
            default_mode(modes)


# ---------------------------------------------------------------------------
# _validate_modes
# ---------------------------------------------------------------------------

class TestValidateModes:
    def test_valid_config_passes(self):
        modes = {
            "pick": _make_pick_mode(),
            "ideas": _make_ideas_mode(),
            "judge": _make_judge_mode(),
        }
        _validate_modes(modes)  # should not raise

    def test_no_default_raises(self):
        modes = {"ideas": _make_ideas_mode()}
        with pytest.raises(ValueError, match="Exactly one mode"):
            _validate_modes(modes)

    def test_two_defaults_raises(self):
        modes = {
            "pick": _make_pick_mode(default=True),
            "ideas": _make_ideas_mode(default=True),
        }
        with pytest.raises(ValueError, match="Exactly one mode"):
            _validate_modes(modes)

    def test_duplicate_alias_raises(self):
        modes = {
            "pick": _make_pick_mode(aliases=["p", "shared"]),
            "ideas": _make_ideas_mode(aliases=["i", "shared"]),
        }
        with pytest.raises(ValueError, match="Duplicate alias"):
            _validate_modes(modes)

    def test_uppercase_alias_raises(self):
        modes = {"pick": _make_pick_mode(aliases=["P", "pick"])}
        with pytest.raises(ValueError, match="lowercase"):
            _validate_modes(modes)

    def test_missing_template_fields_raises(self):
        # ideas mode with empty round1_instruction should fail validation
        bad_ideas = _make_ideas_mode(round1_instruction="")
        modes = {"pick": _make_pick_mode(), "ideas": bad_ideas}
        with pytest.raises(ValueError, match="missing required template fields"):
            _validate_modes(modes)


# ---------------------------------------------------------------------------
# ModeConfig.uses_existing_prompts
# ---------------------------------------------------------------------------

class TestUsesExistingPrompts:
    def test_pick_uses_existing(self):
        cfg = _make_pick_mode()
        assert cfg.uses_existing_prompts is True

    def test_ideas_does_not(self):
        cfg = _make_ideas_mode()
        assert cfg.uses_existing_prompts is False

    def test_judge_does_not(self):
        cfg = _make_judge_mode()
        assert cfg.uses_existing_prompts is False


# ---------------------------------------------------------------------------
# load_config parses modes
# ---------------------------------------------------------------------------

class TestLoadConfigModes:
    def test_modes_parsed(self):
        cfg = load_config()
        assert "pick" in cfg.modes
        assert "ideas" in cfg.modes
        assert "judge" in cfg.modes

    def test_pick_is_default(self):
        cfg = load_config()
        assert cfg.modes["pick"].default is True
        assert cfg.modes["ideas"].default is False
        assert cfg.modes["judge"].default is False

    def test_aliases_loaded(self):
        cfg = load_config()
        assert "p" in cfg.modes["pick"].aliases
        assert "decide" in cfg.modes["pick"].aliases
        assert "i" in cfg.modes["ideas"].aliases
        assert "j" in cfg.modes["judge"].aliases

    def test_persona_mode_directives_parsed(self):
        cfg = load_config()
        assert "ideas" in cfg.persona_mode_directives
        assert "claude" in cfg.persona_mode_directives["ideas"]
        assert "BRAINSTORM" in cfg.persona_mode_directives["ideas"]["claude"]

    def test_judge_directive_for_grok(self):
        cfg = load_config()
        assert "judge" in cfg.persona_mode_directives
        assert "grok" in cfg.persona_mode_directives["judge"]
        assert "ASSESSMENT" in cfg.persona_mode_directives["judge"]["grok"]


# ---------------------------------------------------------------------------
# Prompt assembly — debate
# ---------------------------------------------------------------------------

from config.config_loader import PromptsConfig  # noqa: E402


def _make_prompts() -> PromptsConfig:
    return PromptsConfig(
        initial="{persona}\nInitial: {question}",
        critique="{persona}\nRound {round}. Q: {question}\n{previous_responses_anonymized}",
        synthesis="Synth Q: {question}\n{full_transcript}",
        personas={"claude": "System persona.", "mock": "Mock persona."},
    )


class TestBuildRound1Prompt:
    def test_pick_uses_initial_template(self):
        prompts = _make_prompts()
        result = _build_round1_prompt("claude", "my question", prompts, None, {})
        assert "Initial:" in result
        assert "my question" in result
        assert "System persona." in result

    def test_ideas_uses_mode_templates(self):
        prompts = _make_prompts()
        mode = _make_ideas_mode()
        result = _build_round1_prompt("mock", "my question", prompts, mode, {})
        assert "Generate ideas." in result
        assert "Brainstorm council." in result
        assert "my question" in result
        # Should NOT use the initial template
        assert "Initial:" not in result

    def test_ideas_injects_persona_directive(self):
        prompts = _make_prompts()
        mode = _make_ideas_mode()
        directives = {"mock": "BRAINSTORM: go wild"}
        result = _build_round1_prompt("mock", "my question", prompts, mode, directives)
        assert "CRITICAL INSTRUCTION: BRAINSTORM: go wild" in result

    def test_no_directive_for_provider_omits_prefix(self):
        prompts = _make_prompts()
        mode = _make_ideas_mode()
        result = _build_round1_prompt("claude", "my question", prompts, mode, {})
        assert "CRITICAL INSTRUCTION" not in result

    def test_judge_round1_includes_structure(self):
        prompts = _make_prompts()
        mode = _make_judge_mode()
        result = _build_round1_prompt("mock", "my question", prompts, mode, {})
        assert "Assess honestly." in result
        assert "## Assessment" in result


class TestBuildRound2Prompt:
    def test_pick_uses_critique_template(self):
        prompts = _make_prompts()
        result = _build_round2_prompt(
            "claude", 2, "my question", "--- Proposal A ---\nContent", prompts, None, {}
        )
        assert "Round 2" in result
        assert "Proposal A" in result

    def test_ideas_uses_round2_instruction(self):
        prompts = _make_prompts()
        mode = _make_ideas_mode()
        result = _build_round2_prompt(
            "mock", 2, "my question", "--- Proposal A ---\nContent", prompts, mode, {}
        )
        assert "Cross-pollinate." in result
        assert "Proposal A" in result
        # Should NOT use critique template
        assert "Round 2. Q:" not in result


# ---------------------------------------------------------------------------
# Prompt assembly — synthesis
# ---------------------------------------------------------------------------

class TestBuildSynthesisPrompt:
    def test_pick_uses_prompts_synthesis(self):
        prompts = _make_prompts()
        from src.models import Round  # noqa: PLC0415
        result = _build_synthesis_prompt("my question", "transcript text", [], prompts, None)
        assert "Synth Q: my question" in result
        assert "transcript text" in result

    def test_ideas_uses_synthesis_output(self):
        prompts = _make_prompts()
        mode = _make_ideas_mode()
        result = _build_synthesis_prompt("my question", "transcript text", [], prompts, mode)
        assert "## Inventory" in result
        assert "transcript text" in result
        assert "my question" in result

    def test_judge_uses_synthesis_output(self):
        prompts = _make_prompts()
        mode = _make_judge_mode()
        result = _build_synthesis_prompt("my question", "transcript text", [], prompts, mode)
        assert "## Verdict" in result


# ---------------------------------------------------------------------------
# Auto-detection: _pick_cheapest
# ---------------------------------------------------------------------------

class TestPickCheapest:
    def test_prefers_deepseek(self):
        from tests.conftest import MockProvider  # noqa: PLC0415
        providers = {
            "claude": MockProvider("claude"),
            "gemini": MockProvider("gemini"),
            "deepseek": MockProvider("deepseek"),
        }
        result = _pick_cheapest(providers)
        assert result.name() == "deepseek"

    def test_falls_back_to_gemini(self):
        from tests.conftest import MockProvider  # noqa: PLC0415
        providers = {
            "claude": MockProvider("claude"),
            "gemini": MockProvider("gemini"),
        }
        result = _pick_cheapest(providers)
        assert result.name() == "gemini"

    def test_returns_none_for_empty(self):
        assert _pick_cheapest({}) is None


# ---------------------------------------------------------------------------
# Auto-detection: detect_mode
# ---------------------------------------------------------------------------

class TestDetectMode:
    async def test_returns_detected_mode(self):
        from tests.conftest import MockProvider  # noqa: PLC0415
        provider = MockProvider("gemini", "ideas")
        providers = {"gemini": provider}
        mode, source = await detect_mode("What features?", providers, {"pick", "ideas", "judge"})
        assert mode == "ideas"
        assert "gemini" in source

    async def test_fallback_on_unknown_response(self):
        from tests.conftest import MockProvider  # noqa: PLC0415
        provider = MockProvider("gemini", "something_unknown")
        providers = {"gemini": provider}
        mode, source = await detect_mode("question", providers, {"pick", "ideas", "judge"})
        assert mode == "pick"
        assert "fallback" in source

    async def test_fallback_on_timeout(self):
        from tests.conftest import MockProvider  # noqa: PLC0415
        import asyncio

        provider = MockProvider("gemini")

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(5)
            return provider.generate.return_value

        provider.generate = AsyncMock(side_effect=slow_generate)
        providers = {"gemini": provider}
        mode, source = await detect_mode("question", providers, {"pick", "ideas", "judge"}, timeout_sec=0.05)
        assert mode == "pick"
        assert "timeout" in source

    async def test_fallback_on_empty_providers(self):
        mode, source = await detect_mode("question", {}, {"pick", "ideas", "judge"})
        assert mode == "pick"
        assert "fallback" in source
