"""Integration tests — real API calls, no mocks. Requires .env with 2+ API keys."""

import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip entire module if fewer than 2 API keys are set
_AVAILABLE_KEYS = [
    k for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"]
    if os.environ.get(k, "").strip()
]
pytestmark = pytest.mark.integration

if len(_AVAILABLE_KEYS) < 2:
    pytestmark = pytest.mark.skip(reason=f"Need 2+ API keys, found {len(_AVAILABLE_KEYS)}")


async def test_full_debate_pipeline(tmp_path: Path):
    """Run a real 1-round debate with available providers, verify no crash."""
    from config.config_loader import load_config
    from src.cli import _build_providers, _pick_synthesizer
    from src.debate import run_debate
    from src.models import Question
    from src.output import save_to_file
    from src.synthesis import synthesize

    config = load_config()
    providers = _build_providers(config, requested_models=None)

    assert len(providers) >= 2, f"Need 2+ providers, got {len(providers)}"

    question = Question(
        text="Should a small team use a monorepo or separate repos for a Python microservices project?",
        source="integration_test",
    )

    start = time.monotonic()
    rounds = await run_debate(
        question=question,
        providers=providers,
        prompts=config.prompts,
        num_rounds=1,
    )

    assert len(rounds) == 1
    assert len(rounds[0].responses) >= 1

    for resp in rounds[0].responses:
        assert resp.content, f"Empty content from {resp.provider}"
        assert resp.latency_sec > 0

    synthesizer = _pick_synthesizer(providers, config.defaults.synthesizer)
    result = await synthesize(
        question=question,
        rounds=rounds,
        synthesizer=synthesizer,
        prompts=config.prompts,
        debate_start_time=start,
    )

    assert result.synthesis, "Synthesis content is empty"
    assert result.synthesizer

    saved = save_to_file(result, tmp_path / "output")
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "AI Council Debate" in content
    assert len(content) > 500
