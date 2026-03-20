"""Smoke test: verify required API keys are available.

Run with: pytest -m envcheck
"""

import os

import pytest

pytestmark = pytest.mark.envcheck

# Keys this repo actually uses (from config/settings.yaml api_key_env fields)
REQUIRED_KEYS = [
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "XAI_API_KEY",
]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_api_key_available(key):
    """API key is set in environment (loaded by PS profile from global .env)."""
    value = os.environ.get(key)
    assert value is not None, (
        f"{key} not found. Run 'keys list' in PowerShell. "
        f"Keys should be in Documents/.secrets/.env"
    )
    assert len(value) > 10, f"{key} too short ({len(value)} chars)"
