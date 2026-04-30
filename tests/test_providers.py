"""Unit tests for individual AI providers — all SDK calls mocked, no real API keys needed."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_council.models import ModelResponse
from ai_council.providers.base import ProviderError
from config.config_loader import ModelConfig

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _model_cfg(name: str, base_url: str | None = None) -> ModelConfig:
    return ModelConfig(
        name=name,
        sdk="test",
        model=f"{name}-model-1",
        api_key_env=f"{name.upper()}_API_KEY",
        timeout_sec=30,
        max_tokens=512,
        base_url=base_url,
    )


def _openai_response(content: str = "Hello world", in_tokens: int = 10, out_tokens: int = 20) -> MagicMock:
    """Build a mock openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = in_tokens
    usage.completion_tokens = out_tokens
    usage.total_tokens = in_tokens + out_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-test-anthropic")


def _anthropic_response(content: str = "Claude says hi", in_tokens: int = 5, out_tokens: int = 15) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = content
    usage = MagicMock()
    usage.input_tokens = in_tokens
    usage.output_tokens = out_tokens
    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    return resp


async def test_anthropic_generate_returns_model_response(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-test")
    from ai_council.providers.anthropic import AnthropicProvider

    cfg = _model_cfg("claude")
    cfg = ModelConfig(
        name="claude", sdk="anthropic", model="claude-opus-4-6",
        api_key_env="CLAUDE_API_KEY", timeout_sec=30, max_tokens=512,
    )
    provider = AnthropicProvider(cfg)
    provider._client.messages.create = AsyncMock(return_value=_anthropic_response())

    result = await provider.generate("Test prompt", round_number=1)

    assert isinstance(result, ModelResponse)
    assert result.provider == "claude"
    assert result.content == "Claude says hi"
    assert result.input_tokens == 5
    assert result.output_tokens == 15
    assert result.token_count == 20
    assert result.round_number == 1


async def test_anthropic_generate_propagates_api_error(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-test")
    from ai_council.providers.anthropic import AnthropicProvider

    cfg = ModelConfig(
        name="claude", sdk="anthropic", model="claude-opus-4-6",
        api_key_env="CLAUDE_API_KEY", timeout_sec=30, max_tokens=512,
    )
    provider = AnthropicProvider(cfg)
    provider._client.messages.create = AsyncMock(side_effect=Exception("500 Internal Server Error"))

    with pytest.raises(ProviderError, match="API call failed"):
        await provider.generate("Test prompt", round_number=1)


async def test_anthropic_generate_raises_on_empty_content(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-test")
    from ai_council.providers.anthropic import AnthropicProvider

    cfg = ModelConfig(
        name="claude", sdk="anthropic", model="claude-opus-4-6",
        api_key_env="CLAUDE_API_KEY", timeout_sec=30, max_tokens=512,
    )
    provider = AnthropicProvider(cfg)
    empty_resp = MagicMock()
    empty_resp.content = []
    empty_resp.usage = None
    provider._client.messages.create = AsyncMock(return_value=empty_resp)

    with pytest.raises(ProviderError, match="Empty response content"):
        await provider.generate("Test prompt", round_number=1)


async def test_anthropic_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    from ai_council.providers.anthropic import AnthropicProvider

    cfg = ModelConfig(
        name="claude", sdk="anthropic", model="claude-opus-4-6",
        api_key_env="CLAUDE_API_KEY", timeout_sec=30, max_tokens=512,
    )
    with pytest.raises(ProviderError, match="Missing API key"):
        AnthropicProvider(cfg)


async def test_anthropic_no_usage_gives_none_tokens(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-test")
    from ai_council.providers.anthropic import AnthropicProvider

    cfg = ModelConfig(
        name="claude", sdk="anthropic", model="claude-opus-4-6",
        api_key_env="CLAUDE_API_KEY", timeout_sec=30, max_tokens=512,
    )
    provider = AnthropicProvider(cfg)
    block = MagicMock()
    block.type = "text"
    block.text = "response"
    resp = MagicMock()
    resp.content = [block]
    resp.usage = None
    provider._client.messages.create = AsyncMock(return_value=resp)

    result = await provider.generate("prompt", round_number=2)
    assert result.token_count is None
    assert result.input_tokens is None
    assert result.output_tokens is None


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------


def _gemini_response(text: str = "Gemini says hi", total_tokens: int = 30) -> MagicMock:
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 20
    usage.total_token_count = total_tokens
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = usage
    return resp


async def test_gemini_generate_returns_model_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    from ai_council.providers.gemini import GeminiProvider

    cfg = ModelConfig(
        name="gemini", sdk="google-genai", model="gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY", timeout_sec=30, max_tokens=512,
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=_gemini_response())

    with patch("ai_council.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(cfg)
        result = await provider.generate("Test prompt", round_number=1)

    assert isinstance(result, ModelResponse)
    assert result.provider == "gemini"
    assert result.content == "Gemini says hi"
    assert result.token_count == 30


async def test_gemini_generate_propagates_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from ai_council.providers.gemini import GeminiProvider

    cfg = ModelConfig(
        name="gemini", sdk="google-genai", model="gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("403 Forbidden"))

    with patch("ai_council.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(cfg)
        with pytest.raises(ProviderError, match="API call failed"):
            await provider.generate("Test", round_number=1)


async def test_gemini_empty_text_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from ai_council.providers.gemini import GeminiProvider

    cfg = ModelConfig(
        name="gemini", sdk="google-genai", model="gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    empty_resp = MagicMock()
    empty_resp.text = None
    empty_resp.usage_metadata = None
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=empty_resp)

    with patch("ai_council.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(cfg)
        with pytest.raises(ProviderError, match="Empty response text"):
            await provider.generate("Test", round_number=1)


async def test_gemini_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from ai_council.providers.gemini import GeminiProvider

    cfg = ModelConfig(
        name="gemini", sdk="google-genai", model="gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    with pytest.raises(ProviderError, match="Missing API key"):
        GeminiProvider(cfg)


async def test_gemini_no_usage_metadata_gives_none_tokens(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from ai_council.providers.gemini import GeminiProvider

    cfg = ModelConfig(
        name="gemini", sdk="google-genai", model="gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    resp = MagicMock()
    resp.text = "response text"
    resp.usage_metadata = None
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

    with patch("ai_council.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(cfg)
        result = await provider.generate("prompt", round_number=1)

    assert result.token_count is None


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


async def test_openai_generate_returns_model_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    from ai_council.providers.openai_provider import OpenAIProvider

    cfg = ModelConfig(
        name="openai", sdk="openai", model="gpt-4o",
        api_key_env="OPENAI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    provider = OpenAIProvider(cfg)
    provider._client.chat.completions.create = AsyncMock(
        return_value=_openai_response("GPT says hi")
    )

    result = await provider.generate("Test prompt", round_number=2)

    assert isinstance(result, ModelResponse)
    assert result.provider == "openai"
    assert result.content == "GPT says hi"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.token_count == 30


async def test_openai_generate_propagates_api_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    from ai_council.providers.openai_provider import OpenAIProvider

    cfg = ModelConfig(
        name="openai", sdk="openai", model="gpt-4o",
        api_key_env="OPENAI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    provider = OpenAIProvider(cfg)
    provider._client.chat.completions.create = AsyncMock(side_effect=Exception("429 Too Many Requests"))

    with pytest.raises(ProviderError, match="API call failed"):
        await provider.generate("Test", round_number=1)


async def test_openai_empty_choices_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    from ai_council.providers.openai_provider import OpenAIProvider

    cfg = ModelConfig(
        name="openai", sdk="openai", model="gpt-4o",
        api_key_env="OPENAI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    provider = OpenAIProvider(cfg)
    empty_resp = MagicMock()
    empty_resp.choices = []
    empty_resp.usage = None
    provider._client.chat.completions.create = AsyncMock(return_value=empty_resp)

    with pytest.raises(ProviderError, match="Empty response content"):
        await provider.generate("Test", round_number=1)


async def test_openai_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from ai_council.providers.openai_provider import OpenAIProvider

    cfg = ModelConfig(
        name="openai", sdk="openai", model="gpt-4o",
        api_key_env="OPENAI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    with pytest.raises(ProviderError, match="Missing API key"):
        OpenAIProvider(cfg)


# ---------------------------------------------------------------------------
# XAIProvider (Grok)
# ---------------------------------------------------------------------------


async def test_xai_generate_returns_model_response(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    from ai_council.providers.xai import XAIProvider

    cfg = ModelConfig(
        name="grok", sdk="openai-compat", model="grok-3",
        api_key_env="XAI_API_KEY", timeout_sec=30, max_tokens=512,
        base_url="https://api.x.ai/v1",
    )
    provider = XAIProvider(cfg)
    provider._client.chat.completions.create = AsyncMock(
        return_value=_openai_response("Grok says hi")
    )

    result = await provider.generate("Test", round_number=1)

    assert isinstance(result, ModelResponse)
    assert result.provider == "grok"
    assert result.content == "Grok says hi"
    assert result.token_count == 30


async def test_xai_missing_base_url_raises(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    from ai_council.providers.xai import XAIProvider

    cfg = ModelConfig(
        name="grok", sdk="openai-compat", model="grok-3",
        api_key_env="XAI_API_KEY", timeout_sec=30, max_tokens=512,
        base_url=None,
    )
    with pytest.raises(ProviderError, match="base_url is required"):
        XAIProvider(cfg)


async def test_xai_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    from ai_council.providers.xai import XAIProvider

    cfg = ModelConfig(
        name="grok", sdk="openai-compat", model="grok-3",
        api_key_env="XAI_API_KEY", timeout_sec=30, max_tokens=512,
        base_url="https://api.x.ai/v1",
    )
    with pytest.raises(ProviderError, match="Missing API key"):
        XAIProvider(cfg)


async def test_xai_generate_propagates_api_error(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    from ai_council.providers.xai import XAIProvider

    cfg = ModelConfig(
        name="grok", sdk="openai-compat", model="grok-3",
        api_key_env="XAI_API_KEY", timeout_sec=30, max_tokens=512,
        base_url="https://api.x.ai/v1",
    )
    provider = XAIProvider(cfg)
    provider._client.chat.completions.create = AsyncMock(side_effect=Exception("503 Service Unavailable"))

    with pytest.raises(ProviderError, match="API call failed"):
        await provider.generate("Test", round_number=1)


# ---------------------------------------------------------------------------
# DeepSeekProvider
# ---------------------------------------------------------------------------


async def test_deepseek_generate_returns_model_response(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    from ai_council.providers.deepseek import DeepSeekProvider

    cfg = ModelConfig(
        name="deepseek", sdk="openai-compat", model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY", timeout_sec=30, max_tokens=512,
        base_url="https://api.deepseek.com/v1",
    )
    provider = DeepSeekProvider(cfg)
    provider._client.chat.completions.create = AsyncMock(
        return_value=_openai_response("DeepSeek says hi")
    )

    result = await provider.generate("Test", round_number=3)

    assert isinstance(result, ModelResponse)
    assert result.provider == "deepseek"
    assert result.content == "DeepSeek says hi"
    assert result.round_number == 3


async def test_deepseek_missing_base_url_raises(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    from ai_council.providers.deepseek import DeepSeekProvider

    cfg = ModelConfig(
        name="deepseek", sdk="openai-compat", model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY", timeout_sec=30, max_tokens=512,
        base_url=None,
    )
    with pytest.raises(ProviderError, match="base_url is required"):
        DeepSeekProvider(cfg)


async def test_deepseek_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from ai_council.providers.deepseek import DeepSeekProvider

    cfg = ModelConfig(
        name="deepseek", sdk="openai-compat", model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY", timeout_sec=30, max_tokens=512,
        base_url="https://api.deepseek.com/v1",
    )
    with pytest.raises(ProviderError, match="Missing API key"):
        DeepSeekProvider(cfg)


async def test_deepseek_generate_propagates_api_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    from ai_council.providers.deepseek import DeepSeekProvider

    cfg = ModelConfig(
        name="deepseek", sdk="openai-compat", model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY", timeout_sec=30, max_tokens=512,
        base_url="https://api.deepseek.com/v1",
    )
    provider = DeepSeekProvider(cfg)
    provider._client.chat.completions.create = AsyncMock(side_effect=Exception("Connection refused"))

    with pytest.raises(ProviderError, match="API call failed"):
        await provider.generate("Test", round_number=1)


async def test_deepseek_empty_choices_raises(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    from ai_council.providers.deepseek import DeepSeekProvider

    cfg = ModelConfig(
        name="deepseek", sdk="openai-compat", model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY", timeout_sec=30, max_tokens=512,
        base_url="https://api.deepseek.com/v1",
    )
    provider = DeepSeekProvider(cfg)
    empty_resp = MagicMock()
    empty_resp.choices = []
    empty_resp.usage = None
    provider._client.chat.completions.create = AsyncMock(return_value=empty_resp)

    with pytest.raises(ProviderError, match="Empty response content"):
        await provider.generate("Test", round_number=1)


# ---------------------------------------------------------------------------
# name() and model_string() sanity checks across all providers
# ---------------------------------------------------------------------------


async def test_all_providers_name_and_model_string(monkeypatch):
    """name() and model_string() return values from ModelConfig."""
    monkeypatch.setenv("CLAUDE_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("XAI_API_KEY", "x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")

    from ai_council.providers.anthropic import AnthropicProvider
    from ai_council.providers.deepseek import DeepSeekProvider
    from ai_council.providers.gemini import GeminiProvider
    from ai_council.providers.openai_provider import OpenAIProvider
    from ai_council.providers.xai import XAIProvider

    providers_and_cfgs = [
        (AnthropicProvider, ModelConfig(
            name="claude", sdk="anthropic", model="claude-opus-4-6",
            api_key_env="CLAUDE_API_KEY", timeout_sec=30, max_tokens=512,
        )),
        (OpenAIProvider, ModelConfig(
            name="openai", sdk="openai", model="gpt-4o",
            api_key_env="OPENAI_API_KEY", timeout_sec=30, max_tokens=512,
        )),
        (XAIProvider, ModelConfig(
            name="grok", sdk="openai-compat", model="grok-3",
            api_key_env="XAI_API_KEY", timeout_sec=30, max_tokens=512,
            base_url="https://api.x.ai/v1",
        )),
        (DeepSeekProvider, ModelConfig(
            name="deepseek", sdk="openai-compat", model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY", timeout_sec=30, max_tokens=512,
            base_url="https://api.deepseek.com/v1",
        )),
    ]

    for cls, cfg in providers_and_cfgs:
        p = cls(cfg)
        assert p.name() == cfg.name
        assert p.model_string() == cfg.model

    # Gemini doesn't create client in __init__, just check name/model
    gcfg = ModelConfig(
        name="gemini", sdk="google-genai", model="gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY", timeout_sec=30, max_tokens=512,
    )
    gp = GeminiProvider(gcfg)
    assert gp.name() == "gemini"
    assert gp.model_string() == "gemini-2.5-pro"
