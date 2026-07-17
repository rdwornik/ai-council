"""Tests for the CLI provider adapters (claude, codex) — no real subprocess.

Covers the transport parsing + the seat-equivalence acceptance checks that don't need a live
CLI: I1 (identity gate — unreadable identity raises, never admits), I2 (scratch cwd — never a
repo), I6 (timeout is a hard kill). Live end-to-end is the arc's witnessed closure.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_council.providers.base import ProviderError
from ai_council.providers.cli_base import (
    ClaudeCliProvider,
    CliProvider,
    CodexCliProvider,
)
from config.config_loader import ModelConfig


def _cfg(name: str, cmd: str, model: str = "m", timeout_sec: int = 30) -> ModelConfig:
    return ModelConfig(
        name=name, sdk="cli", model=model, api_key_env="K", timeout_sec=timeout_sec,
        max_tokens=100, backend="cli", cli_command=cmd, cli_model=model,
    )


def _make(cls: type[CliProvider], cmd: str, **cfg_kw: object) -> CliProvider:
    """Construct a CLI provider without touching the real binary (which/version stubbed)."""
    with patch("ai_council.providers.cli_base.shutil.which", return_value=f"{cmd}.CMD"), \
         patch.object(CliProvider, "_read_version", staticmethod(lambda exe: "v-test")):
        return cls(_cfg(cmd, cmd, **cfg_kw))  # type: ignore[arg-type]


def _mock_proc(stdout: bytes, stderr: bytes, returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


# --- construction / resolution ---


def test_missing_cli_command_raises() -> None:
    cfg = ModelConfig(name="x", sdk="cli", model="m", api_key_env="K", timeout_sec=1,
                      max_tokens=1, backend="cli")  # no cli_command
    with pytest.raises(ProviderError, match="requires cli_command"):
        ClaudeCliProvider(cfg)


def test_cli_not_on_path_raises() -> None:
    with patch("ai_council.providers.cli_base.shutil.which", return_value=None):
        with pytest.raises(ProviderError, match="not found on PATH"):
            ClaudeCliProvider(_cfg("claude", "claude"))


# --- claude _extract (identity via .modelUsage) ---


def test_claude_extract_parses_result_and_identity() -> None:
    p = _make(ClaudeCliProvider, "claude")
    doc = {"result": "Hello", "modelUsage": {"claude-haiku-4-5-20251001": {}},
           "usage": {"input_tokens": 5, "output_tokens": 7}}
    out = p._extract(json.dumps(doc), "")
    assert out.content == "Hello"
    assert out.actual_model == "claude-haiku-4-5-20251001"
    assert out.identity_channel == "modelUsage"
    assert (out.input_tokens, out.output_tokens, out.token_count) == (5, 7, 12)


def test_claude_extract_identity_unreadable_raises() -> None:  # I1
    p = _make(ClaudeCliProvider, "claude")
    with pytest.raises(ProviderError, match="identity-unreadable"):
        p._extract(json.dumps({"result": "hi", "modelUsage": {}}), "")


def test_claude_extract_empty_result_raises() -> None:
    p = _make(ClaudeCliProvider, "claude")
    with pytest.raises(ProviderError, match="empty .result"):
        p._extract(json.dumps({"result": "", "modelUsage": {"m": {}}}), "")


def test_claude_extract_bad_json_raises() -> None:
    p = _make(ClaudeCliProvider, "claude")
    with pytest.raises(ProviderError, match="bad JSON"):
        p._extract("not json at all", "")


# --- codex _extract (identity via stderr banner) ---


def test_codex_extract_parses_stdout_and_banner() -> None:
    p = _make(CodexCliProvider, "codex")
    stderr = "OpenAI Codex v0.144.5\nmodel: gpt-5.6-sol\nprovider: openai\ntokens used 1980"
    out = p._extract("OK answer", stderr)
    assert out.content == "OK answer"
    assert out.actual_model == "gpt-5.6-sol"
    assert out.identity_channel == "stderr-banner"
    assert out.token_count == 1980


def test_codex_extract_identity_unreadable_raises() -> None:  # I1
    p = _make(CodexCliProvider, "codex")
    with pytest.raises(ProviderError, match="identity-unreadable"):
        p._extract("answer", "no model banner here")


def test_codex_extract_empty_stdout_raises() -> None:
    p = _make(CodexCliProvider, "codex")
    with pytest.raises(ProviderError, match="empty stdout"):
        p._extract("   ", "model: gpt-5.6-sol")


# --- run() with mocked subprocess ---


async def test_run_process_error_surfaces_exit_code() -> None:
    p = _make(ClaudeCliProvider, "claude")
    proc = _mock_proc(b"", b"kaboom", returncode=1)
    with patch("ai_council.providers.cli_base.asyncio.create_subprocess_exec",
               AsyncMock(return_value=proc)):
        with pytest.raises(ProviderError, match="process error: exit 1"):
            await p.run("hi")


async def test_run_scratch_cwd_is_not_a_repo() -> None:  # I2
    p = _make(ClaudeCliProvider, "claude")
    doc = json.dumps({"result": "ok", "modelUsage": {"m": {}}})
    captured: dict[str, object] = {}

    async def fake_exec(*argv: object, **kwargs: object) -> MagicMock:
        captured["cwd"] = kwargs.get("cwd")
        captured["stdin"] = kwargs.get("stdin")
        return _mock_proc(doc.encode(), b"")

    with patch("ai_council.providers.cli_base.asyncio.create_subprocess_exec", fake_exec):
        await p.run("hi")
    cwd = str(captured["cwd"])
    assert "council-cli-" in cwd  # a fresh scratch temp dir, never the repo
    assert captured["stdin"] == asyncio.subprocess.PIPE  # prompt written then stdin closed (EOF)


async def test_run_timeout_is_hard_kill() -> None:  # I6
    p = _make(ClaudeCliProvider, "claude")
    proc = MagicMock()

    async def never_returns(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        await asyncio.sleep(5)
        return (b"", b"")

    proc.communicate = never_returns
    proc.returncode = None
    proc.wait = AsyncMock()
    with patch("ai_council.providers.cli_base.asyncio.create_subprocess_exec",
               AsyncMock(return_value=proc)), \
         patch("ai_council.providers.cli_base._kill_process_tree") as kill_tree:
        with pytest.raises(ProviderError, match="timed out"):
            await p.run("hi", timeout=0.05)
    kill_tree.assert_called()  # the whole process tree is terminated on timeout


def test_scrubbed_env_is_allowlist_only() -> None:
    """Allowlist: only named non-secret vars survive; ANY credential is dropped by exclusion,
    including odd names a denylist would miss (AWS_*, GOOGLE_APPLICATION_CREDENTIALS, DB URLs)."""
    from ai_council.providers.cli_base import _scrubbed_env
    poison = {
        "OPENAI_API_KEY": "sk", "XAI_API_KEY": "y", "AWS_SECRET_ACCESS_KEY": "a",
        "AWS_ACCESS_KEY_ID": "b", "GOOGLE_APPLICATION_CREDENTIALS": "/c.json",
        "DATABASE_URL": "postgres://u:p@h/db", "MY_TOKEN": "t", "PATH": "/usr/bin",
    }
    with patch.dict("os.environ", poison, clear=True):
        env = _scrubbed_env()
    assert env == {"PATH": "/usr/bin"}  # only the allowlisted, non-secret var survives


def test_scrubbed_env_strips_proxy_userinfo() -> None:
    """A proxy URL's user:pass@ credential is stripped; the host:port is kept for function."""
    from ai_council.providers.cli_base import _scrubbed_env
    with patch.dict("os.environ",
                    {"HTTPS_PROXY": "http://user:s3cret@proxy.corp:8080", "PATH": "/x"}, clear=True):
        env = _scrubbed_env()
    assert env["HTTPS_PROXY"] == "http://proxy.corp:8080"
    assert "user" not in env["HTTPS_PROXY"] and "s3cret" not in env["HTTPS_PROXY"]


async def test_generate_returns_served_identity_as_model() -> None:
    p = _make(ClaudeCliProvider, "claude")
    doc = json.dumps({"result": "answer", "modelUsage": {"claude-opus-4-8": {}}})
    with patch("ai_council.providers.cli_base.asyncio.create_subprocess_exec",
               AsyncMock(return_value=_mock_proc(doc.encode(), b""))):
        resp = await p.generate("hi", round_number=1)
    assert resp.content == "answer"
    assert resp.model == "claude-opus-4-8"  # ModelResponse.model carries the SERVED identity
