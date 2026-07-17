"""CLI-subscription provider adapters (claude, codex) — pure transport behind AIProvider.

ADR-12 safety floor on EVERY call (all witnessed 2026-07-05, re-witnessed 2026-07-17):
- **cwd = a fresh scratch dir, never a repo** — the PRIMARY isolation (`claude -p --tools ""`
  still ingests cwd `CLAUDE.md`; CL-3). A per-call TemporaryDirectory leaves no residue.
- **stdin closed** (`codex exec` prints "Reading additional input from stdin…" then reads EOF;
  an *open* stdin hangs it — CX-1).
- **read-only / tools-off flags** + an **explicit model pin** on every call (per-call pin rule:
  codex otherwise serves its own default, now `gpt-5.6-sol`).
- **`timeout_sec` is the hard kill** (I6).

Identity is read from each CLI's witnessed channel — claude in-band `.modelUsage`; codex
plain-mode **stderr** banner `model:` (the `--json` stream carries none). These adapters do
ONLY transport + identity read and raise a classifiable ProviderError on failure; the
admission gate + same-seat API fallback live in `seat_router.py` (a debate-engine concern,
L-CLI IF#2). Separate classes, no provider-merge (ADR-12 / CLAUDE 5.7).
"""

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from ai_council.models import ModelResponse
from ai_council.providers.base import AIProvider, ProviderError
from config.config_loader import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class CliOutcome:
    """The rich result the seat-router consumes (richer than ModelResponse: carries the
    witnessed served identity + which channel supplied it)."""

    content: str
    actual_model: str
    identity_channel: str  # "modelUsage" | "stderr-banner"
    token_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class CliProvider(AIProvider):
    """Base for CLI-subscription seats. Overrides generate() (the template's API-SDK skeleton
    does not apply to a subprocess); subclasses provide `_argv` + `_extract`."""

    identity_channel: str = ""  # subclass declares its witnessed channel

    def __init__(self, config: ModelConfig) -> None:
        # CLI seats authenticate via the subscription CLI, not an API key — so this does NOT
        # call the AIProvider template __init__ (which requires an API key + builds a client).
        # The same-seat API fallback provider is separate, held by the seat-router.
        self._config = config
        if not config.cli_command:
            raise ProviderError(config.name, "backend=cli requires cli_command (claude|codex)")
        # Resolve to the full executable path (PATHEXT-aware — npm shims are .CMD on Windows).
        # asyncio.create_subprocess_exec does not search PATHEXT for a bare name, so a resolved
        # absolute path is required; the prompt still travels as its own argv element (no shell,
        # no cmd.exe metacharacter exposure).
        exe = shutil.which(config.cli_command)
        if exe is None:
            raise ProviderError(config.name, f"CLI not found on PATH: {config.cli_command}")
        self._exe = exe
        self._cli_model = config.cli_model or config.model
        self._version = self._read_version(self._exe)

    @staticmethod
    def _read_version(exe: str) -> str | None:
        """Capture the CLI version at construction (F5 forensic anchor / seats[].cli.version)."""
        try:
            out = subprocess.run(
                [exe, "--version"],
                capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
            )
            return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else None
        except Exception:  # version is best-effort telemetry; never fatal
            return None

    @property
    def version(self) -> str | None:
        return self._version

    async def run(self, prompt: str, *, timeout: float | None = None) -> CliOutcome:
        """Execute the CLI once under the ADR-12 safety floor and return a CliOutcome.

        Raises ProviderError on any failure (timeout, nonzero exit, unparseable output,
        unreadable identity) with a message that classify_cli_failure maps to a cause token.
        """
        effective_timeout = timeout if timeout is not None else self._config.timeout_sec
        argv = [self._exe, *self._argv(prompt, self._cli_model)]
        # Fresh scratch cwd per call — PRIMARY isolation; auto-removed (no leftover).
        with tempfile.TemporaryDirectory(prefix="council-cli-") as scratch:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=scratch,
                    stdin=subprocess.DEVNULL,  # closed — codex hangs on open stdin (CX-1)
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as exc:  # spawn failure (binary missing, etc.)
                raise ProviderError(self._config.name, f"CLI process error: {exc}") from exc

            try:
                raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                proc.kill()
                await proc.wait()
                raise ProviderError(
                    self._config.name, f"CLI timed out after {effective_timeout}s"
                ) from exc

        stdout = raw_out.decode("utf-8", errors="replace")
        stderr = raw_err.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise ProviderError(
                self._config.name,
                f"CLI process error: exit {proc.returncode}: {stderr.strip()[:200]}",
            )
        return self._extract(stdout, stderr)

    async def generate(
        self, prompt: str, round_number: int, *, timeout: float | None = None
    ) -> ModelResponse:
        """AIProvider contract: run the CLI and adapt to a ModelResponse. ``model`` carries
        the actual SERVED identity (from the witnessed channel), not the requested pin."""
        start = time.monotonic()
        outcome = await self.run(prompt, timeout=timeout)
        latency = time.monotonic() - start
        logger.info(
            "%s (cli=%s) round %d: %.2fs, served=%s",
            self._config.name, self._config.cli_command, round_number, latency, outcome.actual_model,
        )
        return ModelResponse(
            provider=self._config.name,
            model=outcome.actual_model,
            round_number=round_number,
            content=outcome.content,
            latency_sec=latency,
            token_count=outcome.token_count,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
        )

    # --- ABC hooks: unused (generate is overridden for subprocess transport) ---
    async def _invoke(self, prompt: str) -> Any:  # pragma: no cover
        raise NotImplementedError("CliProvider overrides generate(); _invoke is unused")

    def _parse(self, raw: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("CliProvider overrides generate(); _parse is unused")

    # --- subclass hooks ---
    def _argv(self, prompt: str, model: str) -> list[str]:
        raise NotImplementedError

    def _extract(self, stdout: str, stderr: str) -> CliOutcome:
        raise NotImplementedError


class ClaudeCliProvider(CliProvider):
    """claude CLI seat. Identity in-band via `.modelUsage` (witnessed CL-2, re-witnessed 2026-07-17)."""

    identity_channel = "modelUsage"

    def _argv(self, prompt: str, model: str) -> list[str]:
        return ["-p", prompt, "--output-format", "json", "--tools", "", "--model", model]

    def _extract(self, stdout: str, stderr: str) -> CliOutcome:
        try:
            doc = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(self._config.name, f"CLI parse error: bad JSON: {exc}") from exc
        content = str(doc.get("result") or "").strip()
        if not content:
            raise ProviderError(self._config.name, "CLI parse error: empty .result")
        model_usage = doc.get("modelUsage")
        if not isinstance(model_usage, dict) or not model_usage:
            raise ProviderError(
                self._config.name, "identity-unreadable: no served model in .modelUsage"
            )
        actual_model = next(iter(model_usage))
        usage = doc.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        token_count = (
            (input_tokens or 0) + (output_tokens or 0)
            if (input_tokens is not None or output_tokens is not None)
            else None
        )
        return CliOutcome(
            content=content,
            actual_model=actual_model,
            identity_channel=self.identity_channel,
            token_count=token_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class CodexCliProvider(CliProvider):
    """codex CLI seat. Plain mode: answer on stdout, identity from the stderr `model:` banner
    (the `--json` stream carries no identity — witnessed CX-2, re-witnessed 2026-07-17)."""

    identity_channel = "stderr-banner"
    _MODEL_RE = re.compile(r"^\s*model:\s*(\S+)", re.MULTILINE)
    _TOKENS_RE = re.compile(r"tokens used[:\s]+([\d,]+)", re.IGNORECASE)

    def _argv(self, prompt: str, model: str) -> list[str]:
        return ["exec", "--sandbox", "read-only", "--skip-git-repo-check", "-m", model, prompt]

    def _extract(self, stdout: str, stderr: str) -> CliOutcome:
        content = stdout.strip()
        if not content:
            raise ProviderError(self._config.name, "CLI parse error: empty stdout answer")
        match = self._MODEL_RE.search(stderr)
        if not match:
            raise ProviderError(
                self._config.name, "identity-unreadable: no served model in stderr banner"
            )
        actual_model = match.group(1)
        tok = self._TOKENS_RE.search(stderr)
        token_count = int(tok.group(1).replace(",", "")) if tok else None
        return CliOutcome(
            content=content,
            actual_model=actual_model,
            identity_channel=self.identity_channel,
            token_count=token_count,
        )
