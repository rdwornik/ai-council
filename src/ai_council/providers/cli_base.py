"""CLI-subscription provider adapters (claude, codex) — pure transport behind AIProvider.

ADR-12 safety floor on EVERY call (all witnessed 2026-07-05, re-witnessed 2026-07-17):
- **cwd = a fresh scratch dir, never a repo** — the PRIMARY isolation (`claude -p --tools ""`
  still ingests cwd `CLAUDE.md`; CL-3). A per-call TemporaryDirectory leaves no residue.
- **prompt delivered via stdin, then EOF** — NOT as an argv element. Discovered at build time
  (2026-07-17): a multi-line prompt passed as a command-line arg is mangled by the Windows npm
  `.CMD` shim (cmd.exe treats embedded newlines as command terminators), so the seat would
  receive a truncated prompt — a prompt-parity (I3) break. Both `claude -p` and `codex exec`
  read the prompt from stdin when no prompt arg is given; writing it and closing stdin restores
  parity AND keeps codex from hanging (the CX-1 hang is on an *open* stdin, not a closed one).
- **read-only / tools-off flags** + an **explicit model pin** on every call (per-call pin rule:
  codex otherwise serves its own default, now `gpt-5.6-sol`).
- **`timeout_sec` is the hard kill** (I6).

Identity is read from each CLI's witnessed channel — claude in-band `.modelUsage`; codex
plain-mode **stderr** banner `model:` (the `--json` stream carries none). These adapters do
ONLY transport + identity read and raise a classifiable ProviderError on failure; the
admission gate + same-seat API fallback live in `seat_router.py` (a debate-engine concern,
L-CLI IF#2). Separate classes, no provider-merge (ADR-12 / CLAUDE 5.7).

Security posture (defense in depth):
- **Allowlisted subprocess env** — the CLI child receives ONLY a minimal set of non-secret
  system variables (`_ENV_ALLOWLIST`); every credential is dropped regardless of its name. This
  forces subscription auth (the cost-lane intent — ADR-12's "key-strip guard") AND denies a
  prompt-injected agent any inherited secret to exfiltrate.
- **Process-tree kill** on timeout/cancellation — the Windows `.CMD` shim spawns a node child;
  killing only the shim would orphan it, so we terminate the whole tree.
- **Residual (accepted + logged, ADR-12 §3):** codex `exec` is an agent whose `--sandbox
  read-only` policy blocks writes but permits file *reads*. A malicious prompt could, in
  principle, read a local file and surface its contents in the answer. Containment is
  doctrinal: `important`/untrusted content stays on the API lane (ADR-12 §4, all-API), so
  untrusted inbox briefs must not be routed to CLI seats. A stricter no-read sandbox is not
  offered by codex exec; hardening beyond this is out of this arc's scope.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from ai_council.models import ModelResponse
from ai_council.providers.base import AIProvider, ProviderError
from config.config_loader import ModelConfig

logger = logging.getLogger(__name__)

# The ONLY (non-secret) environment variables passed to a CLI subprocess. An ALLOWLIST, not a
# denylist: anything not named here — every API key, AWS/GCP/DB credential, token, regardless
# of its name — is dropped, so a prompt-injected agent has no inherited secret to exfiltrate.
_ENV_ALLOWLIST = frozenset({
    "PATH", "PATHEXT", "COMSPEC", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP",
    "TZ", "LANG", "LC_ALL", "HOME", "HOMEDRIVE", "HOMEPATH", "USERPROFILE", "USERNAME",
    "USER", "LOGNAME", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "NODE_EXTRA_CA_CERTS",
})


# Proxy vars are allowlisted for network functionality, but a proxy URL may embed
# `user:password@` userinfo — a credential. Strip the userinfo, keep the host:port.
_PROXY_USERINFO = re.compile(r"://[^/@]*@")
_PROXY_VARS = frozenset({"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"})


def _scrubbed_env() -> dict[str, str]:
    """The minimal non-secret environment a CLI subprocess needs (an allowlist). Every
    credential is dropped by exclusion; proxy URLs additionally have any `user:pass@` userinfo
    stripped. The CLIs auth via their own subscription stores (~/.claude, ~/.codex), never env
    keys — verified live 2026-07-17 that both run under this allowlist."""
    env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_ALLOWLIST}
    for key, value in env.items():
        if key.upper() in _PROXY_VARS:
            env[key] = _PROXY_USERINFO.sub("://", value)
    return env


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate the child AND its descendants. On Windows a `.CMD` shim's node child would
    otherwise survive `proc.kill()`; on POSIX the process group is signalled."""
    if proc.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # best-effort cleanup; never mask the original failure
        try:
            proc.kill()
        except Exception:
            pass


def _usage_int(provider: str, usage: dict, field: str) -> int | None:
    """Read one integer token field, or raise ProviderError (P1-2).

    A CLI that books a token count as a string ("1200") used to escape _extract as a bare
    TypeError on the sum. Returns None for an absent/null field so the caller's existing
    `or 0` / None-means-unknown semantics are preserved exactly.
    """
    raw = usage.get(field)
    if raw is None:
        return None
    # bool is an int subclass — a JSON `true` here is malformed, not a count of 1.
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ProviderError(
            provider,
            f"CLI parse error: unreadable .usage.{field} "
            f"(expected int, got {type(raw).__name__})",
        )
    return raw


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
                env=_scrubbed_env(),  # same credential boundary as run() — never inherit secrets
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
        argv = [self._exe, *self._argv(self._cli_model)]
        # New process group so the whole tree (shim + node child) is terminable together.
        group_kwargs: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        # Fresh scratch cwd per call — PRIMARY isolation; auto-removed (no leftover).
        with tempfile.TemporaryDirectory(prefix="council-cli-") as scratch:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=scratch,
                    env=_scrubbed_env(),  # strip council API keys -> subscription auth, no exfil
                    stdin=subprocess.PIPE,  # prompt is written then stdin is closed (EOF)
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **group_kwargs,
                )
            except Exception as exc:  # spawn failure (binary missing, etc.)
                raise ProviderError(self._config.name, f"CLI process error: {exc}") from exc

            try:
                raw_out, raw_err = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode("utf-8")), timeout=effective_timeout
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                _kill_process_tree(proc)
                await proc.wait()
                raise ProviderError(
                    self._config.name, f"CLI timed out after {effective_timeout}s"
                ) from exc
            finally:
                # On cancellation (or any early exit) the process must not be orphaned.
                if proc.returncode is None:
                    _kill_process_tree(proc)

        stdout = raw_out.decode("utf-8", errors="replace")
        stderr = raw_err.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise ProviderError(
                self._config.name,
                f"CLI process error: exit {proc.returncode}: {stderr.strip()[:200]}",
            )
        # P1-2: _extract sat OUTSIDE every guard here, so a raw parser exception escaped run()
        # -> generate() -> try_cli's `except ProviderError` (no match) -> the debate's gather,
        # cancelling every sibling seat with no API fallback and no fallback_event. The subclass
        # parsers raise ProviderError themselves; this envelope is the structural backstop so a
        # future parser bug degrades ONE seat instead of the round. ProviderError passes through
        # unwrapped — its message is what classify_cli_failure maps to a cause token.
        try:
            return self._extract(stdout, stderr)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - contract envelope, see above
            raise ProviderError(
                self._config.name, f"CLI parse error: {type(exc).__name__}: {exc}"
            ) from exc

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
            backend="cli",  # subscription lane — $0 marginal cost in metrics
        )

    # --- ABC hooks: unused (generate is overridden for subprocess transport) ---
    async def _invoke(self, prompt: str) -> Any:  # pragma: no cover
        raise NotImplementedError("CliProvider overrides generate(); _invoke is unused")

    def _parse(self, raw: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("CliProvider overrides generate(); _parse is unused")

    # --- subclass hooks ---
    def _argv(self, model: str) -> list[str]:
        """The CLI flags (WITHOUT the prompt — the prompt is delivered on stdin)."""
        raise NotImplementedError

    def _extract(self, stdout: str, stderr: str) -> CliOutcome:
        raise NotImplementedError


class ClaudeCliProvider(CliProvider):
    """claude CLI seat. Identity in-band via `.modelUsage` (witnessed CL-2, re-witnessed 2026-07-17)."""

    identity_channel = "modelUsage"

    def _argv(self, model: str) -> list[str]:
        return ["-p", "--output-format", "json", "--tools", "", "--model", model]

    def _extract(self, stdout: str, stderr: str) -> CliOutcome:
        try:
            doc = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(self._config.name, f"CLI parse error: bad JSON: {exc}") from exc
        # json.loads guarantees VALID JSON, not an OBJECT. An array or scalar on a CLI
        # error/telemetry path used to reach doc.get() and raise a bare AttributeError (P1-2).
        if not isinstance(doc, dict):
            raise ProviderError(
                self._config.name,
                f"CLI parse error: expected a JSON object, got {type(doc).__name__}",
            )
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
        # `or {}` only rescues FALSY values — a truthy non-dict reached .get() below and raised
        # a bare AttributeError; non-numeric fields raised a bare TypeError on the sum (P1-2).
        if not isinstance(usage, dict):
            raise ProviderError(
                self._config.name,
                f"CLI parse error: unreadable .usage (expected object, got {type(usage).__name__})",
            )
        # Real prompt input = fresh input + newly-cached + cache-read: the claude CLI books most
        # of a multi-paragraph prompt to cache_creation_input_tokens, so the bare usage.input_tokens
        # under-reports it (witnessed: input_tokens=1 while cache_creation_input_tokens=4641 for a
        # real prompt — F-M2). cache_read is ~0 for a single-shot seat call (--tools "", one turn),
        # so the agentic ~8x cache-read inflation caveat does not apply; this is a token COUNT, not
        # a spend cap (CLI cost is $0 regardless).
        name = self._config.name
        input_tokens: int | None = (
            (_usage_int(name, usage, "input_tokens") or 0)
            + (_usage_int(name, usage, "cache_creation_input_tokens") or 0)
            + (_usage_int(name, usage, "cache_read_input_tokens") or 0)
        ) if usage else None
        output_tokens = _usage_int(name, usage, "output_tokens") if usage else None
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

    def _argv(self, model: str) -> list[str]:
        return ["exec", "--sandbox", "read-only", "--skip-git-repo-check", "-m", model]

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
        # codex 0.144.5 prints "tokens used\n<N>" (the count on its own line — the [:\s]+ class
        # spans the newline). It is a SINGLE combined total with no input/output split, so record
        # it as output_tokens: metrics.build_call_metrics sums input_tokens/output_tokens and never
        # reads token_count, so a codex call would otherwise book 0 tokens in the sidecar (F-M1).
        tok = self._TOKENS_RE.search(stderr)
        # The ([\d,]+) class matches a digit-free COMMA RUN, so "tokens used: ," produced
        # int("") -> a bare ValueError that killed the round (P1-2). `if tok else None` guards
        # only a MISSING match, never a matched-but-digit-free one. An unreadable count is
        # telemetry, not the answer — degrade it to None and keep the seat.
        digits = tok.group(1).replace(",", "") if tok else ""
        token_count = int(digits) if digits.isdigit() else None
        return CliOutcome(
            content=content,
            actual_model=actual_model,
            identity_channel=self.identity_channel,
            token_count=token_count,
            output_tokens=token_count,
        )
