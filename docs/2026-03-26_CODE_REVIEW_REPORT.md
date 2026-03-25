# Code Review Report — AI Council

**Date:** 2026-03-26
**Branch:** `main` (post Phase 2 + Phase 3 merge)
**Reviewer:** Claude Sonnet 4.6
**Scope:** Phase 2 (cost tracking, CouncilRunner, RunPolicy) + Phase 3 (retry logic, graceful degradation)

---

## Summary

```
REPO:          ai-council
TESTS:         121 unit passed, 0 failed (not integration, not envcheck)
RUFF:          32 errors — 25 auto-fixable, 7 manual
COMMITS:       4 clean squashed commits on main (d56279d → f4f2329)
FILES CHANGED: src/runner.py, src/policy.py, src/metrics.py, src/models.py,
               src/debate.py, src/synthesis.py, src/output.py, src/cli.py,
               config/config_loader.py, config/settings.yaml, tests/ (10 files)
```

---

## Gotcha Compliance

Checked against `~/.claude/skills/gotchas/gotchas.md` (ai-council section) and `CLAUDE.md`:

| Gotcha | Status | Notes |
|--------|--------|-------|
| Windows cp1252: no Unicode in Rich progress callbacks | ✅ PASS | runner.py progress uses `"OK"`, `"Running debate rounds..."` — ASCII only |
| MockProvider ABC: `generate` in class body + shadowed by AsyncMock | ✅ PASS | conftest.py:125 has class-body definition; `__init__` shadows at runtime |
| pytest-asyncio: `asyncio_mode = auto` in pytest.ini | ✅ PASS | pytest.ini confirmed |
| Critique template uses `{previous_responses_anonymized}` | ✅ PASS | debate.py:155 |
| google-genai: use `client.aio.models.generate_content()` | ✅ PASS | gemini.py unchanged; no regression |
| Provider isolation: do NOT merge xai.py / openai_provider.py | ✅ PASS | both files untouched; no shared base beyond ABC |

---

## Linting Issues (ruff)

### Source files — 2 fixable

**`src/runner.py`** — 2 unused imports (F401)

| Line | Import | Why it's there | Action |
|------|--------|----------------|--------|
| 5 | `ModelConfig` | Carried over from pre-refactor cli.py | Remove |
| 9 | `RunPolicy` | Came in with Phase 2; used indirectly via RunRequest | Remove |

These are safe auto-fixes: `py -m ruff check src/runner.py --fix`

### Test files — 30 errors (low priority)

Most are unused imports left behind as tests evolved. Categories:

| File | Issues | Examples |
|------|--------|---------|
| `tests/conftest.py` | F401 (os, MagicMock), E402 (by design) | `os` and `MagicMock` never referenced |
| `tests/test_metrics.py` | F401 (dataclass, field, Path, pytest, DebateMetrics) | Imports from planning; none used in final tests |
| `tests/test_output.py` | F401 (ModelResponse, Question, Round) | Fixtures cover these; direct imports redundant |
| `tests/test_policy.py` | F401 (pytest), I001 (import order) | `pytest` imported but no `pytest.raises` used |
| `tests/test_runner.py` | I001 (import order), E501 (1 line) | Line 249 is 124 chars |
| `tests/test_cli.py` | I001 (import order) | stdlib before third-party |
| `tests/test_config.py` | F401 (os) | Never used |
| `tests/test_synthesis.py` | F401 (Question) | Unused after fixture refactor |
| `tests/test_inbox.py` | F401 (pytest) | No raises/marks used |

**Note on conftest.py E402:** The 3 E402 violations are intentional — `load_dotenv()` must run before importing `config.config_loader` (which reads `os.environ` at import time). This is a known pattern in the project. Consider adding `# noqa: E402` to suppress false positives.

### Line length (E501) — 4 violations

| File | Char count | Line content |
|------|-----------|-------------|
| `config/config_loader.py:57` | 125 | `InboxConfig` default factory lambda |
| `src/cli.py:98` | 122 | `--inbox-dir` click option help string |
| `tests/conftest.py:37` | 126 | Critique template string in fixture |
| `tests/test_runner.py:249` | 124 | Async test function signature |

---

## Design Review

### 1. `RunPolicy.should_abort()` is dead code

**File:** `src/policy.py:23-33` / `src/debate.py`

`RunPolicy.should_abort()` is defined but never called. `run_debate()` directly checks `if not responses` and `if round_num == 1` instead of delegating to the policy. The `abort_if_round1_below` field on `RunPolicy` is therefore also inert.

```python
# policy.py — defined but unused by debate.py
def should_abort(self, active_count: int, round_number: int) -> bool:
    if round_number == 1 and active_count < self.abort_if_round1_below:
        return True
    ...
```

**Risk:** Low (correct behavior, just inconsistent abstraction — the policy doesn't govern abort decisions).
**Recommendation:** Either wire `_policy.should_abort(len(responses), round_num)` into `debate.py`, or remove `should_abort()` and `abort_if_round1_below` from `RunPolicy` to avoid confusion.

---

### 2. `_call_provider` mutates provider config directly (timeout patch)

**File:** `src/debate.py:52-92`

The retry logic temporarily patches `provider._config.timeout_sec` to get 1.5x timeout on the retry call:

```python
cfg = getattr(provider, "_config", None)
if cfg is not None and hasattr(cfg, "timeout_sec"):
    original_timeout = cfg.timeout_sec
    cfg.timeout_sec = int(original_timeout * 1.5)
    ...
try:
    return await provider.generate(prompt, round_number)
finally:
    cfg.timeout_sec = original_timeout  # restored
```

**Risk:** Low in current architecture (single-threaded asyncio, finally block restores). **Medium if providers are ever shared across concurrent runs** — a second task could observe the mutated timeout during the retry window.

**Observation:** This relies on `_config` being a mutable object with a `timeout_sec` attribute — a convention, not a contract. If a provider doesn't have `_config` (e.g. a mock or a future provider), the retry falls through silently with no timeout bump (correct fallback behaviour).

**Recommendation:** No immediate action required, but document the timeout-mutation pattern in `AIProvider` base class or policy docstring so future provider authors know to expose `_config`.

---

### 3. `provider_statuses` semantics: "ok" = succeeded at least once

**File:** `src/debate.py:169-172`

The status is flipped to `"ok"` when a provider succeeds in **any** round, not just all rounds. A provider that fails round 1 but succeeds round 2 will show `"ok"`. This is intentional and reasonable, but the semantics aren't documented.

**Recommendation:** Add a one-line comment at the flip site: `# "ok" means at least one response succeeded across all rounds`

---

### 4. `token_count` vs `input_tokens`/`output_tokens` split in `ModelResponse`

**File:** `src/models.py:25-27`

```python
token_count: int | None  # combined total; kept for display/backward compat
input_tokens: int | None = None
output_tokens: int | None = None
```

`token_count` is the original field (kept for display/backward compat). `input_tokens` and `output_tokens` were added for cost calculation. They can be inconsistent: `token_count` may not equal `input_tokens + output_tokens` if a provider populates them independently.

**Risk:** Low (display only vs cost calculation are separate paths).
**Recommendation:** No immediate change needed. Document in `ModelResponse` that `token_count` is display-only and cost logic uses `input_tokens`/`output_tokens`.

---

## Issues Not Found

The following were checked and found clean:

- ✅ No bare `except:` clauses — all `except Exception` or `except ProviderError`
- ✅ No `print()` in src/ — all output via `logger` or Rich `console`
- ✅ No hardcoded API keys or paths
- ✅ `field(default_factory=dict/list)` used correctly for mutable defaults in all dataclasses
- ✅ Async safety: `asyncio.gather()` used for parallel provider calls; no blocking calls in async context
- ✅ `synthesize()` degradation passthrough correct — `degraded`, `degradation_summary`, `provider_statuses` flow from `DebateOutcome` → `synthesize()` → `DebateResult`
- ✅ Output (Rich + markdown) shows degradation banner when `result.degraded is True`
- ✅ `save_to_file()` correctly writes failed providers list when `result.degraded`

---

## Action Items

| Priority | File | Issue | Fix |
|----------|------|-------|-----|
| P1 | `src/runner.py` | Unused imports: `ModelConfig`, `RunPolicy` (F401) | `py -m ruff check src/runner.py --fix` |
| P2 | `src/policy.py` | `should_abort()` never called; `abort_if_round1_below` inert | Wire into `debate.py` or remove |
| P3 | `tests/conftest.py:15-17` | E402 false positives | Add `# noqa: E402` to suppress |
| P3 | Multiple test files | Unused imports (F401, I001) | `py -m ruff check tests/ --fix` |
| P4 | `src/debate.py:172` | Undocumented "ok" semantics | Add inline comment |
| P4 | `src/models.py` | `token_count` vs split tokens undocumented | Add docstring clarification |

---

## Test Count Correction

CLAUDE.md previously stated 72 (pre-Phase-2) then was incorrectly updated to 79. Actual count as of 2026-03-26:

```
pytest tests/ -m "not integration and not envcheck"
121 passed, 6 deselected
```

CLAUDE.md has been corrected to 121.
