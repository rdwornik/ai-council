# Codex Review — debate-resilience-final

**Date:** 2026-07-21
**Branch:** `worktree-fix-debate-resilience`
**HEAD:** `3dacaec`
**Diff range:** `4badad0..HEAD`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

FINAL verification pass. Your previous follow-up (docs/audits/2026-07-21-codex-debate-resilience-followup.md) raised one HIGH: build_failed_synthesis_result read .response by duck-typing, so an SDK exception carrying an HTTP response under that name would raise AttributeError inside the preservation handler and lose all artifacts.
Confirm: (1) the isinstance(error, EmptySynthesisError) fix fully closes that hole and cannot be defeated by a subclass or any other exception shape; (2) no NEW defect is introduced; (3) nothing else in build_failed_synthesis_result can raise while constructing the preserved result -- walk every attribute access and call on that path, since anything that raises there masks the original error and loses the artifacts.

---

## Findings
## CRITICAL

(none)

## HIGH

### HIGH — src/ai_council/synthesis.py:91 — `isinstance` still trusts hostile subclasses

**What:** An `EmptySynthesisError` subclass can override `.response` or supply a non-`ModelResponse`, causing preservation to raise during attribute access or metrics construction.

**Why:** This masks the original synthesis failure and loses the completed debate artifacts; the added test covers foreign exceptions but not subclasses.

**Fix direction:** Accept the billed response only from the exact `EmptySynthesisError` type and validate it as a `ModelResponse`; add hostile-subclass coverage.

### HIGH — src/ai_council/synthesis.py:70 — exception classification/stringification can raise

**What:** `classify_error(error)` and `f"{error}"` invoke exception stringification without protection; an exception with a failing `__str__` aborts result construction.

**Why:** Any such SDK or custom exception masks the original failure before artifacts can be preserved.

**Fix direction:** Derive classification and diagnostic text through a non-raising helper with a constant fallback, ensuring both classification and formatting reuse the safe value.

## MEDIUM

(none)

## LOW

(none)
