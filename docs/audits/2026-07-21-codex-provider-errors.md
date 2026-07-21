# Codex Review — provider-errors

**Date:** 2026-07-21
**Branch:** `worktree-fix-provider-errors`
**HEAD:** `da69153`
**Diff range:** `main..worktree-fix-provider-errors`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

- P1-1: _parse now wrapped in its own try in base.py generate(). Does the ProviderError contract now hold for ALL paths? Any remaining unwrapped raise out of generate()?
- P1-3: AIProvider._client_for_loop caches an SDK client keyed on the running event loop, compared by object identity. Is there any way a stale client survives a loop swap, or a client leaks across loops? Is holding a strong ref to the loop a problem?
- P1-7: classify_error rewritten as billing-markers then typed dispatch over the __cause__ chain then hardened string fallback. Check the ordering is sound and that no category became unreachable. Word-boundary regex for HTTP codes - any false negative?
- Behavior change: retry eligibility at debate.py:68 flips for some inputs. Is that correctly scoped?
- Test seam _bind_client sets private _client/_client_loop. Is that fragile?

---

## Findings
## Critical

(none)

## High

### [HIGH] src/ai_council/providers/base.py:107 — Sentence-ending periods hide HTTP status codes

**What:** `_http_code_in()` rejects codes adjacent to `.`, so ordinary messages such as `HTTP 429.` or `status 503.` fall through to `unknown`.  
**Why:** `debate.py:68` consequently skips retries for transient rate-limit or server failures, changing runtime behavior.  
**Fix direction:** Exclude dots only when they form numeric/version strings, and add regression tests for status codes followed by sentence punctuation.

### [HIGH] src/ai_council/providers/base.py:142 — Typed 400 dispatch masks content-policy failures

**What:** A `BadRequestError`/HTTP 400 containing `content_policy` or `safety` returns `invalid_request` before the string fallback can classify it as `content_policy`.  
**Why:** Typed OpenAI/Anthropic policy rejections now bypass the documented category and produce incorrect health-check diagnostics.  
**Fix direction:** Detect content-policy markers before generic typed 400 dispatch, after billing, and test a wrapped 400 policy error.

### [HIGH] src/ai_council/providers/base.py:366 — Rebinding abandons the previous async SDK client

**What:** A loop change overwrites `_client` without closing its async HTTP pool, while `_client_loop` retains the closed originating loop.  
**Why:** Health checks and inbox runs use successive `asyncio.run()` loops, so clients can survive their owning loop and cleanup may occur late or against the wrong/closed loop, causing leaked transports or `Event loop is closed` failures.  
**Fix direction:** Add an explicit async client lifecycle that closes each client on its originating loop before that loop exits; test closure as well as reconstruction.

## Medium

(none)

## Low

(none)

Read-only review; tests were not executed. `git diff --check` passed.
