# Codex Review — claude-billing-and-mode-scoped-health-gate

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — merge `be58c36`; billing category (`base.py:29`) + mode-scoped health targets. No open remainder. [ARCHIVED 2026-07-18] _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-05-18
**Branch:** `fix/claude-provider-400-and-research-gating`
**HEAD:** `d511b26`
**Diff range:** `main..fix/claude-provider-400-and-research-gating`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

- Diagnose-vs-fix scope: the 400 is an Anthropic billing condition (operator-handled out of band). Code fixes are: (a) new 'billing' classifier category; (b) mode-scoped health gate routing research to summarizer-only non-blocking check.
- Verify classify_error 'billing' detection is correct + ordered safely vs rate_limit / invalid_request.
- Verify _select_health_check_targets covers the documented cases (research-by-flag/alias only; auto-detect path falls back to debate-pool blocking — pre-existing behaviour preserved).
- _check_summarizer_health: confirm non-blocking semantics are correct and that summarizer outage truly falls back to truncation (research/merger.py:184-186).
- Confirm no regression in --mode pick / ideas / judge: full debate health-check, blocking [Y/n] gate still fires.
- Note pre-existing tests/test_research.py::TestDegradationCLIExitCode failures are caused by the live billing condition (real API calls), not by this branch.

---

## Findings
**Critical**
- (none)

**High**
- (none)

**Medium**
- `src/ai_council/cli.py:117`  
  What: explicit research mode skips the preflight entirely when the configured summarizer is absent from `all_providers`, because `_select_health_check_targets()` returns `{}` and `_check_summarizer_health()` immediately no-ops.  
  Why: `build_all_providers()` drops providers that fail construction, including missing API keys (`src/ai_council/runner.py:15-23`). That means the exact cases this branch is trying to surface early, like missing summarizer credentials or provider init failure, produce no startup warning at all. Research then runs for minutes and only later falls back to truncation in `summarize_report()` (`src/ai_council/research/merger.py:156-157`, `185-186`). The branch stays non-blocking, but it does not reliably surface summarizer outage upfront.  
  Fix direction: when `resolve_mode(...) == "research"`, detect `summary_model not in all_providers` as an explicit warning path rather than treating it as “nothing to check”; print a non-blocking `WARN` that the summarizer is unavailable and research will use truncation fallback.

**Low**
- (none)

Assumptions / verification notes: `classify_error()` ordering for `billing` vs `rate_limit` / `invalid_request` looks safe, and `pick` / `ideas` / `judge` still route through the full blocking health gate. I did not treat `tests/test_research.py::TestDegradationCLIExitCode` as a branch regression; based on the branch intent and current code shape, those failures are consistent with the live Anthropic billing condition rather than this diff.
