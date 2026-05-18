# Codex Review — research-degradation-alarm

**Date:** 2026-05-18
**Branch:** `fix/research-panel-degradation-alarm`
**HEAD:** `c81e840`
**Diff range:** `main..fix/research-panel-degradation-alarm`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

- Verify min_successful_providers threshold logic in merger.py (selected_panel denominator vs built panel)
- Confirm exit code 3 propagates cleanly through CLI interactive + inbox loop paths
- Banner placement in print_research_summary (output.py) and save_research_to_file admonition
- ADR-08 reasoning: alternatives rejection, exit-code convention table
- Test coverage of degradation cases (build-time dropout, at-threshold boundary, banner output)
- Backward compat: existing merge_results callers passing no selected_panel

---

## Findings
**Critical**
- `(none)`

**High**
- **Severity:** High  
  **File:line:** `src/ai_council/research/cache.py:75-90,94-115`  
  **What:** Cached research reports drop the new degradation state. `cache_put()` does not persist `degraded` / `failed_count`, and `cache_get()` reconstructs every cached provider result as a synthetic success (`content="(loaded from cache)"`) before returning a default `MergedResearchReport`.  
  **Why:** On a cache hit, `run_research()` reuses that reconstructed report in [src/ai_council/research/runner.py:160-167], so a previously degraded run loses the console banner, loses the markdown warning block, and both CLI paths skip the new `sys.exit(3)` checks in [src/ai_council/cli.py:431-432] and [src/ai_council/cli.py:484-485]. The second run of the same degraded query therefore looks healthy and exits `0`, which breaks ADR-08’s human-visible and machine-detectable contract.  
  **Fix direction:** Persist degradation metadata in cache (`degraded`, `failed_count`, and enough provider status to reconstruct failures accurately), restore it in `cache_get()`, and add a regression test for a degraded report served from cache still producing the banner and exit code `3`.

**Medium**
- **Severity:** Medium  
  **File:line:** `docs/decisions/ADR-08_research-degradation-alarm.md:30,37-39`; `src/ai_council/cli.py:394-395,431-432`  
  **What:** ADR-08’s exit-code table overstates what the inbox path actually guarantees. The ADR says code `1` means hard error, but the `--inbox` research loop catches per-file exceptions, logs/archive-fails them, and continues; only degradation is surfaced at process exit.  
  **Why:** Anyone automating `council --inbox` from the ADR will assume hard batch failures are reflected in the process status, but the implementation can still return `0` unless some run was degraded. That makes the documented exit-code contract inaccurate for batch mode.  
  **Fix direction:** Either narrow the ADR wording to interactive research plus degraded-batch behavior, or make the inbox loop track hard failures separately and exit non-zero at the end with a documented precedence between hard-failure and degraded statuses.

- **Severity:** Medium  
  **File:line:** `tests/test_research.py:1483-1521`; `tests/test_research.py:246-303`  
  **What:** The new coverage exercises fresh interactive CLI runs, threshold boundaries, and banner rendering, but it does not cover the two highest-risk propagation paths: degraded cache hits and `--inbox` batch exit `3`.  
  **Why:** The cache regression above slipped through precisely because no test asserts that a degraded report remains degraded after `cache_put()` / `cache_get()`, and there is no test for the separate inbox aggregation path (`inbox_any_degraded`). Both are public ADR-08 behaviors.  
  **Fix direction:** Add tests that 1) round-trip a degraded `MergedResearchReport` through cache and assert banner/exit `3` on cache hit, and 2) invoke `cli.py --inbox` with mixed files and assert it finishes the batch and exits `3` when any research item is degraded.

**Low**
- `(none)`
