# ADR-08: Research-panel degradation alarm

**Date:** 2026-05-18
**Status:** Accepted

## Context

The 2026-05-18 research-provider health-check audit (`docs/audits/2026-05-18-research-provider-health-check.md` §"Systemic finding — loud or silent") documented that the research subsystem **detects** provider failures (red ✗ in the live status table, per-provider error lines in the console summary, status column in the saved markdown) but does **not alarm**:

> If `openai_mini` fails on every call and `grok` fails on every call, the default 4-provider research panel has been silently running on `perplexity` + `gemini` only — degraded by half, still producing a clean-looking report. The degradation is logged and displayed, not hidden, but there is no aggregate alarm (e.g. "≥ N failures" warning, no metric, no exit code change). Operators relying on the summary section without reading the status section would not notice.

Two specific gaps:

- `merger.py` silently filters errored results out of the merged document (`successful = [r for r in results if not r.error and r.content]`).
- `runner.py` raises `RuntimeError` **only if zero providers can be built** (e.g. all API keys missing). It does not raise when 4 of 4 *built* providers fail at the API call.

A reader of only the summary block — or an automation harness invoking the council programmatically — receives a clean-looking report and exit code 0 from a half-dead panel.

## Decision

Make degradation **un-missable** for humans (banner) and **machine-detectable** for automation (non-zero exit code). The run still completes and emits whatever report the surviving providers produced — no hard abort.

**Mechanism**

- Add `min_successful_providers` (default `3`) to `config/settings.yaml` under `research:`.
- After the parallel research run, compare the count of successful providers against the threshold. If below: set `MergedResearchReport.degraded = True` and populate `failed_count`.
- When `report.degraded`:
  - **Console banner.** `print_research_summary` emits a red `Rule` and one-line summary ("X of N providers failed; report based on Y survivor(s); process will exit with code 3") plus the names of providers that failed at API-call time.
  - **Markdown banner.** `save_research_to_file` inserts a `> [!WARNING] Degraded research panel — …` admonition block at the top of the saved markdown, above the Provider Summary table.
  - **CLI exit code.** Interactive research command exits with code `3`. The `--inbox` batch loop tracks `inbox_any_degraded` and exits `3` after the batch completes (does **not** abort mid-batch).

**Exit-code convention** (verified against Click 8.x in this repo on 2026-05-18):

| Code | Meaning |
|------|---------|
| 0 | Run completed cleanly |
| 1 | Hard error (RuntimeError, missing research config, no providers buildable, `click.Abort`) |
| 2 | Click usage error — framework-reserved (`click.UsageError.exit_code == 2`) |
| 3 | Run completed but degraded (new) |

Code `3` was chosen specifically to avoid collision with Click's framework-reserved `2`.

**Scope note — inbox batch mode.** The exit-code table above applies cleanly to **interactive** research runs. In `--inbox` batch mode the loop catches per-file exceptions, logs them, archives the failed file as `*.failed.md`, and continues processing the next file — hard per-file failures do **not** propagate to the process exit code today. The batch loop currently surfaces *only* degradation: if any research item in the batch returned `degraded=True`, the process exits `3` after the batch completes; otherwise `0`. Tracking hard batch failures as a separate exit-code dimension is out of scope for ADR-08 and is a candidate for a future amendment if operators need it.

**Denominator — selected panel, not built panel**

The threshold is compared against the **selected** panel (the list of provider names selected for this invocation, post `--models` filter), **not** the built panel (providers that successfully instantiated). A provider that drops at build time because its API key is missing is therefore counted as a non-success, identical to an API-call failure.

This boundary is deliberate:

- A configured 4-provider default panel where `XAI_API_KEY` is unset would otherwise build only 3 providers, all 3 succeed → naïve check sees "3/3 = healthy" and the silently missing Grok goes unnoticed. The same operator-visible symptom (a degraded panel producing a clean report) the alarm was built to prevent.
- The trade-off is that running `council -M r --models perplexity` against a default `min_successful_providers: 3` will always trip the alarm. This is the correct behaviour — a 1-provider invocation against a min of 3 *should* warn loudly. Operators who want a single-provider invocation are expected to either lower the threshold for that invocation (future work, not in scope here) or accept the alarm.

## Alternatives considered

- **(a) Hard abort / raise on threshold breach.** Rejected: a single transient failure during a 20-minute deep-research run would kill the entire run and discard partial results. A partial report from surviving providers still has value, especially when summary cost is small relative to the cost already incurred by the survivors.
- **(b) Banner only, no exit-code change.** Rejected: visible to a human reading the summary, but not detectable by automation invoking the council programmatically (e.g. an inbox sweep cron, a scripted research pipeline). The whole point is detectability *outside* the live console.
- **(c) Fraction-based threshold ("≥ 50% must succeed").** Rejected for now in favour of an absolute integer for simplicity. Revisit if panel sizes diverge widely between default (4) and deep (5) or if operators want different sensitivities per-panel.
- **(d) Denominator = built providers only.** Rejected per the "selected panel" reasoning above — would silently hide missing-API-key dropouts, which is the exact failure mode the alarm exists to prevent.

## Consequences

- Degradation is now un-missable for humans (banner appears in both console and saved markdown) and machine-detectable (exit code `3` distinct from Click's `2` and the hard-error `1`).
- The threshold is tunable per deployment via `config/settings.yaml`.
- The inbox loop also exits non-zero on degraded batches, so cron-driven inbox sweeps will surface degradation to whatever wrapper invokes them.
- **Risk:** automation that previously treated council research runs as always-success may now see exit code `3`. Wrappers may need updating to either tolerate `3` as "partial success" or escalate it as a real failure depending on their policy.
- Existing callers of `merge_results()` that pass no `selected_panel` / `min_successful` are unaffected — `degraded` defaults to `False`, `failed_count` falls back to counting errored results for informational use only.

## Evaluation / revisit triggers

Reasons this decision could prove wrong and worth revisiting:

- Operators still miss degradation despite the banner — the banner copy or placement is insufficiently prominent.
- The non-zero exit causes spurious failures in downstream automation that cannot tolerate `3`, and the cost of updating wrappers proves higher than expected.
- A hard abort (alternative a) proves more appropriate in practice — e.g. an automation pipeline acting on partial results does more harm than discarding the run.
- The single absolute threshold is too coarse — operators want fraction-based or per-panel thresholds.
- Tuning data accumulates suggesting the default `3` is wrong for the typical workload.

## Related

- 2026-05-18 audit: `docs/audits/2026-05-18-research-provider-health-check.md`
- ADR-05 (research-mode integration) — establishes the research subsystem this ADR amends
- Per ecosystem ADR-34, new local ADRs use the underscore-naming convention; ADR-01..07 are grandfathered per ecosystem ADR-29
