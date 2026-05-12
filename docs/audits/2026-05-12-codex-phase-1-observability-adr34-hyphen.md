# Codex Review — phase-1-observability-adr34-hyphen

**Date:** 2026-05-12
**Branch:** `feat/phase-1-and-hyphen-cli`
**HEAD:** `9baf8d4`
**Diff range:** `main..feat/phase-1-and-hyphen-cli`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

(none specified)

---

## Findings
I couldn’t find a repo-local `AGENTS.md` in this checkout, so I used the requested `Critical / High / Medium / Low` bands directly.

**Critical**
- (none)

**High**
- (none)

**Medium**
- Severity: `Medium` | File: `src/ai_council/synthesis.py:99` | What: the new synthesis observability is dropped entirely on synthesis failures. | Why: the `ProviderError` path logs and re-raises before any `DebateResult.synthesis_metrics` is created, and `_metrics.json` only emits synthesis data when `result.synthesis_metrics` exists (`src/ai_council/output.py:303`). Timeouts and rate limits are the cases operators most need to inspect, but this branch preserves them only in logs, despite the changelog/journal claiming emitted per-synthesis metrics. | Fix direction: persist a failure-state synthesis metrics record before re-raising, or introduce a failure artifact/result shape that carries synthesis observability even when synthesis aborts.

**Low**
- Severity: `Low` | File: `src/ai_council/synthesis.py:117` | What: `synth_timeout_flag` is hard-coded to `False`, so the new field never reflects reality. | Why: the branch adds `SynthesisMetrics.synth_timeout_flag` as an explicit observability datum (`src/ai_council/models.py:96`), but the success path always writes `False` and the timeout path never serializes a metrics object at all. That makes the field misleading and effectively dead. | Fix direction: derive the flag from actual timeout behavior or remove it until the code can populate it truthfully on both success and failure paths.

- Severity: `Low` | File: `src/ai_council/synthesis.py:116` | What: `synth_latency_seconds` stores provider-reported `latency_sec` instead of the wall-clock latency the function just measured. | Why: the code computes `time.monotonic() - synth_start` on both paths (`src/ai_council/synthesis.py:100`, `111`) but then ignores it on success and records `synthesis_response.latency_sec`. If provider latency excludes retries/client overhead, the new observability metric underreports the actual end-to-end synthesis time and is inconsistent with the failure-path logging. | Fix direction: record the measured wall-clock duration in `SynthesisMetrics`, and keep provider-reported latency as a separate field only if that distinction is useful.

- Severity: `Low` | File: `docs/SYNTHESIS-QUALITY-RUBRIC.md:24` | What: the new rubric links to a hyphenated `2026-05-11` transcript path that likely did not exist yet under the branch’s own migration rules. | Why: the same diff says the filename flip is “going forward only” and that “historical transcripts [are] unchanged” (`CHANGELOG.md:10`), so a transcript generated on `2026-05-11` would still be expected to use the old `council_out_...` form. That leaves the provenance pointer misleading or dead. | Fix direction: point to the actual historical filename, or make the reference non-format-specific.
