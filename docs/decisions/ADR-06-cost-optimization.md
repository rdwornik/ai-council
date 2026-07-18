# ADR-06: Cost Optimization Strategy

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — cost tracking (`metrics.py`); Qwen/OpenRouter deferred by decision. No open remainder. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-03-29
**Status:** Revised 2026-05-11 (Qwen trial deferred/abandoned)
**Decision:** Switch synthesizer to Sonnet 4.6; trial Qwen via OpenRouter as DeepSeek alternative.

**Actions taken:**
- Synthesizer: Opus 4.6 → Sonnet 4.6 (5x cost reduction, equivalent quality in testing)
- Cost tracking: per-call token counts + USD estimates in `DebateMetrics` (surfaced in output)

**Subsequent change (see ADR-01 revision 2026-04-30):** Synthesizer switched from Sonnet 4.6 to Gemini for reliability — Sonnet timed out on 5-model transcripts.

**Deferred/Abandoned:**
- Qwen 3.5 shadow trial — deferred/abandoned 2026-05-11 per Council debate (synthesizer/panel refresh). Reopen trigger: DeepSeek round-blocking failure rate exceeds 2% per JOURNAL data analysis. Until then, current 5-provider panel composition remains operative.
- OpenRouter as hedge against direct API reliability issues — not implemented

**Implementation:** `src/metrics.py` (cost rates); `config/settings.yaml` (model IDs); `src/runner.py` (synthesizer pick).
