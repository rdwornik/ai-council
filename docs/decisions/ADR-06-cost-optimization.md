# ADR-06: Cost Optimization Strategy

**Date:** 2026-03-29
**Status:** Accepted (partial — Qwen trial deferred; synthesizer subsequently changed)
**Decision:** Switch synthesizer to Sonnet 4.6; trial Qwen via OpenRouter as DeepSeek alternative.

**Actions taken:**
- Synthesizer: Opus 4.6 → Sonnet 4.6 (5x cost reduction, equivalent quality in testing)
- Cost tracking: per-call token counts + USD estimates in `DebateMetrics` (surfaced in output)

**Subsequent change (see ADR-01 revision 2026-04-30):** Synthesizer switched from Sonnet 4.6 to Gemini for reliability — Sonnet timed out on 5-model transcripts.

**Deferred:**
- Qwen 3.5 shadow trial — not completed; deferred indefinitely
- OpenRouter as hedge against direct API reliability issues — not implemented

**Implementation:** `src/metrics.py` (cost rates); `config/settings.yaml` (model IDs); `src/runner.py` (synthesizer pick).
