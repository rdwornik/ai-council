# ADR-06: Cost Optimization Strategy

**Date:** 2026-03-29
**Status:** Accepted (partial — Qwen trial pending)
**Decision:** Switch synthesizer to Sonnet 4.6; trial Qwen via OpenRouter as DeepSeek alternative.

**Actions taken:**
- Synthesizer: Opus 4.6 → Sonnet 4.6 (5x cost reduction, equivalent quality in testing)
- Cost tracking: per-call token counts + USD estimates in `DebateMetrics` (surfaced in output)

**Pending:**
- Qwen 3.5 shadow trial — run parallel with DeepSeek, compare output quality vs cost
- OpenRouter as hedge against direct API reliability issues

**Implementation:** `src/metrics.py` (cost rates); `config/settings.yaml` (model IDs); `src/runner.py` (synthesizer pick).
