# ADR-01: Synthesizer Selection

**Date:** 2026-02-25
**Status:** Revised (2026-03-29)
**Decision:** Claude Sonnet 4.6 as default non-participating synthesizer.

**Context:**
Initial Council vote selected Claude Opus 4.6 — judicial temperament, low sycophancy, best for impartial synthesis.
Revised 2026-03-29: switched to Sonnet 4.6 for 5x cost reduction with equivalent synthesis quality.

**Implementation:** `pick_synthesizer()` in `src/runner.py`; provider `claude-sonnet` in `src/providers/anthropic.py`.

Revised 2026-04-30: switched to Gemini for reliability — Sonnet timed out on 5-model transcripts.

**Fallback:** If all panel models are Claude, synthesizer falls back with `is_participant=True` and a warning.
