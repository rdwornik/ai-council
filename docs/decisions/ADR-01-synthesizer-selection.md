# ADR-01: Synthesizer Selection

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — `pick_synthesizer()`/`exclude_synthesizer_from_panel()` (`runner.py`); default openai per Revised 2026-07-18 (`settings.yaml`); residual: #2/#3 (amendment codification). _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-02-25
**Status:** Revised (2026-07-18)
**Decision:** Gemini as default non-participating synthesizer.

**Context:**
Initial Council vote selected Claude Opus 4.6 — judicial temperament, low sycophancy, best for impartial synthesis.
Revised 2026-03-29: switched to Sonnet 4.6 for 5x cost reduction with equivalent synthesis quality.

**Implementation:** `pick_synthesizer()` in `src/runner.py`; provider `claude-sonnet` in `src/providers/anthropic.py`.

Revised 2026-04-30: switched to Gemini for reliability — Sonnet timed out on 5-model transcripts.

**Fallback:** If all panel models are Claude, synthesizer falls back with `is_participant=True` and a warning.

Revised 2026-07-18 (Epic B — BACKLOG #2/#3): switched the default synthesizer **Gemini → OpenAI**, an evidence-based flip per the operator ruling recorded in `docs/audits/2026-07-17-synthesizer-ruling-gemini-to-openai.md` (grounded in the EPI-1 scoring corpus; #24 closed by that ruling). Branch A shipped as the durable config default — `config/settings.yaml` `synthesizer: openai`. **Cost-optimization principle (#3):** the default is chosen by balancing measured synthesis quality against per-run cost, not by assumption. The line-5 **Decision** text is retained unedited per the immutability rule (CLAUDE.md §5 item 3, ADR-14); this Revised marker is the live default of record.
