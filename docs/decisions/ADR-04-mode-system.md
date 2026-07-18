# ADR-04: Mode System (pick/ideas/judge/research)

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — `detect_mode()` (`mode_detector.py:25`); modes block in `settings.yaml`. No open remainder. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-03-22
**Status:** Accepted
**Decision:** Four debate modes with auto-detection from question text.

**Modes:**
| Mode | Purpose | Default Rounds |
|------|---------|----------------|
| `pick` | Choose between options | 2 |
| `ideas` | Brainstorm / surface unknowns | 1 |
| `judge` | Evaluate a proposal | 2 |
| `research` | Multi-source web research | — |

**Implementation:** `detect_mode()` in `src/mode_detector.py`; mode templates in `config/settings.yaml` under `modes:`.
`-M` flag; aliases `p/i/j/r`; auto-detect via cheap LLM call with 5s interactive confirm.

**Trade-off:** Auto-detection adds latency (~1s) and LLM cost; accepted — UX benefit outweighs cost.
