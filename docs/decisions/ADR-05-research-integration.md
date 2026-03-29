# ADR-05: Research Mode Integration

**Date:** 2026-03-25
**Status:** Accepted
**Decision:** Option C staged approach — o4-mini as MVP, o3-deep as `--deep` only.

**Options considered:**
- A: Single provider (Perplexity only) — insufficient depth
- B: All providers default — too slow/expensive for routine use
- C: 3 fast providers default + o3-deep behind `--deep` flag

**Default providers:** Perplexity sonar-pro, o4-mini-deep-research, Gemini + Google Search grounding.
**Deep provider:** o3-deep-research (~45 min, $10+, `--deep` flag only).

**Implementation:** `src/research/runner.py`; bypasses debate pipeline entirely; file cache under
`~/.ai-council/research_cache/` with 7-day TTL; `asyncio.wait()` + progressive Rich display.
