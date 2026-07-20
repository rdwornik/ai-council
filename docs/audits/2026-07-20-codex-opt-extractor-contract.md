# Codex Review — opt-extractor-contract

**Date:** 2026-07-20
**Branch:** `fix/opt-extractor-contract`
**HEAD:** `e18c940`
**Diff range:** `main..fix/opt-extractor-contract`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

- _top_level_bullets rewrite: does _BULLET_RE consume the EXACT list marker and never a payload character? Try 3D, 2026, 401k, C++, 50%.
- _unwrap_emphasis: is the paired-delimiter regex safe? Check intra-word underscores (snake_case), lone operators (3 * 4), unpaired delimiters, nesting, and catastrophic backtracking on pathological input.
- _THEMATIC_BREAK_RE: correct and non-overreaching?
- Did the change preserve the {items, source, heading} value shape and _extracted_options signature?
- Any regression vs the old character-strip behaviour on inputs the new tests do not cover?

---

## Findings
```markdown
## CRITICAL

(none)

## HIGH

### [HIGH] src/ai_council/output.py:789 — Emphasis unwrapping corrupts literal Markdown and code

**What:** `_EMPHASIS_RE` ignores escaping and code-span boundaries; for example, `` `__init__` `` becomes `init`, `\*literal\*` becomes `\literal\`, and `x *** y` becomes `x * y`.
**Why:** Valid option payload is silently altered on the delegation surface, including identifiers and literal operators.
**Fix direction:** Use a Markdown-aware linear scanner that preserves escaped delimiters and treats code-span contents atomically.

### [HIGH] src/ai_council/output.py:789 — Pathological input causes quadratic regex runtime

**What:** The unanchored lazy `(.+?)` retries from every potential opener when no valid closer exists; a 30,000-character `" *a"` sequence took approximately 4.4 seconds.
**Why:** Model output permits up to 16,384 tokens, so a delimiter-heavy single-line option can stall synchronous verdict generation for many seconds.
**Fix direction:** Replace the backtracking regex with a single-pass delimiter scanner or otherwise guarantee linear-time matching.

### [HIGH] src/ai_council/output.py:825 — List-wrapped thematic breaks leak into options

**What:** The thematic-break check runs before removing the list marker, so `- * * *` and `* - - -` become extracted items; the old implementation discarded both.
**Why:** Formatting-only separators can appear as fabricated `options_considered` entries.
**Fix direction:** After matching the list marker, also reject payloads consisting solely of a valid thematic break before emphasis unwrapping.

## MEDIUM

(none)

## LOW

(none)
```

The hinted `3D`, `2026`, `401k`, `C++`, and `50%` payloads are preserved. The `{items, source, heading}` shape and `_extracted_options` signature are unchanged. All nine new targeted tests passed.
