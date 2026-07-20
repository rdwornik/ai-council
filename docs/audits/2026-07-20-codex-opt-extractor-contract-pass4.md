# Codex Review — opt-extractor-contract-pass4

**Date:** 2026-07-20
**Branch:** `fix/opt-extractor-contract`
**HEAD:** `66d3230`
**Diff range:** `main..fix/opt-extractor-contract`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

Pass 4. Prior passes found 3 HIGH each (regex emphasis; then the replacement scanner; then escape-blind code-span pre-pass, whitespace-only flanking, and the \d+ ordered marker). All repaired.
Review the CURRENT state ONLY for convergence:
- _pair_code_spans, _unwrap_emphasis, _is_punctuation, _top_level_bullets.
- Report ONLY defects demonstrable with a concrete input string, showing actual vs expected output.
- Do NOT restate findings that are already fixed. If you find nothing demonstrable, say so plainly.
State explicitly whether the {items, source, heading} shape and _extracted_options signature are intact.

---

## Findings
```markdown
## CRITICAL

### CRITICAL src/ai_council/output.py:926 — Non-escapable backslashes cause an infinite loop

**What:** `_top_level_bullets("- Use C:\\Temp\\file.txt\n")` never returns; expected `["Use C:\\Temp\\file.txt"]`.
**Why:** The fallback scan stops at `\` without advancing `i`, hanging verdict generation for ordinary Windows paths.
**Fix direction:** Explicitly emit and advance past backslashes that do not introduce Markdown escapes.

## HIGH

### HIGH src/ai_council/output.py:821 — Backticks preceded by backslashes inside code spans are misclassified

**What:** `_top_level_bullets("- `C:\\`\n")` returns ``["`C:`"]``; expected `["C:\\"]`.
**Why:** Markdown backslash escaping does not apply inside code spans, so the closing backtick is incorrectly skipped and payload is corrupted.
**Fix direction:** Apply escape handling only while identifying code-span openers outside spans; treat backticks inside an open span according to code-span rules.

### HIGH src/ai_council/output.py:799 — Unicode symbols are omitted from punctuation flanking

**What:** `_top_level_bullets("- a*€*b\n")` returns `["a€b"]`; expected `["a*€*b"]`.
**Why:** CommonMark punctuation includes Unicode symbol categories (`S*`) as well as punctuation categories (`P*`); misclassification deletes literal asterisks.
**Fix direction:** Recognize Unicode `S*` categories in `_is_punctuation`.

### HIGH src/ai_council/output.py:780 — Unicode whitespace fabricates list items

**What:** `_top_level_bullets("-\u00a0ordinary prose\n")` returns `["ordinary prose"]`; expected `[]`.
**Why:** `\s+` accepts non-breaking and other Unicode spaces that are not Markdown list-marker separators.
**Fix direction:** Restrict marker separation to Markdown-supported ASCII spaces and tabs.

## MEDIUM

(none)

## LOW

(none)

Contract confirmation: the `{items, source, heading}` shape is intact. `_extracted_options(sections, question_sections=None) -> dict` is also intact and unchanged from `main`.
```
