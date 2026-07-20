# Codex Review — opt-extractor-contract-pass6

**Date:** 2026-07-20
**Branch:** `fix/opt-extractor-contract`
**HEAD:** `7ac8978`
**Diff range:** `main..fix/opt-extractor-contract`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

Pass 6, final convergence check. Passes 1-5 each found real defects (payload loss, a hang, fabricated items); all repaired, each pinned by a named test, plus fuzz guards for termination and non-fabrication.
Review the CURRENT state of _pair_code_spans, _unwrap_emphasis, _is_punctuation, _top_level_bullets.
Report ONLY defects demonstrable with a concrete input, in these classes:
 (a) non-termination / superlinear runtime, (b) payload loss, (c) fabricated items.
For each, state how plausible the triggering input is in real LLM synthesis output (a bulleted list of decision options), and say plainly if it is contrived.
If nothing demonstrable remains in those classes, say so plainly -- do not manufacture findings.

---

## Findings
No demonstrable non-termination or superlinear-runtime defect remains.

## Critical

(none)

## High

### HIGH src/ai_council/output.py:1020 — multiline option payload is discarded

**What:** `"- Adopt PostgreSQL for\n  durable transactions\n- Keep SQLite"` produces `["Adopt PostgreSQL for", "Keep SQLite"]`, losing the continuation text.  
**Why:** Valid multiline list items are silently truncated in `options_considered`.  
**Fix direction:** Accumulate continuation lines into their parent item while continuing to exclude genuine nested sublists.  
**Plausibility:** Plausible in LLM output and not contrived, particularly for options with longer explanations.

### HIGH src/ai_council/output.py:1030 — fenced-code lines fabricate options

**What:** A fenced diff containing `- delete legacy` and `+ add replacement` produces both lines as options, although neither is a Markdown list item outside the code block.  
**Why:** Technical examples can silently become authoritative `options_considered` entries.  
**Fix direction:** Make extraction block-aware so content inside fenced code blocks is ignored.  
**Plausibility:** Uncommon but realistic in technical synthesis output; not purely contrived.

### HIGH src/ai_council/output.py:837 — multi-backtick code span loses a trailing backslash

**What:** ```- Use ``C:\`` path``` produces ```["Use ``C:`` path"]```, deleting the backslash and leaving the fences.  
**Why:** `_pair_code_spans` splits a backslash-adjacent closing run into separate runs, after which `_unwrap_emphasis` interprets the backslash as an escape and removes it.  
**Fix direction:** Distinguish an escaped opener outside a code span from a maximal closing run inside an already-open equal-length span.  
**Plausibility:** Contrived: it requires a multi-backtick code span whose payload ends in a backslash. It is valid Markdown but unlikely in ordinary option lists.

## Medium

(none)

## Low

(none)
