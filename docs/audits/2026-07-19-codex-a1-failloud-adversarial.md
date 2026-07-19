# Codex Review — a1-failloud-adversarial

**Date:** 2026-07-19
**Branch:** `main`
**HEAD:** `5bb5455`
**Diff range:** `2492371..a48080f`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

ADVERSARIAL pass on Lane A1's MERGED diff. Do NOT re-derive the design; assume the
record-and-aggregate shape is settled. Attack the merged result.

CONTEXT YOU MUST USE. This lane's independent derivation was done by a Plan subagent
of the same model family as the builder, so the independence property was LOST. Its
blast radius is the largest of the three lanes: every deliverable's write path, under
the 1.0 delegation surface. It has ALREADY shipped one footgun its own green gate could
not see: OutputRoutingError.__init__ did list(failures) with no type guard, so a str --
being iterable -- was silently shredded into one 'deliverable' per character. Four call
sites passed a string; only ONE had an assertion strong enough to notice. That defect
was found at integration, not by the lane.

PRIMARY QUESTION: where ELSE in this diff is a wrong TYPE or wrong SHAPE accepted
silently rather than rejected? Look for the same class:
- iterables treated as collections without a type guard (str/bytes/Path/generator)
- dataclass fields populated from unvalidated callers
- Optional/None flowing into a branch that assumes presence
- a list-vs-single-item ambiguity anywhere in RoutingFailure / routing_failures plumbing
- accumulator mode (routing_failures passed) vs direct mode (raises) diverging in a way
  that makes one path silently weaker than the other

ALSO ATTACK:
1. The accumulator contract. A writer called WITH an accumulator records; WITHOUT one it
   raises. Is that invariant actually held on EVERY path in this diff, including error
   and early-return paths? Any path where a failure is recorded into an accumulator that
   nobody later inspects is a SILENT LOSS -- the exact defect class this lane closes.
2. raise_for_routing_failures and the __cause__ chain. The lane claims the accumulator
   path chains a real traceback. Verify. A lost __cause__ in the path the orchestrators
   actually use is a real finding.
3. Whether any CANONICAL artifact can still be lost. The lane's whole justification is
   that canonical writes always land before any raise. Find a path where they do not.
4. #60's _extracted_options rewrite: the nested-sub-bullet skip and the emphasis
   stripping. Adversarial input -- what markdown shapes produce wrong options?

Report findings with severity and file:line. Do NOT propose or apply fixes.

---

## Findings
Excluding the already-known `OutputRoutingError` constructor defect, I found eight High-severity issues.

## Critical

(none)

## High

### [HIGH] src/ai_council/output.py:246 — Optional secondary failure aborts canonical artifact production

**What:** The best-effort `secondary_dir` write occurs outside error handling.  
**Why:** An existing but unwritable secondary directory aborts before canonical metrics, minority, and verdict artifacts are emitted, bypassing the accumulator entirely.  
**Fix direction:** Contain secondary-write failures according to their best-effort policy and continue the canonical sequence.

### [HIGH] src/ai_council/output.py:481 — Direct mode raises before writing canonical metrics

**What:** Without an accumulator, `_write_routed` raises before `_save_metrics_json` at line 496, whereas accumulator mode continues.  
**Why:** A required return-dir failure unnecessarily suppresses an otherwise writable canonical metrics sidecar, making direct mode materially weaker.  
**Fix direction:** Defer direct-mode routing failure until all canonical work owned by `save_to_file` has been attempted.

### [HIGH] src/ai_council/output.py:283 — `target_paths` silently accepts destructive iterable shapes

**What:** `target_paths` is iterated without validating or materializing a `list[Path]`; strings or `list[str]` become swallowed per-item errors, while a generator is exhausted by the first writer.  
**Why:** A generator reused by `CouncilRunner` can mirror only the transcript while silently omitting minority and verdict artifacts, and string-shaped inputs merely produce warnings instead of a type error.  
**Fix direction:** Validate and materialize the collection and every element once before invoking any writer.

### [HIGH] src/ai_council/orchestrator.py:224 — A failed metrics sidecar can still enter the manifest

**What:** After `_save_metrics_json` reports failure, the orchestrator treats `metrics_path.exists()` as proof of a successful artifact, and `output.py:899` repeats the same test.  
**Why:** A directory at that pathname—or a partial file left by a failed write—will be advertised as a valid metrics artifact despite the recorded degradation.  
**Fix direction:** Base manifest inclusion on explicit write success and validated regular-file content, not pathname existence.

### [HIGH] src/ai_council/output.py:988 — Verdict manifest predicts the return copy before it exists

**What:** `guaranteed_dirs` includes `return_dir` before `_write_routed` runs, so the canonical verdict package unconditionally advertises that future path.  
**Why:** In accumulator mode, a failed return write leaves a durable canonical package naming a nonexistent verdict copy; checker leg L11 detects this but `scripts/verify_output_writes.py:567` still exits successfully because GAP does not fail the checker.  
**Fix direction:** Populate the verdict’s own manifest entry only from verified writes and make the contract check fail when a claimed path is absent.

### [HIGH] src/ai_council/output.py:735 — Valid Markdown list shapes are dropped or truncated

**What:** Any indentation is classified as nesting, but `body.strip()` at line 764 removes indentation only from the first item; valid `+` bullets and `1)` ordered items are also unsupported.  
**Why:** For example, two equally indented top-level bullets produce only the first option, while `+` and `1)` lists produce none and may trigger a stale question fallback.  
**Fix direction:** Parse list indentation and markers consistently with supported Markdown syntax.

### [HIGH] src/ai_council/output.py:740 — Marker and emphasis stripping corrupts option text

**What:** Character-set `lstrip` removes payload characters after the marker, while edge-only emphasis stripping mishandles emphasized labels followed by descriptions.  
**Why:** `- 3D printing` becomes `D printing`, `- 2026 roadmap` becomes `roadmap`, and `- **Alpha** — fast` becomes `Alpha** — fast` in the authoritative verdict package.  
**Fix direction:** Remove the exact list marker first, then unwrap actual Markdown emphasis delimiters without stripping content characters.

### [HIGH] src/ai_council/output.py:759 — Continued scanning promotes unrelated sections to options

**What:** `_options_with_items` continues through the broad `"considered"` marker after a prose-only alternatives section.  
**Why:** A later bulleted `## Risks Considered` section is returned as `options_considered`, preventing the legitimate question-options fallback.  
**Fix direction:** Continue scanning only headings that semantically identify alternatives or options.

## Medium

(none)

## Low

(none)

The production accumulator paths do inspect their lists on normal completion, and `raise_for_routing_failures` preserves the first available originating exception as `__cause__`.
