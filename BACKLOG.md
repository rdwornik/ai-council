# BACKLOG — ai-council

<!-- schema: ADR-41 as relaxed by ADR-47 | grooming: per-handoff (~2 min) + quarterly deep (first 2026-07-01) -->

Canonical cross-session pending items. Single source of truth for all actionable work.
Handoff Future State references items here by stream + title (pointers, not duplication).

---

## Stream: Phase 2 — Synthesizer Refresh

### [P1] [open] Step 5 smoke test execution (Gemini synth scoring on ~15 transcripts)
- **What:** Run current Gemini synthesizer against ~15 historical transcripts; score quality with the synthesis rubric
- **Why:** Generates the empirical data needed to decide whether Phase 3 ADR-01 amendment triggers (Branch A: swap synthesizer; Branch B: keep Gemini)
- **Added:** 2026-05-12 by operator
- **Status:** open

---

## Stream: Phase 3 — Synthesizer Refresh

### [P1] [open] Step 6 Phase 3 conditional implementation (ADR-01 amendment + Branch A/B)
- **What:** Amend ADR-01 with cost-optimization principle; implement Branch A (new synthesizer) or Branch B (keep Gemini) based on Step 5 data
- **Why:** Cost-optimization principle needs codification regardless of branch outcome; blocked on Step 5 smoke test data
- **Blocked:** Depends on Step 5 smoke test data — cannot proceed until operator scores ~15 historical transcripts with synthesis rubric.
- **Added:** 2026-05-12 by operator
- **Status:** open

### [P1] [open] Codify cost-optimization principle in ADR-01 amendment
- **What:** Add cost-optimization framing to ADR-01 amendment text (synthesizer should balance quality vs. cost)
- **Why:** Folded into Step 6 ADR-01 amendment scope; listed separately for tracking visibility
- **Added:** 2026-05-12 by operator
- **Status:** open

---

## Stream: Test Coverage

### [P2] [open] openai_deep_research integration test gap
- **What:** `openai_deep_research.py` (o3-deep-research, ~45 min timeout) is wired but untested end-to-end; add integration test
- **Why:** Migrated from tasks/todo.md; cannot run in CI due to cost (~$10+/run) but needs at least a manual integration path documented
- **Added:** 2026-05-12 by backlog-migration (was tasks/todo.md)
- **Status:** open

---

## Stream: Quality Automation

### [P2] [open] CI enforcement of hyphen-only separator rule (ADR-34)
- **What:** Add a CI check (pre-commit hook or ruff plugin) that rejects new files in `docs/` and `src/` with UPPERCASE letters or underscores in the slug
- **Why:** I5 fresh violation (SYNTHESIS-QUALITY-RUBRIC.md added same day ADR-34 was enforced on other files) — empirical evidence that pattern needs automated enforcement, not just documentation
- **Added:** 2026-05-12 by audit (strażnik finding I5)
- **Status:** open

### [P3] [open] ADR-34 timestamp-underscore case in council-out emitter output

- **What:** AI Council CLI emitter produces filenames with underscore in the `YYYYMMDD_HHMMSS` timestamp portion (e.g. `council-out-20260513_102702-research-question-...md`). The emitter uses Python's `%Y%m%d_%H%M%S` strftime pattern; the `_` between date and time is preserved in output filenames.
- **Why:** Strict reading of ADR-34 (hyphen-only separator) would catch this as a violation. Practical reading might exempt ISO-style timestamps as programmatically-generated date-time data, not naming-slug separators. The case is real but unresolved.
- **Methodology decision needed before action:** does ADR-34 apply to ISO timestamps inside filenames, or are they exempt?
  - If applies: fix emitter to use `%Y%m%d-%H%M%S` (or equivalent hyphen format); decide whether to rename existing transcripts or grandfather them.
  - If exempt: amend ADR-34 with an explicit clause noting ISO-timestamp exemption so future audits don't re-raise the question.
- **Cross-link:** related to existing P2 "CI enforcement of hyphen-only separator rule (ADR-34)" above. The P2 CI check will surface this case once implemented, but the methodology decision needs to be made before that to know whether the check should flag or allow.
- **Added:** 2026-05-13 (architect noted during ADR-45 review when referencing council transcript filenames).
- **Status:** open

---

## Stream: Provider Reliability

### [P3] [open] DeepSeek replacement decision
- **What:** Evaluate whether DeepSeek should be replaced or demoted from the default full panel
- **Why:** Reactive trigger: round-blocking failure rate >2% per JOURNAL data; low priority until threshold crossed
- **Added:** 2026-05-12 by operator
- **Status:** open

---

## Stream: Methodology

### [P3] [open] Synthesis quality rubric refinement — faithfulness sub-clarification
- **What:** Refine the faithfulness criterion in `docs/synthesis-quality-rubric.md` to clarify additive meta-analysis cases
- **Why:** Emerged from N=1 scoring exercise (synthesizer additive meta-analysis question); rubric wording is ambiguous when synthesizer adds cross-model synthesis beyond raw transcript content
- **Added:** 2026-05-12 by operator
- **Status:** open

---

## Stream: Governance

### [P3] [open] ADR-02 amendment (panelist/synthesizer overlap policy)
- **What:** Amend ADR-02 to codify the cost-reframed policy on panelist/synthesizer overlap
- **Why:** Conditional on Phase 3 Branch A (Opus chosen as synthesizer) — likely won't trigger after cost reframe if Gemini retained
- **Added:** 2026-05-12 by operator
- **Status:** open

### [P3] [open] Implement ADR-67 AI-Council gated loop (/council-question template + question-gate + council.return_dir)
- **What:** Build ai-council's downstream pieces of the ADR-67 six-step gated Council loop: a Council-question template (mandates exactly one decision + options + constraints/invariants + relevant prior ADRs), a question-quality gate that validates a filled question before release (required sections, one decision, options enumerated, ADR context attached), and deterministic known-path I/O reading the operator's return directory from `~/.claude` global config (`council.return_dir`). Trigger: `/council-question` generates + self-gates the question.
- **Why:** ADR-67 (`.dev-knowledge`, Accepted 2026-06-01) defines the contract; its cross-domain table assigns the template + gate + return-path implementation to `ai-council`. Currently unbuilt.
- **Deferred:** Downstream work — do NOT build before the canonical-baseline / Phase-2 universalization settles (mirrors `.dev-knowledge` BACKLOG #70 deferral). Tracked here per ADR-41 (child-repo execution items live in the child repo).
- **Added:** 2026-06-02 by universalization-conformance audit (ADR-67 obligation)
- **Status:** open

---

## Cross-stream / Ecosystem

### [P3] [open] Cross-stream P2 — handshake = 1 round trip codification
- **What:** Codify the "bilateral handshake = 1 round trip" methodology in `.dev-knowledge` LESSONS.md; ai-council provides data points (cycle 1 + cycle 2 retrospective)
- **Why:** `.dev-knowledge` owns the codification; ai-council's contribution is empirical evidence from two completed strażnik review cycles
- **Added:** 2026-05-12 by operator
- **Status:** open
