# BACKLOG — ai-council

## Big picture

`ai-council` is the ecosystem's multi-model deliberation engine: it runs structured
debate/research across a configurable AI panel and produces the verdicts that become
binding ADRs. The backlog advances four themes toward a more evidence-based, reliable,
and self-enforcing tool.

**Themes (backbone):** Synthesizer refresh · Provider reliability & test coverage · Naming & quality automation · Council process & methodology

---

## Synthesizer refresh
> As the tool owner, I want the default synthesizer chosen on real scoring data and the choice codified, so the verdict author is evidence-based and cost-aware.

### Decide the synthesizer on real data, then codify it
So that the ADR-01 default rests on measured synthesis quality, not assumption.
- [#1] [P1][M] Run the current Gemini synthesizer against ~15 historical transcripts and score with the synthesis rubric · Done when: ~15 transcripts scored + the Phase-3 branch trigger is decided (Branch A swap / Branch B keep) · refs Phase-2 smoke test
- [#2] [P1][M] Implement the Phase-3 conditional: amend ADR-01 (cost-optimization principle) + execute Branch A (new synthesizer) or Branch B (keep Gemini) · Done when: ADR-01 amended + the chosen branch shipped · refs ADR-01 · BLOCKED on #1 (needs the smoke-test scores)
- [#3] [P1][S] Codify the cost-optimization principle in the ADR-01 amendment text (balance quality vs cost) · Done when: the principle is written into the ADR-01 amendment · refs ADR-01 (folded into #2 scope, tracked separately)

### Settle the panelist/synthesizer overlap policy
So that overlap rules are explicit if the synthesizer ever joins the panel.
- [#4] [P3][S] Amend ADR-02 to codify the cost-reframed panelist/synthesizer overlap policy · Done when: ADR-02 amendment lands (or is closed as not-needed if Gemini retained) · refs ADR-02 · conditional on #2 Branch A

---

## Provider reliability & test coverage
> As the tool owner, I want every wired provider to have a known-good, tested path, so reliability is measured rather than assumed.

### Close the untested and unreliable provider paths
So that no provider is wired-but-unverified or silently degrading the panel.
- [#5] [P2][M] Add an integration path for `openai_deep_research.py` (o3-deep-research, ~45 min, ~$10+/run — cannot run in CI) · Done when: a manual integration test path is documented + runnable · refs migrated from tasks/todo.md
- [#6] [P3][M] Evaluate whether DeepSeek should be replaced or demoted from the default full panel · Done when: a replace/keep/demote decision is recorded · refs reactive trigger: round-blocking failure rate >2% per JOURNAL data

---

## Naming & quality automation
> As the tool owner, I want the ADR-34 naming convention enforced mechanically and its edge cases resolved, so violations are caught by CI, not reviewer luck.

### Enforce hyphen-only naming and resolve its timestamp edge case
So that new files cannot drift from ADR-34 and the ISO-timestamp ambiguity is settled.
- [#7] [P2][M] Add a CI check (pre-commit hook or ruff plugin) rejecting new `docs/`/`src/` files with UPPERCASE or underscores in the slug · Done when: a non-conforming new filename is blocked · refs strażnik finding I5 (SYNTHESIS-QUALITY-RUBRIC.md violation)
- [#8] [P3][S] Decide whether ADR-34 applies to ISO timestamps inside filenames (the `council-out-YYYYMMDD_HHMMSS` underscore) — fix the emitter to hyphens, or amend ADR-34 with an ISO-timestamp exemption · Done when: the methodology decision is recorded + applied · refs ADR-34 · surfaced by #7 once built

---

## Council process & methodology
> As the tool owner, I want the ADR-67 gated loop implemented and the synthesis rubric sharpened, so the Council process runs deterministically and its quality is measurable.

### Build the ADR-67 downstream pieces and sharpen the rubric
So that `/council-question` generates + self-gates questions and synthesis quality is unambiguous.
- [#9] [P3][L] Implement ai-council's ADR-67 pieces: the `/council-question` template (one decision + options + constraints + prior-ADR context), the question-quality gate, and deterministic `council.return_dir` I/O from `~/.claude` config · Done when: the gated loop runs end-to-end · refs ADR-67 · DEFERRED — do NOT build before the canonical-baseline settles (mirrors `.dev-knowledge` #70)
- [#10] [P3][S] Refine the faithfulness criterion in `docs/synthesis-quality-rubric.md` to clarify additive meta-analysis cases · Done when: the rubric wording disambiguates synthesizer cross-model synthesis vs raw-transcript content · refs N=1 scoring exercise
- [#11] [P3][S] Provide ai-council's two data points (cycle-1 + cycle-2 retrospective) for the "bilateral handshake = 1 round trip" codification owned by `.dev-knowledge` · Done when: the data points are handed to `.dev-knowledge/LESSONS.md` · refs cross-stream (codification lives in `.dev-knowledge`)

---

**About this file** — ADR-66 story-map (Big Picture → Theme → User Story → Task), migrated
2026-06-02 from the ADR-41/47 stream schema per ADR-38 A6 (canonical backlog form, all
repos). Stories are human (goal + `So that`); tasks carry `[#id] [P][size] · Done when · refs`.
Done tasks **leave** (ADR-65); git is the implementation record. Conformance is checked
read-only by `.dev-knowledge/scripts/audit.py`.

**Grooming log:** 2026-05-12 (stream-format seed) · 2026-06-02 (story-map migration, all 11 items preserved). Next quarterly: 2026-07-01.
