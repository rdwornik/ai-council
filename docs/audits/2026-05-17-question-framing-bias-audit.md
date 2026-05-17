# Audit — Question-Framing Bias in Past Council Debates

**Date:** 2026-05-17
**Branch:** `docs/question-framing-bias-audit`
**HEAD (audit-time):** `1527c9769d07ffda1edcbd4921ca37e2daf4f720`
**Rubric source:** `docs/council-question-guide.md` § *Neutralizing bias in question framing* (6-point pre-flight self-check + 7 framing biases: leading, anchoring, asker-leakage, false dichotomy, loaded terminology, choice-set, availability)
**Mode:** read-only analysis; no question or transcript was modified
**Scope:** ai-council only; `.dev-knowledge` was read-only

---

## Purpose and boundaries

This report is **measurement before optimization**. It scores past Council debate questions against the bias rubric that was just shipped in `council-question-guide.md`, and reports the distribution of framing biases found. It does **not** decide whether to act — the operator decides whether the findings warrant a deeper research/debate on the guide.

---

## Corpus

Two sources were inventoried:

| Source | Path | Files | Question-extractable? |
|---|---|---|---|
| Curated decision transcripts | `.dev-knowledge/docs/decisions/transcripts/` | 21 (`council-out-2026*.md`, excluding `archive/`) | research: **yes**; pick/judge: **no** |
| ai-council operational transcripts | `ai-council/output/` | 70+ debate `.md` files (pre-curation; many are early-stage drafts) | mixed; same per-mode pattern as above |

**Corpus selection:** the 21 curated `.dev-knowledge` transcripts. These are the questions whose decisions became binding ADRs; they are the consequential corpus.

### Extractability — a finding in itself

| Mode | Count in corpus | Original question text in transcript |
|---|---|---|
| `research` | 10 | **Full question body preserved** (`**Query:**` block, Current State, Sub-questions, Constraints) |
| `pick` | 10 | **Only the truncated title** (~70 chars after `# AI Council Debate:`) |
| `judge` | 1 | Only the truncated title |

For `pick` and `judge`, the `Source:` field points at `~/Downloads/*.md` or temp files that no longer exist. Full question text is **not recoverable** from the transcript alone.

**Finding F-0 (corpus quality):** transcripts for `pick` and `judge` modes do not preserve the full question prompt — only a ~70-char truncated headline. The most consequential debates (binding ADR picks) are the ones with the least audit trail of the original framing. This is a separate issue from the bias rubric but blocks deep scoring of 11 of 21 transcripts.

For those 11, this audit can score only the visible headline plus the option set (recoverable from model responses citing "Option A/B/C"). For the 10 research transcripts, full scoring is possible.

---

## Per-question scorecard — 10 research debates (full text available)

Legend for 6-point check: **P** = pass, **F** = fail, **B** = borderline (one-line rationale below). Biases column lists each present bias with a short evidence quote.

| # | Transcript (date · slug) | 6-check (1·2·3·4·5·6) | Biases present (evidence) |
|---|---|---|---|
| R1 | 2026-04-30 · cross-repo audit prior art | P·B·N/A·P·B·P | **anchoring** ("Scrum Master" / `central 'brain' repository` metaphor inserted into headline pre-empts non-hub architectures); **loaded terminology** ("brain") |
| R2 | 2026-04-30 · BACKLOG placement | P·P·B·B·F·P | **loaded terminology** (option 3 described as "distributed across per-session handoffs *without a canonical view*" — biases against it); **choice-set** (no explicit "none of these / different pattern" escape) |
| R3 | 2026-04-30 · scale tier S/M/L | P·B·N/A·P·P·P | **asker-leakage** (`Insight from .dev-knowledge architect: complexity grows exponentially, not linearly. Tier transitions therefore need explicit, auditable triggers, not vibe-based assessment` — pre-commits to the conclusion the research is supposed to surface) |
| R4 | 2026-05-09 · ai-council architect brief | F·F·N/A·F·F·F | **leading** (`Brief for ai-council architect` — it is a directive, not a question); **asker-leakage** ("Required reading / In scope / Out of scope" pre-specifies the answer); **loaded terminology** ("violates", "non-compliance", "non-measurable", "unblocking step"); **choice-set** (no options); **availability** (frames around the 2026-04-30 audit finding). **This is a misclassified question — research mode but no research question.** |
| R5 | 2026-05-09 · handoff artifacts | B·F·N/A·P·F·P | **leading** (headline presupposes "handoff artifacts" *are* the solution); **asker-leakage** (Context lists "Rob flagged it as wrong" with bullet pain points — pre-states the asker's diagnosis); **loaded terminology** ("wrong", "over-structured", "friction"); **availability** (one failed audit's faults dominate the framing) |
| R6 | 2026-05-13 · memory architecture (isolated processes) | P·B·N/A·P·P·B | **anchoring** (every sub-question ends with option **E: Hybrid combining multiple above** — pre-tilts toward "hybrid wins" before evidence arrives); **choice-set** (the hybrid escape biases every question identically) |
| R7 | 2026-05-14 · cross-session handoff optimization | P·F·N/A·B·F·P | **anchoring** (current v3.3 design used as reference frame); **asker-leakage** ("Self-identified gaps in departing-session Stage 2 response" pre-names 7 specific failures the research is asked about); **loaded terminology** ("confusing", "burdens", "drifted into multiple failure modes"); **availability** (one 5-hour session on 2026-05-14 frames the question) |
| R8 | 2026-05-15 · dated-entries format | P·B·P·P·P·P | mild **choice-set** (date format pre-binarized to "ISO vs human-readable vs heading-only" — other formats not invited). Otherwise the cleanest research question in the corpus. |
| R9 | 2026-05-15 · BACKLOG organization | P·P·P·P·P·P | minimal — anti-patterns are surfaced symmetrically; no detected biases beyond very mild anchoring (existing Stream A/B/C named as prior art) |
| R10 | 2026-04-30 · audit-tool prior-art (shorter variant of R1) — same file 12:50:39 | counted as R1 | — |

**R10 is the same transcript as R1** (`council-out-20260430-125039-...`) — kept as a single row. The corpus is therefore **9 distinct research questions**, not 10.

### 6-check per-question failure rate (research, n=9)

| Check | Pass | Borderline | Fail | Fail-or-borderline |
|---|---|---|---|---|
| 1. Headline = problem, not pre-chosen answer | 7 | 1 | 1 | 22% |
| 2. No wording reveals my preference | 3 | 4 | 2 | 67% |
| 3. Every option equal length / charity | 1 of 2 with options (R2 borderline) | 1 | 0 | n/a small-n |
| 4. Could a reasonable option be missing | 6 | 2 | 1 | 33% |
| 5. Value-laden adjectives | 5 | 1 | 3 | 44% |
| 6. Fast unanimous agreement would surprise me | 7 | 1 | 1 | 22% |

**At least one fail or borderline:** 9 of 9 (100%). **At least one outright fail:** 6 of 9 (67%).

### Bias frequency across the 9 research questions

| Bias | Count | Notes |
|---|---|---|
| Asker-leakage | 5 | Context sections routinely state the asker's diagnosis or "insight" |
| Loaded terminology | 5 | "violates", "wrong", "brain", "without a canonical view", "drifted", "confusing" |
| Anchoring | 4 | Metaphors ("Scrum Master", "central brain") and prior designs (v3.3, current ADR) used as frame |
| Availability | 3 | Recent specific incidents (one 5-hour session, one failed audit, one migration) drive the framing |
| Choice-set | 3 | No explicit "different approach" escape; or hybrid pre-suggested |
| Leading | 2 | Headline presupposes answer (R4 directive, R5 "handoff artifacts" framing) |
| False dichotomy | 0 | Not detected in research mode — most questions offer 3-5+ options |

**Dominant biases (top 3):** asker-leakage (5), loaded terminology (5), anchoring (4). These three account for the bulk of bias incidents and almost always co-occur in the same question.

---

## Pick / judge — headline-only scoring (n=11)

Full text was not recovered. Only the truncated headline + the option set (visible via model response citations) is in evidence. Scoring is therefore partial — focused on Check 1 (headline names problem vs pre-chosen answer) and any biases visible from the headline alone.

| # | Date · slug (truncated) | Headline framing | Visible biases |
|---|---|---|---|
| P1 | 2026-04-28 · format-and-structure-of-visionmd | "Format and structure of VISION.md … as universal LLM platf…" | **leading** (presupposes VISION.md and universal scope) |
| P2 | 2026-04-28 · adr33-vision-universalization | "Pattern dissemination — universal VISION.md template …" | **leading** ("universalization" pre-decided) |
| P3 | 2026-04-29 · adr34-file-naming-convention | "File naming convention across the .dev-knowledge ecosystem" | neutral |
| P4 | 2026-04-29 · adr35-lessons-base-activation | "Lessons base activation — retrieval, promotion, querying mechanism" | mild **anchoring** (mechanism types pre-named) |
| P5 | 2026-04-30 · adr36-audit-tool-architecture | "Audit tool architecture — .dev-knowledge as Scrum Master / auditor" | **anchoring** ("Scrum Master" metaphor in headline) |
| P6 | 2026-04-30 · adr37-two-phase-handoff | "Two-phase handoff format — current state + future state as universal pattern" | **leading** ("two-phase" + "universal" pre-decided) |
| P7 | 2026-04-30 · adr38-scrum-framework | "Scrum metaphor as universal framework for chat sessions, handoffs, backlog manag…" | **leading** (Scrum metaphor adoption pre-framed); **anchoring** |
| P8 | 2026-05-11 · ecosystem-separator | "What is the canonical filename / foldername separator across the De…" | neutral |
| P9 | 2026-05-13 · handoff-architecture-pick | "Which architectural changes should replace ADR-42 v3.2 handoff proc…" | **leading** ("should replace" pre-decides replacement) |
| P10 | 2026-05-15 · cross-repo dated-entries format | "Council pick — cross-repo dated-entries format" | neutral |
| P11 | 2026-05-15 · cross-repo BACKLOG organization | "Council pick — cross-repo BACKLOG organization" | neutral |
| J1 | 2026-05-11 · adr-01 synthesizer panel refresh | "Should ai-council refresh its default debate panel and synthesizer for the 202…" | **leading** (judge-mode yes/no on a "refresh" — value-laden verb) |

**Pick/judge headline-level summary:** 6 of 12 headlines (50%) carry a visible leading or anchoring bias *in the headline alone*. Because the rest of the question text is unrecoverable, this is a **floor**, not a ceiling — actual bias rate is almost certainly higher (research-mode data suggests the Context sections are where most leakage occurs).

---

## Aggregate findings

- **Research mode, n=9, full-text scored:** 100% of questions had at least one borderline or failed check; 67% had at least one outright failure. Three biases dominate — **asker-leakage**, **loaded terminology**, **anchoring** — and they tend to co-occur in the Context section, not the headline.
- **Pick/judge mode, n=12, headline-only:** 50% of headlines carry leading or anchoring at the title alone. Full-question scoring is impossible (F-0).
- **One question (R4) was misclassified** as research mode but is in fact an architect directive with no research question. It fails all six checks.
- **Two questions (R8, R9) score cleanly** — both are the most recent in the corpus (2026-05-15) and both follow a structure (Research question / Context / Sub-questions / Constraints / Anti-patterns) very close to the template the new guide now codifies. This is consistent with — though not proof of — improving framing over time.
- **No false dichotomies were found in research mode.** Research-mode questions in the corpus consistently offered 3-5+ options or open-ended areas. The opposite failure (anchoring via too many sub-options) was more common (R6).
- **F-0 is the most actionable corpus-level finding.** Pick/judge transcripts do not preserve the original question, which means the most consequential framings (those that produced binding ADRs) are not auditable post-hoc.

---

## Findings summary

The evidence supports the conclusion that **past Council debates carried framing bias**. Specifically:

1. **Research-mode questions almost universally leaked the asker's view** — through "insight from architect", pre-listed gaps, named pain points, and value-laden context. The dominant biases are **asker-leakage**, **loaded terminology**, and **anchoring**, and they co-occur.
2. **Headlines were mostly well-formed**; framing failures concentrated in the **Context** section.
3. **Pick/judge transcripts cannot be deeply audited** because the original question is not preserved in the transcript file (F-0). What is visible in the headlines (50% leading/anchoring) is consistent with the research-mode finding.
4. **Recent questions (2026-05-15) show notably cleaner framing**, suggesting framing quality has been improving even before the rubric was written down.

---

## Note on next step

This report is **evidence**, not a decision. Whether the rubric is sufficient as written, whether to commission a follow-on research debate on bias-elimination methodology, or whether to take no further action — that judgement is the operator's, not this report's. The findings above are the input.

Two operational items also surfaced and are flagged here as evidence, not recommendations:

- **F-0 (transcript truncation):** pick/judge transcripts store only the truncated debate title, not the full prompt. A future operator decision about whether to fix this is independent of the bias-rubric question.
- **R4 (misclassified mode):** at least one transcript in the corpus is a directive submitted as `research` mode. Auto-mode-detection or guidance on what is *not* a research question is one place the operator might choose to look next.

---
