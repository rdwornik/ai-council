# EPI-1 archaeology pack — condensation (essence of the retained reversal instrument)

**Class:** audits (condensation record) · **Date:** 2026-07-19 · **Status:** ACTIVE INSTRUMENT — retained, unscored
**Purpose:** capture the essence of the EPI-1 blind-scoring pack so the pack itself need not be opened to know
what it is, what it can overturn, and how to use it. **Additive — nothing existing was edited; the pack was read
read-only and the sealed key was NOT opened.**

---

## 1. What the pack is

A **40-item blind synthesis-quality scoring pack**, built overnight 2026-07-16 as PREP for BACKLOG **#24**
(EPI-1 archaeology). It holds 40 historical AI-Council syntheses — the verdict paragraphs a *non-participating*
model wrote to close past debates — with the author identity redacted, for the operator to score blind against
the 5-criterion `protocols/SYNTHESIS_QUALITY_RUBRIC.md`.

Method canon: `docs/intake/2026-07-06-lane-epi-functional-design.md` §3 (Q3). Operator rulings applied at build
time: **r1** comparative gemini-vs-openai · **r3** operator scores blind, LLM-judge is second opinion only ·
**r5** corpus = full local `output/` with hub dedupe, no curation.

## 2. Which ruling it can reverse, and on what evidence

It is the standing validation instrument for the **G3 / Epic-B synthesizer ruling** recorded in
`docs/audits/2026-07-17-synthesizer-ruling-gemini-to-openai.md`: the default synthesizer switched
**gemini → openai**, now durable in `config/settings.yaml` (`synthesizer: openai`).

That ruling rests on operator judgment, quoted verbatim in the ruling record as its basis:

> the operator observes a recent quality decline in Gemini models.

The night-batch E2E runs were explicitly **corroboration, not the basis**. The pack is the *empirical* path the
ruling was deliberately left open to — the ruling record states it plainly:

> **This ruling is reversible against that evidence at any time** — the archaeology can still be scored to
> confirm or overturn it.

So: scoring the pack produces per-segment pass-rates that either confirm the switch or overturn it on data.
**Nothing else in the repo can do that** — this is the only built instrument for the question.

## 3. Corpus shape

**Full mining pass:** 239 local `output/` files → **138** identity-readable syntheses (visible
`## Synthesis (by X, role)` header), reconciling exactly with the lane doc's witnessed 2026-07-06 tally.
Authors across those 138: claude 53 · openai 53 · gemini 21 · claude-sonnet 7 · plus 4 flagged anomalies
(3 openai-as-participant, 1 pre-label-format). The remaining ~101 are research-mode outputs, `_inbox-run`
logs, and headerless/older files — not identity-readable, so out of the comparison.

**The two compared strata** (r1: gemini vs openai, non-participant only):

| Segment | Identity-header | Decision-mode pool | Selected into pack |
|---|---|---|---|
| gemini / non-participant | 21 | 20 | **20 (all)** |
| openai / non-participant | 53 | 50 | **20 (matched sample)** |

Both clear the protocol floor of **n ≥ 10 per segment**. Documented exclusions (disclosed as method, not
curation-for-outcome): 3 research-mode files (no decision synthesis to score) and 1 FAILED run.

**Matching:** gemini contributes all 20 (zero selection on the incumbent); openai is sampled to match gemini on
three covariates — mode = decision, panel size = 4-model, and era by calendar month to gemini's exact
distribution `{2026-03: 4, 2026-04: 3, 2026-05: 3, 2026-06: 10}` — evenly spaced by sorted filename within each
month bucket. Items are relabelled `ITEM-01..40` and shuffled so order carries no author signal. Reproducible:
`SEED=20260716`, builder `build_pack.py`. **Both segments are decision-mode, 4-model, era-matched — the
confounds the lane doc warned about are controlled by construction.**

**Scoring-sheet structure:** one markdown table, 40 rows (`ITEM-01`…`ITEM-40`) × 5 yes/no criterion columns —
**C1** position representation · **C2** no hallucinated consensus · **C3** scannability · **C4** faithfulness ·
**C5** verbosity proportionality — plus a free-text notes column for any **N**. The rubric's operating principle
is that a **No** is the signal to record the failure mode. **The sheet is entirely blank — every one of the 200
cells is empty.** That is the physical confirmation the pack was retained *unscored*.

**What the sealed key holds (structure only — not opened, contents not revealed):** a per-item mapping from each
`ITEM-NN` to its segment (`gemini` / `openai`), alongside the recorded covariates `date`, `month`, `debate_mode`,
`panel_size`, and `author`. It is the single artifact that de-blinds the pack; ~13 KB of JSON. Opening it before
scoring destroys the instrument's value, which is why it is sealed, gitignored, and untouched here.

**Disclosed residual caveats** (carried from the manifest, so the pack's honesty travels with this condensation):
1. **Blind residual tell** — the synthesizer is always the model *absent* from the 4 panelists, so it is in
   principle deducible by enumerating panel headers. Mitigated by instruction, not by mechanism; canonical Q3 §5
   blind would be header-stripped + shuffled.
2. Era is matched at **month** grain, so within-month topic/date variation remains.
3. **n = 20 per segment** clears the floor but is modest — treat large criterion gaps as signal and small gaps as
   inconclusive.

## 4. The operator ruling that retained it unscored

At the 2026-07-17 G3 event the operator resolved the synthesizer question **by direct authority** rather than by
waiting on the scoring, and retained the pack instead of consuming or discarding it:

> The **EPI-1 40-item blind-scoring pack + the sealed identity key remain retained, unscored**, as the standing
> validation instrument … both gitignored.

The plan-of-record had defined G3 as "the operator's ruling on the EPI-1 archaeology report (#24)". The gate was
satisfied **by ruling**, with #24's blind scoring left available as the confirming/overturning evidence path.
Consequently **#24 is closed** while the instrument stays live — those are not in tension: the *decision* is made,
the *audit trail for reversing it* is preserved.

A second, independent retention decision followed on **2026-07-18** during the consolidation/archival pass —
see §6.

## 5. How to reopen it if the synthesizer decision ever needs reversing

The pack is self-sufficient; its own runbook is `OPERATOR-SCORING-README.md` ("you need no prior context").
The sequence:

1. Open `OPERATOR-SCORING-README.md` — the one-screen runbook.
2. Score `items/ITEM-01.md` … `ITEM-40.md` blind into `scoring-sheet.md`, Y/N on C1–C5, a one-line note on any N.
   Order is pre-shuffled; score in any order. **Do not deduce the synthesizer from the panel roster** (§3 caveat 1).
3. Only after all 40 are scored, open the sealed key to un-blind, then tally per-segment pass-rate per criterion.
4. Cross-check against the segregated **LLM-judge second opinion** — a *secondary* signal, never the verdict
   (ruling r3). It is stored outside the pack and is deliberately **not** summarized here: reading it before
   scoring would bias the blind. *(Not opened in preparing this condensation, for the same reason.)*
5. Produce the single-recommendation report — **Branch A** (swap synthesizer) or **Branch B** (keep) — which is
   #24's original done-when. That report, not this pack, is what would overturn the ruling.
6. **Seal expiry:** once scored and the un-blinding is recorded, the seal has expired by prior ruling — the key
   and the judge report are then un-ignored and committed as the finalized evidence bundle
   ("reproducibility beats secrecy after the fact"). Until then both stay gitignored.

## 6. Location — and the standing ruling that it does not move

**Current canonical location (unchanged as of this writing):**

| Artifact | Path |
|---|---|
| Pack (manifest, runbook, scoring sheet, `items/` ×40) | `docs/audits/2026-07-17-epi1-archaeology/` |
| Sealed identity key (~13 KB) | `docs/audits/2026-07-17-epi1-archaeology-KEY-SEALED.json` |
| LLM-judge second opinion (segregated) | `docs/audits/2026-07-17-epi1-archaeology-SECOND-OPINION-judge.md` |

All three are **gitignored and untracked** (`.gitignore:60-62`) — deliberately, while the seal is live, to preserve
the blind and audit immutability (CLAUDE.md §5.3).

**A relocation to `docs/audits/archive/` was considered on 2026-07-19 and NOT executed**, because the record
already contains a contrary operator ruling from the 2026-07-18 consolidation pass — the very pass that created
`docs/audits/archive/` as the preservation archive. Rider (a) of that ruling:

> the `docs/audits/2026-07-17-epi1-archaeology/` instrument (sealed KEY + 40-item pack) does **not** move; only
> its top-level `-SECOND-OPINION-judge.md` was in scope but is **gitignored/untracked** (can't `git mv`), so it
> stayed too → audit-move count 18→17.

Two further reasons point the same way, recorded here so a future session does not re-litigate them from scratch:

- **Destination semantics.** `docs/audits/archive/README.md` admits "**completed** audit artifacts whose findings
  are fully DEPLOYED with no open remainder". EPI-1 has no findings — it is unscored by design and explicitly
  retained as actionable. Filing a live instrument in a completed-artifact archive mislabels it.
- **Seal-exposure hazard.** The `.gitignore` rules at lines 60–62 are **path-literal**. Moving the artifacts
  without simultaneously rewriting those rules would leave the sealed key untracked-but-no-longer-ignored — the
  exact condition behind the 2026-07-18 near-miss when `git add -A` staged a `SEALED-KEY.json`. Any future
  relocation must rewrite `.gitignore` in the same change and verify with `git check-ignore` before staging
  anything.

**If the pack is ever relocated,** this section is the pointer to update, and the move must carry: byte-identical
contents, a filesystem move (not `git mv` — the artifacts are untracked), the `.gitignore` rewrite above, a
`git check-ignore` verification, and an empty-directory check at the origin.

## 7. References

- `docs/audits/2026-07-17-synthesizer-ruling-gemini-to-openai.md` — the ruling this instrument can reverse
- `docs/intake/2026-07-06-lane-epi-functional-design.md` §3 (Q3) — method canon
- `protocols/SYNTHESIS_QUALITY_RUBRIC.md` — the 5 criteria
- `docs/audits/archive/2026-07-17-consolidation-session-report.md` — the 2026-07-18 archival pass + rider (a)
- JOURNAL 2026-07-18 (consolidation/archival pass) — the destination ruling in full
- BACKLOG #24 (closed by ruling), #2/#3 (ADR-01 amendment text)
