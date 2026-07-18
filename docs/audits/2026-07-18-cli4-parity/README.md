# CLI-4 Parity Run (#27) — Blinded Scoring Instrument

**Status:** Phase 2 complete 2026-07-18 — awaiting operator scoring (Phase 3, non-delegable).
**Instrument for:** BACKLOG #27 (CLI-4 parity run → ADR-12 §5 flip/retire). Corpus: `../2026-07-18-cli4-parity-corpus.md` (frozen).

## What this is

24 blinded debate transcripts = 12 frozen briefs × 2 arms, each arm run on the **same** models
(`claude-opus-4-8` + `gpt-5.6-terra`) and the **same** trial-scoped synthesizer (`gemini`) —
**transport is the only manipulation** (Arm A / Arm B = CLI-backed / all-API, per pair randomized).

Backend tells were stripped from every artifact: the `**Date:**`/`**Cost:**`/`**Duration:**`
headers and the per-response `*Latency: … | Tokens: …*` footers (CLI subprocess runs slower and
its seats cost $0 — those were the discriminators). All structural headers are identical within
each pair; only the verdict/transcript **content** differs. Verified: no residual metadata tell.

## How to score (Phase 3 — you, blind)

1. Open `blinded/<ID>-A.md` and `blinded/<ID>-B.md` for each of the 12 pairs.
2. Fill `SCORING-SHEET.md` — mark PASS/FAIL for A and B on each of the 5 rubric items
   (`../../../protocols/SYNTHESIS_QUALITY_RUBRIC.md`). Any FAIL = investigate the failure mode
   before concluding (rubric operating principle).
3. **Do NOT open `SEALED-KEY.json`** (gitignored, kept local) until all 12 pairs are scored — it
   maps A/B → cli/api per pair. Opening early defeats the blind.

## Then (Phase 4 — unseal + decide)

Unseal, tally per-item CLI-vs-API failures. **Threshold:** CLI fails ≤1 more pair than API per
item (margin 1/12), **zero margin on items 2 (no hallucinated consensus) and 4 (faithfulness)**.
PASS → ratify DRAFT-CLI-3 (amend ADR-12 §5, flip `standard` default to CLI). FAIL on 2/4 →
per the kill condition (regression persisting across two attempts) retire the flip; one failure →
pause + diagnose. The parity report must document the **gemini trial-scoped synthesizer** condition.

## Provenance

Pins/witness: JOURNAL 2026-07-18 "#27 WITNESS PASS" (`gpt-5.6-terra` = the medium 5.6 tier;
both CLI seats witnessed at $0, zero fallback). Phase 1: all 24 runs succeeded, rider-(d)
fallback-abort never triggered (every CLI arm served both seats via CLI). Sealed key + raw
(un-blinded) outputs are local-only.
