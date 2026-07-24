# #27 CLI-4 Parity — Blind Scoring Reading Index

**One-stop instrument.** Open this, read the rubric + margin rules below once, then work
the 12 pairs top-to-bottom. Record rubric PASS/FAIL in `SCORING-SHEET.md` and the per-pair
which-wins call in `SCORING-RECORD.md`. **Do NOT open `SEALED-KEY.json` until all 12 are scored.**

- **What each pair is:** one frozen brief run twice on the *same* models (`claude-opus-4-8` +
  `gpt-5.6-terra`) and the *same* trial synthesizer (`gemini`). **Transport is the only
  manipulation** — Arm A / Arm B = CLI-backed / all-API, randomized per pair. You do not know which.
- **Headers are normalized** (`**Source:** cli` is a constant on every file, backend tells stripped);
  only the verdict/transcript **content** differs within a pair.

## Rubric (inline — `protocols/SYNTHESIS_QUALITY_RUBRIC.md`)

Score each artifact A and B **binary PASS/FAIL** on all 5 items. Any FAIL = investigate the
failure mode before concluding.

1. **Position representation** — every seated model's stance is present and fairly stated.
2. **No hallucinated consensus** *(ZERO-margin)* — no agreement asserted that the transcript does not support.
3. **Scannability** — verdict is skimmable (structure, headers, decision up front).
4. **Faithfulness** *(ZERO-margin)* — claims trace to what the panel actually said; no invention.
5. **Verbosity proportionality** — length fits the question; no padding, no starvation.

## Margin rules (the decision threshold)

- **Overall:** CLI may fail at most **1 more pair than API per item** — margin **1/12**.
- **Items 2 and 4 are ZERO-margin** — CLI failing *any* pair that API passes on item 2 or item 4
  fails parity on that item.
- PASS across all items → ratify DRAFT-CLI-3 (amend ADR-12 §5, flip `standard` default to CLI).
  Zero-margin FAIL on item 2 or 4 → kill/pause per the README's Phase-4 kill condition.

## The 12 pairs (score in this order)

### Judge mode (J1–J6)

| # | Pair | Brief | Option A | Option B |
|---|---|---|---|---|
| 1 | J1 | Solo dev: split work between browser-chat AI and terminal agent | `blinded/J1-A.md` | `blinded/J1-B.md` |
| 2 | J2 | Refresh ai-council's default debate panel + synthesizer for 2026 | `blinded/J2-A.md` | `blinded/J2-B.md` |
| 3 | J3 | REST vs gRPC for an internal mesh of ~20 microservices | `blinded/J3-A.md` | `blinded/J3-B.md` |
| 4 | J4 | Monorepo vs polyrepo for a 5-engineer startup | `blinded/J4-A.md` | `blinded/J4-B.md` |
| 5 | J5 | Feature flags vs short-lived trunk-based branches | `blinded/J5-A.md` | `blinded/J5-B.md` |
| 6 | J6 | PostgreSQL vs document store for an early-stage app | `blinded/J6-A.md` | `blinded/J6-B.md` |

### Pick mode (P1–P6)

| # | Pair | Brief | Option A | Option B |
|---|---|---|---|---|
| 7 | P1 | Canonical filename/foldername separator across the Dev ecosystem | `blinded/P1-A.md` | `blinded/P1-B.md` |
| 8 | P2 | Mechanism to host the recurring cross-repo fleet-baseline run | `blinded/P2-A.md` | `blinded/P2-B.md` |
| 9 | P3 | What determines the context a handoff bundle carries | `blinded/P3-A.md` | `blinded/P3-B.md` |
| 10 | P4 | Mechanism for a handoff to confirm a fresh chat internalized context | `blinded/P4-A.md` | `blinded/P4-B.md` |
| 11 | P5 | Single-binary CLI config default: TOML vs YAML vs JSON | `blinded/P5-A.md` | `blinded/P5-B.md` |
| 12 | P6 | Small Python library dependency policy: lockfile vs … | `blinded/P6-A.md` | `blinded/P6-B.md` |

## After scoring

1. Both sheets filled for all 12 pairs.
2. **Then** open `SEALED-KEY.json` (maps A/B → cli/api per pair) — Phase 4.
3. Tally per-item CLI-vs-API failures against the margin rules above; write the parity report
   (documenting the gemini trial-scoped synthesizer condition). Corpus exits to `archive/`
   only after scoring **and** unseal complete (per `docs/audits/README.md`).
