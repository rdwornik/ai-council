# #27 CLI-4 Parity — Per-Pair Scoring Record (BLIND)

Blank record the operator fills while scoring. One row per pair, in reading order.
This is the **which-wins summary**; the per-rubric-item PASS/FAIL detail goes in `SCORING-SHEET.md`.
You do NOT know which of A/B is which backend — do not open `SEALED-KEY.json` until all 12 rows are done.

**Columns**
- **Option A / Option B** — overall verdict on that artifact (e.g. PASS / FAIL, or a 1-line quality note).
- **Which wins** — A, B, or TIE (your blind preference for the better synthesis).
- **Margin notes** — how close it was; flag any item-2 (no hallucinated consensus) or item-4
  (faithfulness) FAIL here, since those are ZERO-margin.

| # | Pair | Mode | Option A | Option B | Which wins (A/B/TIE) | Margin notes |
|---|---|---|---|---|---|---|
| 1 | J1 | judge |  |  |  |  |
| 2 | J2 | judge |  |  |  |  |
| 3 | J3 | judge |  |  |  |  |
| 4 | J4 | judge |  |  |  |  |
| 5 | J5 | judge |  |  |  |  |
| 6 | J6 | judge |  |  |  |  |
| 7 | P1 | pick |  |  |  |  |
| 8 | P2 | pick |  |  |  |  |
| 9 | P3 | pick |  |  |  |  |
| 10 | P4 | pick |  |  |  |  |
| 11 | P5 | pick |  |  |  |  |
| 12 | P6 | pick |  |  |  |  |

## Tally (fill after unseal — Phase 4)

Margin rule: CLI may fail at most **1 more pair than API per item (1/12)**; items **2 and 4 are ZERO-margin**.

| Rubric item | CLI fails | API fails | Δ (CLI−API) | Within margin? |
|---|---|---|---|---|
| 1 Position representation |  |  |  |  |
| 2 No hallucinated consensus (ZERO-margin) |  |  |  |  |
| 3 Scannability |  |  |  |  |
| 4 Faithfulness (ZERO-margin) |  |  |  |  |
| 5 Verbosity proportionality |  |  |  |  |
