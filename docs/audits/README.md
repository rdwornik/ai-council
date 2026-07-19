# Audits

Archive of audit artifacts produced for ai-council — Codex reviews, code review reports, and similar quality assessments.

## Directory invariant (ruling 2026-07-19)

**The invariant is NOT "markdown only."** This directory holds exactly three classes of entry:

| # | Class | Rule |
|---|---|---|
| a | **Date-slug markdown records** | `YYYY-MM-DD-<topic>.md` — the normal case |
| b | **`archive/`** | The preservation archive, governed by its own `archive/README.md`. Taxonomy, not clutter — never swept |
| c | **A registered live corpus** | A folder holding raw trial/scoring material, permitted **only while live** |

A corpus may sit here **only if both hold**: it is **still live**, **and** it has (1) an **essence markdown at this
root** and (2) a **row in the Live corpora table below**. A corpus that is no longer live, or that lacks either the
essence markdown or the registry row, does not belong here — it exits to `archive/` per its own exit condition.

Registration is what makes a corpus legible. An unregistered folder is indistinguishable from a leftover.

## Live corpora

| Path | What it is | Ruling that keeps it here | Essence markdown | Exit condition |
|---|---|---|---|---|
| `2026-07-17-epi1-archaeology/` | 40-item blind synthesis-scoring pack + sealed identity key; retained unscored | Reversal instrument for the G3 synthesizer ruling (`2026-07-17-synthesizer-ruling-gemini-to-openai.md`); stay-in-place reaffirmed by rider (a), 2026-07-18 | `2026-07-19-epi1-archaeology-pack-condensation.md` | The G3 ruling is reversed or permanently settled |
| `2026-07-18-cli4-parity/` | Live #27 blind backend-parity trial; contains a sealed key (exclusion zone) | #27 scoring is open; the seal must not be disturbed while blind | The parity report written at unseal | Scoring **and** unseal complete → exits to `archive/` |

## Convention

- Naming: `YYYY-MM-DD-<topic>.md` (date prefix enables chronological sort by filename)
- One file per audit cycle
- Earlier `docs/archive/` consolidated here as of 2026-05-11 (commit history preserved via `git mv`)

## What goes here

- Codex `/review` output for branches with 3+ files (per Playbook §15)
- Standalone code review reports
- Targeted audits (e.g., security review, performance audit)

## Archive convention

Pre-ADR-34 audit reports with underscore+UPPERCASE filenames are archived in `archive/legacy/`.
Original filenames preserved for historical accuracy. Current-format audits (hyphen+lowercase) live at this level.

## What doesn't go here

- Cross-repo handoff artifacts → `docs/handoffs/archive/`
- ADR debate transcripts → owned by `.dev-knowledge` strażnik per ADR convention
- Test reports → not archived (live in CI / test runner output)
