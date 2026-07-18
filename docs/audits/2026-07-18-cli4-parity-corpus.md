# CLI-4 Parity-Run Corpus (#27) — FROZEN

**Status:** FROZEN 2026-07-18 · **Instrument for:** BACKLOG #27 (CLI-4 parity run → ADR-12 §5 flip/retire).
**Lock (rider b):** these 12 briefs are frozen; no swap once any Phase-1 pair has run.
**Provenance:** 6 real past inbox briefs (`council_inbox/archive/`) + 6 fresh (avoids training-on-the-test). The 3 night-batch `uc*` briefs are deliberately excluded (prior-evidence contamination).

## Fixed trial conditions (identical across both arms of every pair)

- **Panel:** `claude` (`claude-opus-4-8`) + `openai` (`gpt-5.6-terra`) — the two seats under test.
- **Synthesizer:** `gemini` — **trial-scoped only** (frees the openai API seat so codex pairs transport-only; the durable ratified default `openai` in `config/settings.yaml` is untouched). Identical across both arms → internal validity holds (operator condition, 2026-07-18).
- **Rounds:** 2. **min_panel_size:** 2.
- **Arm A** = CLI-backed both seats (`claude` CLI + `codex` CLI, pins carried from the witness). **Arm B** = all-API both seats (same models via API). Transport is the ONLY manipulation (F4).
- **Pin provenance:** OpenAI API `gpt-5.6` tiers = `[luna, sol, terra]`; `sol` = flagship (codex config default), **`terra` = the medium tier (the pin)**; `claude-opus-4-8` per newest-models policy. Witnessed at $0 (JOURNAL 2026-07-18 "#27 WITNESS PASS").

## The 12 briefs

### PICK (6)

- **P1** *(real — ecosystem-separator, 2026-05-11)*: What is the canonical filename / foldername separator across the Dev ecosystem?
- **P2** *(real — local-scheduled-tier, 2026-06-06)*: What mechanism should host the recurring local cross-repo fleet-baseline run?
- **P3** *(real — handoff-Q2-bundle-composition, 2026-05-26)*: What determines which context a handoff bundle carries — a fixed set every time, or one selected for the work the next session will do?
- **P4** *(real — handoff-Q1-internalization-assurance, 2026-05-25)*: What mechanism should a handoff use to confirm a fresh chat has internalized the bundle — rather than only paraphrased it — before it begins substantive work?
- **P5** *(fresh)*: For a single-binary CLI's config, pick one default format: TOML, YAML, or JSON.
- **P6** *(fresh)*: For a small Python library's dependency policy, pick one: a committed lockfile, or floating minimum-version ranges.

### JUDGE (6)

- **J1** *(real — topic-2-handoff-synergy, 2026-04-27)*: How should a solo developer split work between a browser chat AI and a terminal-based coding agent, and what handoff format minimizes friction transferring work between them?
- **J2** *(real — adr-01-synthesizer-panel-refresh, 2026-05-11)*: Should ai-council refresh its default debate panel and synthesizer for the 2026 model landscape?
- **J3** *(fresh)*: Judge whether REST or gRPC better fits an internal mesh of ~20 microservices — weighing latency, tooling maturity, and debuggability.
- **J4** *(fresh)*: Judge whether a monorepo or polyrepo better serves a 5-engineer startup — weighing CI cost, code sharing, and release independence.
- **J5** *(fresh)*: Judge whether feature flags or short-lived trunk-based branches better control release risk for a weekly-shipping web app — weighing rollback speed, review overhead, and tech-debt accrual.
- **J6** *(fresh)*: Judge whether PostgreSQL or a document store better fits an early-stage app with an evolving schema — weighing iteration speed, query power, and operational burden.

## Scoring rubric (5-item binary; `protocols/SYNTHESIS_QUALITY_RUBRIC.md`)

1. Position representation — all major positions represented?
2. **No hallucinated consensus** — avoids claiming agreement not reached? *(zero-margin item)*
3. Scannability — verdict + trade-offs extractable without re-reading?
4. **Faithfulness** — accurately reflects what each panelist said, no distortion/invention? *(zero-margin item)*
5. Verbosity proportionality — length appropriate to substance?

## Success threshold (ratifies the ADR-12 §5 flip)

Per rubric item, the CLI arm fails **at most one more pair** than the API arm (non-inferiority margin 1/12) — **except items 2 and 4, where the margin is zero**. **Kill condition:** an item-2/4 regression persisting across two attempts (separated by a containment fix) retires the flip; `backend: cli` stays per-seat opt-in. One failure → pause + diagnose.
