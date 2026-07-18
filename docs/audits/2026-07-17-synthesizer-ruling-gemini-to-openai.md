# Operator ruling — synthesizer gemini → openai (G3 / Epic-B gate event)

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — `settings.yaml` `synthesizer: openai`; #2 Branch A shipped 2026-07-18; #24 closed; residual: #2/#3 (ADR-01 amendment, landed this session). _(Additive inventory stamp; body below unchanged.)_

**Class:** audits (governance ruling record) · **Date:** 2026-07-17 · **Status:** RULED (operator authority) · **Filing only** — no durable config change tonight.
**Recorded by:** Claude Code (night batch), from the architect's commission on the operator's authority. This note records a ruling the operator issued; it does not make the decision.

## The ruling
The default **synthesizer switches gemini → openai** (API-backed). The synthesizer-never-CLI guard stands: the synthesizer stays on the API lane (the CLI cost lane already covers the panel, so synthesizer cost stays marginal — openai `gpt-5.4` synthesis measured at ~$0.07/debate tonight).

## Basis (verbatim)
> the operator observes a recent quality decline in Gemini models.

## Instrument retained (reversible)
The **EPI-1 40-item blind-scoring pack + the sealed identity key remain retained, unscored**, as the standing validation instrument (`docs/audits/2026-07-17-epi1-archaeology/`, key at `…-epi1-archaeology-KEY-SEALED.json`, both gitignored). **This ruling is reversible against that evidence at any time** — the archaeology can still be scored to confirm or overturn it.

## Effect
This ruling is the **Epic-B gate event (G3)** — the event that un-gates #18/#19/#9, D12/D13, and the v2 crux-resolver ranking (ADR-13). (Per the plan-of-record §2 gates, G3 was defined as "the operator's ruling on the EPI-1 archaeology report (#24)"; the operator has instead resolved the synthesizer question by direct authority and retained the pack as the reversible instrument. The gate is satisfied by ruling; #24's blind-scoring remains available as the confirming/overturning evidence path.)

## Supporting evidence gathered this session (not the basis — corroboration)
Tonight's live E2E batch ran **openai `gpt-5.4` as the synthesizer across 4 real debates** (UC1 Rama 1, UC2 Rama 3, UC3 DeepSeek, UC4 model-currency). All four verdicts were **decisive, on-brief, and faithful to the transcript** (correct per-panelist attribution, explicit strongest/weakest-argument calls, `error_class=none`), and in UC2 the synthesizer *overrode* a leading design lean with a de-risked alternative rather than rubber-stamping. See `docs/audits/2026-07-17-night-batch-empirical-e2e-audit.md` §2.3. This is corroboration, not the basis; the basis is the operator's observation above.

## Explicitly deferred (NOT tonight)
- The **durable `config/settings.yaml` synthesizer swap** (`defaults.synthesizer: gemini → openai`) is **not** made tonight. It lands as a supervised micro-arc tomorrow, after the operator reviews this night-run evidence. Tonight's runs applied `synthesizer: openai` per-brief via frontmatter only; the durable default remains `gemini` (verified: `config/settings.yaml` byte-identical at session end).
- Any ADR-01 amendment text (Branch A/B codification, #2/#3) — a supervised arc, not tonight.
