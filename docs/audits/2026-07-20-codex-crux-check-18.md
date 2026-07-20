# Codex Review — crux-check-18

**Date:** 2026-07-20
**Branch:** `feat/crux-check`
**HEAD:** `b463fd9`
**Diff range:** `main..feat/crux-check`
**Codex version:** codex-cli 0.144.5
**Mode:** diff-review

---

## Focus

- ADR-03 anonymization boundary: can ANY provider/model attribution reach the Round-2 evidence block? CruxCheckService.check takes only the already-anonymized str block; _build_evidence_block deliberately avoids report.merged_report/summary_2500 because those carry per-provider headers and three research provider names (gemini, grok, openai) collide with PANEL model names.
- Three-outcome contract: is retrieval ever fabricated when no empirical crux exists? no_empirical_crux must short-circuit BEFORE the executor is awaited. Can the debate abort on any crux failure path? It must always degrade, never raise.
- contract_version must stay "1.0" and the crux artifact must NOT reach _build_verdict_payload (output.py:1141). output.py should be byte-identical to main.
- Duplicated fanout in research/headless.py vs run_research_with_display in display.py: drift risk in the exception taxonomy. _error_result is imported (shared), only the try/except split is duplicated.
- Source-compat of the run_debate (new trailing crux_check param) and build_debate_metrics (new trailing extra_calls param) signature changes across all existing call sites and AsyncMock patches.
- Round-2 prompt injection: evidence is a separate param, never concatenated into anon_block. Check the pick-mode compose-after-format path and the ideas/judge parts-list insertion for correctness.

---

## Findings
## Critical

### [CRITICAL] src/ai_council/crux_check.py:94 — Raw model attribution can enter Round 2

**What:** The evidence block copies the extractor’s claim and each `ResearchResult.content` verbatim; either model can self-identify or emit provider/model headers even though `merged_report` is avoided.  
**Why:** Names such as Gemini, Grok, or OpenAI can reach the evidence block and violate the ADR-03 anonymization boundary.  
**Fix direction:** Mechanically validate or normalize generated claim/evidence content against provider/model attribution before injection, degrading to `retrieval_unavailable` when attribution cannot be safely removed.

## High

### [HIGH] src/ai_council/crux_check.py:155 — Malformed extraction is reported as a valid no-crux outcome

**What:** `_parse_crux()` returns `None` for both explicit `NONE` and empty, malformed, or missing-heading responses; `check()` maps every case to `NO_EMPIRICAL_CRUX`. Broad prefixes also reject checkable negative claims such as “There is no statistically significant difference…”.  
**Why:** Extraction failures or valid empirical claims silently skip retrieval while falsely reporting a successful no-crux determination.  
**Fix direction:** Return distinct parse states for claim, explicit `NONE`, and invalid output; map invalid output to `RETRIEVAL_UNAVAILABLE` and recognize only the exact sentinel.

### [HIGH] src/ai_council/crux_check.py:146 — Configured size bounds are not enforced

**What:** `CruxCheckConfig.max_tokens` is never passed to the extractor, so the synthesizer’s normal 16,384-token limit applies; `_MAX_EVIDENCE_CHARS` also caps only the research body, excluding the claim, header, URLs, and footer.  
**Why:** The supposedly bounded step can generate and inject a much larger artifact than configured, increasing latency, cost, and Round-2 context usage.  
**Fix direction:** Add a real per-call output-token override and cap the fully assembled evidence artifact, including the claim and source list.

### [HIGH] src/ai_council/debate.py:243 — Service exceptions erase the third outcome

**What:** The defensive exception path sets `crux_artifact` to `None`, the same state used when the feature is disabled.  
**Why:** Although the debate continues, the failure is absent from `DebateOutcome` and downstream status output, breaking the enabled feature’s three-outcome contract.  
**Fix direction:** Convert caught exceptions into a `CruxArtifact` with `RETRIEVAL_UNAVAILABLE` and diagnostic detail.

## Medium

(none)

## Low

(none)
