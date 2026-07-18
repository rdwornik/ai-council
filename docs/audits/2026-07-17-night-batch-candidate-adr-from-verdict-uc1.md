# Consumption test — candidate ADR drafted from a verdict package ALONE (UC1 / Rama 1)

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — DRAFT-INT-1 validated; DRAFT-INT-2 ratified as ADR-13 this session; open: #18 (crux-check), #38 (verdict→ADR read-back). _(Additive inventory stamp; body below unchanged.)_

**Class:** audits (night-batch artifact, NOT a `docs/decisions/` ADR) · **Date:** 2026-07-17
**Method (amendment 2):** this ADR was drafted **solely** from `output/council-verdict-20260717_230406-pick-uc1-rama1-crux-grounding.json`. The transcript (`council-out-…-uc1-…md`) and the minority report were **deliberately NOT read**. The purpose is to measure the transcript-free contract (DRAFT-INT-1, L-INT R2) empirically and feed #38 (verdict→ADR read-back guide).

---

## CANDIDATE ADR-NN — Bounded crux-check step for empirical grounding in decision-mode debates

**Status:** Proposed (candidate — baseline-gated per DRAFT-EPI-3 / reserved ADR-13; do NOT ratify tonight)
**Verdict author:** openai `gpt-5.4` (non-participant synthesizer) · **Panel:** claude, deepseek, gemini (all seated) · **Dissent:** non-unanimous (minority report exists) · **Run:** `council-out-20260717_230406-pick-uc1-rama1-crux-grounding` · **exit_semantics:** 0

### Context
Decision-mode debates (pick/judge) have zero retrieval; empirical sub-claims arising mid-debate are settled by the most confident voice, not by evidence (audit G6). The council can be confidently wrong on a checkable fact. The decision is which mechanism grounds a flagged empirical crux before Round-2 voting, under hard constraints: rounds ≤ 2 stay capped; blind voting (ADR-03) inviolable (no label→provider map to the resolver; evidence identical across all Round-2 prompts); ~$0.50/debate cost gate; determinism + ephemeral-brief inbox I/O preserved; memory-only answers count as UNRESOLVED.

### Decision
Adopt **option (c): a discrete, bounded crux-check step between rounds**, implemented as a **centralized pipeline stage that emits one canonical evidence artifact injected identically into all Round-2 prompts** (not delegated to individual panelists).

### Rationale (from the package's "Argument Quality Assessment")
- Best-reasoned proposal on system design separated the **mechanism** (option c) from the **resolver backends**, specified hard budgets, typed outcomes, and a canonical artifact, and explicitly handled `UNRESOLVED`.
- The **single strongest argument** in the debate: empirical grounding must be **centralized into one canonical, identical evidence artifact** injected into every Round-2 prompt rather than delegated to panelists — it solves the motivating failure mode, preserves blind voting *structurally*, keeps grounding bounded/auditable, and cleanly separates decision debates from open-ended research.
- The **single weakest argument**: the implicit claim that a lightweight extractor can reliably pick the top ≤3 decision-critical empirical cruxes without additional machinery — several participants flagged this as the likely first failure mode with no empirical support offered.
- Rejected framings flagged as weak: "source grounding is deterministic if cached", "read-only means safe", and an absolutist "no CLI at all."

### Dissent (unresolved)
Non-unanimous. Recorded crux (gist): **"Should the crux-check be allowed to use CLI execution for repo/local facts?"** — full minority reasoning is in a *separate* artifact (`council-minority-…-uc1-…md`), not in the verdict JSON.

### Consequences
- v1 must treat the ≤3-crux extractor as the primary risk surface and measure it (falsifier: does evidence injection ever move a Round-2 position?).
- Resolver backends are pluggable behind the mechanism; memory-only = `unresolved(no-source)`.
- Preserves ADR-03 by construction (identical injection, no label map to the resolver).

---

## Fields the verdict package did NOT provide (empirical finding for #38 / DRAFT-INT-1)

Drafting the above from JSON alone was **possible and produced a usable ADR**, but the following had to be worked around — raw input for the verdict→ADR read-back guide (#38) and possible DRAFT-INT-1 hardening:

1. **`options_considered.items` is EMPTY (`[]`).** The structured "which options were weighed and why rejected" is absent. I recovered the options (a)/(b)/(c) from the raw `question` field, not from a synthesized options field. → **Gap:** the package carries the *question's* options but not the panel's *structured weighing* of them.
2. **No per-option vote tally / no "who favored what".** The `rationale` names proposals qualitatively ("strongest on system design") but there is no structured vote or option→support mapping. An ADR "Options considered / rejected because" section must be reconstructed from prose.
3. **Dissent is a one-line gist + a pointer.** `dissent.gist` gives only the first dissent question; the actual minority *argument* lives in a separate file. A transcript-free caller gets the *existence and topic* of dissent, not its substance, from the JSON alone.
4. **Only two synthesis sections are extracted as fields.** `decision.value` ← "Recommended Decision"; `rationale` ← "Argument Quality Assessment". The synthesis's **Consensus**, **Risks**, and **Action Items** sections are NOT in the package — an ADR "Consequences/Action Items" section has to be inferred (I inferred Consequences from the rationale's weakest-argument call-out). → **Gap:** consider extracting Risks/Action-Items into the package.
5. **No confidence field.** No machine-readable confidence/margin for the decision.
6. **`contract_version` is `null`** (known: DRAFT-INT-2 versioning not yet stamped).

**Net verdict on the transcript-free contract:** **usable** — a competent caller can draft a defensible ADR from the verdict JSON alone for a `pick` debate. The decision, rationale spine, panel, author, and dissent-existence are all present and faithful. The gaps are in *structured options* and *dissent substance*, both currently requiring the separate question/minority artifacts. This matches DRAFT-INT-1's design (the JSON is the machine-authoritative decision record; the minority report is a discrete artifact) but argues #38's read-back guide should tell the caller: "for options-considered and dissent substance, open the `question` and `council-minority-*` artifacts named in `artifacts[]`."
