# L-EPI — Epistemic Quality / Bias Reduction: Functional Design

**Date:** 2026-07-06 · **Lane:** L-EPI (pillar 1 — the mission) · **Mode:** DESIGN — functional architecture only; zero source/config changes; no ADR numbers claimed
**Designer:** Fable 5 (one of five parallel lane sessions; worktree `lane-epi`)
**Base state:** ratified merge `5c81e71` (ADR-11 + ADR-12 Accepted; `COUNCIL_INVOCATION_CONTRACT.md` live; Wave-0 GUIDE fixes)
**Sources:** fleet-recon 2026-07-05 (P1–P4, F1–F12), Fable audit 2026-07-04 (D1–D14, ADR-13 draft), GUIDE, RUBRIC, ADR-01..12, BACKLOG (read-only)
**Checkpoint:** skeleton approved by operator 2026-07-06 ("Approve — write the document")

---

## 1. Charter & current state (verified live)

Entry-state claims from the lane charter, each verified against the live tree at `5c81e71`:

| Charter claim | Verdict | Live evidence |
|---|---|---|
| Rubric exists | ✅ | `protocols/SYNTHESIS_QUALITY_RUBRIC.md` — 5 yes/no criteria, rollback trigger |
| GUIDE research-mode section exists (~:342-447) | ✅ (shifted) | Content-located at `## Research-mode questions` (recognition test, format, quick template) — now ~:345-467 after Wave-0 edits |
| Contamination vector closed going forward (Wave-0 C1) | ✅ | GUIDE examples no longer hardcode `synthesizer: openai`; the sole remaining mention is an explicitly-labeled override example ("This is the only reason to set it"); the synthesizer rule text now correctly describes eviction |
| History contaminated | ✅ **and quantified — see Delta 2** | Corpus tally below |
| #18/#19/#9 unbuilt + baseline-gated | ✅ | `BACKLOG.md` Epic F: #9 "DEFERRED — do NOT build before the canonical-baseline settles"; #18/#19 "baseline-gated (Epic B)" |
| Minority report shipped (#15) | ✅ | `output.py` `extract_dissent`/`save_minority_report` live; grooming log records #15 closed (commit `f1a4b74`) |

**Delta 1 (housekeeping).** This worktree branch was cut from `75006db`, *before* the ratification merge. Fast-forwarded to `5c81e71` (no local commits existed) so the design runs against ratified state.

**Delta 2 (changes Q3 materially).** The charter's identity-archaeology framing assumed forensic recovery. Live inspection shows identity is **in-band since the first pipeline commit**: the `## Synthesis (by X, participant|non-participant)` transcript header dates to 2026-02-21 (`11ed62a`), and `synthesis_metrics.synthesizer_model` in the `_metrics.json` sidecar dates to 2026-05-12 (`555ebf0`). Witnessed corpus tally (2026-07-06):

| Corpus | Files | With identity header | openai | claude | gemini | claude-sonnet | anomalies |
|---|---|---|---|---|---|---|---|
| Local `output/` (primary checkout) | 239 | ~138 | 56 | 54 | 21 | 7 | 3 × `participant` synthesizer (openai); 1 × pre-label-format `(by claude)` |
| Hub `transcripts/` (curated copies) | 48 | 36 | 21 | — | 10 | 5 | 12 × no header (all research-mode — no synthesis section by design) |

Consequences: (a) the contamination is **majority-scale** — openai authored ~40% of identity-readable syntheses, gemini only ~15%; (b) BACKLOG #1's "~15 historical transcripts" premise maps to a gemini segment of n=10 (hub) / n=21 (local) — usable, but **segmentation is mandatory, not optional**; (c) the contaminated history is also an *accidental A/B experiment* — see Q3 and operator question OQ-1; (d) the D5 scoring guard **will** fire — F10's deciding evidence is effectively already in (see §5).

---

## 2. Functional target state

How pillar 1 looks and behaves when this lane is done — written for a non-implementer to validate. The through-line: **every defense is falsifiable, every verdict author is accountable, every alarm is advisory-loud rather than blocking.**

**Authoring.** The operator (or a Lane-A caller per ADR-11) writes a brief. A question gate — available both as an authoring aid where the caller works and as a validator inside the council — checks the brief against the GUIDE's structural and framing rules and returns an itemized PASS/WARN/FAIL verdict keyed to GUIDE sections. A WARN never blocks; the author sees *which* rule tripped and the GUIDE fix for it. (Gated: shape settled here, build after Epic B.)

**Run.** The debate runs as today (blind critique per ADR-03, rounds ≤ 2). Between round 1 and round 2, if panelists on opposing sides flagged the same checkable factual claim, a bounded **crux check** fires: the claim is grounded by an evidence lookup, and the evidence — source-attributed, panelist-anonymous, marked tool-derived — is injected identically into every round-2 prompt. Resolution failure never blocks the debate; "crux flagged, unresolved" is itself recorded signal. (Gated; shape = the audit's ADR-13, sharpened in Q4.)

**Verdict.** The synthesis is produced by the intended, recorded author. The transcript carries a visible "Synthesized by X (non-participant)" line (already true) **and** the metrics sidecar carries a first-class `synthesis` block: intended author, which precedence source set the intent, actual author, actual model ID, participant flag, and an `intent_match` boolean. An intent/actual mismatch prints a loud warning and annotates the transcript — it never blocks. If the run's consensus pattern-matches a framing artifact (unanimity echoing the brief's own framing, anchor echo, unused escape option), a **framing alarm banner** appears on the output and in the artifacts: the verdict is still delivered, marked "treat as advisory — consensus may be a framing artifact," with the trigger named and the GUIDE fix suggested. (Alarm gated; identity block actionable now.)

**Record.** Every debate leaves: transcript with author line (and any alarm/crux annotations), minority report when dissent, metrics with the `synthesis` block. A future scorer can segment any corpus by verdict author *from the artifacts alone* — the EPI-1 protocol (Q3) is executable today and stays executable forever.

**Measurement.** No defense is faith-based. Each mechanism in the bias registry (Q1) names the observation that would falsify its value (Q2), and the registry is the standing checklist for periodic epistemic audits — the same way the recon's liveness matrix is the standing checklist for health.

---

## 3. Design answers

### Q1 — Bias taxonomy in scope

The registry: bias → vector → existing mechanism → designed mechanism → residual exposure no mechanism covers.

| # | Bias | Vector | Existing defense | Designed defense | Residual exposure |
|---|---|---|---|---|---|
| B1 | **Inherited framing** | The brief itself leans; every panelist reads the same lean — the one failure mode with no downstream safety net (GUIDE's own analysis) | GUIDE neutralization section + pre-flight self-check (prose discipline); hub's manual six-question pre-flight | #9 gate (form-level screen); #19 alarm (runtime symptom detector) | A well-formed but subtly leading question passes any form check; only a human re-framing or a symmetric rerun catches it |
| B2 | **Anchoring** | First/longest/most-charitable option becomes the reference point; order effects in critique reading | GUIDE option rules (equal length, neutral order, no editorializing); ADR-03 shuffle randomizes critique-round reading order | #19 trigger (ii): anchor-echo detection | Round-1 proposals all read the same option order in the brief; the synthesizer reads responses in a fixed order — no shuffle at synthesis time |
| B3 | **Identity-driven sycophancy** | Panelists defer to (or pile on) a recognized peer's position rather than its content | ADR-03 blind critique: `_anonymize_responses()` shuffle + A/B/C relabel (contract-protected per CLAUDE.md §10) | — (measurement only: Q2/M3 style-leakage probe) | Style leakage: models may *guess* authorship from prose style; nothing measures whether the blind actually blinds. ADR-12 CLI seats add vendor-harness style as a new fingerprint |
| B4 | **Persuasion-over-evidence** | The most confident voice settles empirical sub-claims; decision modes have zero retrieval (audit G6) | none | #18 crux grounding (Q4) | Unflagged cruxes (panel agrees confidently and wrongly — no disagreement to trigger on); value cruxes (not evidence-checkable by design); resolver-wrong-answer risk |
| B5 | **False / hallucinated consensus** | Synthesis claims agreement not actually reached; or agreement is real but manufactured by B1/B2 | RUBRIC criterion 2 (manual, post-hoc); minority report #15 surfaces *recorded* dissent | #19 false-consensus alarm (Q5); D13 structured-dissent hardening (Epic B, not this lane's to design) | Minority detection is heading-heuristic — couples to synthesizer discipline; consensus-on-contaminated-framing is invisible post-hoc without the alarm |
| B6 | **Breadth-sprawl** | Too many questions/facets dilute evidence depth; each answered shallowly | GUIDE size rules (3–7 questions, 3-facet research cap, one-sentence rule) — prose only | #9 gate (mechanical count checks) | Sprawl *inside* one question (compound options) passes count checks |
| B7 | **Ownership bias** | The asker scores their own debate; a participating synthesizer judges a debate it argued in | Non-participating synthesizer (ADR-01 eviction — the verdict author holds no position in the debate); GUIDE recognition test keeps the asker from smuggling a decision into research mode | EPI-1 blind scoring protocol (Q3 — scorer blinded to author segment); DRAFT-EPI-2 `participant` flag makes the anomaly machine-visible | Operator rubric-scoring of live runs is unblinded today; the 3 witnessed `participant` runs show the eviction invariant has been bypassable via frontmatter |
| B8 | **Verdict-author drift** | Invocation habit (frontmatter examples) silently swaps who writes the verdict — the audit's real finding | Wave-0 C1 doc fix (vector closed at the source) | DRAFT-EPI-2: intended vs actual author recorded + checked per run (Q7) | Closed-going-forward is a doc promise, not a mechanism, until EPI-2 records intent/actual on every run; history stays contaminated (handled by Q3 segmentation, not repair) |

Boundary note: B3's CLI-seat fingerprint and harness-contamination containment are L-CLI's charter (their Q6); this lane only registers the exposure. The taxonomy is the umbrella DRAFT-EPI-1's normative content: **a mechanism that defends no named bias, or a named bias with no falsifier, is a registry violation.**

### Q2 — Measurement: what would falsify each mechanism's value

G5 baseline philosophy applied: for each defense, the observation that would show it is *not* reducing bias. These are functional measurement designs, not tooling plans.

- **M1 — GUIDE prose discipline (B1/B6).** Measure: framing-failure and mis-mode rates in the inbox archive (the F9 evidence — an archaeology pass can piggyback this audit: same corpus, one extra scoring column). Falsified if failure rates are no lower after the GUIDE's bias section landed (2026-05-17) than before.
- **M2 — Eviction / non-participating synthesizer (B7).** Measure: EPI-1 segment comparison — do `participant`-authored syntheses score worse on RUBRIC criteria 2 (hallucinated consensus) and 4 (faithfulness)? Witnessed n=3 is too small to rule; recorded as a standing segment that accumulates if the anomaly ever recurs. Falsified if, at usable n, participant syntheses score no worse — which would license reopening the ADR-02 overlap policy (#4) with evidence.
- **M3 — ADR-03 blind critique (B3).** Measure: a deanonymization probe — present anonymized round-1 blocks (they exist verbatim in transcripts) to an offline judge asked to attribute authorship; chance is the baseline. Falsified-in-part if attribution accuracy is far above chance (the blind leaks); falsified-in-full if, additionally, critique agreement patterns track *guessed* identity prestige. Zero live-run cost — runs entirely on the existing corpus.
- **M4 — Minority report (B5).** Measure: missed-dissent rate — manual audit of a transcript sample where round-2 positions disagree, checking whether a minority artifact was emitted. Falsified if dissent-bearing debates regularly emit nothing (this is also D13's deciding evidence, feeding Epic B's synthesis-contract work — measured here, fixed there).
- **M5 — #18 crux grounding (B4).** Measure: (a) inertness check — do round-2 positions ever change on injected evidence? (b) spot-check resolver answers against ground truth. Falsified if evidence injection is inert (positions never move) or the resolver is unreliable (>~1 in 5 spot-checks wrong) — either kills the mechanism's premise at v1 cost, before v2 investment.
- **M6 — #19 framing alarm (B1/B2/B5).** Measure: seeded-corpus calibration — a set of deliberately biased briefs (GUIDE's own bias catalog as the generator) plus matched clean briefs; measure FP/FN per trigger. Falsified if FP rate makes operators ignore alarms (the alarm's only value is being heeded) or FN rate ≈ no-alarm baseline.
- **M7 — #9 gate (B1/B6).** Measure: correlation between gate verdicts and downstream quality (rubric scores, alarm firings) on the same runs. Falsified if FAIL-verdict questions, run anyway, produce debates indistinguishable from PASS questions.
- **M8 — DRAFT-EPI-2 identity integrity (B8).** Measure: `intent_match` is itself the metric — a standing count of mismatches. Falsified (as *unnecessary*, not harmful) if the mismatch count stays zero for a long horizon AND no fallback path can swap the author; the recon's P4 note ("missing check with no owner") plus the witnessed ADR-12 fallback semantics (CLI seat → API retry) argue a swap path will exist, so the check earns its keep at near-zero cost.

### Q3 — EPI-1 archaeology protocol (the executable core)

Purpose: produce the Branch A/B ruling for BACKLOG #1/#2 from historical evidence, segmented by verdict author so the ruling is built on clean data (D5 guard, now protocol).

**1. Corpus assembly.** Primary: local `output/` in the primary checkout (239 files — the complete production record). Secondary: hub `transcripts/` (48 curated copies) + both `archive/` subdirs. Dedupe hub-vs-local by filename slug + timestamp (hub files are routed copies). Unit of analysis: one debate transcript.

**2. Identity recovery — sources ranked (all witnessed live):**
   1. **Transcript synthesis header** `## Synthesis (by X, participant|non-participant)` — authoritative for *actual* author; present since 2026-02-21; covers ~138/239.
   2. **`_metrics.json` sidecar** `synthesis_metrics.synthesizer_model` — adds model-level identity (which claude, which gemini); present since 2026-05-12; tiebreaker and precision upgrade where it exists.
   3. **Brief frontmatter** `synthesizer:` from archived inbox briefs — recovers *intent*, enabling the intent-vs-actual classification retroactively (default-follower vs override-run).
   4. **Run logs** (`output/_inbox-run-*.log`) — batch-run corroboration where headers are ambiguous.
   5. **Git history of `config/settings.yaml`** defaults — era reconstruction (e.g. the claude-authored era, n=54, predates the gemini default) so "default at the time" is decidable per transcript.
**3. Unrecoverable handling.** Research-mode outputs: excluded by protocol (no synthesis section to score — witnessed: all 12 headerless hub files are research). Decision-mode transcripts with no recoverable author: `author-unknown` segment — counted in coverage stats, excluded from the ruling, never guessed. Anomalies (3 × participant, 1 × pre-label-format): quarantined into their own labeled segments with reason codes.
**4. Segments.** `gemini/non-participant` (the incumbent, n≈21) · `openai/non-participant` (the accidental challenger, n≈56 — sample down to era/mode-matched pairs) · `claude/non-participant` (n≈54, earlier era — context segment, not ruling input, since era confounds model) · `claude-sonnet` (n≈7) · anomaly + unknown segments.
**5. Scoring protocol.** RUBRIC's 5 yes/no criteria per synthesis; decision-mode only; **scorer is blind to segment** — the synthesis header is stripped and transcripts are presented in shuffled order (the same blind principle L-CLI's CLI-4 uses; ownership bias B7 applies to scoring too). Mode, rounds, panel size recorded as covariates. Scoring authority: operator per the rubric's origin; an LLM-judge second pass is admissible as a *secondary* opinion, never the ruling authority (OQ-3).
**6. Minimum n.** For a usable Branch A/B ruling: **n ≥ 10 scored decision-mode syntheses per compared segment** (matches the rubric's own 25–50 smoke-test guidance at the low end when two segments are compared). Witnessed inventory clears this: gemini ≈ 21, openai ≈ 56 pre-filter. If any compared segment lands under 10 after filtering, the protocol falls back to **prospective scoring** — fresh paired runs on archived briefs — rather than ruling on thin data.
**7. Report format (what the operator rules on).** A dated `docs/audits/` report: (a) coverage table (recovered / excluded / unknown, per source); (b) criterion × segment pass-rate matrix; (c) failure-mode notes per failed criterion (the rubric's own operating principle: investigate before flipping); (d) covariate caveats; (e) a single explicit recommendation line — "Branch A (swap) / Branch B (keep gemini)" — with the evidence sentence. The operator's ruling on this report **is** the Epic B baseline event that un-gates #18/#19/#9 and L-CLI's v2 resolver ranking (seam §C.3).

**Bonus finding for the ruling design (OQ-1):** because openai authored the majority segment, the contaminated history is a free natural experiment — the protocol can compare incumbent vs challenger on real historical debates *without spending a token*. Recommendation: run #1 as this comparative scoring, not as gemini-only scoring.

### Q4 — #18 crux grounding (DRAFT-GATED — shape only)

Sharpening the audit's ADR-13 draft at the functional level; nothing below reopens its ratified shape (bounded step between rounds, ≤3 cruxes, pick/judge only, pluggable resolver, fail-open).

- **When does a disagreement become a crux?** Three conjunctive conditions: (a) **empirical** — checkable by evidence lookup, not a values/priorities difference; (b) **load-bearing** — at least two panelists on opposing sides of the verdict cite it as a reason for their position; (c) **surviving round 1** — still contested after critique. Panelists supply candidates via the "checkable claims" field (ADR-13 §2); conditions (b)/(c) are evaluated mechanically.
- **Who/what adjudicates cruxhood?** v1: mechanical intersection — a claim flagged from both sides of a disagreement, capped at 3 by flag count. No LLM judge in the loop (keeps the step bounded and deterministic-ish; an adjudicator model would itself be a new bias surface). If mechanical intersection proves too coarse, that evidence — not anticipation — buys an adjudicator later.
- **What "grounded" means:** the resolver returns evidence with **source attribution** and a confidence marker; an answer from model memory without sources does not count as grounded and is recorded as `unresolved(no-source)`. v1 resolver = existing research pool, single-shot, cached via `make_cache_key` (in `research/merger.py`); v2 = read-only CLI agent per ADR-13, ranking per the recon (codex > claude > grok, by witnessed sandbox posture).
- **Transcript representation:** a first-class `## Crux check` block between the round sections: claim text (quoted from the anonymized label that raised it), flag provenance ("raised by Proposals B, D"), resolution status (`resolved` / `unresolved` / `timeout` / `no-source`), evidence summary + sources, marked **tool-derived, attributed to no panelist**. Metrics record crux count + statuses + resolver cost.
- **Anonymization interaction:** crux flags travel under ADR-03 labels end-to-end; the evidence block is injected *identically* into every round-2 prompt (no panelist gets private evidence); the label→provider mapping never enters the resolver's input, so grounding cannot deanonymize. The `_anonymize_responses()` contract is untouched.

### Q5 — #19 framing alarm (DRAFT-GATED — shape only)

- **Trigger conditions (v1 set, all computable from run artifacts):**
  - **T1 — echoed unanimity:** round-1 positions are unanimous AND materially echo the brief's own framing language (the panel is agreeing with the asker, not each other).
  - **T2 — anchor echo:** all panelists converge on the option the brief described at materially greater length or charity (B2's runtime symptom; the GUIDE's equal-length rule, measured).
  - **T3 — dead escape hatch:** a pick debate offers an escape option ("a different approach") and no panelist engages it across both rounds while consensus forms fast — pattern-matches choice-set capture.
  - Explicitly out of v1: semantic counterfactuals (rerun with reframed question) — right idea, wrong cost tier; available to the operator as the *response* to an alarm, not the detector.
- **Operator experience of an alarm:** the run completes and the verdict is delivered — an alarm is **advisory-loud, never blocking**: (a) an ALARM banner on CLI output naming the trigger (mirroring the ADR-08 degradation-alarm pattern; ASCII-only per repo rules); (b) a "treat as advisory — consensus may be a framing artifact" annotation in the transcript header zone; (c) a `framing` entry in metrics (trigger IDs + measures); (d) the banner suggests the specific GUIDE fix and the natural response (reframe and rerun). No new exit code: exit-code semantics are ratified (ADR-08, §B) and an alarmed verdict is still a delivered verdict — Lane-A callers read the metrics field, not a new code.
- **False-positive posture: annoying-but-safe, by design.** Asymmetry: an FP costs seconds of operator attention on an advisory banner; an FN lets a framing artifact become a binding ADR. But heed-ability is the real budget (M6): triggers ship threshold-tunable, calibrated on the seeded corpus *before* default-on, and any trigger whose measured FP rate would train operators to ignore banners stays off-by-default until tuned. Deciding evidence for default-on: OQ-2.

### Q6 — #9 question gate (DRAFT-GATED — shape only, placement-agnostic per F9)

- **What the gate checks** — one checklist, derived clause-by-clause from the GUIDE so gate and doc cannot drift apart (each check carries its GUIDE section key):
  - *Structural (mechanical, cheap):* one-sentence headline; required sections present **for the mode as recognized** (decision format vs research format — the recognition test is the first check); option counts 2–5; question count 3–7; research facet cap 3; source-rules present in research briefs; size bounds; constraints phrased as eliminators (heuristic: does each constraint kill ≥1 option class).
  - *Framing (judgment, WARN-class):* leading headline (names a candidate solution, not a problem); asker-leakage phrases ("I think", "obviously", "ideally"); loaded adjectives where observable facts belong; materially unequal option depth; missing escape option on consequential picks; diagnosis-stated-as-background in Context.
- **Placement design (so either F9 answer fits the ratified CONTRACT):** the gate is specified as a **pure function**: brief in → verdict out, `{PASS | WARN | FAIL, findings[]}`, each finding keyed to a GUIDE section and carrying the fix hint. Machine-readable (Lane-A callers consume it) and human-readable (authors act on it). Because it is placement-free by construction: **caller-side** (`/council-question` skill) runs it as an interactive authoring aid; **council-side** runs the identical checklist at parse time in *both* lanes (defense-in-depth — inbox briefs and `--file` briefs get the same screen). Recommendation: **both**, with council-side WARN-only at first — structural FAILs may block eventually, framing findings never mechanically block (the asker outranks the gate on judgment calls; the alarm (#19) is the runtime backstop).
- **F9's deciding evidence, sharpened:** where bad questions actually originate — the Q3 archaeology pass piggybacks a framing/mis-mode audit of archived briefs (M1) at near-zero marginal cost. If failures cluster in operator-authored inbox briefs, council-side placement matters most; if in agent-authored Lane-A briefs, the caller-side skill matters most.

### Q7 — Verdict-author identity: the `synthesis` metrics namespace (seam §C.1, this lane's side)

Owned here per the seam contract; L-CLI's `seats[]` namespace untouched. Functional field design for the `synthesis` block in `*_metrics.json`:

| Field | Meaning |
|---|---|
| `intended_author` | The synthesizer the run *should* have used, resolved through the ratified precedence chain (CLI flag > frontmatter > config default) |
| `intent_source` | Which precedence link set it: `flag` / `frontmatter` / `default` — makes B8 (habit-driven drift) visible in aggregate, not just per-run |
| `actual_author` | The provider that actually produced the synthesis |
| `actual_model` | Model ID actually served (extends the existing `synthesizer_model`; for a future CLI-backed world this is the identity-channel readout — though ADR-12 already forbids CLI synthesizer seats) |
| `participant` | Whether the actual author also debated (the witnessed n=3 anomaly, made machine-countable) |
| `intent_match` | `actual_author == intended_author` — P4's "missing check with no owner," now owned |

**Check semantics:** computed at synthesis time, recorded always. On mismatch: loud CLI warning + a transcript-header annotation + `intent_match: false` — **record-and-alarm, never block** (by the time it is detectable the verdict exists; blocking would discard work; the alarm makes silent drift impossible, which is the actual failure mode). Mismatch is expected to be rare — its plausible sources are fallback paths (e.g. ADR-12's CLI→API same-seat retry semantics, or a future synthesizer-failure fallback), which is exactly why the check must exist *before* those paths do. **Transcript visibility (F10):** the existing "Synthesis (by X)" line is retained and normatively required — metadata-only was a live option only while the deciding evidence was out; Delta 2 settles it (see §5).

---

## 4. Draft ADRs

> Per master discipline: no real numbers claimed (ADR-13 stays reserved for the crux-resolver draft); `DRAFT-GATED(...)` names the ratifying evidence. "Epic B #2 ruling" = the operator's Branch A/B decision on the Q3 report (the seam §C.3 gate, a.k.a. EPI-1/2).

### DRAFT-EPI-1 — Bias-defense architecture (umbrella registry) · Status: DRAFT

**Decision.** The council's epistemic mechanisms are governed as one registry: bias → vector → defense → measurement → falsifier (§3 Q1/Q2 is the seed content). Two normative rules: (1) **no defense without a falsifier** — any new epistemic mechanism must name the observation that would show it isn't working, before it ships; (2) **no orphan bias** — a bias named in the registry with no defense and no explicit accepted-exposure note is a standing design debt, visible in every epistemic audit. The registry lives with the protocols surface (exact placement is the technical architect's call). **Why umbrella rather than per-mechanism:** the mechanisms individually are already governed (ADR-03; the gated drafts below); what nothing governs is the *coverage* — the map from biases to defenses and the obligation to measure. Per-mechanism ADRs would re-fragment exactly that. Gated mechanisms still get their own ADRs (EPI-3/4/5) *under* this umbrella; the umbrella itself is baseline-independent and ratifiable now.

### DRAFT-EPI-2 — Verdict-author identity integrity · Status: DRAFT (actionable-now class)

**Decision.** (1) The `synthesis` metrics namespace per §3 Q7 (six fields, intent-vs-actual semantics, record-and-alarm on mismatch). (2) The visible transcript author line is normatively required, not incidental. (3) The `participant` anomaly is recorded, counted, and alarmed the same way. **Rationale:** closes P4's ownerless check; converts Wave-0's doc promise (B8 closed going forward) into a mechanism; the archaeology protocol's per-run future equivalent. Baseline-independent (same separability argument as ADR-10/11): it *records* the author, decides nothing about who the author should be — it is what makes the Epic B decision *auditable* after the fact.

### DRAFT-EPI-3 — Bounded crux grounding (#18) · Status: DRAFT-GATED(Epic B #2 ruling)

Shape per §3 Q4, refining the reserved crux-resolver draft (ADR-13): conjunctive cruxhood test (empirical + load-bearing + surviving-critique), mechanical intersection adjudication (no LLM judge in v1), grounded = source-attributed evidence (memory answers = `unresolved(no-source)`), first-class transcript block, anonymization-preserving injection, fail-open. Ratifies when the baseline ruling lands; v2 resolver additionally waits on ADR-12's adapters (L-CLI seam).

### DRAFT-EPI-4 — Framing alarm & false-consensus detection (#19) · Status: DRAFT-GATED(Epic B #2 ruling)

Shape per §3 Q5: trigger family T1–T3, advisory-loud-never-blocking operator experience, no new exit code (metrics `framing` block is the machine channel), annoying-but-safe posture bounded by the M6 heed-ability calibration, seeded-corpus FP/FN measurement before default-on.

### DRAFT-EPI-5 — Question gate (#9) · Status: DRAFT-GATED(Epic B #2 ruling)

Shape per §3 Q6: one GUIDE-derived checklist; gate as a pure function with a machine+human-readable verdict schema; dual placement (caller-side authoring aid + council-side both-lanes validator), council-side WARN-only initially; framing findings never mechanically block. Targets the ratified ADR-11 CONTRACT surface in either placement.

---

## 5. Refined forks

- **F9 (research-mode recognition: prose discipline vs mandatory `mode:` frontmatter)** — sharpened, not decided. The Q3 archaeology corpus gives the deciding evidence a concrete vehicle and cost (~zero marginal: one extra scoring column, M1). Sub-recommendation: if the measured mis-mode rate is low, F9 resolves to status quo *and* the gate's recognition-test check (Q6) becomes the mechanism of record; mandatory frontmatter only if mis-modes cluster where prose discipline demonstrably failed.
- **F10 (verdict-author visibility: metadata-only vs visible line)** — **evidence is in; recommend closing for the visible line.** The recon's deciding evidence was "whether the D5 guard fires during Epic B scoring." Delta 2 shows it fires before scoring even starts: the majority of the historical corpus is openai-authored, and the *only* reason archaeology is cheap is that the visible line existed from day 1. Metadata-only would have made the same history unrecoverable at scale. DRAFT-EPI-2 codifies the line as required.
- **New sub-fork FE-1 (archaeology corpus scope):** hub-curated only (48; cleaner, operator-blessed) vs full local `output/` (239; complete, includes failures and era diversity). **Recommendation: full local with hub dedupe** — the curated set over-represents successful debates (survivorship bias in a bias audit is self-defeating); coverage stats report both populations.

## 6. Questions for the operator

- **OQ-1 — Shape of the #1 scoring run:** gemini-only baseline (letter of BACKLOG #1) vs **comparative gemini-vs-openai segment scoring** (uses the contamination as a free A/B experiment; same rubric, same corpus, materially stronger Branch A/B evidence). *Recommendation: comparative.* Genuine fork because it re-scopes a P1 backlog item's done-when.
- **OQ-2 — #19 default posture at ship:** alarms default-on after seeded-corpus calibration vs opt-in flag until a live-run FP baseline exists. *Recommendation: default-on post-calibration* — an opt-in alarm protects nobody by default, and the posture is already advisory-only.
- **OQ-3 — Scoring authority for EPI-1:** operator scores all (rubric's origin assumption; slow) vs operator scores a calibration subset + LLM-judge extends (faster; introduces a judge-model bias surface into the mission lane's foundational measurement). *Recommendation: operator scores the ruling segments (≈2×10–20 syntheses, blind); LLM judge admissible as a recorded second opinion only.*

## 7. Inputs forward for the technical architect (requirements & constraints — NOT backlog items)

1. **`synthesis` metrics block** — six fields + semantics per Q7; namespace boundary per seam §C.1 (never touch `seats[]`); mismatch alarm is CLI-visible and transcript-annotated; never blocks.
2. **Archaeology tooling needs (EPI-1):** header parser for both label formats (`(by X)` and `(by X, role)`); hub/local dedupe by slug+timestamp; era reconstruction from `settings.yaml` git history; blind-presentation packager (header-stripped, shuffled); coverage accounting. Corpus is read-only evidence — no transcript is ever edited (append-only records discipline; L-GOV owns the wider policy).
3. **Gate verdict schema (when un-gated):** `{PASS|WARN|FAIL, findings[{check_id, guide_section, severity, fix_hint}]}` — identical schema in both placements; machine-readable for Lane-A callers under the ADR-11 CONTRACT.
4. **Crux block contract (when un-gated):** transcript block fields per Q4; metrics carry crux count/status/cost; resolver behind a small protocol with the v1 research-pool implementation using the existing cache (`make_cache_key` lives in `research/merger.py` — repo gotcha); injection path must not receive the ADR-03 label map.
5. **Alarm surfaces (when un-gated):** ASCII-only banner (Windows cp1252 rule); `framing` metrics entry; no exit-code changes (ADR-08 ratified); transcript annotation co-located with the author line.
6. **Constraints inherited:** rounds ≤ 2 untouched (crux check is a step, not a round); `_anonymize_responses()` contract untouched; no hub edits; research providers stay API-only (ADR-12); inbox/interactive feature parity applies to any gate/alarm CLI surface (repo anti-pattern list).
7. **Sequencing facts:** DRAFT-EPI-1/2 are baseline-independent and buildable first; EPI-1 archaeology is executable with zero code changes (manual protocol) but is cheaper with input 2's tooling; EPI-3/4/5 stop at shape until the Epic B #2 ruling; the ruling itself consumes the Q3 report.

---

**Done-contract check:** charter questions Q1–Q7 each answered explicitly ✔ · entry-state verified with two deltas flagged at checkpoint ✔ · five draft ADRs, gates named, no numbers claimed ✔ · F9/F10 refined + one new sub-fork ✔ · three operator questions with recommendations ✔ · zero writes outside this document ✔ · functional level held (no signatures, no diffs) ✔
