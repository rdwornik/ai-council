# Audit — Does AI Council embody what the multi-agent-debate research shows *works*?

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — ADR-seed (never ratified); G7 to #15, G6 to #18, G5 to #55 filed; G1–G4 closed as superseded-brainstorm (2026-07-18); open: #55, #18. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-06-16
**Branch:** `docs/audit-council-vs-research`
**HEAD (audit-time):** `9eeedfc`
**Status:** Draft — ADR seed (pre-decision)
**Mode:** read-only analysis; no council code was modified
**Scope:** `ai-council` only (`src/ai_council/`, `config/settings.yaml`). External research cited by source.

---

## Purpose and boundaries

This report validates the **live `ai-council` implementation** against the external multi-agent-debate
(MAD) literature: it inventories what the tool actually does, correlates each research-validated
mechanism against the code, surfaces gaps, and proposes research-grounded improvements.

**This is a pre-decision brainstorm, not a ratified decision.** It deliberately keeps ≥2 live options on
every contested point and does **not** converge on a single recommended design. It is an *ADR seed* — the
input to a future numbered ADR in `docs/decisions/`, not the ADR itself. The operator decides whether,
and in which direction, to act.

Every "implemented / partial / absent" claim cites a file and symbol read during this audit. Every
research claim cites its source.

---

## Mode split — verifiable vs subjective (read this before any skeptic critique)

The MAD literature splits cleanly along a fault line that the council straddles, and **the audit scopes
every finding to the regime where it holds:**

| Regime | Definition | The council's success metric | What the council does here |
|---|---|---|---|
| **Verifiable-answer** | A sub-claim has ground truth — math, a fact, a checkable assertion. | **Accuracy** against ground truth. | Mostly in `research` mode and any empirical crux inside a `pick`/`judge` debate. |
| **Subjective / no-ground-truth** | Architecture decisions, trade-off picks. **No ground truth exists.** | **Crux & dissent surfacing** — naming the real disagreement and its load-bearing assumption. | The `pick`/`judge` debate path — the tool's reason for existing. |

The widely-cited skeptic results — **"Stop Overvaluing MAD"** (Zhang et al., `arXiv:2502.08788`),
**Huang et al.** (`arXiv:2310.01798`), and the **entropy-collapse** finding (`arXiv:2406.06461`) — all
measure **accuracy on ground-truth benchmarks** (math, MMLU). Their verdict ("MAD often loses to
Self-Consistency at matched compute") is imported as **binding only for the verifiable regime.**

For architecture decisions there is no ground truth to be accurate *about*; the council's value is
surfacing the crux and the minority report, which Self-Consistency cannot produce by construction. So the
audit does **not** transfer the accuracy-benchmark verdict to the subjective regime. Where a finding
applies to only one regime, it says so explicitly.

---

## Current state — what the council does today

Capabilities map. Every row cites a file:symbol read this audit.

| Capability | State | Evidence (file:symbol) | How it actually works |
|---|---|---|---|
| **Heterogeneous panel** | ✅ | `config/settings.yaml:34-91` (`models:`), `:10-11` (`default_panel`/`full_panel`) | 6 models across 5 lineages (Anthropic, Google, OpenAI, xAI, DeepSeek). Default panel = 3 (claude/gemini/openai); `--full` = 5 (adds deepseek/grok). |
| **Persona differentiation** | ✅ | `config/settings.yaml:93-122` (`personas:`) | Each provider gets a distinct *lens* (claude=Systems, gemini=Security, deepseek=Performance, openai=Product, grok=Contrarian) — diversity by role, not just by weights. |
| **Multi-round debate loop** | ✅ | `debate.py:186-295` (`run_debate`) | Default 2 rounds (`settings.yaml:2`, max 3). Round 1 independent answers; round 2+ critique. |
| **Full prior-round refeed** | ✅ (tension — see below) | `debate.py:227-236`, `_build_round2_prompt` `:146-183`, `settings.yaml:315` (`{previous_responses_anonymized}`) | Round 2+ injects the **whole** prior round verbatim (anonymized), not scores/summaries. |
| **Blind voting / anonymization** | ✅ | `debate.py:19-34` (`_anonymize_responses`) | Shuffles prior answers, relabels A/B/C, strips provider identity (ADR-03). |
| **Steelman-first critique** | ✅ | `config/settings.yaml:319` (`critique` prompt) | Each member must state the strongest version of each proposal *before* assessing it, plus surface "hidden assumptions" (`:327`). |
| **Non-participating synthesizer** | ✅ | `runner.py:59-75` (`pick_synthesizer`), `:44-56` (`exclude_synthesizer_from_panel`) | Synthesizer preferred from outside the panel; default gemini (ADR-01). Falls back to participant only if no outsider exists. |
| **Quality-weighted synthesis** | ✅ | `synthesis.py:14-22` (`_format_full_transcript`), `settings.yaml:339-384` (`synthesis` prompt) | Judge weighs **argument quality, not vote count**; "minority position backed by strong evidence should outweigh a majority" (`:344-346`). |
| **Groupthink/false-consensus flag** | ⚠️ Partial | `config/settings.yaml:356-358` (synthesis prompt) | Synthesizer is *prompted* to note when consensus was groupthink vs shared evidence. Delta-to-Full: prompt-only, post-hoc; no mechanical detector, no out-of-family skeptic spawned. |
| **Dissent / minority handling** | ⚠️ Partial | `synthesis.py:14-22`; `settings.yaml:360-363` ("Unresolved Disagreements" + crux) | Full transcript reaches synthesizer; "Unresolved Disagreements" names the crux. Delta-to-Full: dissent is preserved *inside* the synthesis narrative, not emitted as a first-class minority-report artifact that survives averaging. |
| **Confidence / calibration** | ⚠️ Partial | `config/settings.yaml:234-235` (judge `round1_structure`), `:246-247` (judge synthesis) | Confidence (high/med/low + "what would change it") exists in **judge mode only**. Delta-to-Full: no cross-session calibration tracking, no accuracy scoring, no reputation weighting. |
| **Cost / token tracking** | ✅ | `metrics.py:7-67` (`compute_call_cost`, `build_debate_metrics`), `models.py` (`ProviderCallMetrics`) | Per-call USD + token totals across rounds + synthesis. |
| **Context-budget enforcement** | ❌ | `settings.yaml:151` (`token_budget: 1500`) | `token_budget` is declared per mode but **informational only** — no code reads it to gate or truncate. |
| **Debate-gating ("earn the debate")** | ❌ | `debate.py:269-280` (warn-only quality gate), `policy.py:23-33` (`should_abort`) | No trigger decides *whether* to debate. Only a low-participation warning and an all-fail abort. |
| **Influence/trust-gated turns** | ❌ | `debate.py:240-244` (all providers called every round) | Every panelist speaks every round; equal weight; no speak-by-exception. |
| **Shared retrieval pool** | ⚠️ Partial | `research/runner.py:24-65` (`build_research_providers`), `research/merger.py` (`make_cache_key`) | A shared research pool + file cache exists — but **only in `research` mode**. Delta-to-Full: the `pick`/`judge` debate panel has **no** retrieval; debaters reason from parametric memory only. |
| **Institutional / cross-session memory** | ❌ | (no symbol — debate path is request→panel→synthesis) | No lookup of past debates/ADRs at debate time; each run is independent. |
| **Single-model baseline comparison** | ❌ | (no symbol in `debate.py` / `orchestrator.py`) | The council never checks its synthesis against one strong model or a self-consistency run. |
| **Output routing** | ✅ | `routing.py:21-70` (`TargetResolver`) | Per-invocation transcript mirroring to named target projects (ADR-43). |
| **Tests (anonymization, rounds, synthesis)** | ✅ | `tests/test_debate.py`, `tests/test_synthesis.py` | Cover anonymization stripping, round flow, retry, quality-gate warning, synthesis metrics. No test asserts diversity preservation or baseline-beat. |

**Reading of the map:** the council strongly implements the *diversity + blind + quality-weighted-judge*
half of the research (heterogeneity, anonymization, steelman, non-participating synthesizer, minority
protection in synthesis). It largely lacks the *measurement + control* half (baseline comparison,
calibration, debate-gating, influence-gating, retrieval-for-debate, context-budget enforcement).

---

## Correlation — research mechanism × implemented?

Each research-validated mechanism (and each failure-mode guard) scored against the code. **Every
`Partial` carries an explicit delta-to-`Full` with file:symbol evidence.** Regime column marks where the
mechanism's evidence applies (V = verifiable, S = subjective, both = applies in either).

### What research says *works*

| Mechanism (source) | Regime | State | Evidence + delta-to-Full |
|---|---|---|---|
| **Model heterogeneity / lineage diversity** (Du et al. `arXiv:2305.14325`) | both | ✅ Implemented | 5 lineages, 5 distinct personas — `settings.yaml:34-122`. Correlated-blind-spot risk is structurally mitigated. |
| **Anonymizing response sources** (Choi et al. 2025) | both | ✅ Implemented | `debate.py:19-34` strips identity, shuffles, relabels A/B/C; tests assert no provider/model string leaks (`tests/test_debate.py`). |
| **Score-based aggregation over the trajectory** (Free-MAD, Cui et al. `arXiv:2509.11035`) | V | ❌ Absent | No per-turn scoring. Aggregation is a single qualitative judge pass — `synthesis.py:_format_full_transcript` + `settings.yaml:339-384`. There is no numeric trajectory score, so no token savings / dropout-robustness Free-MAD reports. |
| **Influence-/trust-gated turns** (Sun et al. 2025) | both | ❌ Absent | All providers speak every round at equal weight — `debate.py:240-244`. No ~70% input-length cut available. |
| **Shared retrieval pool** (cognitive-islands escape) | both | ⚠️ Partial | Pool + cache exist in `research/runner.py:24-65` / `merger.py make_cache_key`. **Delta:** debate panel (`pick`/`judge`) has **zero** retrieval — debaters cannot ground a crux, only assert. |
| **Confidence calibration as first-class signal** (ConfMAD, Lin et al. 2025) | both | ⚠️ Partial | Confidence field in **judge mode only** — `settings.yaml:234-235,246-247`. **Delta:** no `pick`-mode confidence, no cross-session calibration curve, no accuracy back-test. |
| **Debate-gating — earn the debate** (iMAD `arXiv:2511.11306`) | V | ❌ Absent | No single-agent-first check; the council always runs the full panel — `debate.py:219-244`. iMAD's warning (debate can *override a correct single answer*) is unguarded in the verifiable regime. |
| **Quality-weighted (not majority) adjudication** | S | ✅ Implemented | `settings.yaml:344-346` — minority-with-evidence outweighs majority-by-assertion. This is the council's strongest alignment with subjective-regime best practice. |
| **Steelman before critique** (debate-quality hygiene) | both | ✅ Implemented | `settings.yaml:319` forces strongest-version restatement; `:327` forces hidden-assumption surfacing. |

### Failure-mode guards

| Failure mode (source) | Regime | Guarded? | Evidence + delta-to-Full |
|---|---|---|---|
| **Entropy collapse / tunneling** via dependent re-sampling (`arXiv:2406.06461`) | V | ⚠️ Partial (and contested) | Full prior-round refeed — `debate.py:227-236`. Anonymization + steelman + the Contrarian persona (`settings.yaml:117-122`) push back on imitation; but no temperature/framing re-injection, no diversity metric. **In the subjective regime the same full refeed is a *feature*, not a failure — see Gap analysis.** |
| **Echo chambers** (shared lineage → confirmation) | both | ✅ Guarded | Cross-lineage panel + Contrarian persona — `settings.yaml:34-91,117-122`. |
| **Persuasion ≠ truth** (`arXiv:2510.13912`) | V | ⚠️ Partial | Anonymization removes identity-persuasion; quality-weighted judge resists style. **Delta:** no *hard adjudicator* — the judge is itself an LLM (`synthesis.py:95-98`), so a confident-but-wrong argument on an empirical crux is not tool-checked. |
| **Does it beat the baseline at all?** (`arXiv:2502.08788`, `arXiv:2310.01798`) | V | ❌ Absent | No baseline comparison anywhere — see Current State. Cannot answer the skeptics on its own data in the verifiable regime. |
| **Security / Sybil conformity** (compromised agents propagate falsehood) | both | ⚠️ Partial | Heterogeneity + no trust-weighting means no single agent dominates; but no anomaly detection and no influence cap — `debate.py` treats every response equally. |

**Validation verdict (held in tension, not converged):** on the **subjective** axis the council is well
aligned with the literature (diversity, blind, steelman, quality-weighted judge, minority protection). On
the **verifiable** axis it is missing exactly the mechanisms the skeptic papers say matter — baseline
comparison, score-based aggregation, debate-gating, and a hard (tool-grounded) adjudicator.

---

## Gap analysis

**Lens (restated):** each gap is scoped to the regime where its supporting research holds. A skeptic
finding measured on math/MMLU accuracy is a real gap **in the verifiable regime** and is *not*
automatically a defect in the subjective regime, where the metric is crux/dissent surfacing.

### G1 — Full prior-round refeed: a trade-off, not a defect

- **Now:** round 2+ injects the entire prior round verbatim (`debate.py:227-236`; `_build_round2_prompt`
  `:146-183`).
- **Reading (a) — verifiable regime:** dependent re-sampling reduces answer diversity each round
  (`arXiv:2406.06461`); full-text exposure lets a weaker model converge by imitation onto a confident
  wrong answer. Here the refeed is a **risk**.
- **Reading (b) — subjective regime:** full-transcript cross-examination is the **point** of
  deliberation — a member must steelman (`settings.yaml:319`) and attack the *actual* argument, which
  scores/summaries would strip. Here the refeed is a **feature**, and richer than Free-MAD's numeric
  trajectory.
- **Both readings stay live.** The gap is not "remove the refeed" — it is "the implementation applies one
  refeed policy to both regimes." Partial mitigations already present: anonymization, steelman mandate,
  Contrarian persona. Absent: entropy/diversity instrumentation, per-regime refeed policy.

### G2 — No score-based aggregation (verifiable regime)

- **Now:** one qualitative judge pass over the full transcript (`synthesis.py`, `settings.yaml:339-384`).
- **Research:** Free-MAD (`arXiv:2509.11035`) scores the whole trajectory — ~half the tokens, robust to
  agent dropout, anti-conformity.
- **Risk:** on verifiable cruxes the council pays full-transcript token cost and inherits single-judge
  fragility; a judge outage loses the whole aggregation. *In the subjective regime the qualitative pass
  is defensible — there is no score to compute.*

### G3 — No debate-gating (verifiable regime)

- **Now:** the full panel always runs (`debate.py:219-244`); only a low-participation warning exists
  (`:269-280`).
- **Research:** iMAD (`arXiv:2511.11306`) — single-agent is often already right; triggering debate can
  *override a correct answer*.
- **Risk:** on easy verifiable questions the council spends 3–5× compute and can talk itself out of a
  correct first answer. *Subjective questions arguably always merit the debate — gating matters far less
  there.*

### G4 — No calibration / cross-session memory (both regimes, asymmetric value)

- **Now:** confidence only in judge mode (`settings.yaml:234-235`); no history (`debate.py` is
  stateless run-to-run).
- **Research:** ConfMAD (Lin et al. 2025) treats calibrated confidence as a first-class signal;
  standing-council designs learn their own failure patterns.
- **Risk:** the tool cannot say "last time we converged this fast on this class of question we were
  wrong," and cannot weight an over-confident-and-wrong model down over time.

### G5 — No baseline comparison (verifiable regime — the load-bearing gap)

- **Now:** absent everywhere.
- **Research:** the entire skeptic line (`arXiv:2502.08788`, `arXiv:2310.01798`) is an argument that MAD
  must be measured against Self-Consistency at matched compute.
- **Risk:** in the verifiable regime the council **cannot demonstrate it is net-positive** on its own
  outputs. This is the precondition for trusting (or discounting) every other verifiable-regime
  improvement — see Open Question #1.

### G6 — No retrieval for the debate panel (both regimes)

- **Now:** retrieval exists only in `research` mode (`research/runner.py:24-65`); debaters reason from
  parametric memory.
- **Research:** shared retrieval pools let agents escape "cognitive islands" / private-context bias.
- **Risk:** an empirical crux inside a `pick`/`judge` debate is resolved by the most persuasive assertion
  (`arXiv:2510.13912`), not by evidence — because no debater (and no judge) can look anything up.

### G7 — Minority report is not a first-class output (subjective regime)

- **Now:** dissent is preserved inside the synthesis narrative ("Unresolved Disagreements",
  `settings.yaml:360-363`) but the final artifact is a single decision summary.
- **Research / design intent:** strong unresolved dissent should never be averaged away; it is signal,
  not noise.
- **Risk:** a decision carrying a strong minority objection looks identical, downstream, to a unanimous
  one — the dissent is recoverable only by re-reading the transcript.

---

## Invention bank (fantasy waters)

Research-grounded proposals. Each is tagged **basis · effect · cost/context · conflicts-with**. These are
options to weigh, **not** a recommended set — several conflict by design.

> **Contingency flag:** every invention aimed at the **verifiable regime** (marked ⓥ) is contingent on
> **Open Question #1** — until we know the council beats a self-consistency baseline at matched compute on
> verifiable sub-questions, these are bets, not settled wins. Subjective-regime inventions (marked ⓢ) are
> not gated by #1 because their metric is crux-surfacing, not accuracy.

### Cluster A — Fight tunneling / preserve diversity
- **Entropy injector ⓥⓢ** — each round inject fresh framings / contrarian seeds / temperature bump.
  *Basis:* entropy collapse `arXiv:2406.06461`. *Effect:* slows imitation-convergence. *Cost:* +tokens
  per round. *Conflicts-with:* Speak-by-exception, Context-budget market (injecting more vs. speaking
  less), and the "refeed-as-feature" reading of G1.
- **False-consensus detector ⓢ** — suspiciously-high agreement given shared lineage → spawn an
  out-of-family skeptic. *Basis:* echo-chamber literature; upgrades the existing prompt-only groupthink
  flag (`settings.yaml:356-358`) into a mechanism. *Effect:* catches premature convergence. *Cost:* +1
  agent when triggered. *Conflicts-with:* Debate-gating (one adds agents, the other removes them).
- **Cognitive basis-vector panel selection ⓢ** — pick personas to *span* the reasoning-style space,
  optimizing divergence not agreement. *Basis:* heterogeneity > homogeneity (Du et al.). *Effect:*
  formalizes the persona set already in `settings.yaml:93-122`. *Cost:* selection overhead.
  *Conflicts-with:* Self-MoA provocation (diverse panel vs. one model's samples).

### Cluster B — Earn / shrink the debate (verifiable-leaning)
- **Debate-gating / speak-by-exception ⓥ** — single-agent answers first; debate only triggers if a
  cheap check flags disagreement; silence is the default turn. *Basis:* iMAD `arXiv:2511.11306`, Sun et
  al. 2025. *Effect:* large compute cut on easy questions. *Cost:* a gating classifier. *Conflicts-with:*
  Entropy injector, False-consensus detector (both *add* deliberation).
- **Context-budget market ⓢ** — agents bid context; surviving an argument refunds budget, being refuted
  costs it. *Basis:* influence-gating `Sun et al.`; orchestration-cost reality (AutoGen/CrewAI 5–10× token
  burn). *Effect:* ties airtime to demonstrated value. *Cost:* accounting complexity. *Conflicts-with:*
  "every voice every round" (current `debate.py:240-244`), Entropy injector.

### Cluster C — Ground the crux (resolve, don't out-argue)
- **Recursive crux-zooming + tool adjudication ⓥ** — drill to the simplest checkable disputed claim,
  then **check it with retrieval/code-exec**, not a vote. *Basis:* scalable-oversight debate
  (`arXiv:1805.00899`); persuasion≠truth (`arXiv:2510.13912`). *Effect:* the most context-economical
  debate primitive; closes G6 for empirical cruxes. *Cost:* tool wiring; the research pool already exists
  to borrow. *Conflicts-with:* nothing structurally — strongest standalone candidate.
- **Obfuscation guard ⓥ** — an argument whose validity can't be cheaply checked loses by default.
  *Basis:* obfuscated-argument problem (Barnes & Christiano 2020). *Effect:* blocks expensive-to-refute
  sophistry. *Cost:* needs a checkability heuristic. *Conflicts-with:* subjective-regime decisions, where
  *nothing* is cheaply checkable — would mis-fire if applied there (scope to ⓥ only).
- **Asymmetric-information debate ⓥ** — shard context across agents so each knows something the judge
  doesn't. *Basis:* Khan et al. 2024 (info asymmetry raises judge accuracy). *Effect:* turns the context
  constraint into a feature. *Cost:* context-sharding plumbing. *Conflicts-with:* Shared retrieval pool
  (shard vs. pool the same context).

### Cluster D — Anti-sycophancy / decision integrity (subjective-leaning)
- **"What would change our mind" mandate ⓢ** — no decision without a falsifiable flip-condition per
  side. *Basis:* persuasion≠truth; core anti-sycophancy. *Effect:* forces real cruxes; extends the
  existing "Signals to Revisit" (`settings.yaml:296-297`). *Cost:* minimal. *Conflicts-with:* nothing.
- **Minority report as first-class output ⓢ** — strong unresolved dissent is emitted as its own
  artifact, never averaged away. *Basis:* design intent; closes G7. *Effect:* downstream sees dissent.
  *Cost:* output-schema change. *Conflicts-with:* nothing.
- **Steelman-swap ⓢ** — argue the opponent's case better than its author before defending your own.
  *Basis:* debate-quality hygiene; upgrades the current steelman-first (`settings.yaml:319`). *Effect:*
  deeper engagement. *Cost:* +tokens. *Conflicts-with:* Speak-by-exception, Context-budget market.

### Cluster E — Standing institution / memory
- **Calibrated reputation across sessions ⓥ** — over-confident-and-wrong penalized over time; weight by
  earned calibration, not eloquence. *Basis:* ConfMAD. *Effect:* closes G4. *Cost:* persistent store +
  outcome labeling (needs ground truth → ⓥ). *Conflicts-with:* anonymization (reputation needs identity;
  blind voting hides it) — a **direct** conflict to resolve.
- **Standing council with institutional memory ⓢ** — learns failure patterns from past ADRs/journal
  ("last time we converged this fast, we were wrong"). *Basis:* orchestration reality (eval/observability
  > framework). *Effect:* closes G4 on the subjective side without needing ground truth. *Cost:* retrieval
  over `docs/decisions/` + `JOURNAL.md`. *Conflicts-with:* anonymization (weakly — pattern memory can be
  identity-free).
- **Premortem agent ⓢ** / **Negative-space agent ⓢ** — one narrates the decision's failure before
  commit; the other asks "what question did none of us ask?". *Basis:* blind-spot guards. *Effect:*
  cheap insurance on the existing "Blind Spots" synthesis section (`settings.yaml:370-372`). *Cost:* +1
  agent each. *Conflicts-with:* Debate-gating (adds agents).

### Cluster F — Reframes of "council" itself
- **Council as context compiler ⓢ** — output is *denser* than input; net-negative entropy on working
  memory. *Basis:* context-economy goal. *Effect:* the debate pays for itself in compression.
  *Conflicts-with:* Entropy injector, Steelman-swap (both expand tokens).
- **Branchable debates ⓢ** — git-for-arguments: pause / rewind / fork alternative-assumption worlds.
  *Basis:* exploration of the decision space. *Cost:* state model. *Conflicts-with:* single-pass synthesis.
- **Provocation — Self-MoA ⓥ** — maybe the ideal "council" is sometimes ONE strong model run as an
  enforced internal dialectic (Self-MoA, Li et al. 2025: aggregating one strong model's samples can beat a
  heterogeneous mix unless specialties match task heterogeneity). *Effect:* questions multi-provider as the
  default. *Conflicts-with:* the entire heterogeneous-panel design — kept live deliberately.

---

## Tension map — the design is a choice of a point, not an answer

The "ideal council" is **a point selected on these axes**, not a single answer. Each axis lists ≥2 live
options; none is pre-resolved here.

| Axis | Pole A | Pole B | What pulls each way |
|---|---|---|---|
| **Anti-conformity ↔ Consensus** | Inject entropy, spawn skeptics, reward dissent | Converge fast, reward agreement | Diversity preserves signal (`arXiv:2406.06461`) ↔ decisions must actually land. |
| **Diversity ↔ Cost** | Full 5-lineage panel + extra skeptic agents | Lean 3-model or Self-MoA | Heterogeneity beats homogeneity (Du) ↔ orchestration 5–10× token burn. |
| **More rounds ↔ Entropy collapse** | More critique rounds = deeper engagement | Fewer rounds = preserved diversity | Cross-examination value (G1-b) ↔ imitation-convergence (G1-a). |
| **Full refeed ↔ Score/summary refeed** | Whole-transcript cross-exam (subjective feature) | Numeric trajectory (Free-MAD economy) | Richness for subjective cruxes ↔ token economy + dropout-robustness for verifiable ones. |
| **Multi-provider ↔ Self-MoA** | Heterogeneous panel (current default) | One strong model, internal dialectic | Lineage-diversity guard ↔ Self-MoA can win when specialties don't match task. |
| **Always-debate ↔ Debate-gating** | Council always runs (current) | Earn the debate per iMAD | Subjective questions merit debate ↔ verifiable easy ones don't. |
| **Anonymized ↔ Reputation-weighted** | Blind voting (ADR-03, current) | Calibrated cross-session reputation | Kills identity-sycophancy ↔ can't weight earned calibration without identity. **Direct conflict.** |
| **Model judge ↔ Tool-grounded judge** | LLM synthesizer (current) | Retrieval/code-exec adjudicates empirical cruxes | Works for value trade-offs ↔ only tools settle facts (`arXiv:2510.13912`). |

**The mode split chooses the point.** A reasonable reading: lean **Pole B** (gate, score, tool-ground,
maybe Self-MoA) for the **verifiable** regime, and lean **Pole A** (full refeed, dissent-first, blind,
LLM-judge) for the **subjective** regime — i.e. *the single biggest design move may be making refeed /
gating / judging policy regime-aware rather than picking one global point.* This is stated as **one live
option, not a recommendation.**

---

## Open questions

1. **(Load-bearing, scopes all ⓥ inventions)** On **verifiable** sub-questions, is the council
   net-positive over a **self-consistency baseline at matched compute**? Until measured, every
   verifiable-regime invention (Cluster B/C, reputation, Self-MoA) is a bet, not a settled win. This is
   the prerequisite experiment — closing G5 answers it.
2. How would the council **measure crux/dissent surfacing** in the subjective regime, where accuracy is
   undefined? Without a metric here, "the subjective regime is fine" is an assertion, not a finding.
3. **Anonymization vs. reputation** is a direct conflict (tension-map row 7). Can earned calibration be
   carried without re-introducing identity-driven sycophancy (e.g. reputation applied *after* blind
   rounds, at synthesis only)?
4. Is the right unit of change a **global** redesign or a **regime-aware** policy (refeed / gating /
   judging that branches on verifiable-vs-subjective)? The audit leans toward the latter as a live
   option but does not decide it.
5. Which single invention has the best **effort-to-value** ratio for a first experiment? Recursive
   crux-zooming + tool adjudication (Cluster C) is the only proposal with no structural conflict and
   reuses the existing research pool — a candidate, not a conclusion.

---

## Note on next step

This report is **evidence, not a decision.** It validates that `ai-council` is strongly aligned with the
*subjective-deliberation* half of the MAD research and structurally missing the *verifiable-measurement*
half — and it deliberately keeps every contested design choice open (see the tension map).

Whether to act, and in which direction, is the operator's call. The natural first move suggested by the
evidence (not mandated by it) is **Open Question #1** — a baseline experiment — because it gates the
credibility of every verifiable-regime improvement.

If the operator chooses to act, this audit is the **seed for a future numbered ADR in `docs/decisions/`**
(no number is assigned here — ADR numbering is the operator's, and this brainstorm has not converged on a
decision to record). Candidate ADR framing: *"Regime-aware council policy: when to debate, how to refeed,
how to adjudicate."*

---

## References

- Du et al., *Improving Factuality and Reasoning via Multiagent Debate* — `arXiv:2305.14325`
- Irving, Christiano, Amodei, *AI Safety via Debate* — `arXiv:1805.00899`
- Barnes & Christiano, *Obfuscated Arguments* (2020)
- Khan et al., *Debate with information asymmetry* (2024)
- Wang et al., *Mixture-of-Agents* — `arXiv:2406.04692`
- Li et al., *Self-MoA* (2025)
- Choi et al., *Anonymizing response sources removes identity-sycophancy* (2025)
- Cui et al., *Free-MAD: score-based aggregation* — `arXiv:2509.11035`
- Sun et al., *Influence-/trust-gated turns* (2025)
- Lin et al., *ConfMAD: confidence calibration* (2025)
- *iMAD: debate-gating* — `arXiv:2511.11306`
- *Entropy collapse in multi-agent debate* — `arXiv:2406.06461`
- *Persuasion ≠ truth* — `arXiv:2510.13912`
- Zhang et al., *Stop Overvaluing MAD* — `arXiv:2502.08788`
- Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* — `arXiv:2310.01798`

---

*Pre-decision brainstorm. Status: Draft. No decision is ratified by this document.*
