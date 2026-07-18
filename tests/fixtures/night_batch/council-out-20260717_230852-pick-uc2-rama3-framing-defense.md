# AI Council Debate: # Decision: How should ai-council upgrade framing-bias defense from author-time 

**Date:** 2026-07-17 23:08:52
**Panel:** claude-haiku-4-5-20251001, gpt-5.6-sol, gemini-3.1-pro-preview
**Synthesizer:** openai (non-participant)
**Rounds:** 2
**Duration:** 179.7s
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** C:\Users\1028120\Documents\Dev\ai-council\council_inbox\uc2-rama3-framing-defense.md
**Cost:** ~$0.1156 (31,052 tokens)

## Verdict Summary

**Decision:** Choose (b): a debate-time framing-challenge ROLE**, implemented as a **bounded, parallel, evidence-citing diagnostic** in shadow mode first.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 3/3
**Verdict author:** openai (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

# Decision: How should ai-council upgrade framing-bias defense from author-time to debate-time? (Rama 3 / #19)

## Context
ai-council has strong AUTHOR-time framing defense (a question guide, a Stage-1a gate, and a "fast unanimous agreement = leading question" self-check applied to the brief BEFORE the debate). But once a debate is running, nothing re-challenges the framing. The operator's own documented recurring failure mode is "accepting inherited framing without verifying against state." The leverage is making the defense active and debate-time.

## The decision
Pick ONE posture for debate-time framing defense.

## Options
- (a) A false-consensus ALARM: mechanise the "fast/tight unanimous agreement" heuristic as an embedding-agreement detector that flags suspiciously fast or tight consensus for review, rather than treating it as confidence. Advisory-loud, never blocking.
- (b) A debate-time framing-challenge ROLE: an agent that re-derives the decision from the actual repo/state and surfaces "you asked X; the state suggests the real decision is Y."
- (c) Keep it STATIC — a deliberate decision that the existing author-time Stage-1a gate is sufficient and the runtime cost/complexity is not warranted.

## Constraints (these eliminate weak options)
- Advisory-only, never blocking: no new exit code; a framing finding surfaces as an ASCII-only banner + a metrics entry, co-located with the author line (ADR-08 exit semantics are ratified; Windows cp1252 = ASCII only).
- False-positive posture is annoying-but-safe, but heed-ability is the real budget: any trigger whose FP rate would train the operator to ignore banners must stay off-by-default until calibrated on a seeded corpus.
- Rounds <=2 and the _anonymize_responses() contract are untouched.
- Determinism + inbox I/O preserved; research providers stay API-only.

## Prior decisions in scope
ADR-03, the question-framing-bias audit, DRAFT-EPI-4, and triggers T1 (echoed unanimity) / T2 (anchor echo) / T3 (dead escape hatch). Escape hatch: option (c) is the explicit "do nothing / keep static" choice.

## Round 1: Initial Responses

### Claude (claude-haiku-4-5-20251001)

## Position
Recommend **(a) false-consensus alarm** — mechanize the existing "fast agreement = leading question" heuristic as an embedding-agreement detector. Keep it off-by-default until calibrated on a seeded corpus, then enable advisory-loud with tight thresholds to minimize false positives that erode heedability.

## Alternatives Considered

**Option (b) — debate-time re-derive role:** Rejected because it trades a weak-but-predictable signal (consensus speed) for a stronger-but-opaque one (agent re-derivation). At 2am, an oncall engineer debugging a framing concern needs to understand *why* the system flagged it. "The agent re-derived the decision and disagreed" is a black box that requires trusting the re-derivation logic — and that logic is easier to get subtly wrong (context it didn't see, decision precedent it misses) than a consensus-timing rule.

**Option (c) — keep it static:** Rejected because the operator has a documented recurring failure mode (accepting inherited framing without verification). This is not a hypothetical risk; it's a known gap in the current system. Accepting it as "author-time gate is sufficient" is accepting known operational debt that your own incident patterns have already highlighted. The leverage is literally there.

## Decision Criteria

1. **Addresses documented failure mode** — must reduce the frequency of "inherited framing accepted without verification" incidents.
2. **Heedability under false positives** — must not train the operator to ignore the signal through excessive noise.
3. **Operational clarity at 2am** — must be understandable without deep reasoning about inference or agent behavior.
4. **Implementation realism** — must fit within Determinism + inbox I/O constraints and not couple to research APIs.
5. **Reversibility** — must be safe to disable or tune down if calibration reveals it's unworkable.

## Assumptions

- The existing "fast unanimous agreement" intuition is a *real* signal of potential group-think or inherited framing, not an artifact of measurement. (If consensus speed doesn't correlate with framing problems in practice, this recommendation breaks.)
- Calibration on a seeded corpus is feasible before this goes live. (If you can't run debates with injected framing problems to set thresholds, the FP rate will be unpredictable.)
- Advisory banners, when well-calibrated, are *heeded* by the operator in practice. (This is a bet that the operator responds to warnings; if they don't, no amount of mechanization helps.)

## Recommendation

**Now:**
1. Instrument debate-time consensus to measure embedding-agreement speed and tightness (e.g., "all responses agree within 2 rounds, with <5% variance in top-3 candidates").
2. Seed a test corpus of 10–20 debates with deliberate framing problems (e.g., "should we use Pattern A or B?" when the state actually suggests "neither, use C").
3. Measure consensus metrics on this seeded corpus *and* on normal debates. Set thresholds (e.g., "flag if all rounds reach >95% agreement within 2 consensus rounds") that catch ≥70% of seeded problems while maintaining <20% FP rate on normal debates.
4. Implement the alarm as an advisory banner (ASCII, ADR-08 safe) co-located with the author line. Off-by-default behind an opt-in flag.

**Not now:**
- Do not build a re-derive agent or add repo-state coupling.
- Do not make it blocking or affect exit codes.
- Do not enable it in production until FP rate is validated <20% on representative corpus.

**Defer:**
- If FP rate stays >20% after tuning, demote this to a metrics-only entry (logged but no banner) and re-evaluate whether (b) or (c) is better.
- Debate whether "fast agreement" should be measured as speed (round N) vs. consistency (variance) vs. both; this can iterate once real data exists.

## Rationale

- **Reuses existing intuition:** The "fast unanimous agreement" heuristic already lives in the Stage-1a brief review and the self-check gate. You're not inventing a new failure mode; you're automating a pattern the author already recognizes. This reduces cognitive load on the operator because it feels *calibrated* to their existing mental model.

- **Catches the most common case:** If the operator's documented failure mode is "accepting inherited framing," the most likely cause is *consensus signaling false confidence*. A tight consensus on a poorly-framed question is a strong signal of group-think. Fast embedding agreement is cheaper to detect than re-derivation and more reliable (consensus-timing metrics are deterministic; re-derivation logic has edge cases).

- **Stays lightweight and reversible:** An advisory banner costs almost nothing computationally, fits ADR-08 constraints, and can be disabled with a single flag if it becomes noise. A debate-time role agent is harder to disable once operators start depending on it.

- **Heedability is the constraint, not sensitivity:** A false-consensus alarm with a 20% FP rate is tolerable *if* each false positive is comprehensible (e.g., "everyone agreed fast, but the question was actually well-framed"). That's information the operator can act on or dismiss. A re-derivation that says "the agent thinks you should do X instead" requires the operator to trust or debug the agent, which is a higher cognitive bar and more likely to train them to ignore it.

- **Calibration path is clear:** You have a concrete way to measure success: seed problematic debates, measure whether the alarm catches them, and track FP rate on normal debates. This gives you a concrete go/no-go threshold.

## Trade-offs

- **You do not get deep framing re-validation.** The alarm is a *heuristic* (consensus speed) not a *derivation* (re-checking from state). If a debate has tight consensus on a perfectly-correct framing, you'll get a false positive. This is acceptable *if* the FP rate is low enough.
- **Advisory-only means operator discretion.** The banner can be ignored. You're betting that an operator with a documented failure mode will respond to a well-calibrated signal. If they don't, this doesn't fix the problem.
- **Delayed feedback loop.** The alarm fires during debate, not at brief stage. This means the framing error surfaces *after* debate has started, requiring possible re-work. (But this is strictly better than never re-checking, which is the current state.)

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| False-positive rate >20% on production debates | Medium (consensus speed is noisy) | High (operator learns to ignore banner) | Calibrate on seeded corpus before shipping; start at conservative thresholds (>95% agreement); add calibration telemetry to refine over 2-3 rounds |
| Seeded corpus doesn't represent real framing problems | Medium (framing bias is subtle) | High (threshold tuning is invalid) | Require at least 2-3 recent framing incidents from real debates to be included in seed set; validate on a holdout of recent debates |
| Operator doesn't heed the banner in practice | Low (it's designed to be advisory and visible) | High (doesn't address the failure mode) | Log banner ignores as a metric; if >50% of flagged debates are ignored, demote to logs-only and re-evaluate (b) |
| Determinism broken by embedding variance | Very low (embeddings are deterministic given a model) | High (makes debate results non-reproducible) | Measure and log embedding-space distance metrics separately; ensure thresholds are based on discrete signals (round count, response agreement %) not embeddings |

## Signals to Revisit

- **Calibration validates:** If seeded-corpus testing shows FP rate ≤15% and catches ≥70% of seeded framing problems, proceed to (a) off-by-default in production.
- **Calibration fails:** If FP rate is >25% even with tight thresholds, demote to logs-only metrics and re-evaluate whether (b) (re-derive role, accepting higher complexity) or accepting (c) is the right trade-off.
- **Real incidents drop:** After 3 months, measure whether framing-related incident rate drops after (a) is enabled. If it doesn't, the signal isn't working and (c) becomes the honest default.
- **Operator behavior shifts:** If heedability metrics show operators are ignoring the banner in >50% of cases, even after FP calibration, it's a signal that advisory-only isn't enough and a blocking gate (option b re-designed as a hard check) might be necessary.

*Latency: 56.67s | Tokens: 3059*

### Deepseek (gpt-5.6-sol)

## Position

Choose **(b), a debate-time framing-challenge role**. Run it as a parallel, state-grounded diagnostic rather than another debate round, initially in metrics-only shadow mode and later as an advisory ASCII banner once precision is calibrated.

## Alternatives Considered

- **(a) False-consensus alarm:** Rejected as the primary defense because agreement speed and embedding similarity detect a symptom, not whether the framing contradicts repository state. It also introduces threshold calibration and alert-fatigue risk before demonstrating useful precision.
- **(c) Keep it static:** Rejected because it leaves the documented failure mode intact. Author-time checks cannot catch framing invalidated by newly inspected state or assumptions echoed by every debate participant.

## Decision Criteria

- Must identify state-inconsistent framing, not merely correlated response patterns.
- Must remain advisory-only with no exit-code or round-count changes.
- Must preserve `_anonymize_responses()`, deterministic orchestration, and inbox I/O.
- Must keep false-positive banners below the operator’s heed-ability threshold.
- Must add tolerable token cost and tail latency under a maximum of two rounds.

## Assumptions

- Debate participants already receive, or can deterministically inspect, sufficient repository/state evidence to re-derive the decision.
- One additional parallel model invocation is operationally acceptable if it adds roughly 15-30% to debate token usage.
- A seeded corpus can provide at least 50-100 labeled cases, including T1/T2/T3 positives and legitimate unanimous decisions.
- The framing role can remain outside the anonymized peer-response set and produce a separate diagnostic artifact.

## Recommendation

**What to do now:** Add one dedicated framing-challenge role to the first-round parallel fanout. Give it a narrow structured task: restate the requested decision, derive the decision implied by repository/state evidence, identify unsupported premises, test the explicit escape hatch, and emit either `NO_FINDING` or a finding containing the proposed real decision and cited evidence.

Persist its result through normal inbox I/O and record metrics such as execution latency, finding type, confidence, and whether the operator accepted or dismissed it. During calibration, keep banners off by default. After the seeded corpus meets a defined threshold—recommended precision at least 80% and a benign-case banner rate below 5-10%—surface findings as an ASCII-only banner co-located with the author line. Do not alter exit status.

**What not to do now:** Do not add an embedding-agreement service, change debate rounds, inject the role’s output into anonymized peer responses, or let it rewrite the question automatically.

**What to defer until later:** After collecting labeled production outcomes, evaluate T1/T2/T3 as cheap secondary signals. Enable an agreement alarm only if it provides measurable recall beyond the framing role without exceeding the banner budget.

## Rationale

- The first failure point is semantic, not statistical: all agents can agree because they inherited the same false premise. A state-grounded role can expose that directly; embedding similarity cannot distinguish coordinated correctness from coordinated framing error.
- Running the role in parallel avoids an extra sequential round. For a typical 3-5-call fanout, adding one comparable call should increase token cost by roughly 15-30%, while increasing wall-clock p95 by approximately 5-15% because completion time is governed by the slowest parallel call.
- A structured `NO_FINDING`/finding contract limits downstream complexity and keeps the diagnostic out of `_anonymize_responses()`, avoiding accidental influence on the debate it is auditing.
- Shadow mode protects heed-ability. With 100 labeled cases, an observed 80% precision still has substantial uncertainty, but it is enough to reject obviously noisy prompts before exposing banners.
- The role directly operationalizes the known failure mode: “you asked X; repository state supports Y.” That produces actionable evidence, unlike an alarm that only says consensus looks suspicious.

## Trade-offs

- Every debate pays for an additional model call, even when framing is sound: approximately 15-30% more tokens and potentially 5-15% higher p95 latency.
- A specialized role may duplicate repository inspection already performed by other agents.
- Keeping findings separate preserves debate integrity but prevents peers from correcting course within the same run; the operator must initiate a reframed run.
- Shadow calibration delays visible protection until enough labeled cases exist.

## Risks

- **Role echoes the original framing instead of challenging it — medium likelihood, high impact.** Require explicit state citations, an independently derived decision statement, and evaluation against seeded adversarial cases.
- **Alert fatigue after activation — medium likelihood, high impact.** Gate activation on precision and banner-rate thresholds; retain per-trigger disable controls and log dismissals.
- **Added call becomes the tail-latency bottleneck — medium likelihood, medium impact.** Cap its evidence budget and output size, run it concurrently, and record separate p50/p95/p99 latency.
- **Diagnostic leaks into debate responses or breaks anonymization — low likelihood, high impact.** Store it as control-plane metadata and add contract tests proving it never enters `_anonymize_responses()`.
- **Repository evidence is incomplete or stale — medium likelihood, high impact.** Require the banner to name the inspected state and downgrade unsupported conclusions to metrics-only observations.

## Signals to Revisit

- Framing-role precision remains below 80% or benign-case banner rate exceeds 10% after at least 100 labeled cases.
- The role adds more than 20% to p95 wall-clock latency or more than 30% to median token cost.
- T1/T2/T3 signals demonstrate at least a 10-percentage-point recall gain on missed framing cases at acceptable precision.
- Operators dismiss more than 25% of banners without reframing, indicating declining heed-ability.
- State-grounded challenges rarely change decisions—below roughly 2% of debates over a representative sample—making the runtime cost unjustified.

*Latency: 51.86s | Tokens: 7273*

### Gemini (gemini-3.1-pro-preview)

## Position
Implement Option (a), the false-consensus ALARM, using a local embedding model to detect suspiciously tight agent agreement. This provides active, debate-time anomaly detection without expanding the API attack surface or granting a new external agent unbounded access to your repository state.

## Alternatives Considered
- **Option (b) (Framing-challenge ROLE):** Rejected because granting a new LLM agent broad, unstructured read access to "actual repo/state" to re-derive decisions massively expands the data-exfiltration attack surface and external API dependency. If this agent is compromised or hallucinating, its blast radius includes injecting malicious context into the core debate.
- **Option (c) (Keep STATIC):** Rejected because the operator's documented failure mode is "accepting inherited framing." Relying solely on author-time defenses violates defense-in-depth; if the Stage-1a gate fails, the system currently has zero runtime detection for inherited bias. 

## Decision Criteria
1. **Blast Radius:** The potential damage (state corruption, data leakage, malicious injection) if the framing-defense mechanism is compromised or behaves unexpectedly.
2. **Attack Surface & API Boundaries:** The volume and sensitivity of raw state data that must cross external trust boundaries to research providers.
3. **Heed-ability (Alarm Fatigue):** The ability to tune the mechanism to avoid high false-positive (FP) rates that train operators to ignore warnings.
4. **Constraint Adherence:** Must remain advisory-only (ASCII banner), preserve the <=2 round limit, and not alter the `_anonymize_responses()` contract.

## Assumptions
- "Actual repo/state" contains sensitive intellectual property or configuration data that should be strictly scoped on a least-privilege basis.
- The system can run a lightweight embedding model locally, or the existing trusted AI API boundary can securely handle embedding requests without logging them for training.
- The operator possesses a seeded corpus of both genuine consensus and false/forced consensus to calibrate the alarm.

## Recommendation
- **What to do now:** Implement Option (a) as a mechanistic, vector-distance check on the agents' Stage-1 outputs. Use a lightweight, local embedding model (e.g., `all-MiniLM-L6-v2`) to compute semantic similarity between responses. Output findings purely as an ASCII banner co-located with the author line, and log a metrics entry.
- **What not to do now:** Do not implement Option (b). Do not introduce any new agent roles that require parsing raw repository state. Do not block the execution pipeline or alter exit codes, regardless of how tight the consensus appears.
- **What to defer until later:** Do not enable the banner in production UI immediately. Defer default-enablement until the mechanism has run in "shadow mode" against the seeded corpus to calibrate the distance threshold and prove a low false-positive rate.

## Rationale
- **Minimizes API Blast Radius:** Option (a) operates purely on the *outputs* the agents have already generated. It requires zero additional context from the repository, adhering to the principle of least privilege.
- **Prevents Supply Chain Data Exfiltration:** By using a local embedding model for the consensus check, you add a runtime defense layer without sending unstructured repository state to an external, API-only research provider.
- **Acts as an IDS for Cognitive Bias:** In security architecture, when preventative controls (Stage-1a gate) fail, we rely on Intrusion Detection Systems (IDS). Option (a) mechanizes anomaly detection (T1/T2 triggers) for suspiciously tight semantic agreement, treating it as an indicator of compromise (IoC) for framing bias.
- **Preserves Determinism and I/O:** Calculating embedding similarity is a deterministic mathematical operation that does not introduce the stochastic variability, prompt-injection risks, or round-limit complications of adding a new LLM agent.

## Trade-offs
- **Semantic Depth:** You trade the explicit, human-readable counter-argument that Option (b) would provide for operational security. The alarm will tell you *that* the agents agree too tightly, but the operator must manually investigate *why* the framing was flawed.
- **Compute Overhead:** Introduces a minor computational cost to generate embeddings and calculate vector distances at debate-time, though this is negligible if localized.

## Risks
- **Risk:** High false-positive rates lead to alarm fatigue, causing the operator to blindly ignore the ASCII banner. 
  - **Mitigation:** Strictly enforce the "off-by-default" constraint. Target a false-positive rate of <5% during the seeded corpus calibration phase before enabling the banner in the operator's view.
- **Risk:** The embedding model exhibits its own biases, failing to recognize tight consensus if the agents use diverse vocabulary to say the exact same thing.
  - **Mitigation:** Calibrate the vector-distance threshold specifically against the T1 (echoed unanimity) and T2 (anchor echo) failure modes documented in ADR-03, rather than generic semantic similarity benchmarks.

## Signals to Revisit
- Operator telemetry indicates the ASCII banner is ignored (no subsequent state verification) in >90% of cases where it triggers, meaning the defense has failed behaviorally despite technical success.
- Future architectural shifts bring highly trusted, strictly localized Small Language Models (SLMs) into the environment, lowering the exfiltration risk of Option (b) enough to justify granting an agent read-access to repo state.

*Latency: 29.59s | Tokens: 3537*

## Round 2: Critique

### Claude (claude-haiku-4-5-20251001)

# Architectural Assessment: Framing-Bias Defense

I'll evaluate each proposal against the standard: what does an on-call engineer actually face at 2am, and is this maintainable in three years when the builder isn't here?

---

## Proposal A (Framing-Challenge Role)

**Steelman:** This directly operationalizes the failure mode: ask the agent "what does the state *actually* suggest?" and surface disagreement. It bypasses the symptom (tight consensus) and attacks the semantic root. Parallel execution avoids sequentiality. The structured `NO_FINDING`/finding contract keeps the diagnostic isolated. Seeded corpus + shadow-mode calibration is methodical and empirically grounded.

**Assessment:** **Partially disagree with Position.**

**Strongest point:** "The first failure point is semantic, not statistical" — this is exactly right. All agents can agree because they inherited the same false premise. Embedding similarity cannot catch that.

**Weakest assumption:** 
> "Debate participants already receive, or can deterministically inspect, sufficient repository/state evidence to re-derive the decision."

This does too much work. The re-derivation agent has the same input constraints and perceptual biases as the original debate participants. It's not magic reasoning from state; it's another LLM reading the same repository signals. If the state is ambiguous, contradictory, or genuinely underdetermined, the role doesn't resolve it—it just produces a different hallucination. **What breaks:** You've moved the framing error from "5 agents agreed on X" to "the re-derivation agent said Y, so X was wrong"—but if Y is also wrong, you've just layered confusion.

**Hidden assumptions:**
1. The role's re-derivation is more trustworthy than the original participants' reasoning, despite having identical inputs.
2. A structured contract (`NO_FINDING`/finding) prevents hallucination-escape (it doesn't).
3. The operator can debug "the framing role disagreed" at 2am without spending 30 minutes reasoning about what the agent actually saw.

**Overlooked risks:**
- **Role echoes original consensus by different path:** If all participants misread the state in the same way, the role will too. You've added latency without catching coordinated reasoning errors.
- **Black-box at 2am:** An on-call engineer sees "framing-role found: real decision is Y." They now have two contradictory re-derivations and no way to resolve it without re-running the debate. That's higher cognitive load, not lower.
- **State-grounded reasoning is brittle:** Many framing problems don't stem from wrong state; they stem from *ambiguous* state that different stakeholders interpret differently. A re-derivation agent can't magically disambiguate that.

---

## Proposal B (False-Consensus Alarm – Security Lens)

**Steelman:** This is defense in depth. The Stage-1a gate can fail. When it does, you want a runtime IDS with a narrow trust boundary: operate only on debate outputs, use local embeddings, minimize API surface, and avoid shipping repository state to external providers. It's a weak signal but a *safe* one, and safety is non-negotiable for systems running at production scale.

**Assessment:** **Agree with the principle, partially disagree with Position (a) being the right choice for this context.**

**Strongest point:** The blast-radius analysis is correct. A new agent role with "broad, unstructured read access to actual repo/state" is a supply-chain and exfiltration risk that shouldn't be dismissed. Local embedding-distance calculation is deterministic and bounded.

**Weakest assumption:**
> "The embedding model exhibits its own biases, failing to recognize tight consensus if the agents use diverse vocabulary to say the exact same thing."

This is acknowledged but *not solved*. You're accepting high false-positive risk if the model can't distinguish coordinated framing error (diverse vocabulary, same false premise) from legitimate consensus (diverse vocabulary, same correct conclusion). **What breaks:** You calibrate the threshold to catch tight semantic consensus, you miss cases where agents paraphrase the inherited premise, and you get false negatives on your actual failure mode.

**Hidden assumptions:**
1. Embedding distance is a *causal signal* for framing error, not just a *correlation* (unvalidated).
2. The seeded corpus used for calibration represents the *actual distribution* of framing problems (in reality, you don't yet know what that distribution is).
3. The operator will act on an advisory banner telling them "consensus looks tight, verify the framing" (behavioral assumption; not guaranteed).

**Overlooked risks:**
- **False negatives on paraphrase:** Agents use different vocabulary to express the same false premise → embeddings show diversity → alarm doesn't fire. You miss the exact failure mode.
- **Calibration bias:** Which framing problems do you seed? If you seed only the obvious ones, the threshold won't catch subtle inherited bias.
- **"Tight consensus" ≠ "wrong framing":** Fast agreement could indicate clarity or domain expertise. High false-positive rate trains operators to ignore the banner.

---

## Proposal C (False-Consensus Alarm – Operability Lens)

**Steelman:** This reuses the existing human intuition ("fast agreement = leading question") that already lives in the Stage-1a brief review. It's lighter than role-based re-derivation. It's reversible. It gives clear, actionable feedback: "all agents converged quickly on <5% variance; verify the framing." An engineer at 2am understands that without trusting a black-box agent. Most importantly, it's honest about its assumption: "if consensus speed correlates with framing error, this works; if not, it doesn't." That's testable.

**Assessment:** **Agree with Position (a), and agree with the methodology more strongly than Proposals A or B.**

**Strongest point:** The decision criteria explicitly include "Operational clarity at 2am." This is the right lens. The banner should be comprehensible without trusting agent re-derivation logic or statistical models. "Everyone agreed in 2 rounds with <5% variance on the decision" is a fact an engineer can verify by reading the debate. Either they re-checked the framing or they didn't.

**Weakest assumption:**
> "The existing 'fast unanimous agreement' intuition is a *real* signal of potential group-think or inherited framing, not an artifact of measurement."

This is empirically testable (which Proposal C acknowledges), but it's not obvious. Tight consensus could mean the question was well-framed or the domain expert in the room dominated. **What breaks:** If your data shows tight consensus correlates *weakly* or *equally* with correct framing, then the alarm becomes noise, heedability collapses, and you're no better than before.

**Hidden assumptions:**
1. Round count and variance are good proxies for framing speed (they are simpler proxies than semantics, but still proxies).
2. A calibration corpus can be built without biasing what counts as "tight" consensus.
3. The operator's failure mode is "they didn't re-check" rather than "they re-checked and still got it wrong"—if it's the latter, (a) doesn't help.

**Overlooked risks:**
- **Variance metric fragility:** Response variance depends on participant verbosity and diversity of expression. It's noisier than you think.
- **Consensus speed in distributed decision-making:** If participants are on different time zones or batching responses, "fast" becomes measurement artifact, not a cognitive signal.
- **Operator fatigue if FP rate is high:** If the alarm fires on 20% of debates and only 3% of them have framing problems, the operator learns to ignore it.

---

# My Revised Recommendation

**I recommend Option (a) – False-Consensus Alarm, using Proposal C's methodology** – but with a critical pre-condition: **validate the core assumption before shipping.**

**Why not (b)?**

Proposal A's semantic re-derivation is more direct, but it introduces two unacceptable 2am costs:
1. **Black-box reasoning:** The operator sees "framing-role disagreed" and must either trust the agent or spend 30 minutes re-reasoning the state themselves.
2. **New trust boundary:** A dedicated agent with repository-state access is a blast radius you don't need. Consensus-speed detection operates only on debate text and requires no new external state access.

The semantic directness of (b) doesn't overcome its operability cost.

**Why (a) over (c)?**

Actually, I don't disagree with (c). Proposal C *is* the right methodology. But I need to state the position clearly:

**Option (a) – False-Consensus Alarm – is the right *form*, but it only works if the assumption is true.**

The assumption is: **Tight consensus on a question, reached quickly, correlates strongly with framing error.** This is plausible (group-think is real) but empirically untested in your context.

**What I would do right now:**

1. **Instrument before alarming (next 2 weeks):** Measure consensus-speed metrics (round count, embedding variance, response overlap) on 20-30 real debates *without* surfacing any banner. Build a baseline.

2. **Seed adversarially (week 3):** Take 10-15 recent debates that were later identified as having framing issues. Rerun them in simulation with the original brief. Do they show tight early consensus? If <60% do, the signal is too weak; stop here and reconsider (b) or accept (c).

3. **Seed positive controls (week 3):** Take 10-15 debates with sound framing that reached consensus quickly. Confirm the alarm would fire on these too (expected false positives). Set your threshold to keep FP rate <15% on this set.

4. **Validate once more (week 4):** Run both seeded sets through the tuned alarm. If it catches ≥70% of known framing problems with <15% FP rate on sound debates, proceed to off-by-default production. If not, demote to metrics-only and re-evaluate (b).

5. **Measure real behavior (first month live):** Log which operators heed the banner and whether the flagged debates actually had framing issues. If >50% of banners are dismissed without follow-up, the signal isn't working—don't spend more cycles on it.

**What I would NOT do:**

- Ship (b) without accepting that re-derivation is another opinion, not truth.
- Assume tight consensus *is* correlated with framing error; measure it first.
- Enable the banner until you have data showing it catches real problems at acceptable precision.

**The honest statement:**

All three proposals are betting on operators noticing and acting on a signal. (a) and (c) bet on them acting on a heuristic (consensus speed). (b) bets on them trusting an agent's re-derivation. My position is: (a)'s bet is *easier to understand at 2am*, so if it's empirically valid, do (a). If it's not (if tight consensus is a weak predictor), then accepting (c) and staying with author-time gates is more honest than adding noise.

**Signals to revisit:**
- If seeded-corpus data shows consensus speed has <0.65 correlation with framing error, don't ship (a); move to (c) or reconsider (b).
- If production shows >25% of banners dismissed, demote to logs-only and re-evaluate.
- If (a) goes live and framing-related incidents don't drop after 3 months, it's not working—stop maintaining it.

*Latency: 84.97s | Tokens: 5373*

### Deepseek (gpt-5.6-sol)

## Proposal A

**Steelman first:** This is exactly right if repository state is the authoritative way to detect framing errors, the role can inspect that state with bounded and already-approved access, and it can produce independently verifiable citations rather than another unsupported opinion. Its incremental cost and tail latency must also remain within explicit budgets.

**Your assessment:** **Agree.** Option (b) is the only posture that tests the actual failure condition: whether the question’s premises match reality. Consensus similarity is, at best, a trigger correlated with that condition.

The rollout thresholds are statistically inadequate, however. If framing errors occur in 2% of debates, a detector with 70% recall and a 5% benign false-positive rate produces only about 22% precision:

`(0.02 × 0.70) / ((0.02 × 0.70) + (0.98 × 0.05)) ≈ 22%`

Roughly four out of five banners would be false. Achieving 80% precision at that prevalence and recall requires a benign false-positive rate below approximately 0.36%. A corpus of 100 cases cannot establish that.

**Strongest point:** It distinguishes a causal test from a proxy. “State contradicts premise X” is actionable; “responses are semantically similar” is not.

**Weakest assumption:** The role can independently inspect sufficiently complete, current state. If it receives the same curated evidence and inherited premises as the debating agents, it becomes another correlated voter and adds cost without independent detection power.

**Hidden assumptions:**

- State inspection is already within the same trust boundary; the proposal does not quantify whether the new call exposes any additional data.
- Evidence citations can be validated cheaply and deterministically. A plausible-looking but incorrect citation would make the banner actively misleading.

**Overlooked risks:**

- Adding one call to a 3-call fanout is a 33% call-count increase, not 15–30%; for five calls it is 20%. Token cost must be measured from actual bounded prompts.
- Parallel execution does not guarantee only a 5–15% p95 increase. The framing call becomes the critical path whenever it is the slowest call; its timeout and latency distribution matter more than its mean.
- The role could systematically oppose sound questions to appear useful. Calibration needs “correct framing with surprising unanimity” negatives.
- Production prevalence drift can destroy precision even when seeded-corpus precision looks good.
- Metrics need separate outcomes for “premise contradicted,” “missing evidence,” and “alternative worth considering.” Only the first should eventually qualify for a loud banner.

## Proposal B

**Steelman first:** This would be exactly right if false framing reliably caused unusually tight semantic agreement, correct framing rarely did, local embeddings were cheap and operationally supportable, and state access truly introduced a new unacceptable trust boundary. Under those conditions, option (a) would be a low-cost, high-recall intrusion detector.

**Your assessment:** **Disagree.** The detector measures semantic convergence, not framing validity. True decisions can have unanimous answers, while a bad framing can produce diverse recommendations inside the same false premise. No embedding threshold repairs that identifiability problem.

Its security argument also attacks an implementation not required by option (b). A framing diagnostic can be isolated from peer responses, read-only, evidence-bounded, and prohibited from injecting anything into the debate. If existing council agents already inspect repository state through the same model boundary, the claimed expansion in exfiltration surface may be zero.

**Strongest point:** It correctly emphasizes that any new state-reading role needs an explicit data-flow and trust-boundary analysis. Proposal A assumes this rather than proving it.

**Weakest assumption:** Tight agreement is sufficiently predictive of framing error. If legitimate consensus is common, the base rate dominates and the alarm becomes noise.

**Hidden assumptions:**

- A local embedding runtime is already available and operationally acceptable. Shipping a model introduces artifact size, cold-start latency, dependency, patching, and model-version concerns.
- Response similarity is comparable across providers, response lengths, question types, and council sizes.

**Overlooked risks:**

- “Fast” is poorly defined with at most two rounds. Wall-clock completion time primarily reflects provider latency and answer length, not cognitive independence.
- Pairwise similarity estimates are unstable with only a few responses. Three agents provide only three non-independent pairs.
- A single dissenting response can defeat a tightness threshold even when every response accepts the same bad premise.
- Embeddings can classify differently after model or tokenizer upgrades unless artifacts, preprocessing, and execution settings are pinned.
- The alarm provides no evidence about what the operator should verify, increasing investigation time per alert.
- The claimed prompt-injection blast radius is backwards if the role remains control-plane metadata; the embedding detector still consumes potentially adversarial model output.

## Proposal C

**Steelman first:** This would be exactly right if the existing heuristic had already demonstrated strong empirical correlation with actual framing incidents, a representative labeled corpus existed, and a conservative threshold could achieve a very low production false-positive rate without sacrificing most recall.

**Your assessment:** **Disagree.** This proposal accepts an alarm false-positive rate of 15–20%, which directly violates the stated heed-ability constraint. At one banner per five to seven benign qualifying debates, habituation is not a secondary risk; it is the expected operating condition.

The proposed 10–20-debate corpus is far too small. With zero false positives in 20 benign cases, the one-sided 95% upper bound on the true false-positive rate is still roughly 14%. That dataset cannot validate even a 5% target, much less the sub-1% rate likely needed at a low framing-error prevalence.

**Strongest point:** It makes calibration, reversibility, and operator behavior first-class operational requirements.

**Weakest assumption:** The author assumes a 20% false-positive rate is tolerable because alerts are understandable. The constraint says the opposite: heed-ability, not explainability, is the limiting budget.

**Hidden assumptions:**

- The system exposes “top-3 candidates,” percentage agreement, and multiple “consensus rounds.” Those structures are not established in the question and may not exist.
- Seeded examples reproduce the linguistic and prevalence characteristics of organic framing failures.

**Overlooked risks:**

- Threshold tuning on 10–20 examples will overfit badly; a separate holdout cannot rescue such a small sample.
- “Agreement within two rounds” has almost no temporal resolution when the system is capped at two rounds.
- The proposal conflates detection recall on a balanced seeded set with positive predictive value in low-prevalence production traffic.
- Its revisit suggestion of a blocking check contradicts the ratified advisory-only constraint.
- “If incidents do not drop, choose static” is a false conclusion: the implementation or operator response may have failed even though debate-time defense remains necessary.

## Revised recommendation

I choose **(b), the debate-time framing-challenge role**. The other proposals do not change my position because neither establishes that response similarity identifies state-inconsistent framing. They optimize the cost and determinism of measuring the wrong variable.

Implement the role as a bounded, parallel, read-only control-plane diagnostic:

- Require an independently derived decision, explicit unsupported premises, inspected-state identifiers, and exact evidence citations.
- Keep its output outside `_anonymize_responses()` and never feed it back into the running debate.
- Preserve inbox I/O, two-round behavior, and exit semantics.
- Initially record metrics only. A model’s self-reported confidence must not control banner eligibility.
- Eventually banner only evidence-backed premise contradictions, not generic disagreement or missing context.

Do not use “80% precision on 100 cases” as the launch gate. Define the expected production prevalence and banner budget first. If framing failures occur near 2%, an 80% precision target with 70% recall requires approximately **0.36% benign FPR**. Demonstrating that order of magnitude requires hundreds to thousands of representative benign cases, depending on the confidence required—not 50–100.

Performance budgets should be concrete:

- Incremental median tokens: no more than 25%.
- Incremental p95 wall time: no more than 20%.
- A hard diagnostic timeout that returns no finding and cannot delay debate completion beyond the budget.
- Separate p50/p95/p99 latency, timeout rate, evidence-validation rate, benign FPR, recall, and banners per 100 debates.

What breaks first is not compute. It is operator trust. Option (a) spends that scarce budget on a proxy; option (b) can spend it on a cited contradiction the operator can verify.

*Latency: 58.88s | Tokens: 11440*

### Gemini (gemini-3.1-pro-preview)

Here is my architectural evaluation of the proposals, followed by my final recommendation. As a Security & Integration Architect, my evaluation is rooted in blast radius, API boundaries, and supply chain integrity. 

---

### --- Proposal A ---

**Steelman first:** For this proposal to be exactly right, it must be true that the root cause of framing bias is fundamentally semantic, and therefore only a semantic system (an LLM) evaluating ground-truth state can detect it. Furthermore, it assumes that the security and privacy boundaries of the system are already configured to safely transmit unconstrained repository state to the external research provider API without exposing the system to prompt injection or IP exfiltration.

**Your assessment:** **Disagree with Position.** Option (b) is a security nightmare. Granting a new, unconstrained agent broad read-access to repository state to "re-derive" a decision massively expands the attack surface. It funnels unstructured, potentially untrusted repository state directly into an external API provider. If the repo contains malicious strings, this agent becomes a vector for prompt injection that could poison the debate diagnostic. 

**Strongest point:** "Embedding similarity cannot distinguish coordinated correctness from coordinated framing error." This is a highly accurate critique of Option (a). A false-consensus alarm only detects the *symptom* (agreement), not the *disease* (bad framing).

**Weakest assumption:** The assumption that "One additional parallel model invocation is operationally acceptable." The author only measures token cost and latency, completely ignoring the security cost of pushing raw internal state across an external API boundary. 

**Hidden assumptions:** 
1. The author assumes the LLM has the reasoning capability to correctly re-derive complex decisions from raw state without hallucinating a worse framing. 
2. The author assumes "actual repo/state" can be neatly scoped and packaged into a prompt without leaking sensitive environment variables, secrets, or proprietary business logic to the API provider.

**Overlooked risks:** Supply chain data exfiltration and Indirect Prompt Injection. If the external research provider is compromised, or if a bad actor commits a malicious payload into the repository state, this framing-challenge role will execute it and surface the resulting payload into the operator's inbox as a "diagnostic."

---

### --- Proposal B ---

**Steelman first:** For this proposal to be exactly right, it must be true that anomalous consensus (tight/fast agreement) is a statistically reliable indicator of inherited framing bias, and that running local, deterministic anomaly detection provides enough of a behavioral "speed bump" to force the operator to manually verify the state themselves, completely avoiding the need to send state to an external API.

**Your assessment:** **Agree with Position.** This is the only proposal that correctly identifies the blast radius implications of Option (b). By framing the problem as an Intrusion Detection System (IDS) and relying on local, deterministic embedding models over already-generated outputs, it satisfies the defense-in-depth requirement without expanding the API attack surface.

**Strongest point:** "Minimizes API Blast Radius... operates purely on the outputs the agents have already generated. It requires zero additional context from the repository." This perfectly adheres to the principle of least privilege.

**Weakest assumption:** The assumption that the operator will actually investigate *why* the framing was flawed based solely on an alarm. The proposal delegates the hardest part of the work (re-derivation) back to the human operator. 

**Hidden assumptions:** 
1. Assumes the local environment has the secure compute provisioning to run an embedding model (e.g., `all-MiniLM-L6-v2`) without introducing new dependency vulnerabilities (e.g., pulling unverified HuggingFace models into the runtime).
2. Assumes the existing outputs contain enough semantic variance in *correct* unanimous decisions to clearly separate them from *forced/biased* unanimous decisions.

**Overlooked risks:** The operational risk of dependency poisoning in the local embedding model itself. If we implement local embeddings, we must pin and hash the model weights; otherwise, we've just shifted the supply-chain risk from the external API to the local model registry.

---

### --- Proposal C ---

**Steelman first:** For this proposal to be exactly right, it must be true that human operators respond best to mechanized versions of their own existing heuristics, and that a 20% false-positive rate is an acceptable operational trade-off for catching 70% of framing errors in a non-blocking advisory capacity.

**Your assessment:** **Partially Agree with Position, Disagree with execution.** I agree with selecting Option (a), but the author's proposed thresholds are reckless. In security alerting, a 20% false-positive rate on a non-blocking ASCII banner guarantees alert fatigue. The operator will build muscle memory to ignore it within a week.

**Strongest point:** "Operational clarity at 2am... The agent re-derived the decision and disagreed is a black box that requires trusting the re-derivation logic." This brilliantly highlights the operational danger of LLM-based diagnostics during incident response. 

**Weakest assumption:** The assumption that "Advisory banners, when well-calibrated [at <20% FP], are heeded." A 1-in-5 false alarm rate is not well-calibrated; it is noise. 

**Hidden assumptions:** 
1. Assumes that injecting fake framing problems into a test corpus is computationally straightforward and accurately mimics the subtle, organic ways framing bias manifests in production.
2. Assumes embedding distance is perfectly correlated with round-over-round response agreement.

**Overlooked risks:** The author introduces a critical contradiction: they advocate for an embedding-agreement detector, but in the Risks section, they state "ensure thresholds are based on discrete signals (round count, response agreement %) not embeddings." This architectural confusion will lead to a botched implementation that fails to actually measure semantic tightness.

---

### Revised Recommendation

**I mandate Option (a) — the false-consensus ALARM, implemented strictly via local anomaly detection.** 

I am firmly rejecting Option (b). Proposal A’s advocacy for a "state-grounded diagnostic agent" completely ignores the API boundary. Sending unstructured repository state to an external LLM API provider to double-check a debate introduces unacceptable exfiltration risks and exposes the orchestration pipeline to indirect prompt injection from the codebase. The blast radius of a compromised framing agent is total context-poisoning. 

I am adopting Proposal B's architectural posture. We will treat framing bias as a behavioral anomaly and implement Option (a) as a local Intrusion Detection System (IDS). 

**Implementation Directives:**
1. **Zero-API Expansion:** The consensus check must operate strictly on the existing Stage-1 outputs. It will use a lightweight, local embedding model (`all-MiniLM-L6-v2`). The model weights must be pinned, hashed, and stored in our internal artifact registry to neutralize local supply-chain poisoning. No external API calls will be made to calculate consensus.
2. **Thresholds & Alert Fatigue:** I reject Proposal C's 20% false-positive tolerance. The banner remains strictly off-by-default (shadow mode) until calibration proves a FP rate of **<5%**. An ignored security banner is worse than no banner, as it normalizes the bypassing of controls. 
3. **Immutability of Debate Constraints:** The alarm will be an ASCII-only banner co-located with the author line. It will not alter exit codes, it will not increment rounds beyond the <=2 limit, and it will strictly operate outside the `_anonymize_responses()` pipeline. 

By mechanizing the "fast/tight agreement" heuristic locally, we break the operator's cognitive bias (the documented failure mode) without handing our raw repository state over to an external LLM. We contain the blast radius to zero while adding an active, debate-time defense layer.

*Latency: 36.62s | Tokens: 8825*

## Synthesis (by openai, non-participant)

## Consensus

- **Do not choose (c) STATIC.**  
  All participants agreed the documented failure mode is real: author-time framing defense alone leaves a known runtime gap. This was strong consensus backed by the stated operator failure pattern, not mere echoing.

- **Any runtime defense must be advisory-only and heavily constrained.**  
  Broad agreement that it must:
  - not alter exit codes,
  - stay outside `_anonymize_responses()`,
  - preserve `<=2` rounds,
  - surface as ASCII-only banner + metrics,
  - be introduced cautiously to protect heed-ability.

- **Heed-ability is the key budget.**  
  Most participants independently recognized that a noisy banner is worse than none. This was one of the strongest shared conclusions and was supported by the explicit constraint, not groupthink.

- **Calibration/shadow mode is necessary before visible enablement.**  
  Nearly everyone converged on “measure first, banner later.” This was evidence-driven.

## Unresolved Disagreements

### 1) Proxy signal vs direct state check
- **Crux:** Is debate-time framing defense better implemented as a **proxy detector** of suspicious consensus ((a)) or a **direct test** of whether the question contradicts repo/state ((b))?
- **Side favoring (a):**
  - Simpler, safer, more understandable at 2am.
  - Lower trust burden: “everyone agreed unusually fast/tightly” is inspectable.
  - Avoids adding a new state-reading agent and possible blast-radius/security concerns.
- **Side favoring (b):**
  - It measures the actual thing that matters: whether framing matches reality.
  - Consensus similarity is only a symptom and may be weakly correlated.
  - A good detector should surface “you asked X, but state implies Y,” not “this felt too unanimous.”

- **Stronger side:** **(b) had the stronger substantive argument.**  
  The best reasoning in the debate was that the failure mode is **semantic/state-grounded**, not statistical. If all agents inherit a bad premise, consensus alarms may miss it or flag many correct cases. That is a core identifiability problem, not just a tuning problem.

### 2) Operability/security cost of a framing role
- **Crux:** Does (b)’s directness outweigh its complexity, opacity, and possible trust-boundary expansion?
- **Objections to (b):**
  - It may become “just another opinion” with the same blind spots as other agents.
  - It may be hard for an on-call operator to adjudicate contradictory outputs.
  - If it requires broader repo/state access to an external model, it may expand exfiltration/prompt-injection risk.
- **Defense of (b):**
  - It can be isolated as control-plane metadata, read-only, parallel, and evidence-bounded.
  - If existing debate agents already inspect repo/state through the same boundary, the trust-boundary expansion may be zero.
  - Evidence citations can make the finding more actionable than a generic alarm.

- **Stronger side:** **Mixed, but objections to a naive version of (b) were valid.**  
  The anti-(b) side was strongest when attacking an **unbounded, black-box re-deriver**. The pro-(b) side was strongest when narrowing it to a **bounded, citation-required diagnostic**. So the better conclusion is not “reject (b),” but “only choose (b) in a tightly scoped form.”

### 3) Whether (a) is viable if carefully calibrated
- **Crux:** Can a consensus alarm be accurate enough to preserve heed-ability?
- **Pro-(a):**
  - It reuses an existing heuristic already trusted at author-time.
  - It is lightweight, reversible, and easy to explain.
- **Anti-(a):**
  - With low prevalence of true framing errors, even modest false-positive rates destroy precision.
  - “Fast/tight consensus” may reflect correctness or expertise, not framing failure.
  - With only 2 rounds and few agents, the signal may be statistically weak.

- **Stronger side:** **Anti-(a) had the stronger technical case.**  
  The base-rate argument was especially strong: if framing failures are rare, a banner based on a weak proxy will likely be mostly false alarms unless false-positive rate is extremely low. Several advocates for (a) proposed tolerances that conflicted with the heed-ability constraint.

## Argument Quality Assessment

### Best-reasoned proposals
- **Best overall reasoning:** the case for **(b)** from deepseek.  
  Why:
  - It tied the choice directly to the actual failure mode.
  - It distinguished causal detection from proxy correlation.
  - It correctly challenged the statistical adequacy of proposed calibration plans for (a), especially under low-prevalence conditions.
  - It offered a constrained implementation shape compatible with the architectural constraints.

- **Best reasoning for (a):** claude’s operability-focused defense.  
  Strong on:
  - operator comprehension,
  - reversibility,
  - caution about noisy black-box diagnostics.  
  But it still depended on an unproven correlation and tolerated false-positive rates that likely violate the stated heed-ability budget.

### Weaker proposals / weaker assumptions
- **Weakest recurring assumption:** that a 10–100 case seeded corpus can validate an advisory banner for a likely low-prevalence problem.  
  This was challenged well and is not credible for production confidence.

- **Weakness in some (a) arguments:** treating explainability of the banner as enough to offset a materially high false-positive rate. The constraints explicitly say heed-ability is the budget.

- **Weakness in some anti-(b) arguments:** assuming (b) necessarily means broad new external state exposure. That attacks one implementation, not the option itself. The better pro-(b) arguments showed it can be bounded.

### Single strongest argument in the debate
- **“The first failure point is semantic, not statistical.”**  
  This is the central reason to prefer (b): the system’s known failure is accepting inherited framing without checking against state. A runtime defense that checks consensus shape instead of state consistency may optimize the wrong variable.

### Single weakest argument in the debate
- **Any suggestion that ~15–20% false-positive rate is acceptable for a visible advisory banner.**  
  This directly conflicts with the stated heed-ability constraint and ignores base-rate effects.

## Blind Spots

1. **No one fully specified what “actual repo/state” means operationally.**  
   The debate argued about state-grounding, but did not define:
   - exact evidence scope,
   - allowed files/sources,
   - citation format,
   - handling of ambiguous or incomplete state.

2. **No one deeply addressed how operators should act on a finding.**  
   A banner only helps if there is a clear next step:
   - rerun with reframed question?
   - pause and inspect cited files?
   - mark finding dismissed with reason?
   This workflow was underdeveloped.

3. **Little discussion of evaluation labels and ground truth.**  
   To calibrate either option, you need a reliable definition of “framing failure” and adjudication rules. The debate assumed labeled corpora exist or can be seeded, but did not solve labeling quality.

4. **Insufficient treatment of ambiguity vs contradiction.**  
   Many framing issues are not “X is false; Y is true” but “the question prematurely narrows the decision.” The best version of (b) should detect unsupported premises and dead escape hatches, not only direct contradictions.

5. **No one seriously considered combining the options, but the prompt forbids that.**  
   Several arguments implicitly pointed to “(b) primary, (a) later secondary.” Since the decision requires one posture, this matters: the best choice is the one that leaves room for later layering.

## Recommended Decision

**Choose (b): a debate-time framing-challenge ROLE**, implemented as a **bounded, parallel, evidence-citing diagnostic** in shadow mode first.

### Rationale
This best matches the documented failure mode: *accepting inherited framing without verifying against state*. Option (b) is the only option that directly performs that verification. Option (a) may be useful later as a secondary heuristic, but the debate did not establish that consensus tightness is a reliable enough primary detector to justify a visible advisory under the heed-ability constraint.

### Why not (a)?
The strongest objections to (a) outweighed its benefits:
- It measures a proxy, not the failure condition.
- With low likely prevalence of true framing failures, precision will collapse unless benign false-positive rate is extremely low.
- Multiple proponents implicitly accepted false-positive rates too high for the stated banner budget.
- `<=2` rounds limits the meaning of “fast consensus,” weakening the core signal.

### Why still accept (b) despite the objections?
The best objections to (b) were about **implementation shape**, not the underlying posture:
- “black box” is mitigated by requiring explicit evidence citations and a narrow output contract;
- trust-boundary concerns are real, but only decisive if this role needs access beyond what existing agents already have;
- operator burden is reduced if the output is not “agent disagrees” but “premise P is unsupported/contradicted by files A, B, C.”

### Strongest objection acknowledged
The strongest objection is that (b) could become another correlated hallucinating voter and/or widen the security surface. I weigh this as serious but manageable because the role can be:
- read-only,
- evidence-bounded,
- isolated from peer debate,
- required to cite inspected state,
- held to launch gates based on evidence-backed contradiction precision rather than self-confidence.

If those safeguards cannot be met, revisit the choice. But based on the debate, **(b) is still the better posture**.

## Risks

1. **The framing role becomes another noisy opinion**
   - **Risk:** It echoes inherited framing or invents counter-framings.
   - **Mitigation:** Require:
     - independently restated decision,
     - explicit unsupported premises,
     - exact citations to inspected state,
     - `NO_FINDING` default when evidence is insufficient.

2. **Operator confusion from contradictory outputs**
   - **Risk:** Debate says X, framing role says Y, operator trusts neither.
   - **Mitigation:** Make the banner narrow: flag only evidence-backed premise contradictions or dead escape hatches, not generic disagreement.

3. **Security / exfiltration / prompt-injection expansion**
   - **Risk:** If the role reads more raw state than current agents, it may widen blast radius.
   - **Mitigation:** Use the same approved trust boundary as existing debate agents where possible; scope inputs minimally; keep output as control-plane metadata; add explicit data-flow review.

4. **Latency and cost creep**
   - **Risk:** Parallel call becomes tail latency bottleneck.
   - **Mitigation:** Hard timeout, bounded prompt budget, separate latency SLOs, and “timeout = no finding.”

5. **Bad calibration due to poor labels**
   - **Risk:** Shadow-mode metrics may look good on synthetic cases but fail in production.
   - **Mitigation:** Build a corpus with real historical framing incidents and benign unanimous decisions; separate contradiction / ambiguity / missing-evidence classes.

## Action Items

1. **Adopt option (b) formally**
   - Record ADR choosing a debate-time framing-challenge role as advisory-only control-plane diagnostic.

2. **Define the role contract**
   - Input: same approved repo/state access envelope as debate agents, no broader unless separately ratified.
   - Output:
     - `NO_FINDING`, or
     - structured finding with:
       - requested decision,
       - independently derived decision,
       - unsupported/contradicted premises,
       - inspected-state identifiers,
       - exact evidence citations.

3. **Keep it isolated**
   - Do not feed output into `_anonymize_responses()`.
   - Do not alter rounds or exit semantics.
   - Surface only as ASCII banner + metrics when enabled.

4. **Launch in shadow mode**
   - Metrics-only first.
   - Collect:
     - finding type,
     - citation presence/validity,
     - operator disposition,
     - latency,
     - token cost,
     - timeout rate.

5. **Create an evaluation corpus**
   - Include:
     - real historical framing failures,
     - benign debates with fast unanimity,
     - T1/T2/T3-style seeded cases,
     - ambiguous-state cases.
   - Label outcomes by type:
     - premise contradicted,
     - premise unsupported,
     - dead escape hatch,
     - no issue.

6. **Set stricter launch gates than proposed for (a)**
   - Visible banner only for evidence-backed premise contradictions.
   - Require low benign-case banner rate on a large representative set, not a tiny seeded sample.
   - Treat unsupported/missing-evidence findings as metrics-only until proven useful.

7. **Define operator workflow**
   - For a banner:
     - inspect cited evidence,
     - choose reframe/rerun or dismiss with reason,
     - record disposition for calibration.

8. **Revisit (a) only later as a secondary signal**
   - After (b) metrics stabilize, evaluate whether T1/T2/T3 or consensus-tightness adds recall without exceeding banner budget.  
   - Do not use (a) as the primary posture now.

In short: **pick (b), but only in a constrained, evidence-citing, shadow-first form.**
