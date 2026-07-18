# AI Council Debate: # Decision: Keep, replace, or demote DeepSeek in ai-council's default full panel

**Date:** 2026-07-17 23:12:20
**Panel:** claude-haiku-4-5-20251001, gpt-5.6-sol, gemini-3.1-pro-preview
**Synthesizer:** openai (non-participant)
**Rounds:** 2
**Duration:** 174.1s
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** C:\Users\1028120\Documents\Dev\ai-council\council_inbox\uc3-deepseek-panel-disposition.md
**Cost:** ~$0.1088 (27,420 tokens)

## Verdict Summary

**Decision:** Decision: DEMOTE DeepSeek from the DEFAULT full panel, keep it opt-in, and keep its research summarizer role only behind an explicit temporary reliability review with fallback work.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 3/3
**Verdict author:** openai (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

# Decision: Keep, replace, or demote DeepSeek in ai-council's default full panel? (#6)

## Context
ai-council's full panel is 5 API models with load-bearing personas: claude (Systems), gemini (Security), deepseek (Performance/algorithmic-cost), openai (Product), grok (Contrarian). The reactive trigger for this review is a round-blocking failure rate above 2%. DeepSeek is the cheapest panelist ($0.30/$0.50 per 1M in/out) and doubles as the summarizer for research mode.

## The decision
Choose ONE disposition for DeepSeek in the DEFAULT full panel.

## Options
- (a) KEEP — retain DeepSeek in the full panel unchanged; its Performance lens and low cost justify its seat.
- (b) REPLACE — swap DeepSeek for a more reliable model in the Performance seat (name the candidate and the reliability evidence that would justify it).
- (c) DEMOTE — remove DeepSeek from the DEFAULT full panel but keep it available via --models opt-in and as the research summarizer.

## Constraints (these eliminate weak options)
- The 5 personas are load-bearing (Systems/Security/Performance/Product/Contrarian); any change must preserve lens coverage, not merely swap a vendor.
- DeepSeek is the research summary_model; any disposition must state the research-path impact.
- The decision must be reversible and evidence-triggered, not vibes — name the metric that would flip it back.
- Cost posture: API dollars are not the constraint; reliability and lens quality are.

## Prior decisions in scope
ADR-02 (default panel composition), ADR-06 (cost optimization). Escape hatch: (a) KEEP is the null option if the evidence does not justify a change.

## Round 1: Initial Responses

### Claude (claude-haiku-4-5-20251001)

## Position
**DEMOTE** DeepSeek from the default full panel, retaining it for opt-in and research. A >2% blocking failure rate is operationally unacceptable for a default panelist, and since cost is not a constraint, there is no economic reason to tolerate this unreliability in the critical path.

## Alternatives Considered
- **KEEP (rejected):** Only defensible if the failure rate were measurement error or environmental (e.g., transient cloud incidents). At >2%, this represents systematic unreliability; "Performance lens" is not a privilege worth blocking rounds. KEEP becomes escape-hatch only if evidence shows the measured rate is outdated.
- **REPLACE (rejected):** Tempting if a candidate offers >99.5% availability AND the "Performance" lens isn't already provided by existing panelists. However, the question doesn't name a proven replacement, and promoting another model to default without evidence of similar reliability would repeat the problem.

## Decision Criteria
1. **On-call operational reality** — what's acceptable to page on at 2am? (>2% is not acceptable; <0.5% is).
2. **Lens irreplaceability** — does removing DeepSeek from default eliminate a capability gap, or just reduce redundancy?
3. **Research-path stability** — is DeepSeek the only viable summarizer, or a pragmatic choice?
4. **Reversibility and evidence** — can we detect and auto-restore if reliability improves?
5. **Cost / effort** — is the demotion worth implementation overhead? (Yes, if failure rate is real; no if it's measurement noise.)

## Assumptions
1. The >2% failure rate is measured across representative traffic, not a transient spike or measurement artifact.
2. The "Performance" lens (algorithmic cost analysis, efficiency reasoning) is a nice-to-have in the default panel, not a must-have; it can degrade to opt-in without losing critical capability.
3. DeepSeek is the summarizer *by pragmatic choice* (cheap, acceptable quality), not because it's technically irreplaceable for research mode.
4. Your SLA for the full panel is 99.5%+ availability; a single panelist at 98% can drag the whole round.

## Recommendation

**Now:**
- Remove DeepSeek from the hardcoded default panel (revert to a 4-model default: Claude, Gemini, OpenAI, Grok).
- Keep DeepSeek available via `--models deepseek` for users who explicitly want cost-optimized or performance-focused reasoning.
- No change to research mode: keep DeepSeek as the summary_model; it doesn't need to be bulletproof there (async, user tolerance for slowness is higher).

**Not now:**
- Don't replace DeepSeek with another model without proof of >99.5% measured availability.
- Don't add a 5th model just to preserve the Performance persona; the other four provide adequate coverage.

**Defer:**
- Evaluate a replacement only if: (i) you have 2+ weeks of production data from a candidate model, AND (ii) it demonstrates 99.8%+ success rate at the scale you run.
- Revisit the decision in 30 days: if DeepSeek reliability improves to <0.3% failure rate, consider re-adding it.

## Rationale
- **Operational math:** A 5-model panel with one 98% reliable member drops to ~90% reliability for the full round (assuming independent failures). Removing it brings you back to the 4-model baseline. A 2% failure rate means one user in fifty hits a failed round—unacceptable for a default experience.
- **Lens redundancy:** Claude already reasons about efficiency and trade-offs; Grok's contrarian stance often includes cost-vs-capability questioning. The Performance lens is *useful*, not *irreplaceable*.
- **Research mode is forgiving:** Async summarization of research output tolerates higher latency and occasional retries. Keep DeepSeek there until it becomes a bottleneck (which >2% failure won't be in a non-realtime context).
- **Cost is a red herring:** At $0.30/$0.50 per 1M tokens, replacing DeepSeek with Claude/GPT-4 costs ~$10–50 more per month in typical use. Demotion cost is zero; pain from a failed round is incalculable (user frustration, retries, 2am pages).
- **Reversibility is cheap:** Add a metric check to the startup path: if 30-day rolling failure rate drops to <0.3%, auto-re-enable with a log entry. This makes the decision data-driven.

## Trade-offs
- **Lost default:** Users accustomed to `ai-council` without flags will get a 4-model response instead of 5. The Performance lens will be quieter. Mitigation: document clearly; add a one-liner to help text.
- **Opt-in friction:** Power users wanting algorithmic-cost analysis must now type `--models deepseek` (low friction, but non-zero).
- **Research UX unchanged:** No cost; research mode already expects retries and has a different SLA.

## Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Failure rate is transient (cloud incident), not systematic | Medium | High—demoting solves the wrong problem | Verify: 2% sustained over 7+ days, not 1-day spike? If spike, KEEP and monitor. |
| Some critical user (internal team) relies on default 5-panel and will complain | Medium | Medium—adds friction to workflow | Pre-announce in release notes; offer `--models deepseek` alias in shell profile. |
| DeepSeek reliability improves in 2 weeks, but you don't catch it | Low | Low—re-enable via metric trigger is automatic | Implement the <0.3% re-enable check in code; log every time it fires. |
| 4-model panel is insufficient for some reasoning task | Low | Medium—users need to hand-add DeepSeek | Treat as feature request; gather user evidence before adding a 5th default. |

## Signals to Revisit
- **Trigger re-enable:** 30-day rolling failure rate drops to <0.3% and holds for 7 days. Auto-re-add and log.
- **Trigger replace:** If user feedback shows "Performance lens missing" in >5 distinct tickets within 30 days, evaluate replacement candidates.
- **Trigger re-evaluate:** If research mode hitting DeepSeek failure rate >1% (indicating load-path issue), explore alternative summarizers.
- **Trigger KEEP-only:** If measured failure rate is actually <0.5%, this was a false alarm; revert and tighten alerting.

*Latency: 49.75s | Tokens: 2810*

### Deepseek (gpt-5.6-sol)

## Position
KEEP DeepSeek in the default full panel for now. The >2% failure rate warrants a controlled bake-off, but no evidence supplied shows that a replacement preserves Performance-lens quality while materially improving end-to-end reliability.

## Alternatives Considered

- **REPLACE with Claude Sonnet 4 in the Performance seat:** Plausible because cost is secondary and Claude is already operationally integrated, but there is no same-workload reliability or blinded lens-quality evidence to justify the swap.
- **DEMOTE DeepSeek:** Rejected because it either removes a load-bearing Performance perspective or requires another panelist to cover two roles, reducing independent analysis. It also leaves the potentially unreliable research summarizer untouched.

## Decision Criteria

- Default-round completion rate, especially failures attributable to each model.
- Preservation of an independent, high-quality Performance/algorithmic-cost lens.
- Candidate evidence from representative ai-council traffic, not vendor-wide reputation.
- Research-mode completion and summary quality.
- Reversibility through configuration and explicit metric thresholds.

## Assumptions

- The reported rate is only slightly above 2%, rather than a sustained severe failure rate such as 10%+.
- Failures currently block rounds because retry, timeout, and partial-result handling are limited.
- No candidate has already completed a statistically meaningful, same-workload bake-off.
- DeepSeek’s Performance responses remain materially differentiated and useful.

## Recommendation

- **Do now:** Retain DeepSeek in both the default Performance seat and `research.summary_model`. Separate provider errors, timeouts, malformed responses, and research-summary failures in telemetry. Run Claude Sonnet 4 as the first shadow replacement candidate using the identical Performance persona.
- **Do not do now:** Do not remove the Performance seat, assign it to a dual-role panelist, or replace DeepSeek based solely on its crossing a review-trigger threshold.
- **Defer until later:** Replace DeepSeek when a candidate completes at least 2,000 representative calls with a blocking-failure rate at or below 0.75%, a 95% confidence upper bound below 1%, and no material regression in blinded Performance-lens scoring. Qualify the research summarizer separately before changing that path.

## Rationale

- A DeepSeek failure rate just above 2% imposes a hard floor of roughly 2% on blocked rounds, but the review trigger is not evidence that another model will improve the complete system.
- With five independent panel calls, even a 0.5% failure rate per model produces approximately `1 - 0.995^5 = 2.48%` failed rounds. DeepSeek replacement alone may therefore fail to achieve the desired round-level reliability.
- At 2,000 candidate calls, an observed 0.5% failure rate has an approximate 95% upper bound near 0.8%, providing credible evidence of improvement over a sustained 2%+ incumbent rate.
- Demotion does not address research-mode exposure because DeepSeek remains the summarizer, while it immediately sacrifices an independent default-panel lens.
- Cost savings are decision-irrelevant here: even a several-fold increase over DeepSeek’s price is likely small compared with blocked-round retries and degraded council output.

## Trade-offs

Keeping DeepSeek accepts at least its observed 2% blocking contribution during the evidence-gathering period. A 2,000-call bake-off delays replacement and incurs duplicate candidate-call cost, but those dollars are explicitly not the binding constraint.

## Risks

- **Persistent blocked rounds — high likelihood, medium impact:** Bound the evaluation window and alert on DeepSeek-attributable blocking failures; escalate immediately if the rate reaches 5% over 200+ calls.
- **Performance-lens quality is overestimated — medium likelihood, high impact:** Use blinded scoring for complexity analysis, bottleneck identification, quantitative reasoning, and actionable recommendations.
- **Failures are workload-specific — medium likelihood, medium impact:** Stratify results by prompt size, timeout, region, research versus standard mode, and concurrency.
- **Research summaries fail independently — medium likelihood, high impact:** Track research-summary completion separately and test a fallback summarizer; do not infer summary reliability from panel-call reliability.

## Signals to Revisit

- Replace DeepSeek if it remains above 2% blocking failures in two consecutive seven-day windows with at least 1,000 calls each, while Claude Sonnet 4 or another candidate meets the ≤0.75% rate, <1% upper-confidence-bound, and lens-quality gates.
- Revisit immediately if DeepSeek reaches 5% failures over 200+ calls, exhibits correlated outages, or materially dominates p95 round latency.
- Keep it without further review if its rolling rate returns below 1% and candidate testing shows no statistically or operationally meaningful improvement.
- Evaluate the research summarizer independently if its completion rate falls below 99% or summary-quality scores regress materially.

*Latency: 60.66s | Tokens: 7994*

### Gemini (gemini-3.1-pro-preview)

## Position
REPLACE DeepSeek in the default panel with OpenAI's `o3-mini` for the Performance/algorithmic-cost seat, and immediately remove DeepSeek from the default research summarization path. A 2% round-blocking failure rate is an unacceptable blast radius for a synchronous dependency; we must excise it from the default critical path entirely.

## Alternatives Considered
- **(a) KEEP**: Rejected. In a synchronous 5-node scatter-gather architecture, a 2% failure rate at a single node translates to a ~10% failure rate for the entire round ($1 - 0.98^5$). Cost savings do not justify corrupting the integrity and reliability of the entire system.
- **(c) DEMOTE**: Rejected. Removing DeepSeek from the panel without a replacement violates the constraint to maintain 5 load-bearing personas. Furthermore, leaving it as the default research summarizer retains a single point of failure (SPOF) in the research pipeline, violating defense-in-depth principles. 

## Decision Criteria
1. **Blast Radius & Pipeline Stability**: The integration must not introduce cascading or round-blocking failures into the composite system.
2. **Persona Integrity**: The replacement must capably fulfill the "Performance/algorithmic-cost" analytical lens.
3. **Supply Chain & Vendor Risk**: The dependency must have a proven track record of API stability and enterprise-grade infrastructure.
4. **Value over Cost**: Optimization for reliability and lens quality, given that API dollars are explicitly not the constraint.

## Assumptions
- The 2% failure rate is occurring on the provider side (e.g., 502s, 529s, long timeouts) and cannot be entirely mitigated by standard client-side retry logic without breaching acceptable latency thresholds.
- The current `ai-council` architecture treats a full panel request as a strict scatter-gather operation where a single model timeout fails the entire synchronous round.
- `o3-mini` can accurately emulate the "Performance/algorithmic-cost" persona through system prompting.

## Recommendation
- **What to do now**: Swap the default Performance seat to `o3-mini` (reliability evidence: OpenAI's historical status metrics and enterprise SLAs guarantee >99.9% uptime, well below the 2% threshold). Swap the research summarizer to `gpt-4o-mini` to eliminate the SPOF in the research path.
- **What not to do now**: Do not permanently delete the DeepSeek integration code. Maintain it via the `--models deepseek` opt-in flag for users willing to explicitly accept the availability risk.
- **What to defer until later**: Architectural refactoring of the `ai-council` runner to support partial-round completions (e.g., gracefully degrading to 4 panelists if one times out) rather than failing the entire round.

## Rationale
- **Dependency Isolation**: Treating every API boundary as a potential point of failure means we cannot tolerate a component that fails 1 in 50 times on a blocking path. Replacing it neutralizes the immediate threat to system stability.
- **Algorithmic Pedigree**: `o3-mini` is highly optimized for coding, performance analysis, and algorithmic reasoning, making it the most functionally accurate drop-in for this specific load-bearing persona.
- **Research Path Integrity**: The research mode requires a highly available summarizer. Swapping to `gpt-4o-mini` ensures the research pipeline is decoupled from DeepSeek’s current API volatility.
- **Constraint Alignment**: The prompt explicitly states API dollars are not the constraint. We are trading an immaterial cost increase for a guaranteed return to <0.1% failure rates on the critical path.

## Trade-offs
- **Vendor Concentration**: By introducing `o3-mini` (Performance) alongside `gpt-4o` (Product), we sacrifice vendor diversity in the panel, mildly increasing our supply chain exposure to OpenAI's infrastructure.
- **Cost**: We abandon DeepSeek's highly disruptive pricing ($0.30/$0.50), accepting higher aggregate run costs in exchange for pipeline reliability. 

## Risks
- **Persona Overlap**: Two OpenAI models on the panel may exhibit correlated reasoning biases, diluting the contrarian value of the council. 
  *Mitigation*: Implement strict, highly opinionated system prompts for the Performance persona to force divergent, optimization-obsessed outputs. Monitor output similarity scores between the Product and Performance seats.
- **Research Summarization Quality**: The replacement summarizer might miss nuances DeepSeek caught.
  *Mitigation*: Evaluate the first 100 research summaries post-migration; if quality degrades, fallback to Anthropic's `claude-3-5-haiku` as the summarizer.

## Signals to Revisit
We will revert this decision and reinstate DeepSeek to the default panel and summarizer roles if:
1. DeepSeek’s API demonstrates a sustained, verifiable round-blocking failure rate of **<0.5% over a trailing 14-day rolling window**. 
2. The core council architecture is updated with circuit breakers and partial-round tolerance, rendering individual API timeouts a localized degradation rather than a global round failure.

*Latency: 46.88s | Tokens: 5944*

## Round 2: Critique

### Claude (claude-haiku-4-5-20251001)

# Analysis of Proposals

## Proposal A: DEMOTE

**Steelman:** If the 2% failure rate is genuine and systematic (not transient), then a synchronous critical path cannot tolerate it. Demotion is the only move that immediately stops user-facing failures while preserving all capability via opt-in. The metric-driven re-enable gate ensures reversibility without manual review. Cost is irrelevant; operational safety is not.

**My assessment:** Partially agree with Position. The demotion logic is sound, but the proposal underestimates risk by leaving an unreliable component as the default research summarizer.

**Strongest point:** The on-call math is iron-clad. "One user in fifty hits a failed round" is unacceptable for a default experience. Demotion stops the bleeding immediately and cheaply.

**Weakest assumption:** That DeepSeek's 2% failure rate is measured consistently across *representative traffic* and not a spike. If it's a 12-hour cloud incident, demotion solves the wrong problem. The proposal gestures at this ("Verify: 2% sustained over 7+ days") but doesn't require verification before acting.

**Hidden assumptions:**
1. The proposal assumes the "Performance lens" is a luxury, not load-bearing. But what if Claude and Grok's cost reasoning are surface-level and DeepSeek's algorithmic analysis is unique? No blinded lens-quality comparison is proposed.
2. The proposal assumes research-mode users have higher latency tolerance than default-panel users. That's plausible but unstated—what if research summaries are feeding into time-sensitive dashboards?

**Overlooked risks:**
- **Research path SPOF:** Leaving DeepSeek as the default research summarizer while acknowledging 2% failures means async research summaries will also fail ~2% of the time. If research is a load-bearing async path (feeding automated dashboards, reports), this is still a reliability hazard.
- **Breakage detection lag:** Metric-based re-enable assumes you're monitoring 30-day rolling failure rate continuously. If alerting is noisy or the metric is slow to compute, you might not catch improvement for weeks.

---

## Proposal B: KEEP

**Steelman:** If the failure rate is only slightly above the trigger threshold and the root cause is unknown, gathering evidence before acting prevents thrashing. A 2,000-call bake-off of Claude Sonnet 4 under identical conditions is the only way to know if you're trading DeepSeek's 2% failures for Claude's 0.5% failures or Claude's 1.8% failures. Lens quality matters—DeepSeek may be irreplaceably good at cost analysis. Rushing to replace risks losing capability while accepting continued failures.

**My assessment:** Disagree with Position. The proposal is too cautious about production harm and under-weights the operational cost of accepting 2% round failures during a multi-week bake-off.

**Strongest point:** The shadow-testing approach is methodologically sound. Blinded lens-quality scoring and workload stratification (prompt size, timeout, region) are the right way to avoid premature replacement. The proposal correctly notes that 0.5% × 5 models still yields ~2.5% round failures, so replacement alone may not solve the problem.

**Weakest assumption:** That the >2% failure rate is "only slightly above 2%" rather than sustained at 2–3%+. If it's been at 2.5% for two weeks, the bake-off delay costs 50–100 failed rounds. The proposal doesn't quantify acceptable production harm during evidence gathering.

**Hidden assumptions:**
1. The proposal assumes the "review trigger" threshold of 2% is itself evidence-based and not arbitrary. If the threshold was set by incident response (e.g., "we got 10 complaints last week"), then *the threshold itself is the evidence*, and deferring action is asking the team to accept more complaints.
2. The proposal assumes DeepSeek's failures are independent node failures, not correlated outages. If DeepSeek is having systematic auth or rate-limit issues, a 2,000-call bake-off might happen during a stable period and miss the real problem.

**Overlooked risks:**
- **Bake-off timing bias:** Shadow testing during a stable period for DeepSeek gives false confidence. If DeepSeek has periodic (daily? weekly?) outages, a 2-week bake-off might miss them.
- **User erosion:** Accepting 2% failures in production for 2–4 weeks while you gather evidence may exceed users' patience. If those users switch to a single-model tool (e.g., just Claude), they don't come back when you fix it.

---

## Proposal C: REPLACE

**Steelman:** In a strict scatter-gather architecture where any model timeout fails the entire round, the math is unavoidable: 1 − 0.98^5 ≈ 9.6% failed rounds. A single unreliable node corrupts the entire system. o3-mini has proven enterprise SLAs (>99.9%) and is purpose-built for algorithmic reasoning. The cost delta (DeepSeek ~$0.40/1M → o3-mini ~$5–10/1M) is immaterial compared to retry cost and user friction. Replace immediately, keep DeepSeek as opt-in, defer architecture refactoring.

**My assessment:** Partially agree with Position. The math and urgency are correct, but the replacement candidate is overspecified and the proposal doesn't verify that o3-mini actually replicates the Performance lens.

**Strongest point:** The scatter-gather math is unavoidable. If the architecture is truly synchronous and failure at any node fails the round, then 2% per-node failure is intolerable. You cannot retry your way out of this if latency SLAs matter.

**Weakest assumption:** That o3-mini can replicate DeepSeek's "Performance/algorithmic-cost" lens through system prompting. There is no blinded lens-quality evidence. o3-mini is optimized for coding and math; DeepSeek's niche was *cost-tradeoff analysis*. They may not be the same capability.

**Hidden assumptions:**
1. The proposal assumes the current architecture *is* strict scatter-gather with no partial degradation. But that's never stated. What if the framework already times out slowly and retries intelligently? Then o3-mini's higher latency could be worse.
2. The proposal implicitly assumes the 2% failure rate is *per-call* and not *per-user-session*. If it's the latter, and each user makes 3 calls, the failure rate per user is actually ~6%, making replacement even more urgent. But this distinction isn't examined.

**Overlooked risks:**
- **Persona homogenization:** Two OpenAI models (Product + Performance) will exhibit correlated reasoning biases. The "contrarian" value of the council drops if half the panel agrees. The mitigation (strict system prompts) is speculative—you're betting on prompt engineering, not tested independence.
- **Latency regression:** o3-mini's reasoning is slower than DeepSeek. If the current 2% failures are timeouts (not crashes), replacing with a slower model might worsen latency even if the error rate drops.
- **Summarizer is still unaddressed:** Proposal C leaves DeepSeek's 2% failures in the research summarizer, just swaps it to gpt-4o-mini without evidence that this is better.

---

## My Revised Recommendation

**Position: DEMOTE with modifications to address research-path risk.**

**Why I'm updating from a pure Proposal A stance:**

Proposal A is operationally sound but incomplete. Proposal B is too cautious. Proposal C is right on urgency but over-commits to an unvalidated replacement.

Here's what I'd actually do:

1. **Remove DeepSeek from the default panel now** (Proposal A's core move). The 2% failure rate is operationally unacceptable for a synchronous critical path, and the math in Proposal C proves it.

2. **Do NOT keep DeepSeek as the default research summarizer** (deviation from Proposal A). Use `claude-3-5-haiku` as the research summarizer instead. Rationale: research mode is async and forgiving, so you don't need o3-mini's cost; haiku is cheap, proven reliable, and won't introduce a second SPOF.

3. **Keep DeepSeek available via `--models deepseek`** (both proposals agree). Users who explicitly want cost-optimized reasoning can opt in.

4. **Shadow-test Claude Sonnet 4 or Gemini 2.0 in the Performance seat for one week of production traffic** (lighter than Proposal B's 2,000 calls, faster than waiting). If either hits <0.5% failures *and* blinded lens-quality scoring shows Performance reasoning at parity, add it as an alternative default. If not, declare the 4-model default sufficient (Proposal A's position).

5. **Implement auto-re-enable** (Proposal A's metric gate): if DeepSeek's rolling failure rate drops to <0.3% and holds for 7 days, automatically re-add it to the default panel with a log entry. This makes the decision reversible.

**Evidence that would flip this:**
- If the measured 2% is actually <0.5%, this was a false alarm. Revert and tighten alerting.
- If a shadow-test candidate hits >99.5% reliability *and* lens-quality is at parity, add it as an alternative default.

**Why this beats the three proposals:**

| Aspect | Proposal A | Proposal B | Proposal C | My rec |
|--------|-----------|-----------|-----------|---------|
| **Stops user failures immediately** | ✓ | ✗ | ✓ | ✓ |
| **Reversible** | ✓ | ~ | ✗ | ✓ |
| **Addresses research SPOF** | ✗ | ✗ | ~ | ✓ |
| **Validates replacement lens-quality** | ✗ | ✓ | ✗ | ✓ |
| **Avoids vendor concentration** | ✓ | ✓ | ✗ | ✓ |
| **Cost-efficient** | ✓ | ~ | ✗ | ✓ |
| **Time to safety** | Days | Weeks | Days | Days |

**What the on-call engineer faces at 2am:**

- User reports: "Round failed. What do I do?"
- Response: "We removed an unreliable model from the default panel. Use `--models deepseek` if you want it, or wait for a retry. We're testing better options."
- No manual override needed. No false hope that a bake-off is imminent. Clear options.

**The key tension I'm resolving:**

Proposal B wants evidence before acting; Proposal C wants to act before evidence. I'm saying: act *immediately* to stop the bleeding (demotion), but gather evidence *concurrently* about a better default (shadow test). Don't slow down the safety move to gather bake-off data.

*Latency: 68.44s | Tokens: 4206*

### Deepseek (gpt-5.6-sol)

## Proposal A

**Steelman first:** DEMOTE is exactly right if DeepSeek’s >2% rate is sustained, attributable to the provider, measured after reasonable retries, and the remaining default panel still executes a distinct Performance persona. Research must also tolerate retries or fallback without materially harming completion time.

**Your assessment: Disagree.** The proposal violates the load-bearing-persona constraint by shrinking the default to four models and declaring Performance merely “nice-to-have.” That is not demotion with preserved coverage; it is removal of required functionality.

Its reliability arithmetic is also wrong. One panelist with 98% success contributes approximately 2% round failure, not 10%. The `1 - 0.98^5 ≈ 9.6%` calculation applies only if all five panelists independently succeed at 98%. If the other four each succeed at 99.5%, total success is:

`0.98 × 0.995^4 ≈ 96.1%`

That is roughly 3.9% round failure—not 10%, but still poor.

**Strongest point:** Cost should buy DeepSeek no tolerance on a blocking path. One failure per roughly 50 calls is operationally significant.

**Weakest assumption:** It assumes Claude and Grok collectively preserve the Performance lens. If false, demotion improves availability by deleting a required analytical capability.

**Hidden assumptions:**

- Every DeepSeek error necessarily fails the round; retries or partial results cannot contain it.
- Research failures matter less because research is asynchronous. Asynchrony changes latency tolerance, not completion reliability.

**Overlooked risks:**

- Automatically changing default composition at startup based on a rolling metric makes runs non-reproducible and can cause fleet oscillation.
- Thirty-day recovery measurement becomes selection-biased after demotion because opt-in traffic may be much smaller and systematically different.
- The proposed research trigger is stricter than the panel trigger: if research must exceed 99% completion, retaining a provider already failing over 2% needs an explicit retry or fallback budget.

---

## Proposal B

**Steelman first:** KEEP is exactly right if the 2% figure is not yet statistically representative, no candidate has proven both better reliability and equivalent Performance-lens quality on the same workload, and continued exposure is bounded by a short evaluation window with an emergency cutoff.

**Your assessment: Agree.** Crossing a review threshold establishes that a decision is needed; it does not establish which replacement is qualified. B is the only proposal that preserves the required lens while demanding workload-specific evidence before changing the default.

The proposed candidate gate is directionally sound, though its statistics need tightening. With 10 failures in 2,000 calls—an observed 0.5% rate—the exact one-sided 95% upper bound is roughly 0.9%, not an unconditional “near 0.8%.” More importantly, sample size should be derived from the required detection power, traffic strata, and tolerated regression—not chosen as a round number.

**Strongest point:** Provider reputation and global uptime do not predict ai-council’s completion rate for its prompt lengths, concurrency, timeouts, regions, and response-validation rules. A same-workload bake-off does.

**Weakest assumption:** It assumes accepting another 2,000 production DeepSeek calls is tolerable. At 2%, that implies about 40 additional DeepSeek failures before the nominal evaluation completes. If every failure blocks a round, the bake-off itself has a material user cost.

**Hidden assumptions:**

- Shadow calls exercise the same throttling, concurrency, timeout, and quota conditions as an active default seat.
- Blinded human scoring can reliably distinguish Performance-lens quality with acceptable inter-rater agreement.

**Overlooked risks:**

- Failure correlation matters more than independent averages. Provider incidents, local networking, shared orchestrator limits, and regional failures can dominate the tail.
- Aggregate failure rates can hide prompt-size or concurrency cliffs.
- Candidate shadowing may understate production latency because its result is not awaited on the critical path.
- Two consecutive seven-day windows delay action unnecessarily once the sequential evidence is already decisive.

---

## Proposal C

**Steelman first:** REPLACE is exactly right if `o3-mini` is currently supported in the deployment, has materially better measured success and tail latency on representative ai-council calls, matches or beats DeepSeek in blinded Performance scoring, and the replacement summarizer independently passes research-quality and reliability gates.

**Your assessment: Disagree.** The proposal jumps from evidence that DeepSeek is unreliable to an unsupported conclusion that `o3-mini` will deliver below 0.1% failures. It names a candidate but provides no relevant reliability evidence. Generic status history or an infrastructure SLA is not evidence about model-specific throttling, timeouts, malformed output, context limits, or application validation failures.

It also repeats Proposal A’s arithmetic error: a single 98%-reliable dependency does not by itself create `1 - 0.98^5` round failure.

**Strongest point:** It correctly recognizes that retaining DeepSeek as the sole research summarizer leaves the same dependency exposed on another default path. Research reliability must be evaluated separately.

**Weakest assumption:** It treats provider-level availability as a guarantee of application-level success below 0.1%. If that premise fails, the project swaps a measured problem for an unmeasured one while claiming the incident is resolved.

**Hidden assumptions:**

- `o3-mini` and `gpt-4o-mini` are stable, supported choices with suitable context, latency, quota, and output behavior for these workloads.
- Adding a second OpenAI model does not create enough correlated quota or provider-outage risk to offset the per-call reliability gain.

**Overlooked risks:**

- Replacing panelist and summarizer simultaneously destroys attribution: regressions cannot cleanly be assigned to either change.
- Reasoning-model latency could make p95/p99 round latency worse even if completion rate improves.
- Product and Performance seats on one provider can fail together during outages or quota exhaustion.
- Evaluating only 100 research summaries has little power to establish sub-1% failure behavior.
- Strict prompting does not guarantee independent reasoning; shared model lineage can produce correlated blind spots.

---

## Revised recommendation: KEEP

I maintain **KEEP DeepSeek unchanged in the default panel and as the research summarizer for the bounded qualification period**. Proposal B reinforces this position; A and C do not supply a constraint-compliant, evidence-qualified alternative.

This is not indefinite tolerance. Run a representative bake-off against a currently supported candidate in the Performance seat, but do not promote it on vendor reputation.

Replace DeepSeek only when all of these gates are met:

- DeepSeek remains above **2% application-level blocking failures** after retries in two representative windows, with at least 1,000 calls total and no single incident dominating the result.
- The candidate’s one-sided 95% upper confidence bound is below **1%**, and the measured improvement over DeepSeek is statistically credible.
- The candidate passes stratified tests for prompt size, concurrency, region, timeout, and rate-limit conditions.
- Candidate p95 and p99 latency do not materially worsen round latency.
- Blinded Performance-lens scoring shows no material regression in complexity analysis, bottleneck identification, quantitative estimates, or failure-mode analysis.

Keep DeepSeek outright if its sustained blocking rate returns below **1%** and no candidate demonstrates a meaningful end-to-end improvement. Trigger immediate accelerated replacement evaluation at **5% over 200 calls**, correlated outage behavior, or material domination of p99 round latency.

Qualify the research summarizer separately. Do not infer research reliability from panel calls, and do not change both roles in one release. The first thing likely to break is not average API uptime; it is the scatter-gather tail under large prompts and concurrency. Measure that directly.

*Latency: 55.53s | Tokens: 10523*

### Gemini (gemini-3.1-pro-preview)

As a Security & Integration Architect, my primary mandate is to contain blast radius, ensure defense-in-depth, and manage supply chain risk. A synchronous architecture where a single dependency exhibits a >2% blocking failure rate is an active critical incident, as it corrupts the availability of the entire composite system.

Here is my evaluation of the proposals.

---

### --- Proposal A ---

**Steelman first:** This proposal is exactly right if the "Performance" persona is actually an illusion, entirely redundant to the other 4 models, and the system architecture inherently supports a variable-sized panel without failing or degrading the consensus mechanism. If adding *any* 5th model introduces more failure risk than analytical value, pruning the dependency tree is the ultimate blast-radius mitigation.

**Your assessment:** **Disagree with Position.** The proposal explicitly violates a hard constraint of the prompt: *The 5 personas are load-bearing... any change must preserve lens coverage.* Deciding a load-bearing constraint is a "nice-to-have" is a product management pivot, not an architectural solution. 

**Strongest point:** The operational math. Highlighting that a 2% failure rate on one node of a 5-model panel degrades the entire round to ~90% reliability is the exact blast-radius calculation an architect needs to see.

**Weakest assumption:** Assuming that asynchronous tasks (research mode) are immune to failure rate friction. A >2% failure rate on a summarizer still creates poisoned pipelines, dead-letter queues, and operational drag, even if the user isn't waiting synchronously. 

**Hidden assumptions:** 
1. The aggregation layer can cleanly handle a missing 5th input without code changes or degraded synthesis.
2. The 2% failure rate is an API availability issue, rather than an adversarial or malformed-output issue (which could manifest as a security vulnerability rather than just a timeout).

**Overlooked risks:** By keeping DeepSeek as the default research summarizer, the proposal fails to mitigate the dependency risk in the research pipeline. If DeepSeek's API is compromised or starts returning maliciously crafted summaries, the research path remains fully exposed.

---

### --- Proposal B ---

**Steelman first:** This proposal is exactly right if replacing DeepSeek with an untested model introduces silent, unmeasured degradation in analytical quality (a "soft" failure), which is vastly more dangerous to the system's integrity than a loud, measurable 2% API error (a "hard" failure). 

**Your assessment:** **Disagree with Position.** While I respect the rigor of demanding a data-driven bake-off, leaving a structurally failing dependency in the synchronous critical path while gathering 2,000 requests of evidence is architectural malpractice. You do not let a burning building burn just to measure the temperature of the fire.

**Strongest point:** The insistence on blinded performance-lens scoring. It correctly identifies that swapping an LLM without validating its persona capability risks compromising the integrity of the council's output.

**Weakest assumption:** The assumption that a 2% blocking failure rate (aggregating to ~10% system-wide failure) is an acceptable user-facing operational floor during the "evidence-gathering period." 

**Hidden assumptions:** 
1. Telemetry is currently sophisticated enough to cleanly separate "DeepSeek provider errors" from "generic timeout/network errors" in real time.
2. The user base will tolerate a 1-in-10 failure rate for the weeks it takes to complete the bake-off.

**Overlooked risks:** Supply chain contagion. A model exhibiting a sustained 2%+ failure rate may be experiencing severe infrastructure distress or under attack. Continuing to blast it with synchronous production traffic exposes our application to cascading latency bottlenecks (e.g., hanging connections tying up worker threads).

---

### --- Proposal C ---

**Steelman first:** This proposal is exactly right if `o3-mini` perfectly mimics the algorithmic/performance lens, and isolating the system from DeepSeek’s instability is the only way to restore the SLA. It correctly treats API boundaries as zero-trust zones that must be excised if they violate operational contracts.

**Your assessment:** **Partially Agree with Position.** I agree entirely with the disposition (REPLACE) and the immediate excision of DeepSeek from both paths. I am highly cautious about the specific replacement (`o3-mini`), as it introduces dangerous supply chain concentration.

**Strongest point:** "Dependency Isolation." It correctly diagnoses that a synchronous scatter-gather architecture cannot tolerate a 2% point-of-failure. It solves the entire blast-radius problem immediately across both the panel and the research pipeline.

**Weakest assumption:** The assumption that `o3-mini` can be easily prompted out of OpenAI's RLHF'd "house style." Introducing a second OpenAI model risks correlated reasoning biases, degrading the independent variance required for a scatter-gather council. 

**Hidden assumptions:** 
1. `o3-mini` does not have hidden rate limits or latency spikes that behave differently under our specific payload sizes compared to OpenAI's general SLA.
2. The cost increase of `o3-mini` (which is significantly more expensive than DeepSeek) truly will not trigger secondary financial constraints, despite the prompt's assurance.

**Overlooked risks:** Correlated vendor outages. By putting `gpt-4o` (Product), `o3-mini` (Performance), and `gpt-4o-mini` (Research) in the critical path, an OpenAI API outage now takes down 40% of the panel and 100% of the research pipeline simultaneously.

---

### Revised Recommendation

**Position: REPLACE DeepSeek in the default panel and the research path.**

Proposal C made the most architecturally sound case by correctly calculating the composite blast radius (~10% round failure) and refusing to tolerate it. However, Proposal C introduces unacceptable supply chain concentration by stacking OpenAI models. 

**My updated implementation:**
1. **Immediate Action (Critical Path Mitigation):** REPLACE DeepSeek in the Performance seat with Anthropic's **Claude 3.5 Haiku** (or Sonnet, if Haiku's reasoning fails the lens check). REPLACE the research summarizer with Anthropic's Haiku. 
    *   *Why Anthropic over OpenAI?* We currently have Gemini (Security), Grok (Contrarian), OpenAI (Product), and Claude (Systems). Adding a second Anthropic model for Performance is still a vendor concentration risk, but Haiku/Sonnet are demonstrably independent from OpenAI's reasoning biases, preserving council variance better than `o3-mini`.
2. **Opt-in Retention:** Demote DeepSeek to `--models deepseek` only.
3. **Architectural Fix (The Real Problem):** The fact that a single API timeout fails the whole round is a severe architectural flaw. We must implement **partial-round tolerance** (circuit breakers with a strict P95 latency timeout). If a model fails, the round must gracefully degrade to a 4-panel consensus with a warning flag, rather than crashing.

**Signals to Revisit (The Reversibility Metric):**
1. **Restore DeepSeek:** We will auto-restore DeepSeek to the default Performance seat if its API demonstrates a sustained round-blocking failure rate of **<0.5% over a 14-day rolling window**, monitored via a shadow-traffic health check.
2. **Restore strictness:** If blinded telemetry shows Claude 3.5 Haiku cannot maintain the contrarian algorithmic rigor of the Performance lens, we will trigger an emergency evaluation of `o3-mini` vs `llama-3.3-70b` for the seat.

*Latency: 38.53s | Tokens: 8032*

## Synthesis (by openai, non-participant)

## Consensus

- **DeepSeek’s measured reliability is bad enough to warrant action.**
  - All participants treated **>2% blocking failure rate** as a serious issue for a default synchronous panelist.
  - This was driven by **strong shared reasoning**, not mere echoing: each side independently noted that in a scatter-gather path, a flaky dependency disproportionately harms round completion.

- **Cost should not protect DeepSeek’s default seat.**
  - Broad agreement that DeepSeek’s low price is irrelevant here because the stated constraint says **reliability and lens quality dominate cost**.

- **Any permanent change must be evidence-triggered and reversible.**
  - All sides proposed some metric gate for reconsideration.
  - This is strong consensus and well aligned with the prompt constraints.

- **Research summarizer reliability must be considered separately.**
  - Multiple participants correctly noted that panel-seat and research-summary reliability are different paths and should not be conflated.

- **A replacement, if chosen, must preserve the Performance lens, not just swap vendors.**
  - There was broad agreement on this principle, though not on whether demotion without replacement is acceptable.

## Unresolved Disagreements

### 1) Keep vs immediate removal from the default panel
- **Crux:** Is crossing the >2% review threshold enough to remove DeepSeek now, or only enough to trigger qualification of alternatives?
- **Sides:**
  - **KEEP** camp: threshold triggers review, not automatic removal; do not change the default without same-workload evidence that a replacement is both more reliable and preserves the Performance lens.
  - **DEMOTE/REPLACE** camp: a blocking dependency above 2% is already unacceptable on the critical path; stop the user harm first, evaluate alternatives second.
- **Stronger argument:** **Remove now** had the stronger argument.
  - Why: the prompt’s reactive trigger is a **round-blocking failure rate above 2%**, and reliability is explicitly the top concern. The KEEP case was methodologically careful, but it underweighted ongoing production harm and effectively asked users to absorb known failures while evidence is gathered.

### 2) Demote vs replace
- **Crux:** Does removing DeepSeek without an immediate replacement violate the “load-bearing personas” constraint?
- **Sides:**
  - **DEMOTE** camp: remove from default and run a 4-model panel; Performance is useful but can move to opt-in temporarily.
  - **REPLACE** camp: the default full panel must preserve all five lenses; dropping the Performance seat is not constraint-compliant.
- **Stronger argument:** **REPLACE** had the stronger argument on the constraint.
  - Why: the prompt explicitly says the **5 personas are load-bearing** and any change must preserve lens coverage. The demotion proposals mostly asserted that the remaining models could absorb Performance, but they did not show that this preserves an independent Performance persona. On the debate record alone, demotion weakly satisfies the stated constraints.

### 3) Whether a specific replacement candidate is justified now
- **Crux:** Is there enough evidence in the debate to name a replacement model today?
- **Sides:**
  - **Specific replacement** camp: suggested `o3-mini` or Anthropic Haiku/Sonnet based on vendor reliability reputation and rough fit for the persona.
  - **Evidence-first** camp: no candidate has presented representative ai-council workload data proving both reliability and lens parity.
- **Stronger argument:** **Evidence-first** had the stronger argument.
  - Why: the prompt requires reliability evidence that would justify replacement. No participant provided actual same-workload measured evidence for any candidate. Vendor status pages, general SLAs, or model reputation are not enough.

### 4) Research-path handling
- **Crux:** Can DeepSeek remain the summarizer if removed from the panel?
- **Sides:**
  - Some demotion arguments said yes because research is async and more tolerant.
  - Others argued no because that leaves a default-path SPOF in place.
- **Stronger argument:** **Do not leave the research path unexamined** had the stronger argument.
  - Why: async changes latency tolerance, not completion reliability. However, the debate did **not** establish enough evidence to mandate an immediate summarizer swap to any named model.

## Argument Quality Assessment

### Best-reasoned proposals
1. **Claude Round 2 (demote with modifications)** was the best overall synthesis.
   - Strengths:
     - Correctly prioritized stopping production harm.
     - Correctly rejected overconfidence in an unvalidated replacement.
     - Correctly called out the research summarizer as an incompletely addressed risk.
   - Weakness:
     - Its final recommendation still leaned on a temporary 4-model default, which is in tension with the load-bearing persona constraint.

2. **DeepSeek Round 2 (KEEP with qualification gates)** was the strongest methodological argument.
   - Strengths:
     - Best critique of weak replacement evidence.
     - Correctly challenged misuse of reliability arithmetic.
     - Emphasized workload-specific measurement, latency, and failure correlation.
   - Weakness:
     - It accepted too much known operational pain while waiting for better evidence.

### Weaker proposals
- **Gemini Round 1 REPLACE with `o3-mini`** had the right urgency but weak evidentiary support.
  - It relied too heavily on generic provider uptime/SLA claims.
  - It asserted “guaranteed return” to very low failure rates without ai-council-specific evidence.
- **Claude Round 1 DEMOTE to 4 models** was operationally practical but constraint-light.
  - It too readily treated the Performance lens as expendable despite the prompt saying personas are load-bearing.

### Single strongest argument in the debate
- **A known >2% blocking failure rate on a synchronous default dependency is operationally unacceptable, regardless of cost.**
  - This is the clearest, best-supported argument and directly answers the review trigger.

### Single weakest argument in the debate
- **That a specific replacement (`o3-mini`, Haiku, Sonnet, etc.) is justified now based on provider reputation or generic SLA rather than representative ai-council traffic.**
  - This failed the prompt’s evidence standard.

## Blind Spots

1. **The decision options and the persona constraint are partially in tension, and nobody resolved that cleanly.**
   - DEMOTE removes DeepSeek from the default panel but appears to leave only 4 active default panelists, conflicting with “5 personas are load-bearing.”
   - Participants noticed this but did not fully reconcile whether temporary reduction is allowed.

2. **The actual round-level failure decomposition is missing.**
   - The debate repeatedly reasoned from DeepSeek’s individual failure rate, but nobody presented:
     - DeepSeek-attributable failures vs total round failures
     - retry behavior
     - timeout budget effects
     - malformed output vs transport errors
   - This matters because the trigger is “round-blocking failure rate above 2%,” not necessarily “DeepSeek API call failure rate above 2%.”

3. **No one proved that the current architecture truly requires all five responses to succeed.**
   - Many arguments assumed strict scatter-gather with no partial tolerance, but this was not established from evidence in the transcript.

4. **No candidate actually met the replacement bar in the debate.**
   - The discussion named plausible candidates, but no one supplied representative production reliability data plus blinded Performance-lens quality evidence.

5. **Nobody quantified the value of the Performance lens in current user outcomes.**
   - The debate treated it as either load-bearing by fiat or expendable in practice, but did not cite user evidence, ticket patterns, or judged output quality deltas.

## Recommended Decision

**Decision: DEMOTE DeepSeek from the DEFAULT full panel, keep it opt-in, and keep its research summarizer role only behind an explicit temporary reliability review with fallback work.**

Why **DEMOTE** rather than KEEP:
- The strongest evidence in the debate is that a **known >2% blocking failure rate is too high for a default synchronous dependency**.
- KEEP was the best argument on replacement rigor, but not strong enough to justify continued default exposure.

Why **DEMOTE** rather than REPLACE:
- The prompt requires an evidence-based decision, and **no replacement candidate was actually justified with representative reliability evidence**.
- REPLACE is conceptually attractive and probably the eventual destination, but the debate did not meet its own bar for naming a replacement.

How I weigh the strongest objection:
- The best objection to DEMOTE is that it may violate the “load-bearing personas” constraint by removing the default Performance seat.
- I still recommend DEMOTE because:
  1. The review trigger is about **active operational failure in the default path**.
  2. The replacement proposals failed to provide admissible evidence for a specific successor.
  3. The escape hatch says **KEEP is the null option if evidence does not justify a change**, but here the evidence **does justify removing the failing dependency from the default critical path** even if it does not justify promoting a successor.
- This means the decision is slightly imperfect against lens coverage, but less imperfect than either:
  - keeping a known unreliable default dependency, or
  - installing an unproven replacement by vibes.

### Research-path impact
- The debate did not justify an immediate permanent summarizer replacement.
- Therefore:
  - **Do not treat DeepSeek as a trusted long-term default summarizer without safeguards.**
  - Keep it temporarily for research summarization **only if** retries/fallback are added immediately and research completion metrics are tracked separately.
  - If those safeguards cannot be added quickly, remove it from the default research path as well.

### Flip-back / reversibility metric
Re-enable DeepSeek to the default panel only if all of the following hold:
- **30-day rolling application-level blocking failure rate <0.5%**
- sustained for **14 consecutive days**
- over at least **1,000 representative default-path calls**
- with no material p95/p99 latency regression
- and no evidence that failures are merely hidden by atypical low-volume opt-in traffic

Promote a replacement into the Performance seat only if a candidate shows:
- **measured blocking failure rate ≤0.75%**
- **one-sided 95% upper confidence bound <1%**
- parity in blinded Performance-lens evaluation on representative prompts

## Risks

1. **Loss of default Performance-lens coverage**
   - Risk: default panel quality regresses on algorithmic-cost/performance analysis.
   - Mitigation: make DeepSeek opt-in easy; run an expedited bake-off for a replacement; track user complaints specifically about missing Performance analysis.

2. **Research summarizer remains a latent reliability problem**
   - Risk: demotion fixes panel rounds but leaves research mode exposed.
   - Mitigation: add retries, fallback summarizer, and separate research completion telemetry immediately.

3. **False positive demotion**
   - Risk: the >2% was a transient incident or instrumentation bug.
   - Mitigation: verify with segmented telemetry before finalizing; if corrected data shows <0.5%, revert.

4. **Temporary default-panel contract ambiguity**
   - Risk: “full panel” semantics become unclear if default no longer has a dedicated Performance seat.
   - Mitigation: document this as a temporary evidence-based reliability exception and prioritize qualification of a replacement.

5. **Biased reinstatement metrics after demotion**
   - Risk: opt-in traffic differs from default traffic, making DeepSeek look healthier than it is.
   - Mitigation: use shadow traffic or representative synthetic probes for reinstatement eligibility.

## Action Items

1. **Demote DeepSeek from the default full panel immediately.**
   - Keep available via `--models deepseek`.

2. **Add/verify telemetry before making further model changes.**
   - Split failures into:
     - provider errors
     - timeouts
     - malformed/validation failures
     - orchestrator/network failures
   - Track separately for panel path and research summarizer path.

3. **Stabilize the research path now.**
   - Add retries and a fallback summarizer.
   - If fallback is not already available, qualify one urgently under the same evidence standard.

4. **Run an expedited replacement bake-off for the Performance seat.**
   - Candidates may include `o3-mini`, Claude Haiku/Sonnet, or others already supported.
   - Use representative prompts, concurrency, regions, and timeout settings.
   - Judge both reliability and blinded Performance-lens quality.

5. **Define explicit reinstatement criteria in code/config, not ad hoc judgment.**
   - DeepSeek return-to-default gate: <0.5% blocking failure, 14 days, 1,000+ representative calls, no latency cliff.

6. **Document the temporary exception clearly.**
   - Explain that DeepSeek was removed from default for reliability, not quality/cost reasons.
   - Note that the Performance lens is temporarily opt-in pending qualification of a reliable default seat.

7. **Prioritize architectural follow-up.**
   - Implement partial-round tolerance / graceful degradation so a single panelist cannot block the whole round in the future. This was not the decision asked for, but it is the most important structural mitigation the debate surfaced.
