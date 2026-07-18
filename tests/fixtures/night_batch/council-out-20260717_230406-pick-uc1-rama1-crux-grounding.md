# AI Council Debate: # Decision: How should ai-council ground empirical cruxes inside a decision-mode

**Date:** 2026-07-17 23:04:06
**Panel:** claude-haiku-4-5-20251001, gpt-5.6-sol, gemini-3.1-pro-preview
**Synthesizer:** openai (non-participant)
**Rounds:** 2
**Duration:** 216.5s
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** C:\Users\1028120\Documents\Dev\ai-council\council_inbox\uc1-rama1-crux-grounding.md
**Cost:** ~$0.1240 (35,062 tokens)

## Verdict Summary

**Decision:** Adopt option (c): a discrete, bounded crux-check step between rounds, implemented as a centralized pipeline stage that emits one canonical evidence artifact for all Round-2 prompts.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 3/3
**Verdict author:** openai (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

# Decision: How should ai-council ground empirical cruxes inside a decision-mode debate? (Rama 1 / #18)

## Context
ai-council runs 2-round debates across a fixed panel to produce ADRs. Decision modes (pick/judge) currently have ZERO retrieval — empirical sub-claims that arise mid-debate ("does provider X support prompt caching?", "does option A collide with the response refeed at debate.py:227?") are settled by the most confident voice, not by evidence (architecture audit finding G6; persuasion != truth). The council can be eloquently, confidently wrong on a checkable fact. This is the highest-value identified gap — the "keystone" on which the debate-literature, the internal audit, and fusion practice all converge.

## The decision
Pick ONE mechanism to ground a flagged empirical crux before Round-2 voting.

## Options
- (a) Reuse the existing research-mode retrieval pool — let a decision-mode debater ground a flagged sub-claim through the same provider path research mode already uses.
- (b) Use a CLI agent as a tool-grounded panelist that can actually RUN the check (read the cited file, run a command) rather than assert an answer — couples to the now-shipped CLI-seat backend.
- (c) Insert a discrete, bounded "crux-check" step BETWEEN rounds: mechanically identify <=3 flagged empirical claims, resolve them with source-attributed evidence, inject the result identically into every Round-2 prompt.

## Constraints (these eliminate weak options)
- Rounds are capped at <=2 and MUST stay capped; grounding is a bounded step, not a new round and not open-ended research inside a debate.
- Blind voting (ADR-03) is inviolable: the label->provider map must never reach the resolver, and any injected evidence must be identical across all Round-2 prompts (no panelist gets private evidence).
- Cost gate is ~$0.50/debate operator judgment; added latency/cost must be justified.
- Determinism + the ephemeral-brief / fire-and-forget inbox I/O model must be preserved.
- "Memory-only" answers (a model asserting from training with no source) must count as UNRESOLVED, not as grounding.

## Prior decisions in scope
ADR-03 (blind voting), ADR-12 (CLI backend now shipped), the reserved ADR-13 (bounded crux-check draft), DRAFT-EPI-3. Escape hatch: recommend "keep decision modes retrieval-free" if the tension is not worth resolving at v1.

## Round 1: Initial Responses

### Claude (claude-haiku-4-5-20251001)

## Position
Recommend **option (c): discrete crux-check step between rounds**. It's the only mechanism that preserves blind voting, caps scope, enforces source grounding, and gives on-call engineers a deterministic audit trail.

## Alternatives Considered

**Option (a) – Reuse research-mode retrieval:** Tempting for reuse, but it folds retrieval back into debate (not a bounded transformation), doesn't address memory-only assertions, and creates precedent for mid-debate information access that will erode your 2-round cap over time. "Just add one more search" becomes a slide.

**Option (b) – CLI agent panelist:** Solves the execution/verification problem at the cost of coupling a new actor type into the debate fabric. Creates operational questions (how does this agent appear in Round-2 prompts? how does its context stay deterministic? does tool execution latency blow the budget?). Harder to audit than a single preprocessing pass.

## Decision Criteria

1. **Blind voting preservation** – evidence must be identical for all panelists, label→provider map never exposed.
2. **Round ceiling** – must stay ≤2 rounds; no creep via "one more search" or "one more verification."
3. **Determinism** – audit trail must be reproducible; fire-and-forget I/O model must hold.
4. **Source grounding** – memory-only assertions must not count; evidence must be traceable.
5. **Cost + latency** – must fit within ~$0.50 budget and not explode Round-2 wait time.

## Assumptions

1. **"Crux-check is not a round"** – identifying and resolving ≤3 claims between Round-1 and Round-2 is a bounded preprocessing step, not a new debate turn. This preserves the 2-round cap in letter and spirit.
2. **Flagging is automatic or lightweight** – you can identify empirical cruxes from Round-1 debate text without human review (regex, NLP signal, or panelist flags that get deduplicated).
3. **Source resolution is deterministic** – the same crux (same wording, same context) yields the same evidence every time, or you can cache the resolution.
4. **Panelists won't escalate**  – allowing crux-check won't trigger requests for "can we also ground assumptions?" or "can we add a live Q&A round?" (cultural/process boundary holds).

## Recommendation

**Implement a discrete crux-check phase that runs between Round-1 conclusion and Round-2 prompt assembly:**

- **Now:** Mechanically identify up to 3 empirical sub-claims flagged during Round-1 (either by panelists explicitly marking them as cruxes or by heuristics on claim density). Require each claim to name a concrete target: a specific provider capability, a code location, a config constraint, etc.
- **Now:** For each flagged claim, issue a single bounded source-grounding query (e.g., "does provider X document prompt caching support?" → consult provider docs/code; "does this response refeed happen at line 227?" → grep and return snippet). Require source attribution; reject any memory-only answers.
- **Now:** Inject resolved evidence as a single context block identical into every panelist's Round-2 prompt. Format: `[EMPIRICAL GROUNDING - Round 1 Cruxes] [Claim 1] Evidence: [source]. [Claim 2] Evidence: [source].`
- **Not now:** Don't expose the resolution process to panelists; don't make crux-check interactive or allow re-opening of cruxes mid-Round-2.
- **Defer:** Additional retrieval loops, panelist-specific evidence, or open-ended research inside the debate.

## Rationale

- **Blind voting held**: Evidence block is identical for all panelists; label→provider remains hidden. No panelist gets private information. The resolver sees all cruxes and evidence simultaneously (no selective revelation).
- **2-round cap preserved**: Crux-check is preprocessing, not a round. It's atomic: happens once, result is locked. No ambiguity about "did we just add a round?"
- **Determinism guaranteed**: The same set of cruxes always produces the same evidence block (or you cache it). Fire-and-forget I/O model holds: one input (Round-1 debate), one output (amended Round-2 prompt), no back-channel state.
- **Memory-only is rejected**: Source-attribution requirement means "I think X based on training" doesn't ground a crux; you force a tool to produce a snippet or document reference. Cost is one focused lookup per crux, not open-ended research.
- **Bounded cost + latency**: Capping cruxes at 3 and requiring concrete targets keeps this to ~1-2 lookups, ~$0.05–0.15 total. Latency is a single serial pass; doesn't slow down Round-2 voting (happens between rounds).

## Trade-offs

- **Less flexibility**: You can't resolve "soft" empirical claims ("is this architecture elegant?"). You're limited to checkable facts. But that's the point—soft claims belong in debate; crux-check is for empirical closure only.
- **Panelist skill required**: Panelists must articulate cruxes clearly (name a target). If they say "I'm not sure if X is true" without saying what X is, it won't be flagged. Training/tooling burden.
- **Potential for gaming**: A panelist could deliberately obscure a crux to avoid grounding. Mitigate: make crux-check transparent (publish the evidence and flagged claims in the ADR), create social pressure to name claims explicitly.
- **Latency addition**: Even bounded crux-check adds 30–60s between rounds. Acceptable if cost cap is met; flag if total debate latency becomes >10 min.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Crux flagging heuristic is noisy (flags non-empirical claims) | Medium | Wastes resolution budget, confuses Round-2 prompt | Require explicit panelist flag + keyword whitelist (provider, config, line, version, API). Audit first 10 crux-checks manually. |
| Source lookup returns nothing (evidence doesn't exist) | Low–Med | Claim stays UNRESOLVED in prompt; panelists may guess anyway | Explicitly mark as unresolved: `[Claim X] Evidence: NOT FOUND [target]`. Treat as "debate continues on assumption" signal. |
| Panelists ignore grounding in Round-2 (anchor on pre-check beliefs) | Medium | Defeats the purpose; most confident voice still wins | Monitor: compare Round-1 vs Round-2 claims. If panelist contradicts injected evidence, flag in ADR. Cost-justified only if evidence actually shifts votes. |
| Determinism breaks if tool execution is non-deterministic | Low | Panelists see different evidence → blind voting violated | Snapshot all tool outputs at crux-check time; version the evidence. If provider API changes mid-debate, freeze resolution at Round-1 close time. |

## Signals to Revisit

- **If >30% of cruxes remain unresolved:** The "source grounding" requirement is too strict, or targets are too vague. Consider allowing lightweight retrieval-pool fallback for unresolved claims.
- **If panelists routinely ignore grounded evidence in Round-2:** Indicates the crux-check is an exercise in box-checking, not actual belief-updating. Abandon or redesign; may mean the debate framing itself is the problem.
- **If cost exceeds $0.20/debate or latency >2 min:** Crux flagging is too aggressive or source lookup is inefficient. Tighten the cap to 1–2 cruxes only.
- **If the CLI backend's tool execution becomes reliable and cheap** (ADR-12 followup): Revisit option (b) as a hybrid—use (c) for primary grounding, (b) as fallback for "run this specific command." But only if it preserves blind voting.
- **If a single crux resolution cascades to >5 downstream debates:** The crux was foundational and re-using it (caching) is justified; no re-evaluation needed.

*Latency: 59.74s | Tokens: 3282*

### Deepseek (gpt-5.6-sol)

## Position

Choose **(c): a discrete, bounded crux-check between rounds**. It is the only mechanism that grounds evidence symmetrically, preserves blind voting, and places hard limits on latency and cost.

## Alternatives Considered

- **(a) Reuse research-mode retrieval inside a debater.** Rejected because evidence acquisition becomes panelist-dependent and potentially private; it also turns debate turns into open-ended research with variable cost and latency.
- **(b) Make a CLI agent a tool-grounded panelist.** Rejected because one panelist gains asymmetric access and influence. It is also incomplete: CLI execution handles repository facts well but cannot reliably verify external provider capabilities.

## Decision Criteria

- Identical evidence must reach every Round-2 voter without exposing the label-to-provider mapping.
- Grounding must be source-attributed; unsupported model recollection is `UNRESOLVED`.
- The debate remains exactly two rounds, with no recursive or open-ended investigation.
- Incremental cost should ordinarily remain below the approximately **$0.50/debate** operator gate.
- Runtime must remain bounded and compatible with deterministic, ephemeral, fire-and-forget I/O.

## Assumptions

- Most debates produce no more than three decision-relevant, empirically checkable cruxes.
- Existing retrieval and CLI-seat capabilities can be invoked without giving the resolver panel identities.
- Local checks usually complete within 10–30 seconds, while external retrieval usually completes within 15–60 seconds.
- A structured unresolved result is more useful than forcing a weak or memory-based answer.

## Recommendation

Implement ADR-13 as a deterministic pipeline stage after Round 1 and before Round 2:

1. Extract candidate empirical claims from the complete, anonymized Round-1 transcript.
2. Rank them by expected decision impact and checkability, then retain at most three.
3. Resolve each through a fixed routing policy:
   - Repository, configuration, or runtime claims → sandboxed CLI check.
   - Provider, API, or published capability claims → existing research retrieval path.
4. Require each resolved claim to contain the normalized claim, verdict, evidence excerpt or command result, source identifier, retrieval timestamp, and confidence.
5. Mark claims `UNRESOLVED` when no attributable source is obtained, sources conflict, or the check exceeds its budget.
6. Serialize one canonical evidence block and inject its exact bytes into every Round-2 prompt.
7. Persist only the bounded evidence artifact required for the ADR and existing ephemeral-brief flow.

Apply hard limits: three claims, one primary check plus at most one corroborating check per claim, approximately 30 seconds per local check, 60 seconds per external check, 90 seconds wall-clock overall, and an incremental budget around **$0.15–$0.25** by default with a hard ceiling below **$0.50**.

Do not allow panelists to initiate retrieval independently, expose resolver provenance tied to panel labels, recursively generate new cruxes from evidence, or interpret command output without retaining the underlying result.

Defer semantic deduplication beyond simple normalization, adaptive budgets, cross-debate evidence caching, parallel multi-source verification, and automatic fact-checking of every assertion until production telemetry shows they are necessary.

## Rationale

- **Symmetry is enforced structurally:** one canonical artifact is constructed before prompt fan-out, making evidence drift between voters mechanically testable rather than dependent on prompting discipline.
- **The first scaling failure is external retrieval tail latency:** three sequential 30–60 second checks could add 90–180 seconds. Running bounded checks concurrently caps expected added wall time near the slowest check, approximately 30–90 seconds.
- **Cost is predictable:** at three claims and no more than two checks each, resolver work has a fixed upper bound. Typical debates with one or two cruxes should add roughly **$0.05–$0.20**, below the operator gate.
- **Routing avoids false universality:** CLI execution is authoritative for local code facts, while source retrieval is appropriate for provider claims. The crux-check stage governs both without pretending either resolver covers all empirical questions.
- **Explicit `UNRESOLVED` prevents fabricated certainty:** timeouts and missing citations remain visible uncertainty rather than being converted into persuasive but unsupported facts.

## Trade-offs

- The pipeline gains another orchestration state, schema, timeout path, and test surface.
- Decision-mode latency likely increases by **20–90 seconds at p95** when checks are triggered; debates without flagged cruxes should incur only extraction overhead.
- Limiting checks to three can leave lower-ranked claims unresolved.
- A retrieved source can still be stale or misleading; this mechanism establishes attributable evidence, not absolute truth.
- Deterministic routing sacrifices potentially better but unbounded exploratory research.

## Risks

- **Crux selection misses the decisive claim — medium likelihood, high impact.** Mitigate by scoring both decision sensitivity and factual checkability, and expose omitted candidates in diagnostics without injecting them as evidence.
- **Resolver output leaks panel identity — low likelihood, critical impact.** Mitigate by accepting only the anonymized transcript, prohibiting provider-map access, and testing serialized prompts for byte-identical evidence.
- **Commands have side effects or escape scope — low likelihood, high impact.** Mitigate with a read-only sandbox, an allowlist of commands, repository-scoped paths, and strict time/output limits.
- **External retrieval dominates tail latency — medium likelihood, medium impact.** Mitigate through concurrent execution, per-check deadlines, an overall 90-second deadline, and graceful `UNRESOLVED` results.
- **Evidence injection anchors all voters on a bad source — low-to-medium likelihood, high impact.** Require primary or authoritative sources where available, preserve conflicting results, and avoid collapsing disagreement into a definitive verdict.
- **Token-heavy evidence exceeds the cost gate — medium likelihood, medium impact.** Cap excerpts and command output, deduplicate sources, and abort additional checks when the incremental budget is exhausted.

## Signals to Revisit

- More than **20%** of debates produce over three high-impact empirical cruxes.
- Crux checking pushes total debate cost above **$0.50 at p95** or adds more than **90 seconds at p95**.
- More than **30%** of selected claims remain unresolved, indicating inadequate routing or source access.
- Post-decision audits show grounded verdicts are wrong or stale in more than **5%** of checked claims.
- CLI-addressable claims dominate and external claims are rare enough that a specialized tool-grounded resolver becomes simpler than maintaining both paths.
- Evidence changes fewer than **2–5%** of outcomes over a meaningful sample, suggesting the operational cost is not justified.

*Latency: 41.76s | Tokens: 7160*

### Gemini (gemini-3.1-pro-preview)

## Position
I strongly recommend Option (c) — inserting a discrete, bounded "crux-check" step between rounds. This approach strictly contains the blast radius of external integrations, enforces the information symmetry required by blind voting (ADR-03), and avoids the catastrophic security risks of unconstrained command execution.

## Alternatives Considered
- **Option (b) Use a CLI agent as a tool-grounded panelist:** Rejected. Granting an LLM panelist arbitrary command execution or file-system read access (even if containerized) based on adversarial or untrusted debate input is a massive attack surface. It introduces severe supply-chain and command injection risks (RCE by design) that violate fundamental defense-in-depth principles.
- **Option (a) Reuse research-mode retrieval pool via individual debaters:** Rejected. Decentralized, debater-initiated retrieval breaks the guarantee of identical, symmetric evidence injection required by ADR-03. It also multiplies the API boundary risk by the number of panelists, complicating auditability and cost controls.

## Decision Criteria
1. **Blast Radius & Containment:** The impact if the integration is manipulated by prompt injection or unexpected input.
2. **Information Symmetry:** Must strictly preserve ADR-03 (blind voting) by guaranteeing identical context across all panelists.
3. **Determinism & Boundary Enforcement:** Must respect the 2-round cap and the ~$0.50/debate budget without risking infinite tool-use loops.
4. **Verifiability:** Must successfully filter out "memory-only" assertions in favor of attributed, source-backed evidence.

## Assumptions
- The system can reliably extract <=3 empirical claims from Round 1 text using a cheap, fast LLM call without blowing the $0.50 margin.
- A bounded, safe, read-only lookup mechanism (e.g., semantic search over local workspace or approved external docs) exists to support the crux-check without requiring arbitrary code execution.
- The Round 2 prompts have sufficient token headroom to absorb the injected evidence block without truncating critical debate history.

## Recommendation
- **What to do now:** Implement Option (c). Build an isolated "crux extractor" component that runs immediately after Round 1. It must parse the R1 transcripts, select up to 3 testable factual disagreements, execute a strictly read-only, source-attributed lookup, and prepend the identical findings (or an explicit "UNRESOLVED" tag if no source is found) to all Round 2 prompts. 
- **What not to do now:** Do not expose the newly shipped CLI-seat backend (ADR-12) to the automated decision-mode panel. Do not allow individual panelists to trigger their own tool calls.
- **What to defer until later:** Complex consensus or voting algorithms to determine what constitutes a "crux." Start with a simple, low-temperature prompt to extract the most frequent empirical collisions from R1.

## Rationale
- **Strictly Controlled API Boundary:** Centralizing retrieval into a single, discrete step between rounds restricts the external dependency to one controlled chokepoint. This drastically reduces the attack surface compared to empowering N individual panelists with tooling.
- **Guaranteed Blind Voting Integrity:** By injecting the exact same evidence block into every R2 prompt simultaneously, Option (c) structurally prevents panelists from deducing identities based on distinct retrieval behaviors, private evidence, or tool-use latency, fully satisfying ADR-03.
- **Mitigation of Command Injection:** Option (b) introduces a fundamentally uncontainable risk. An untrusted prompt manipulating the CLI agent could expose the host environment or internal repositories. Option (c) relies on strictly bounded, read-only lookups over a defined corpus.
- **Cost and Execution Bounding:** Capping extractions at <=3 and executing them as a single parallel batch keeps the operation deterministic. It prevents the sprawling token costs associated with LLMs iterating on failed tool calls mid-debate.

## Trade-offs
- **Depth of Investigation:** We give up the ability for an agent to iteratively dig into a codebase (Option b) to chase down deep architectural dependencies, favoring strict security and speed over exhaustive research.
- **Pipeline Latency:** Adding a sequential step (Extract -> Search -> Inject) between R1 and R2 will add a fixed latency penalty (likely 5-10 seconds) to the overall pipeline.

## Risks
- **Risk 1: Extractor prompt injection or hallucination.** (Likelihood: Medium, Impact: Moderate). A malicious ADR input could trick the extractor into querying for sensitive internal terms. *Mitigation:* Sanitize extractor inputs, use a rigid output schema, and enforce the rule that any extracted claim returning no explicit, safe source is strictly tagged UNRESOLVED.
- **Risk 2: Source corpus pollution.** (Likelihood: Low, Impact: High). If the underlying source used for crux-checking is compromised, all panelists receive identical poisoned data. *Mitigation:* Restrict the search corpus to verified documentation and immutable, read-only source code. Disable general web search entirely.

## Signals to Revisit
- The extraction step fails to find relevant cruxes or returns UNRESOLVED >30% of the time, making the added step dead weight and unjustifiable under the $0.50 budget constraint.
- The latency of the discrete crux-check step routinely exceeds user tolerance, breaking the fire-and-forget I/O model.
- Security isolation of the CLI backend (ADR-12) reaches a high maturity level (e.g., ephemeral, network-isolated sandboxes with strictly typed outputs and zero access to host secrets), at which point Option (b) might become an acceptable risk.

*Latency: 28.39s | Tokens: 3992*

## Round 2: Critique

### Claude (claude-haiku-4-5-20251001)

# Steelman, Assess, and Revise

## PROPOSAL A

**Steelman First:**

For Proposal A to be exactly right, you'd need:
- Cruxes to be rare (≤3 per debate), clearly articulable (panelists name concrete targets), and resolvable in single deterministic lookups.
- Evidence to actually shift votes (measured by audit).
- Panelists to accept bounded crux-check without escalating requests for related capabilities (soft-claim grounding, iterative refinement, Q&A).

If these hold, option (c) is *perfect*: it avoids mission creep by design, enforces blind voting structurally, caps cost/latency, and gives on-call a reproducible audit trail. The framing of crux-check as **preprocessing, not a round** is the load-bearing move—it makes the 2-round cap defensible against future feature creep.

**My Assessment: Agree, with one reservation.**

The decision criteria are well-chosen. The tradeoffs are honestly stated. The recommendation is implementable. I particularly like the refusal to soft-pedal "less flexibility."

**Strongest Point:**

The explicit framing of crux-check as *preprocessing, not a round*. This is the critical boundary. By making the input/output contract clear (input: R1 debate, output: amended R2 prompt), you make it harder for future requests to masquerade as "just preprocessing." The proposal knows this is a cultural boundary and calls it out; most proposals bury that assumption.

**Weakest Assumption:**

**"Panelists won't escalate — allowing crux-check won't trigger requests for 'can we also ground assumptions?'"**

This is fragile. Once crux-check ships and panelists see it working, they *will* ask for related capabilities. The proposal acknowledges this risk but treats it as a matter of willpower ("cultural/process boundary holds"), not structure. Willpower fails. You need a *process rule*: e.g., "Any request to expand crux-check goes back through the decision council; quick escalation is prohibited." Otherwise, the line will erode.

**Hidden Assumptions:**

1. **Panelists can articulate empirical cruxes without training.** The proposal assumes "automatic or lightweight" flagging, but panelists rarely say "Does provider X support prompt caching?" They say "I'm worried about latency" or "The architecture feels brittle." Training is implied but not budgeted.

2. **"Source grounding" resolves conflict symmetrically.** The proposal requires source attribution (good), but doesn't address what happens when sources *conflict* or are *stale*. If provider docs say "feature Y is available" but the implementation changed last month, the mechanism says "resolved" when it's actually *misresolved*. You need a follow-up rule about source staleness and conflicts.

**Overlooked Risks:**

1. **Crux selection can be adversarial.** A panelist who disagrees with the likely grounding can flag cruxes they know will resolve in their favor, or obscure cruxes they fear will undermine them. The proposal mitigates with "transparency + social pressure"—weaker than structural mitigation (e.g., panelist flags stay anonymous; a neutral extractor selects top-3 by decision impact, not by panelist request).

2. **"Memory-only is rejected" gains formality, not truth.** If the resolver tries to ground "Does provider X support feature Y?" and the docs are unclear but the implementation obviously does, the mechanism returns `UNRESOLVED`. The panelist knows this from training and will mentally override it. You've formalized the process without increasing confidence in the outcome.

---

## PROPOSAL B

**Steelman First:**

For Proposal B's operational specifics to be exactly right:
- Crux extraction ranks by decision impact (not just frequency); ≤3 captures the relevant empirical disagreements.
- Routing policy (repo → CLI, external → retrieval) is approximately optimal for cost (CLI ~5–10x cheaper).
- Local checks are enforceably read-only and complete in 30s.
- External retrieval is bounded to 60s without tail-latency surprises.
- 90s wall-clock latency is acceptable (doesn't break fire-and-forget I/O).
- Cost concentrates around $0.15–0.25, not the $0.50 ceiling.

If those hold, Proposal B's spec is precise and defensible.

**My Assessment: Partially agree.**

I agree with the routing policy (different evidence sources warrant different tools) and explicit budgets (good operational hygiene). I *partially* disagree with 90s being acceptable—that depends on the operator's tolerance, which the proposal doesn't validate against actual SLAs.

**Strongest Point:**

**"The first scaling failure is external retrieval tail latency."**

This is sophisticated. Proposal B recognizes that cost is asymmetric (local ~$0.01, external ~$0.10–0.20) and tail latency is the constraint, not mean. Suggesting concurrent execution is sharp. This shows production experience—you're not guessing; you're naming the actual bottleneck.

**Weakest Assumption:**

**"Existing retrieval and CLI-seat capabilities can be invoked without giving the resolver panel identities."**

This is operationally fragile. If a crux is "Does provider X support Y?", the resolver *learns which provider is under debate*. If you execute multiple cruxes—some provider-specific, some generic—panelists who receive correlated evidence can start deducing the mapping. Proposal B doesn't address this leakage. ADR-03 (blind voting) says "label→provider map must never reach the resolver"; this proposal skirts the edge.

**Hidden Assumptions:**

1. **Crux extraction scales with debate heat.** The proposal assumes ≤3 cruxes capture the decision-relevant empirical disagreement. But in adversarial debates, panelists will disagree on more facts. The proposal silently drops lower-ranked claims. You're gambling those drops don't matter—unvalidated.

2. **"Deterministic routing" is actually deterministic.** The proposal says "fixed routing policy," but a claim like "Provider X's latency in the cloud" could be resolvable via CLI (run a benchmark) or retrieval (published benchmarks). The proposal needs explicit conflict resolution. Otherwise, determinism is illusory.

**Overlooked Risks:**

1. **Concurrent execution breaks audit trails.** The proposal recommends parallel checks for latency, but concurrent execution means the resolved evidence is timing-dependent. If one check finishes early and influences another's filtering, you lose determinism. You need to *snapshot* all results at a wall-clock deadline, not report as they arrive.

2. **"Graceful UNRESOLVED" is a trap when frequent.** If >30% of cruxes remain unresolved (acknowledged as a revisit signal), crux-check is expensive infrastructure for a weak signal. The proposal doesn't guide what to do *while* operating above that threshold—keep running it, or fall back to debate-mode retrieval? That's a gap.

---

## PROPOSAL C

**Steelman First:**

For option (c) to be security-sound (and option b to be correctly rejected):
- CLI execution in a decision-mode panel is inherently risky (prompt injection is hard to defend).
- Sandboxing CLI execution is possible but expensive/complex (violates cost constraint).
- Read-only lookup is sufficient to ground most empirical cruxes.
- The blast radius of compromised crux-check (identical evidence injected into all R2 prompts) is smaller than compromised CLI panelist (arbitrary code execution).

If those hold, Proposal C is correct to reject option (b).

**My Assessment: Strongly agree with the security stance; partially agree with the implementation.**

Proposal C is *right* to reject option (b). Giving an LLM control over CLI execution in the context of adversarial debate input is a well-known attack surface (prompt injection → RCE). This is the hardest truth in the council.

However, Proposal C's implementation of option (c) is underdeveloped. It says "bounded, read-only lookup mechanism" without specifying what that is. If it's semantic search over local workspace, it's fast but can hallucinate. If it's API calls to external docs, it's more reliable but slower and still has injection risk (the query could be adversarial). The proposal defers these details ("start with a simple, low-temperature prompt"), which is exactly where security vulnerabilities hide.

**Strongest Point:**

**"An untrusted prompt manipulating the CLI agent could expose the host environment or internal repositories."**

This names the actual risk. Option (b) assumes debate input is trustworthy enough to drive code execution. In security, that's a false assumption. By comparison, option (c) filters debate input through a lightweight extractor before executing anything—much safer. Proposal C deserves credit for naming the zero-trust problem explicitly.

**Weakest Assumption:**

**"A bounded, safe, read-only lookup mechanism... exists to support the crux-check without requiring arbitrary code execution."**

This is deferring the hard part. What does "bounded" mean? If you search a corpus, how large? If you make API calls, which APIs are allowed? The proposal says "start with a simple prompt" — but that's where injection happens. You're hand-waving the security boundary, not drawing it.

**Hidden Assumptions:**

1. **"Read-only" is a security boundary.** Proposal C assumes read-only access is safe. But read-only access to *all* source code (including secrets, config, keys) leaks sensitive information. You need to *constrain what's readable*, not just that it's read-only. The proposal doesn't address corpus scope.

2. **The extractor itself is trustworthy.** Proposal C recommends "a simple, low-temperature prompt to extract collisions from R1." But this prompt reads the debate transcript and outputs a structured list of cruxes. If the debate is adversarial, the extractor could be manipulated to output malicious queries (e.g., designed to leak information or DoS the retrieval system). The proposal doesn't address extractor robustness.

**Overlooked Risks:**

1. **Extractor is an LLM-driven code path.** Proposal C defers "complex consensus, automatic fact-checking" to later, but it draws a line: "Do not allow individual panelists to trigger their own tool calls." That line assumes the extractor (also LLM-driven) is safe. It's safer than giving each panelist a tool, but it's still LLM-driven parsing of adversarial input. That gap needs closure.

---

## REVISED RECOMMENDATION

**I agree with the destination (option c) and disagree with all three proposals on exactly which assumptions are solid.**

All three recommend (c); they differ in what they've hardened and what they've deferred.

**What I'd integrate:**
- **Proposal A's framing:** Crux-check is preprocessing, not a round. Enshrine this as a hard boundary.
- **Proposal B's operations:** Routing (repo → CLI, external → retrieval). Explicit budgets. Determinism as non-negotiable.
- **Proposal C's security:** CLI execution out-of-band. Extractor threat model defined before implementation.

**What I'd reject:**

1. **Proposal A's "panelists won't escalate"** → Replace with: "Any request to expand crux-check goes through the decision council. No quick escalation."

2. **Proposal B's concurrent execution for latency** → Serial execution is required for determinism. Parallelize only if tail latency exceeds measured tolerance. Snapshot all results at wall-clock deadline.

3. **Proposal C's deferred extractor safety** → You can't defer "what is safe about the extractor?" The extractor reads the debate and outputs crux queries. If the debate is adversarial, the extractor is an LLM-driven code path processing untrusted input. Define the threat model before implementation: extractor must be low-temperature, deterministic, schema-validated, and prohibited from executing external APIs.

**Final Position:**

Implement option (c) with these hard constraints:

| Constraint | Why |
|---|---|
| **Atomic boundary** | Crux-check runs once, between R1 close and R2 prompt assembly. No re-opening, no recursion. |
| **Routing** | Repo-local → CLI (read-only, allowlisted commands). External → retrieval (whitelisted sources, no general web search). |
| **Budgets** | ≤3 cruxes, ≤1 primary check per crux. 30s per local, 60s per external, 90s wall-clock. ~$0.20 cost (leave $0.30 headroom under operator gate). |
| **Extractor** | Separate frozen model (not panelist model). Low-temperature. Output schema enforced before downstream execution. No external API calls from extractor. |
| **Evidence injection** | One canonical evidence block. Identical bytes into all R2 prompts. Serialization determinism is non-negotiable. |
| **Audit trail** | Every invocation logs: R1 excerpt, extracted claims, routing decisions, resolved evidence, source, confidence. Core artifact for post-hoc audit. |
| **Escalation gate** | Requests to expand crux-check (soft claims, iterative refinement, panelist-initiated retrieval) require decision council review. No quick escalation. |

**Strongest unresolved problem shared by all three proposals:**

All assume cruxes are factual and rare. In practice, adversarial debates produce cruxes like "Is this maintainable?" or "Does this degrade gracefully?" — design judgments, not empirical facts. The mechanism would mark these `UNRESOLVED`. But panelists will still debate them in R2 without grounding. This is not a problem with option (c); it's a problem with the debate framing itself. Worth monitoring: if >30% of cruxes are unresolvable by design, the mechanism is overhead, not value.

*Latency: 113.36s | Tokens: 7788*

### Deepseek (gpt-5.6-sol)

## Proposal A

**Steelman first:** Proposal A is exactly right if explicit crux flags reliably identify the few decision-changing facts, each fact can be resolved with one authoritative lookup, and resolution can be frozen into a canonical artifact. Under those conditions, a single between-round stage gives symmetric evidence with little orchestration complexity.

**Your assessment:** **Agree with the Position, disagree with important implementation claims.** Option (c) is correct, but A does not make its latency, cost, or determinism case rigorously enough. Three serial checks at 30–60 seconds each imply roughly 90–180 seconds of added wall time, not a harmless “single serial pass.” Source retrieval is not deterministic merely because it is cached afterward; caching makes one debate internally consistent, not reproducible.

**Strongest point:** It draws the correct architectural boundary: empirical verification is a bounded pipeline transformation, not a privilege exercised independently by panelists.

**Weakest assumption:** “The same crux yields the same evidence every time.” Search rankings, documentation, APIs, and model-generated synthesis all change. Without captured source content, tool inputs, versions, timestamps, and raw outputs, replay will not reproduce the result. The claimed deterministic audit trail then collapses into an unrepeatable assertion with a URL attached.

**Hidden assumptions:**

- One lookup is sufficient to determine a verdict, including when sources conflict or documentation is stale.
- Panelists can flag cruxes clearly enough that a lightweight extractor will not spend the three-item budget on merely easy-to-check claims instead of decision-sensitive ones.

**Overlooked risks:**

- Selection is the likely first correctness failure. With a cap of three, even a 20% miss rate on the decisive claim can dominate the value of perfect downstream retrieval.
- A citation can support adjacent text without supporting the normalized claim. Source presence alone is not grounding.
- Serial resolution unnecessarily multiplies tail latency.
- Injected snippets can increase prompt length enough to displace Round-1 context.
- “NOT FOUND” conflates absence of evidence, timeout, access failure, and contradictory evidence; those require distinct statuses.

---

## Proposal B

**Steelman first:** Proposal B is exactly right if the extractor can rank claims by decision sensitivity, the resolver routes each claim to an authoritative substrate, all checks run under enforced budgets, and the resulting canonical artifact is byte-identically injected. It then turns an unbounded epistemic problem into at most three concurrent jobs with explicit timeout and cost ceilings.

**Your assessment:** **Agree.** This is the strongest proposal because it specifies an actual bounded system rather than merely naming option (c). It correctly separates orchestration from resolution: the crux-check is the mechanism, while CLI inspection and retrieval are resolver backends selected by claim type.

The cost estimates are still hypotheses, not capacity planning. A token/check budget and enforced cancellation are necessary; “at most two checks” does not by itself guarantee a dollar ceiling.

**Strongest point:** The canonical evidence artifact makes symmetry mechanically testable. Exact-byte injection is a much stronger invariant than asking multiple prompt builders to include equivalent evidence.

**Weakest assumption:** The system can automatically select the most decision-relevant claims. Checkability and importance are different rankings. An extractor will tend to favor concrete claims such as line numbers while missing broader factual premises that actually flip the decision. If selection fails, the pipeline produces high-quality evidence about the wrong questions.

**Hidden assumptions:**

- Existing retrieval and CLI interfaces can be invoked as bounded, non-interactive functions. A CLI seat designed for agent interaction may not naturally provide one-command, typed, cancellable execution.
- Concurrent checks have independent bottlenecks. If they share a provider rate limit, browser pool, repository lock, or subprocess capacity, concurrency shifts latency into throttling rather than removing it.

**Overlooked risks:**

- A command’s successful execution does not make its interpretation authoritative. Environment-dependent tests can produce false verdicts due to missing dependencies, platform differences, stale checkouts, or configuration.
- Evidence can contain prompt injection, terminal control characters, secrets, or oversized output. Canonicalization must treat resolver output as untrusted data.
- An overall deadline needs cancellation semantics. Otherwise timed-out jobs continue consuming tokens or processes after the debate has moved on.
- Confidence is likely to become fabricated precision unless derived from explicit evidence classes.
- Persisting timestamps weakens replay determinism unless the distinction between immutable artifact determinism and fresh rerun reproducibility is explicit.

---

## Proposal C

**Steelman first:** Proposal C is exactly right if the threat model includes adversarial debate inputs, the CLI backend cannot be strongly sandboxed, all useful cruxes can be answered from a curated read-only corpus, and those lookups reliably finish in 5–10 seconds. In that environment, a centralized read-only resolver minimizes both security exposure and latency variance.

**Your assessment:** **Partially agree.** The Position—option (c)—is correct. The proposed resolver restriction is not. Refusing CLI-backed verification discards the most authoritative mechanism for local code claims, including the example about `debate.py:227`. A read-only, allowlisted CLI check can be a backend inside option (c) without making a CLI agent a panelist.

The security argument also overstates option (b) as necessarily “arbitrary command execution” and “fundamentally uncontainable.” ADR-12 already shipped a CLI backend; the relevant question is its capability envelope, not whether command execution is intrinsically impossible to contain.

**Strongest point:** It recognizes that evidence itself is an attack surface. Centralizing resolution creates one chokepoint where corpus scope, output schema, sanitization, and access policy can be enforced.

**Weakest assumption:** The claimed 5–10 second latency. The proposed flow requires extraction, up to three searches, evidence validation, and serialization. External lookups commonly have multi-second individual latency and heavy tails; a single retry or rate-limit event can push p95 beyond 30–60 seconds. Designing around the optimistic median will violate the operational budget under load.

**Hidden assumptions:**

- A curated corpus is sufficiently complete and current to answer provider-capability questions.
- “Read-only” implies safe. Reads can still exfiltrate secrets, expose proprietary code through generated evidence, or trigger side effects through poorly designed tools and virtual filesystems.

**Overlooked risks:**

- Disabling general web access makes recent provider facts systematically stale or unresolved.
- Source-corpus poisoning is not necessarily low likelihood if repository documents, issue text, or retrieved pages contain untrusted instructions.
- Prompt sanitization cannot reliably neutralize semantic prompt injection; resolver outputs need structural isolation and provenance validation.
- The proposal gives no hard wall-clock, token, output-size, or dollar limits despite claiming bounded execution.
- It ignores selection bias, contradictory sources, timeout cancellation, and context-window pressure.

---

## Revised recommendation

Choose **option (c), implemented substantially as Proposal B specifies**. The other proposals do not change that position; they reinforce it. A correctly designed crux-check is an orchestration boundary, not a single resolver technology.

The first thing likely to break is **crux selection quality**, not retrieval capacity. At only three claims per debate, one misranked candidate consumes 33% of the verification budget. The second failure is **external-retrieval p95 latency**: sequential execution reaches roughly 90–180 seconds for three 30–60 second checks. The third is **false authority**—a cited but non-entailing or environment-specific result being presented as settled fact.

Implement these non-negotiable constraints:

- Accept only the anonymized Round-1 transcript; the resolver must have no provider-map access.
- Select at most three claims using both decision sensitivity and checkability. Preserve the omitted candidate list for diagnostics.
- Route local code/configuration claims to read-only, allowlisted CLI checks and external capability claims to authoritative-source retrieval.
- Run checks concurrently under an overall 90-second deadline, with actual cancellation.
- Enforce a token/check ledger and a hard incremental ceiling below $0.50; stop launching corroboration before the ceiling is threatened.
- Emit typed outcomes: `SUPPORTED`, `REFUTED`, `CONFLICTING`, `NO_SOURCE`, `TIMED_OUT`, and `CHECK_FAILED`. Only the first two are resolved.
- Store normalized claim, exact source or command, bounded raw evidence, environment/version, timestamp, and resolver status. Do not use free-form confidence scores.
- Sanitize and size-limit all evidence as untrusted input.
- Build one canonical artifact, hash it, and inject its exact bytes into every Round-2 prompt.
- Measure p50/p95 latency, incremental cost, unresolved rate, decisive-crux miss rate, evidence correction rate, and outcome-change rate.

Do not ship based on the proposals’ cost estimates. Run a shadow evaluation first. The mechanism earns production use only if it stays below the cost gate at p95 and demonstrably catches decision-changing factual errors.

*Latency: 51.31s | Tokens: 11340*

### Gemini (gemini-3.1-pro-preview)

### Evaluation of Proposal A

**Steelman first:** Proposal A is exactly right if empirical cruxes in architectural debates are discrete, easily identifiable facts (e.g., "Does AWS support X?") that can be accurately extracted by simple heuristics and safely piped into a lookup tool. In this world, an automated preprocessing step flawlessly maintains pipeline speed, budget, and perfectly preserves the blind voting mechanism.

**Your assessment:** **Partially Agree.** I agree with the selection of Option (c) and the insistence on preserving the blind voting architecture. However, I disagree with the assumption that this pipeline is safe out of the box. Proposal A treats the transition from "extracted sub-claim" to "source-grounding query" (e.g., `grep`) as a benign data transformation. From a security perspective, passing an un-sanitized, LLM-generated string into a shell command or external API is a direct injection vector.

**Strongest point:** The structural preservation of blind voting. By insisting that the crux-check is an atomic preprocessing step that outputs a single, identical context block injected into all Round-2 prompts, Proposal A entirely eliminates the risk of asymmetric information leaks that would break ADR-03.

**Weakest assumption:** "Flagging is automatic or lightweight" using regex or NLP heuristics. Assuming you can accurately extract a highly contextual architectural crux via simple regex without either dropping the context or capturing malicious/malformed inputs is extremely naive and will lead to broken, vulnerable queries.

**Hidden assumptions:** 1. The source lookup mechanism inherently sanitizes inputs (e.g., passing a hallucinated string into `grep` won't trigger shell injection). 2. The external corpus being queried is trusted, immutable, and free from poisoned data.

**Overlooked risks:** Command/API injection via the crux extraction phase. If an adversarial or hallucinating debater injects a shell escape sequence or malformed payload into a factual claim, and A blindly pipes it into a lookup tool, the pipeline is compromised. 

---

### Evaluation of Proposal B

**Steelman first:** Proposal B is exactly right if our CLI sandbox is perfectly isolated, completely stateless, and mathematically secure against all forms of escape or side-channel data exfiltration. If true, using a routed approach—sending code claims to a CLI runner and provider claims to a web fetcher—provides the most accurate, high-fidelity grounding possible without manual human intervention.

**Your assessment:** **Disagree with the implementation (while agreeing on Option c).** Proposal B correctly identifies Option (c) as the only structurally sound path, but then it sneaks Option (b)'s most dangerous element—automated CLI execution—into the intermediate pipeline step. Bridging untrusted, LLM-generated debate output into a sandboxed CLI runner without human-in-the-loop validation is a massive, unacceptable expansion of the attack surface. 

**Strongest point:** The emphasis on bounding tail latency. Recognizing that external retrieval is the first scaling failure and mandating concurrent execution with a strict 90-second wall-clock cap is excellent systems engineering.

**Weakest assumption:** "Mitigate with a read-only sandbox." This assumes a read-only sandbox effectively mitigates supply chain and exfiltration risks. Historically, sandboxes executing untrusted commands are porous. If the CLI runs `npm install` or `python -c` based on a hallucinated crux, it can still pull malicious code into the container or leak read-only intellectual property via network side-channels (e.g., DNS exfiltration).

**Hidden assumptions:** 1. The routing LLM is robust against prompt injection designed to force arbitrary web data into the local CLI runner or vice versa. 2. "Anonymized transcripts" strip malicious payloads, not just PII/labels.

**Overlooked risks:** Automated triggering of local vulnerabilities. B uses the CLI runner blindly on extracted claims. Even a read-only sandbox can be weaponized to exhaust resources (CPU/memory DoS via recursive grep/zip bombs) or exfiltrate environment variables if the CLI tool itself has a vulnerability.

---

### Evaluation of Proposal C

**Steelman first:** Proposal C is exactly right if the environment treats all LLM outputs as highly adversarial and untrusted. In this reality, granting any automated agent dynamic execution capabilities guarantees an eventual Remote Code Execution (RCE) or supply chain compromise. The absolute limitation to purely static, read-only document search is the only way to operate securely.

**Your assessment:** **Agree.** Proposal C speaks the language of defense-in-depth. Option (c) is fundamentally an API boundary design choice. By centralizing the retrieval into one discrete chokepoint and stripping away arbitrary execution capabilities, C successfully contains the blast radius of the integration. 

**Strongest point:** Explicitly identifying that giving an LLM arbitrary command execution (Option b, or Proposal B's hybrid) based on unverified debate input is "RCE by design." It prioritizes structural containment over exhaustive investigation.

**Weakest assumption:** That a strictly "read-only, source-attributed lookup" (like semantic search) will actually be capable of resolving architectural cruxes. Resolving a debate over code execution flow (e.g., "does line 227 trigger a refeed?") often inherently requires static analysis or grep-like tooling, which semantic search struggles with. If the lookup fails too often, the feature becomes dead weight.

**Hidden assumptions:** 1. The budget of $0.50 allows for the infrastructure overhead of standing up an isolated, secure extractor service. 2. The Round 1 text isn't so overwhelmingly complex that the extractor LLM gets confused and routinely times out.

**Overlooked risks:** Denial of Service (DoS) via prompt complexity. An attacker could flood R1 with hundreds of synthetic "empirical claims." The extractor component could spend massive compute attempting to parse and isolate these, leading to pipeline timeouts and systemic degradation of the decision mode.

---

### Revised Recommendation

**Update position:** I maintain the core selection of **Option (c)**, strongly aligned with Proposal C's security-first posture, but I am updating the implementation requirements to directly block the vulnerabilities exposed in Proposal A and B. 

**Rationale for the Update & Required Architecture:**
Proposal A was naive to injection risks; Proposal B recklessly attempted to automate CLI execution on untrusted data. I cannot authorize any architecture that bridges an automated LLM extraction step into dynamic command execution. The blast radius of an unconstrained CLI environment processing adversarial strings is a critical failure waiting to happen. 

We will implement the discrete crux-check step (Option c) with the following mandatory security constraints:

1. **Strictly Static Retrieval (No CLI Execution):** The automated crux-checker is barred from accessing the ADR-12 CLI-seat backend. All source grounding must occur via strictly typed, static API calls to verified documentation or bounded semantic search over immutable code indices. No shell execution. No `grep`. No dynamically interpreted languages.
2. **Typed Extraction Boundaries:** The "crux extractor" must output a rigid JSON schema, not natural language queries. Inputs to the lookup integration must be strictly parameterized (e.g., `{"provider": "AWS", "service": "S3", "feature": "caching"}`). Any extracted claim failing this schema validation is instantly dropped.
3. **Network Isolation:** The lookup service must operate in a zero-trust, egress-filtered environment. It can only talk to a whitelisted set of read-only endpoints (internal docs, specific provider APIs). Total ban on general web browsing.
4. **Context Injection Safety:** The returned evidence block injected into Round 2 must be treated as untrusted text. It must be wrapped in explicit delimiter tokens (e.g., `<<<EVIDENCE_START>>>...<<<EVIDENCE_END>>>`) to prevent prompt injection attacks where a polluted external source tries to hijack the Round 2 voting instructions.

We do not split the difference on security. If a factual claim requires dynamic code execution to prove, it remains `UNRESOLVED` in an automated pipeline. The risk of RCE and supply chain poisoning outweighs the benefit of winning an automated architectural debate.

*Latency: 39.59s | Tokens: 8951*

## Synthesis (by openai, non-participant)

## Consensus

- **Choose option (c): a discrete, bounded crux-check between Round 1 and Round 2.**
  - This was unanimous.
  - The consensus appears **substantive, not merely herd behavior**: participants converged from different angles:
    - architecture/process integrity: preserve 2-round structure and blind voting
    - operations: bound cost/latency and create a canonical evidence artifact
    - security: avoid giving debaters private retrieval or unconstrained tool use

- **Evidence injected into Round 2 must be identical for all panelists.**
  - Strongly grounded in ADR-03 and the stated blind-voting constraint.
  - This was one of the most solidly reasoned points in the debate.

- **Memory-only answers do not count as grounding.**
  - Also a strong consensus, directly tied to the motivating failure mode.

- **Grounding must be bounded.**
  - Most participants endorsed limits like:
    - max 3 claims
    - fixed timeout / cost budget
    - no recursive “one more check”
    - unresolved remains unresolved

- **Option (a) alone is insufficient.**
  - Broad agreement that “let a debater retrieve” risks asymmetric evidence, scope creep, and open-endedness.

## Unresolved Disagreements

### 1) Should the crux-check be allowed to use CLI execution for repo/local facts?

- **Crux:** Is a read-only, allowlisted CLI backend sufficiently safe and deterministic to use as a resolver inside option (c), or is any automated CLI execution on debate-derived inputs too risky?
- **Sides:**
  - **Pro-CLI-inside-(c):** Best argued by deepseek and partly by claude R2. Repo/code claims are often best resolved by directly reading files or running bounded local checks; pure retrieval is weaker for claims like “what happens at debate.py:227?”
  - **Anti-CLI:** Best argued by gemini. Debate text is adversarial/untrusted input; bridging LLM extraction to execution expands attack surface materially. If security isolation is immature, this is an unjustified risk.

- **Stronger side:** **Pro-CLI-inside-(c), narrowly.**
  - Why:
    - The decision is not between “CLI panelist” and “no CLI anywhere”; it is between resolver backends inside a centralized crux-check.
    - The strongest anti-CLI arguments often targeted **option (b)** as a panelist with agentic execution, which is a stronger capability than a **non-agentic, allowlisted, read-only local check** inside option (c).
    - The motivating examples explicitly include repository-local questions; forbidding local inspection would weaken the mechanism on a key use case.
  - But this win is conditional: only if implemented as a **strict backend function**, not as an autonomous CLI panelist.

### 2) Parallel vs serial crux resolution

- **Crux:** Is concurrency worth the determinism/audit complexity?
- **Sides:**
  - **Parallel:** deepseek argued tail latency is the practical bottleneck; bounded concurrent checks keep p95 acceptable.
  - **Serial:** claude R2 favored serial for determinism unless measurement proves otherwise.

- **Stronger side:** **Parallel, with canonical snapshotting and cancellation.**
  - Why:
    - Determinism does not require serial execution; it requires a deterministic artifact creation rule.
    - If all results are normalized into one canonical artifact at a fixed deadline, concurrency does not inherently violate blind voting or reproducibility.
    - Given the cost/latency constraint, the operational argument for concurrency was stronger than the abstract caution for serialism.

### 3) How much to rely on automatic extraction/ranking of cruxes?

- **Crux:** Can the system reliably identify the right <=3 empirical claims from Round 1?
- **Sides:**
  - Most accepted extraction as feasible with heuristics/LLM assistance.
  - Several participants also flagged selection as the likely first failure mode.

- **Stronger side:** **Skeptical caution.**
  - Why:
    - Multiple strong critiques noted that “checkable” and “decision-relevant” are different.
    - A perfect resolver on the wrong claims is low value.
    - No participant provided real evidence that extraction/ranking quality is already good enough.

## Argument Quality Assessment

### Best-reasoned proposals

1. **deepseek Round 1 / Round 2**
   - Strongest overall on system design.
   - Good because it:
     - separated the **mechanism** (option c) from the **resolver backends**
     - specified hard budgets, typed outcomes, and a canonical artifact
     - explicitly handled `UNRESOLVED`
     - identified latency tail risk and cancellation needs
     - noted that source attribution alone does not guarantee entailment

2. **claude Round 1 / Round 2**
   - Strong on architectural boundaries and governance.
   - Best contribution was framing crux-check as **preprocessing, not a round**, and emphasizing anti-scope-creep guardrails.
   - Weaker where it assumed stronger determinism than the tooling reality supports.

3. **gemini**
   - Strongest on security threat modeling.
   - Valuable corrective against naive “just run grep/commands from extracted claims.”
   - Weaker because it sometimes treated all CLI use as equivalent to arbitrary agentic execution, which overstates the case and blurs the distinction between option (b) and a tightly bounded backend inside option (c).

### Weaker assumptions / misses

- **“Source grounding is deterministic if cached”** — weak.
  - Caching can freeze one run; it does not make retrieval inherently reproducible or authoritative.
- **“Simple extraction/heuristics will find the right cruxes”** — weak and insufficiently defended.
- **“Read-only means safe”** — weak.
  - Read-only can still leak secrets, create prompt injection paths, or consume resources.
- **“No CLI at all”** — too absolute given the concrete need to inspect local code facts.

### Single strongest argument in the debate

**The strongest argument was that empirical grounding must be centralized into one canonical, identical evidence artifact injected into every Round-2 prompt, rather than delegated to panelists.**

Why it matters:
- It directly solves the motivating failure mode.
- It preserves blind voting structurally.
- It keeps grounding bounded and auditable.
- It cleanly distinguishes decision-mode debates from open-ended research.

### Single weakest argument in the debate

**The weakest recurring argument was the implicit claim that a lightweight extractor can reliably pick the top <=3 decision-critical empirical cruxes without much additional machinery or evidence.**

Why weak:
- It is central to the mechanism’s value.
- Several participants themselves noted it is likely the first failure mode.
- No one offered empirical support that this works well in practice.

## Blind Spots

The debate missed or underexplored several important issues:

1. **Who is the resolver, architecturally?**
   - The debate discussed tools and stages, but not enough about the trust boundary of the resolver service itself:
     - Does it have provider-map access?
     - Where does anonymization happen?
     - How is byte-identical prompt injection verified?
   - This matters for ADR-03 compliance.

2. **What counts as an authoritative source by claim type?**
   - The debate noted docs/code/commands but did not fully define precedence rules:
     - docs vs code vs tests
     - official docs vs third-party docs
     - stale docs vs observed local behavior
   - Without this, “grounding” may still produce misleading certainty.

3. **Prompt/context budget pressure**
   - Mentioned, but not seriously designed around.
   - Adding evidence can crowd out debate context, especially if snippets are verbose.

4. **Outcome evaluation design**
   - Participants suggested telemetry, but no one clearly defined the success metric hierarchy:
     - fewer factual errors?
     - changed votes?
     - audit-detected corrections?
     - operator trust?
   - This matters because vote-change alone may understate value.

5. **Fallback behavior when extraction is low-confidence**
   - The debate mostly assumed the system should always attempt extraction when a debate happens.
   - It did not fully explore a “no safe/selectable crux found → skip crux-check entirely” path.

6. **How to prevent evidence overreach**
   - No one fully addressed the risk that injected evidence causes panelists to overweight a narrow fact and underweight broader architectural reasoning.

## Recommended Decision

**Adopt option (c): a discrete, bounded crux-check step between rounds, implemented as a centralized pipeline stage that emits one canonical evidence artifact for all Round-2 prompts.**

### Rationale

This best satisfies the hard constraints:

- **Preserves 2-round cap**
  - It is a bounded preprocessing stage, not a new debate round.
- **Preserves blind voting**
  - One identical artifact for all panelists; no panelist-specific retrieval.
- **Fixes the identified failure mode**
  - Checkable factual disputes stop being settled by confidence alone.
- **Supports determinism/auditability better than alternatives**
  - Better than panelist-initiated retrieval or a tool-using panelist.
- **Keeps cost and latency governable**
  - Only if hard caps are enforced.

### Implementation choice within option (c)

Use a **routed resolver** inside the crux-check:

- **Repo/local code/config claims** → bounded, read-only, allowlisted local inspection backend
- **External provider/API capability claims** → bounded retrieval against authoritative whitelisted sources

This is materially stronger than:
- pure research retrieval for all claims, which is weak for local code facts
- a CLI panelist, which creates asymmetry and unnecessary agentic surface area

### Why not option (a)?

Because retrieval by a debater or through debate turns:
- risks asymmetric evidence
- invites scope creep toward open-ended mid-debate research
- is harder to cost-cap and audit
- does not itself enforce a shared artifact

### Why not option (b)?

Because making a CLI agent a **panelist** is the wrong shape:
- asymmetric tool access
- more coupling to debate fabric
- more identity/leakage complexity
- larger security and determinism surface

### Strongest objections and why they were outweighed

- **Security objection to any CLI use:** serious and valid, but outweighed because the examples include repo-local checks that retrieval alone cannot resolve well. The mitigation is not “ignore security”; it is “do not make CLI agentic, and constrain it to a typed, allowlisted backend.”
- **Selection-quality objection:** also valid and not fully solved. But it argues for staged rollout and telemetry, not for staying retrieval-free, because the current failure mode is already known and high-value to fix.

## Risks

1. **Wrong cruxes selected**
   - Risk: the system grounds easy facts instead of decision-flipping ones.
   - Mitigation:
     - rank by both decision impact and checkability
     - preserve omitted candidates in logs
     - shadow-evaluate selection quality before broad rollout

2. **Security issues from local inspection backend**
   - Risk: injection, exfiltration, or resource abuse.
   - Mitigation:
     - no autonomous shell
     - typed requests only
     - allowlisted read-only operations
     - path restrictions
     - no network for local checks
     - output sanitization and truncation
     - hard time/process limits

3. **Evidence appears authoritative when it is stale or non-entailing**
   - Risk: bad source anchors all panelists.
   - Mitigation:
     - typed statuses: `SUPPORTED`, `REFUTED`, `CONFLICTING`, `NO_SOURCE`, `TIMED_OUT`, `CHECK_FAILED`
     - prefer primary sources
     - preserve source identifiers and excerpts
     - avoid synthesized overclaiming

4. **Latency/cost exceed budget**
   - Risk: debates become too slow or expensive.
   - Mitigation:
     - max 3 claims
     - strict overall deadline
     - concurrency with cancellation
     - token/output caps
     - hard spend ceiling below operator gate

5. **Prompt-context crowding**
   - Risk: evidence displaces important debate content.
   - Mitigation:
     - compact evidence schema
     - excerpt limits
     - only inject normalized claim + verdict + minimal source snippet

6. **Panelists ignore evidence**
   - Risk: process cost without epistemic benefit.
   - Mitigation:
     - audit contradictions between Round 2 text and evidence artifact
     - measure outcome-change and factual-correction rates

## Action Items

1. **Decide and document ADR-13**
   - State that crux-check is a **single bounded preprocessing stage**, not a round.
   - Reaffirm identical evidence injection and no panelist-private retrieval.

2. **Define the evidence artifact schema**
   - For each selected claim include:
     - normalized claim
     - resolver route
     - status: `SUPPORTED | REFUTED | CONFLICTING | NO_SOURCE | TIMED_OUT | CHECK_FAILED`
     - minimal evidence excerpt/result
     - source identifier / file path / command spec
     - timestamp / environment version
   - Hash the final artifact.

3. **Implement claim selection conservatively**
   - Extract candidate empirical claims from anonymized Round 1 transcript.
   - Rank by:
     - decision sensitivity
     - empirical checkability
   - Cap at 3.
   - Log omitted candidates for evaluation.

4. **Implement two bounded resolver backends**
   - **Local/backend for repo facts:** typed, read-only, allowlisted inspection only.
   - **External/backend for provider facts:** whitelisted authoritative retrieval only; no general web search initially.

5. **Enforce hard execution limits**
   - Overall wall-clock deadline
   - per-check timeouts
   - cancellation semantics
   - output size caps
   - incremental spend cap below $0.50

6. **Sanitize all resolver outputs**
   - Treat evidence as untrusted text.
   - Strip control characters.
   - Delimit clearly in prompts.
   - Prevent prompt injection via raw retrieved text.

7. **Inject exact-byte-identical evidence into all Round-2 prompts**
   - Add automated verification in tests and runtime assertions.

8. **Run a shadow-mode evaluation before full rollout**
   - Measure:
     - selection accuracy
     - unresolved rate
     - latency p50/p95
     - added cost p50/p95
     - evidence correction rate
     - outcome-change rate
   - Do not rely on estimated cost/latency alone.

9. **Set revisit thresholds**
   - Reassess if:
     - unresolved rate stays high
     - p95 cost exceeds gate
     - p95 latency becomes unacceptable
     - selection misses decisive cruxes often
     - local checks dominate enough to justify specialized simplification

10. **Create a governance rule against scope creep**
    - Any expansion beyond bounded crux-check requires explicit council review.
    - No ad hoc panelist-initiated retrieval or iterative follow-up in v1.
