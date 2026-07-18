# AI Council Debate: What determines which context a handoff bundle carries - a fixed set every time,

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** cli

## Verdict Summary

**Decision:** Implement a fixed operational core on every handoff, with the variable payload selected by human judgment based on the declared next action. Do not build an automated field-exclusion manifest.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 2/2
**Verdict author:** gemini (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

What determines which context a handoff bundle carries - a fixed set every time, or one selected for the work the next session will do?

## Round 1: Initial Responses

### Claude (claude-opus-4-8)

## Position

Use a **fixed core bundle plus a small, explicitly-typed variable section** — not a fully fixed set, and not a freely selected one. The core carries invariants every session needs; the variable section carries task-scoped context chosen by declared rules, not ad-hoc judgment.

## Alternatives Considered

**Fixed set every time.** Rejected because a static bundle either bloats to cover every possible next-session need (high cognitive load, most of it irrelevant) or omits task-specific context that the receiving session actually requires. At 2am, an engineer digging through 40 fields of mostly-irrelevant state to find the 3 that matter is a failure mode.

**Fully selected per next-session work.** Rejected because "selected for the work" implies a selector that must correctly predict what the next session needs. When that predictor is wrong, context silently disappears — and the receiving session has no way to know what was dropped. Selection logic also becomes an unbounded, hard-to-test surface. This is the clever-abstraction trap: it optimizes the common case and creates an invisible failure mode in the tail.

## Decision Criteria

1. **Recoverability at handoff:** can the receiving session detect missing context, not just consume present context?
2. **Cognitive load on the reader:** how much irrelevant state must be parsed to act.
3. **Testability of the bundling logic:** can we assert what a bundle contains?
4. **Failure visibility:** silent omission is worse than explicit "not included."
5. **Cost of carrying vs. cost of re-deriving:** context that's cheap to regenerate need not be carried.

## Assumptions

- Handoffs happen frequently enough that per-handoff bundling cost matters, but not so rarely that carrying everything is trivially cheap.
- The receiving session can, in principle, re-fetch or re-derive some context if it knows what's missing (i.e., there's a source of truth beyond the bundle).
- "Context" includes both stable identity/config (rarely changes) and volatile task state (changes every session). These have genuinely different lifecycles.
- A malformed or incomplete bundle causes real harm (wrong action, stalled work), not just inconvenience.

## Recommendation

**Do now:**
- Define a **fixed core schema**: identity, provenance (who/what produced this bundle, when, from what version), invariants, and a **manifest** of what task-scoped keys are included and which were deliberately excluded.
- Define a **typed variable section** governed by explicit inclusion rules keyed to task type — a lookup table, not a heuristic. Task type X pulls fields {a, b, c}; this is data-driven and testable.
- Make **omission explicit**: the manifest records "field Z: not included, reason: not required for task type X" so the receiver can detect and re-fetch.

**Do not do now:**
- Do not build a predictive/learned selector that infers what the next session "probably" needs.
- Do not collapse core and variable into one flat bag "for simplicity" — the differing lifecycles will force a split later at higher cost.

**Defer:**
- Compression/optimization of bundle size until you have measured actual bundle sizes and observed load pressure.
- Dynamic per-task selection rules beyond a coarse task-type table until you see repeated evidence the table is too coarse.

## Rationale

- **The core/variable split maps to lifecycle, not convenience.** Identity and invariants change on a scale of releases; task state changes every session. Bundling them separately means the stable part is trivially cacheable and the volatile part is the only thing under scrutiny during debugging.
- **The manifest converts silent omission into a detectable event.** This is the single highest-leverage decision: a receiver that knows "field Z was excluded" can recover; a receiver that just sees Z's absence cannot distinguish "excluded" from "lost." At 2am this is the difference between a 5-minute re-fetch and a 2-hour root-cause hunt.
- **Rule-table selection is testable; heuristic selection is not.** A task-type→fields table can be unit-tested exhaustively (N task types × expected fields). A predictive selector requires you to test the world.
- **Provenance in the core makes handoff chains debuggable.** When session 3 misbehaves, you can trace which producer/version emitted the bad bundle. Without provenance, every handoff is a fresh mystery.
- **Bounded cognitive load:** the reader always sees a known-size core plus a manifest telling them exactly what task-scoped data to expect. No open-ended parsing.

## Trade-offs

- **You carry some context you don't strictly need** (the fixed core) on every handoff — bundle overhead is higher than a perfectly-selected minimal bundle, plausibly by whatever the core's size is. This is deliberate: predictability over minimalism.
- **The rule table needs maintenance** as task types evolve. This is explicit, visible work rather than an invisible model that drifts.
- **You give up "smart" adaptivity** — the bundle won't automatically learn to include a field the receiver kept needing. You'll find that by observing manifest-driven re-fetches and updating the table manually.

## Risks

- **Rule table goes stale / task types outgrow the coarse buckets** — *Likelihood: medium, Impact: medium.* Mitigate by logging every time a receiver re-fetches an excluded field; a rising rate for a given task type signals the table needs a new entry.
- **Core schema becomes a dumping ground** ("just add it to core, it's easier") — *Likelihood: high, Impact: medium.* Mitigate with an explicit admission rule: core = truly session-invariant only; require justification in review to add a field.
- **Manifest and actual contents drift** (manifest says included, it isn't) — *Likelihood: low, Impact: high.* Mitigate by generating the manifest from the actual serialization, never hand-maintaining it.
- **Provenance overhead dismissed as bureaucracy and dropped** — *Likelihood: medium, Impact: high (at debug time).* Mitigate by keeping it tiny (producer id, version, timestamp) so cost is negligible.

## Signals to Revisit

- **Re-fetch rate for excluded fields exceeds ~10–20% for any task type** → the variable rules are too stingy; expand that task type's field set.
- **Core bundle size grows enough to cause measurable transport/parse latency** → revisit whether some "core" fields are actually volatile and belong in variable, or should be referenced rather than embedded.
- **Task types proliferate past what a static table can express cleanly** (dozens of near-duplicate rows) → consider composable field-groups, still declarative, still not predictive.
- **Handoffs become rare and bundles small** → the whole optimization is moot; collapse to a single fixed bundle and stop maintaining rules.


### Openai (gpt-5.6-terra)

## Position
A handoff bundle should carry a small fixed core every time, plus a task-selected context payload for the next session’s intended work. Fixed-only bundles become noisy; fully dynamic bundles become unreliable.

## Alternatives Considered
1. **Fixed set of context every time** — Rejected because it either omits task-critical detail or grows into a large, stale bundle that wastes attention and token budget.
2. **Entirely work-selected context** — Rejected because the sender may overlook essential operational facts such as current status, decisions already made, constraints, and unresolved risks.

## Decision Criteria
- The next session must be able to resume work without rediscovering basic state.
- Bundle creation and consumption must be practical for the team’s actual discipline and available time.
- Context must be relevant to the explicitly planned next action, not merely historically interesting.
- Handoffs must preserve critical constraints, decisions, ownership, and verification state.
- The process should tolerate imperfect summaries without causing repeated work or unsafe changes.

## Assumptions
- A handoff has a reasonably identifiable next objective or work queue, even if priorities can change.
- Sessions are separated enough that the next operator cannot rely on conversational memory.
- The team can maintain a lightweight handoff template and update a small number of linked artifacts.
- Context size or reader attention is limited enough that indiscriminate inclusion has real cost.

## Recommendation
**What to do now**
- Define a mandatory fixed core for every handoff:
  - current objective and status;
  - what changed since the prior handoff;
  - decisions made and constraints that must not be violated;
  - next recommended action, owner, and priority;
  - blockers, risks, and validation/rollback status;
  - links or references to the authoritative artifacts.
- Add a selected payload based on the work the next session is expected to perform. For example:
  - implementation work: relevant files, interfaces, acceptance criteria, test failures;
  - incident response: timeline, current mitigation, dashboards, runbook steps, customer impact;
  - design decision: options, evidence, open questions, decision deadline;
  - release work: deployment state, environment differences, rollback plan, verification checklist.
- Require the sender to state why each non-core item is included: “needed for next step X.”

**What not to do now**
- Do not attempt to create a universal exhaustive handoff document.
- Do not rely on a free-form narrative as the sole handoff mechanism.
- Do not include raw logs, complete chat histories, or broad repository summaries unless the next task specifically requires them.

**What to defer until later**
- Defer automated context assembly until the team has used the template enough to identify recurring payload types and authoritative sources.
- Defer sophisticated relevance scoring or AI-generated bundles until there is evidence that manual selection is the bottleneck.

## Rationale
- A fixed core prevents the highest-cost failure: a new session acting without knowing the current state, non-negotiable constraints, or whether prior work was actually validated.
- Task-selected context keeps the bundle actionable. A developer fixing a failing migration needs schema state and failed-test output, not the full history of unrelated product decisions.
- In practice, the core should be short—roughly **5–10 fields or 10–20 lines**—while the variable payload should usually contain **3–7 directly relevant references or artifacts**.
- This approach makes quality review straightforward: reviewers can verify the core mechanically and challenge whether the selected material supports the declared next action.
- It avoids the operational burden of maintaining a “complete context” artifact, which tends to become stale faster than teams can curate it.

## Trade-offs
- The sender must exercise judgment when selecting the task-specific payload; this adds roughly **5–15 minutes** to a meaningful handoff.
- A relevant item can still be omitted, especially when the next task is vague or likely to change.
- Recipients may need to retrieve additional source material through links rather than receiving everything inline.
- The process is less simple than a fixed checklist, but substantially more useful for non-routine work.

## Risks
- **Risk: The next task changes after handoff, making selected context less useful.**  
  Likelihood: medium. Impact: medium.  
  Mitigation: make the fixed core self-sufficient for orientation; link to canonical sources rather than embedding only bespoke summaries; update the handoff when priorities materially change.

- **Risk: Senders include excessive “just in case” material, recreating a fixed giant bundle.**  
  Likelihood: high initially. Impact: medium.  
  Mitigation: set a default size expectation and require each selected item to name the next action it supports.

- **Risk: Critical information is omitted because it was treated as optional.**  
  Likelihood: medium. Impact: high for incidents, releases, and security work.  
  Mitigation: make status, constraints, blockers, validation state, and rollback information mandatory core fields; use domain-specific checklists for high-risk handoffs.

- **Risk: The handoff becomes stale before it is consumed.**  
  Likelihood: medium. Impact: medium.  
  Mitigation: timestamp it, identify the authoritative live sources, and require a quick “still current?” check at session start.

## Signals to Revisit
- More than **10–15%** of sessions begin by rediscovering facts that should have been in the handoff.
- Bundles routinely exceed the team’s agreed reviewable size, or recipients ignore large portions of them.
- The next task cannot usually be identified at handoff time; in that case, increase the fixed core and organize variable context around likely workstreams instead.
- Repeated omissions occur in a specific domain, such as release state or incident mitigation; promote those fields into the mandatory core or add a domain template.
- Handoff preparation consistently takes more than **15–20 minutes** for ordinary work, indicating a need for better source links, templates, or automation.


## Round 2: Critique

### Claude (claude-opus-4-8)

# Evaluation

## Proposal A

**Steelman:** If handoffs are frequent, machine-produced, and consumed by an agent (human or LLM) that can re-fetch from a source of truth, then A is close to exactly right. Its core insight — that *silent omission is the killer failure mode* — is the correct thing to optimize for. If you have a system where bundles are serialized artifacts passing through automated pipelines, generating the manifest from actual serialization and carrying provenance turns every 2am mystery into a traceable event. For that world, A is not just good, it's the reference design.

**Your assessment:** **Agree with the Position.** The core/variable split keyed to lifecycle, plus a manifest of deliberate exclusions, plus provenance — this is exactly what the on-call engineer needs. A's Position matches my priors and improves on them: I hadn't weighted "detect missing context" as heavily as it deserves.

**Strongest point:** The manifest as the single highest-leverage decision. "A receiver that knows 'field Z was excluded' can recover; a receiver that just sees Z's absence cannot distinguish 'excluded' from 'lost.'" This is the whole ballgame. It converts an unbounded root-cause hunt into a bounded re-fetch. Everything else in A is good hygiene; this is the load-bearing wall.

**Weakest assumption:** *"The receiving session can re-fetch or re-derive context if it knows what's missing (i.e., there's a source of truth beyond the bundle)."* If the bundle **is** the source of truth — if the volatile task state exists nowhere else because it was reasoning, intent, or ephemeral working state — then the manifest tells you what's missing but you have no way to recover it. The manifest degrades from "recovery mechanism" to "tombstone." Still better than nothing (you know you're blind), but the entire recoverability argument rests on re-fetchability, and for the most valuable context that may not hold.

**Hidden assumptions:**
1. **That "task type" is a stable, enumerable classification.** The rule table assumes work sorts into discrete types with knowable field requirements. In practice much handoff-worthy work is cross-type or novel — the exact cases where a static table is coarsest are the non-routine cases where handoff matters most.
2. **That the producer knows the task type at bundle-creation time.** The selector keys on task type, but the *sending* session may not know what the *receiving* session will do. A implicitly assumes handoff-time knowledge of next-session intent — the same predictor problem it rejects in the "fully selected" alternative, just relocated into the table lookup.

**Overlooked risks:**
- **Manifest-driven complacency.** Engineers learn to trust "not included, reason: X" and stop questioning whether X was the right call. The manifest documents the selection logic's decisions; it does not validate them. A silent-but-honest wrong exclusion is still a wrong exclusion.
- **Provenance without semantic versioning discipline.** Recording "producer id, version" only helps if versions map to meaningful behavioral differences. A doesn't address what happens when the same version produces different bundle behavior due to upstream config drift.

---

## Proposal B

**Steelman:** If handoffs are *human-authored* by an engineer with 5–15 minutes and judgment, then B is exactly right. Its mandatory core — status, constraints, decisions, blockers, validation/rollback state, links to authoritative sources — is precisely the orientation packet the 2am on-call needs. The domain templates (incident, release, implementation, design) are the practical form of A's abstract "task-type table," and B has the honesty to bound sizes (5–10 core fields, 3–7 references) so bundles stay reviewable.

**Your assessment:** **Partially agree with the Position.** The Position is nearly identical to A's — fixed core plus task-selected payload — and the core field list is more concretely operational than A's. But B relies on human discipline where A relies on mechanism, and discipline is the thing that fails first at 3-year horizons and 2am hours.

**Strongest point:** The mandatory core field list is the best single artifact in either proposal. "Whether prior work was actually validated" and "rollback status" as *mandatory* fields directly address the worst on-call failure: acting on state you believe is safe but isn't. B names the fields; A abstracts them into "invariants."

**Weakest assumption:** *"The sender exercises judgment to select the payload and states why each item is included."* This is a discipline tax paid at handoff time — often the moment of least energy (end of session, context-switching out). The 5–15 minute estimate assumes a conscientious sender with time. On-call reality: the person handing off is exhausted, the incident is ongoing, and the "why included" annotations get skipped. When judgment is the mechanism, judgment erodes under exactly the pressure handoffs are meant to survive.

**Hidden assumptions:**
1. **That a human is in the loop authoring the bundle.** B's entire cost model (minutes, review, "still current?" check) presupposes human authorship. If handoffs are automated or agent-to-agent, "require the sender to state why" has no enforcement surface and no one to feel the friction.
2. **That "require each item to name the next action it supports" survives contact with routine.** For the 80% routine handoff, this annotation becomes ritual box-checking ("needed for next step"), providing no signal. B assumes the discipline stays meaningful; repetition guarantees it won't.

**Overlooked risks:**
- **No omission-detection mechanism.** This is B's critical gap versus A. B's mitigation for "critical info omitted" is "make it mandatory core" — but for *variable* payload omissions, the receiver has no way to know something relevant was left out. B has A's exact failure mode and no manifest to catch it. This is the difference that matters at 2am.
- **Staleness has no enforcement.** "Timestamp it and require a quick 'still current?' check" is a human ritual with no teeth. A timestamp tells you when it was made, not whether it's still true.

---

# Revised Recommendation

**I maintain my architectural position, and I update it in one specific way by merging A and B into a single design. The update: adopt B's concrete mandatory-core field list as the definition of A's abstract "invariants," and treat the manifest as non-negotiable.**

My original lens — *what does the on-call engineer face at 2am* — sorts these cleanly:

**A is the correct mechanism. B is the correct content.** A's manifest-of-exclusions is the one feature that turns a 2-hour mystery into a 5-minute re-fetch, and B has no equivalent. But A's "invariants" are dangerously abstract — an on-call engineer can't act on "invariants," they act on "was this validated? what's the rollback state? what constraints must I not violate?" B names those fields precisely. A without B's field list will have its core schema filled in by whoever's building it, badly.

**What caused the update:** B's mandatory core field list. I came in valuing explicit contracts, and B's list *is* the explicit contract A left as a placeholder. Specifically "validation state" and "rollback status" as mandatory fields — these are the fields whose absence causes destructive 2am actions. A's schema would have gotten there eventually; B got there now.

**What did NOT change my mind:** B's reliance on human judgment and per-item justification. This is the load-bearing assumption that fails under exactly the conditions handoffs exist for — exhaustion, time pressure, context-switching. High cognitive load at authoring time *is* technical debt: it's a cost that gets paid by skipping the step, and the debt surfaces at the receiver. A's data-driven table and generated manifest push the discipline into mechanism, where it survives the 3-year horizon.

**The concrete design:**

1. **Fixed core = B's field list, as A's typed schema:** objective, status, changed-since-last, decisions/constraints, next-action+owner, blockers/risks, **validation state, rollback status**, provenance (producer/version/timestamp), links to authoritative sources.

2. **Variable section = A's data-driven task-type table**, seeded with B's four domain templates (implementation, incident, release, design). Declarative, not heuristic, not predictive.

3. **Manifest generated from actual serialization** — never hand-authored — recording included keys and explicitly-excluded keys with reason. This is the non-negotiable.

4. **Resolve A's weakest assumption directly:** for volatile context that exists *nowhere but the bundle*, the manifest must mark it `source: bundle-only, not re-fetchable`. This is the one case where exclusion is unrecoverable, and the receiver must be told loudly.

**Where I hold firm against both:** Do not require senders to annotate "why included" per item (B) — it decays to ritual. Do not build a predictive selector (both reject this; correct). And drop B's "quick 'still current?' check" as a staleness control — it has no teeth. Staleness is handled by timestamp plus mandatory authoritative-source links, so the receiver validates against truth, not against a ritual.

The failure mode I am optimizing away: the on-call engineer at 2am who cannot tell whether a missing field was *left out on purpose* or *lost in transit*. A solves that. B tells me which fields must never be the ones missing. Neither alone is sufficient.


### Openai (gpt-5.6-terra)

## Proposal A

### Steelman first

Proposal A is exactly right if handoffs are frequent, consequential, and sufficiently structured that the team already has:

- a stable, finite vocabulary of task types;
- a known catalog of context fields;
- reliable canonical sources from which omitted fields can be re-fetched;
- serialization and validation tooling;
- enough engineering capacity to maintain schemas, task-to-field rules, and telemetry.

In that environment, the largest operational danger is not writer effort but silent context loss. A fixed core, declarative task-type rules, provenance, and a generated manifest make handoffs inspectable and reproducible. The receiving session can distinguish “this was deliberately excluded” from “the producer failed to include it,” which is a major reliability improvement for regulated, incident-heavy, or highly automated workflows.

### Your assessment: Partially agree

I agree with the position that neither a wholly fixed bundle nor unconstrained ad-hoc selection is sufficient. A fixed core plus work-relevant context is the right overall shape.

I do **not** agree that an explicit field-level exclusion manifest and task-type-to-field rule table should be the default starting point. This proposal assumes a level of schema maturity, taxonomy stability, and tooling investment that most teams do not have when they are first solving handoffs. It turns a lightweight coordination mechanism into a data-modeling and governance system before proving that this is the actual bottleneck.

For a small team, the likely outcome is not reliable declarative selection. It is stale task types, rules that no longer reflect real work, and people bypassing the process because updating the table is harder than writing a useful handoff.

### Strongest point

The strongest point is that **silent omission is materially worse than explicit absence**.

A receiver that knows context was intentionally omitted can recover from a canonical source. A receiver that cannot distinguish omission, corruption, or irrelevance wastes time and may act incorrectly. Proposal A correctly treats recoverability and debuggability as first-class requirements rather than assuming context selection will be correct.

### Weakest assumption

The weakest assumption is that work can be cleanly categorized into stable task types with predictable field requirements.

In real teams, work routinely crosses categories: a release becomes an incident; an implementation task exposes a product decision; an incident requires a migration; a design decision is blocked by production behavior. If the taxonomy is wrong or coarse, the rule table gives a false sense of completeness. The danger is worse than an openly manual handoff: people trust the generated bundle because it appears systematic.

### Hidden assumptions

1. **There is a bounded, known universe of context fields.**  
   A manifest of excluded fields only has meaning if the system knows which fields could have existed. In an evolving product, newly relevant context is often not a predefined field at all.

2. **The team has authoritative, accessible sources for re-fetching.**  
   “Excluded; re-fetch it” is only useful if the source is current, permissioned, searchable, and understandable by the receiving operator.

### Overlooked risks

- **Manifest overload and false precision.** Recording every excluded field can itself become unreadable, expensive, or meaningless. A giant list of “not included” entries does not help an operator at 2am.
- **Rule-table ownership failure.** The proposal says the table needs maintenance, but does not establish who owns it, how changes are reviewed, or what happens when a new task type appears during urgent work.
- **Schema evolution compatibility.** Producer and receiver versions may disagree on fields, task types, or meanings. Provenance helps diagnose this but does not prevent the operational break.
- **Automation-before-learning.** Instrumenting re-fetches and maintaining generated manifests is a reasonable later-stage system, but it may delay shipping a useful handoff practice now.
- **Sensitive-context propagation.** More systematic bundles can make it easier to copy secrets, customer data, incident details, or privileged operational information into more locations than necessary.

---

## Proposal B

### Steelman first

Proposal B is exactly right if the team is small to medium-sized, handoffs are mostly human-authored, and the next session usually has an identifiable objective. The team can sustain a short template but cannot sustainably operate a field registry, task taxonomy, or selection engine.

Under those conditions, the real failure mode is not imperfect optimization of payload contents. It is that handoffs do not happen, are too long to read, or contain no actionable statement of what to do next. A short mandatory core preserves orientation and safety; selected context tied to a declared next action makes the handoff useful without requiring new platform work.

### Your assessment: Agree

I agree with Proposal B’s position.

The determining factor should be **the explicitly declared next work**, constrained by a small fixed core that always carries orientation, decisions, operational safety state, and links to authority. This is the best balance between usefulness and what a real team can consistently produce and consume.

Proposal B is operationally credible. It can be introduced with a template, a review habit, and a few domain checklists. It does not require building a context-selection platform before the team has demonstrated a need for one.

The key qualifier is that “selected” must not mean “whatever the sender happens to remember.” Selection should be anchored to a declared next action and be reviewable: each included item must support that action, and high-risk domains must have mandatory checks.

### Strongest point

The strongest point is that it treats handoff quality as an **operational practice**, not primarily as a schema-design problem.

A handoff system that theoretically captures every exclusion but is too burdensome for engineers to maintain will fail. Proposal B starts with a process a team can actually execute: a short core, a next action, a limited number of relevant references, and direct links to canonical sources.

### Weakest assumption

The weakest assumption is that there is a reasonably identifiable next objective at handoff time.

When priorities are volatile, queues are shared, or an incident is still unfolding, “the next recommended action” may be speculative or invalid by the time the next session starts. If that assumption fails, a narrowly selected payload can overfit to a task that no longer matters, leaving the recipient without enough context to pivot.

The consequence is that recipients must rediscover broader state, and the handoff becomes a misleading plan rather than a reliable starting point.

### Hidden assumptions

1. **Senders have enough understanding and discipline to select relevant material.**  
   A template does not create judgment. Junior operators, exhausted incident responders, or people handing off unfamiliar work may not know which evidence matters.

2. **The referenced artifacts are authoritative and remain available.**  
   Links are only a substitute for embedded context if dashboards, tickets, documents, repositories, and runbooks are current, accessible, and not contradictory.

### Overlooked risks

- **Rubber-stamped relevance rationales.** “Needed for next step X” can become boilerplate unless reviewers or recipients use it to challenge irrelevant payloads.
- **High-volume handoff noncompliance.** Even 5–15 minutes per handoff is significant if handoffs occur many times per day. The proposal needs a distinction between ordinary checkpoint handoffs and formal shift/ownership transfers.
- **Stale or misleading selected evidence.** A copied test failure, dashboard state, or deployment observation may be obsolete by consumption time. Links and timestamps help, but the recipient needs an explicit freshness check.
- **No escalation path for uncertainty.** The proposal should state what to do when the sender cannot identify the next task: name likely workstreams, state uncertainty, and include the minimum context needed to choose among them.
- **Security and privacy boundaries.** The recommendation discourages raw logs, but it does not specify rules for secrets, customer data, or incident-sensitive material in bundles and linked artifacts.

---

# Revised recommendation

I recommend: **a fixed operational core on every handoff, with the rest selected for the explicitly declared next work. Start manual and template-driven. Do not build a field-level exclusion manifest or task-type rule engine yet.**

This is not a compromise between fixed and selected approaches. The decision rule is clear:

> **The next planned work determines the variable context. A fixed core exists only to ensure the recipient can safely orient, verify freshness, and change course when that plan is wrong.**

## Do now

Use one lightweight handoff template:

### Mandatory fixed core

1. **Timestamp, owner, and handoff source**
2. **Current objective and status**
3. **What changed since the last meaningful checkpoint**
4. **Decisions and non-negotiable constraints**
5. **Known blockers, risks, and unresolved questions**
6. **Validation state and rollback/safety status where applicable**
7. **Declared next action or likely workstream**
8. **Links to authoritative live sources**

This must be short enough to read in under two minutes.

### Work-selected payload

Include only the artifacts or references needed to execute or assess the declared next action. Each item must say what it supports.

Examples:

- “Migration test failure and affected schema: needed to implement and verify the rollback-safe migration.”
- “Current deployment dashboard and rollback runbook: needed to decide whether mitigation is holding.”
- “Decision options and customer evidence: needed to make the pricing decision by Friday.”

Default target: **three to seven relevant references or artifacts**, not a broad history dump.

### Add mandatory domain gates, not a universal rule table

For incident, release, security, and data-changing work, require a small domain checklist. For example:

- Incident: customer impact, current mitigation, owner, last verified time, escalation status.
- Release: deployed version, environment, verification result, rollback path.
- Data change: migration status, backup/rollback status, validation result.

These address the highest-cost omissions without requiring a generalized context registry.

### Recipient responsibility

At session start, the receiver should perform a quick freshness check:

- Is the declared next action still valid?
- Are the linked sources current and accessible?
- Has an operational state changed since the handoff timestamp?

If not, the receiver updates the handoff or creates a new one before taking irreversible action.

## Do not do now

- Do not require a manifest listing every excluded context field.
- Do not introduce a task-type-to-field selection matrix as a default process.
- Do not build automated assembly, relevance scoring, or predictive selection.
- Do not require long free-form narratives or copy raw logs and chat history into ordinary bundles.

## What would justify upgrading toward Proposal A

Adopt declarative payload rules and richer omission tracking only if evidence shows manual selection is repeatedly failing in a measurable, recurring way:

- the same domain-specific omission happens despite templates and checklists;
- handoffs are frequent enough that manual preparation is a material delivery cost;
- task categories are demonstrably stable;
- canonical source systems and field definitions are mature;
- the team has a clear owner for maintaining the rules.

Until then, Proposal A’s mechanisms are premature operational complexity. Proposal B’s model is the one a normal team can ship this week, use consistently, and improve from observed failures.


## Synthesis (by gemini, non-participant)

## Consensus

Both participants converged on the same fundamental architectural shape: **a handoff bundle must contain a fixed core plus a variable section.** 

This consensus was driven by strong shared reasoning regarding cognitive load and system lifecycles. Both successfully argued that a completely fixed bundle grows into an unreadable, bloated artifact, while a completely dynamic bundle inevitably omits critical safety state. 

Furthermore, both participants agreed to reject predictive or AI-inferred selection mechanisms for the variable context, recognizing them as unbounded, hard-to-test surfaces that create invisible failure modes. In Round 2, Participant A explicitly adopted Participant B’s concrete list of operational fields (status, validation state, rollback status) as the definition of its "fixed core," demonstrating evidence-driven consensus.

## Unresolved Disagreements

**Mechanism vs. Human Discipline for Variable Context**
*   **The Crux:** Should the variable context be selected via a static, declarative rule table mapped to predefined "task types" (Proposal A), or via human judgment based on the explicitly declared next intended action (Proposal B)?
*   **Stronger Argument:** Proposal B (Terra). Proposal A correctly identifies that human discipline decays under 2am exhaustion, but A's proposed solution—an automated exclusion manifest generated from a task-type rule table—rests on a fatal assumption. As B points out, a manifest of *excluded* fields is only possible if there is a known, bounded universe of *all possible* fields, and if real-world work fits neatly into a static taxonomy of task types. In practice, messy operational work routinely crosses boundaries, and new context often doesn't exist in a predefined schema. B correctly identifies A's approach as premature optimization that demands massive data-modeling overhead.

**Handling of Silent Omissions**
*   **The Crux:** How does the receiving session know if a crucial piece of context was deliberately left out versus accidentally lost/forgotten?
*   **Stronger Argument:** Split. Proposal A makes the theoretically superior argument: distinguishing between "excluded" and "lost" is the highest-leverage factor in recovering from bad handoffs. However, B successfully argues that A's mechanism to solve this (listing every excluded field) is operationally unfeasible. Conversely, B's solution to omission relies entirely on human review, which A correctly diagnoses as prone to ritualistic rubber-stamping.

## Argument Quality Assessment

**Best Reasoned Proposal:** Proposal B (Terra) remained the most grounded in operational reality. It recognized the limits of human attention without demanding an impossible-to-maintain schema registry. 

**Strongest Argument:** Proposal A’s insight that *"silent omission is the killer failure mode... a receiver that knows 'field Z was excluded' can recover; a receiver that just sees Z's absence cannot distinguish 'excluded' from 'lost.'"* This perfectly articulates the exact risk of dynamic context payloads.

**Weakest Argument:** Proposal B's insistence that senders manually annotate exactly why each item is included in the variable payload ("Needed for next step X"). Proposal A expertly dismantled this, noting that for routine work, this discipline will immediately decay into thoughtless, ritualistic box-checking that provides zero actual signal.

## Blind Spots

*   **The Nature of the Handoff Medium:** The debate largely ignored the underlying medium of the handoff. Are these payloads passed between human engineers via Slack/Jira, or JSON artifacts passed between automated autonomous agents? Proposal A's rule tables make perfect sense for agentic workflows; Proposal B's templates make perfect sense for humans. 
*   **Ephemeral "Working Memory":** Both heavily relied on "linking to canonical sources" for omitted context. Neither addressed what happens when the context is ephemeral working memory, intent, or half-finished thought processes that *have no canonical source yet*. If that data isn't in the bundle, it is permanently destroyed, regardless of what any manifest or template says.

## Recommended Decision

**Implement a fixed operational core on every handoff, with the variable payload selected by human judgment based on the declared next action. Do not build an automated field-exclusion manifest.**

**Rationale:**
The fixed core guarantees that non-negotiable orientation and safety constraints (like rollback status) survive every handoff. The variable payload must remain flexible because operational work is too unpredictable to map into rigid "task type" schemas (weighing B's reality check heavier than A's desire for strict mechanism). 

While A's fear of "silent omission" is the most acute risk identified in the debate, A's proposed solution (a schema-driven rule table) requires an unrealistic level of infrastructure. We will mitigate the silent omission risk by adopting B's "domain gates" (mandatory checklists for high-risk domains like Incidents or Releases) which act as localized, enforceable schemas, without requiring a universal field registry.

Weighing A's critique of B, we will **drop** B's requirement for manual per-item justifications ("needed for next step X"), as A is correct that this introduces friction that will inevitably degrade into ritualistic compliance.

## Risks

*   **Risk:** The intended next action changes immediately after the handoff, rendering the variable context useless.
    *   *Mitigation:* The mandatory fixed core must always contain links to authoritative live sources (dashboards, repos, tickets) so the receiver can quickly re-orient and pull new context.
*   **Risk:** Critical domain-specific information is forgotten by exhausted engineers.
    *   *Mitigation:* Enforce hard checklists for known high-risk domains (Incidents, Data Migrations, Releases) that act as an expanded fixed core for those specific scenarios.
*   **Risk:** Context overload (the sender dumps everything just in case).
    *   *Mitigation:* Establish a hard cultural norm and template limit: the core should take < 2 minutes to read, and the variable payload should ideally contain 3-7 referenced artifacts, not raw log dumps.

## Action Items

1.  **Deploy a standardized Handoff Template** containing the unified fixed core: 
    * Timestamp/Provenance
    * Current objective & status
    * Changes since last checkpoint
    * Decisions & non-negotiable constraints
    * Validation state & rollback/safety status
    * Declared next action/workstream
    * Links to authoritative sources
2.  **Define Domain Gates:** Create 3 specific mini-checklists for the most common high-risk work: Incidents, Releases, and Data Migrations. These must be filled out if the task touches these domains.
3.  **Train the Team on Variable Selection:** Establish the rule that variable context (logs, snippets, specific files) is included *only* if it serves the "Declared next action." 
4.  **Defer Automation:** Do not attempt to build automated task-type rule tables, payload serializers, or generated exclusion manifests unless manual template usage shows a >15% failure rate in providing necessary context over a sustained period.
