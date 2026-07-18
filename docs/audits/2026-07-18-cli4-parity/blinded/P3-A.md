# AI Council Debate: What determines which context a handoff bundle carries - a fixed set every time,

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** cli

## Verdict Summary

**Decision:** Implement a strict Two-Layer Handoff Bundle: A Fixed Skeleton + Task-Selected Payload.
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

Use a **fixed core skeleton plus a task-selected payload** — the bundle is not one-or-the-other. A small, always-present invariant set (identity, task goal, active constraints, open thread state) is carried every time; everything else is selected by relevance to the next session's declared work. Purely fixed bundles rot; purely selected bundles drop the load-bearing invariants nobody flagged as "relevant."

## Alternatives Considered

- **Fully fixed set every time** — Rejected. A fixed template guarantees completeness but scales badly: it either grows to cover every scenario (bloated, high cognitive load, wastes the very context budget it's trying to preserve) or stays small and silently omits what this particular task needed. The failure mode is quiet: the next session doesn't know what was dropped.
- **Fully selected per next-session work** — Rejected as the sole mechanism. Selection assumes you can predict what the next session will need, but handoffs happen precisely when work is incomplete and the next step is uncertain. Relevance-selection reliably drops cross-cutting invariants (a security constraint, a "don't touch X" decision) that aren't topically adjacent to the next task but are still binding.

## Decision Criteria

1. **Recoverability at 2am** — can the next session reconstruct *why*, not just *what*, without the prior session present?
2. **Completeness of invariants** — are binding constraints and decisions carried regardless of topical relevance?
3. **Context budget efficiency** — bundle must fit the window with room to work; every token spent on carry-over isn't spent on the task.
4. **Predictability of failure** — when the bundle is wrong, is it loud (missing-but-flagged) or silent (missing-and-unknown)?
5. **Authoring cost** — a scheme so elaborate that sessions skip it is worse than an imperfect automatic one.

## Assumptions

- Handoffs occur mid-task with genuine uncertainty about the next step (not clean, fully-planned segment boundaries).
- Context budget is a real constraint — you cannot just carry everything.
- The next session's *general* work area is knowable (the current goal), even if the exact next action isn't.
- Some constraints are cross-cutting and won't surface via topical relevance matching.

## Recommendation

**Now:**
- Define a **fixed skeleton** of ~5 slots that appear every handoff: (1) task goal / definition of done, (2) active constraints and hard "do-not" decisions, (3) current state — what's done, what's in flight, (4) the open question or next intended action, (5) pointers (files/functions touched, not their contents).
- Add a **selected payload**: content pulled in because it's relevant to slot (4), the next intended action — code excerpts, prior reasoning, external facts.
- Make omission **loud**: the skeleton includes a "dropped for space — retrievable here" line so the next session knows what exists but wasn't carried.

**Not now:**
- Don't build a learned/predictive relevance ranker for the payload. Start with explicit author-tagged relevance; earn the ML.
- Don't let the skeleton grow past a handful of slots — resist adding "just in case" fields.

**Defer:**
- Automated relevance scoring, per-task-type skeleton variants, and compression tuning until you have handoff-failure data showing where bundles actually fall short.

## Rationale

- **The invariants are the load-bearing wall, and they're topically invisible.** A "don't call the prod API" decision won't be selected by relevance to "write the parser" — but violating it is the expensive 2am failure. Fixed skeleton guarantees it survives.
- **The payload is where budget is won or lost.** State + constraints are small (a few hundred tokens); the expensive mass is code/reasoning context. Selecting *that* by next-task relevance is where a fixed scheme wastes 60–80% of carry-over budget on irrelevant material.
- **Silent omission is the dominant failure mode of pure selection** — and it's unfixable by the victim, because they don't know what's missing. The "dropped-but-pointered" line converts silent loss into a cheap retrieval.
- **Uncertainty at handoff time defeats prediction.** You hand off *because* the path is unclear; a scheme that assumes you know the next task will misfire exactly when handoffs matter most. The fixed skeleton is the hedge against your own bad prediction.
- **Two mechanisms, clear ownership:** skeleton = correctness guarantee, payload = efficiency optimization. Debugging a bad handoff becomes "was the invariant in the skeleton?" (spec bug) vs. "was the right context selected?" (tuning bug) — separable, not tangled.

## Trade-offs

- You give up the simplicity of a single mechanism — two things to reason about, not one.
- The fixed skeleton spends a fixed ~200–500 tokens every handoff even when trivially small tasks wouldn't need all slots.
- Author-tagged selection (vs. automated) puts modest burden on the handing-off session to mark relevance — the price of not building a ranker yet.

## Risks

- **Skeleton bloat over time** (likelihood: high, impact: medium) — teams add "one more field." Mitigate: cap the slot count as a hard rule; new fields must displace an old one.
- **Payload under-selection drops something needed** (likelihood: medium, impact: medium) — the "dropped-but-pointered" line makes it recoverable rather than fatal.
- **Skeleton becomes stale boilerplate** — copied forward unchanged, no longer reflecting true state (likelihood: medium, impact: high). Mitigate: require the "current state" and "next action" slots to be rewritten each handoff, not carried verbatim; a skeleton that reads identically to last time is a smell.

## Signals to Revisit

- Handoff-failure logs show the *skeleton* invariants are consistently sufficient and payload is rarely used → collapse toward a leaner fixed set.
- Failures cluster in "wrong payload selected" → invest in the deferred automated relevance ranker.
- Context budgets grow large enough that carrying everything costs less than the selection machinery → drop selection entirely and go fixed-and-full.
- Handoffs shift to clean, planned boundaries (next task genuinely known) → weight moves toward selection, skeleton shrinks.


### Openai (gpt-5.6-terra)

## Position

Select context for the next session’s intended work; do not carry a fixed bundle every time. Retain only a small mandatory baseline for continuity and safety.

## Alternatives Considered

- Fixed comprehensive bundle every handoff: rejected because it creates context bloat, obscures the active task, and becomes increasingly stale.
- Minimal generic summary only: rejected because it often omits the concrete state, decisions, and artifacts needed to resume technical work safely.

## Decision Criteria

- The next session can resume productive work without rediscovery.
- Context fits the available window without crowding out new reasoning.
- Handoff content is current, verifiable, and operationally useful.
- The producing team can create it consistently with low overhead.
- Safety constraints, open risks, and irreversible decisions survive every handoff.

## Assumptions

- The handoff identifies an intended next action or a bounded set of likely next actions.
- Source systems—repository, tickets, dashboards, and documents—remain available for reference.
- Sessions have finite context capacity.
- Work spans multiple domains, so not every prior detail is equally relevant.

## Recommendation

What to do now: define a two-layer bundle:

- Mandatory baseline: current objective, status, completed work, current blockers, key decisions and their rationale, changed artifacts/locations, validation performed, and immediate risks.
- Work-selected context: details chosen specifically for the next session’s planned task—for example relevant architecture constraints, API contracts, failing test output, incident timelines, or stakeholder requirements.

What not to do now: do not attach every prior discussion, complete logs, or all repository history by default.

What to defer until later: automate selection only after the team has several handoffs’ worth of evidence about omissions and recurring retrieval needs.

## Rationale

- A session fixing a failing deployment needs environment state, recent changes, and logs; a session designing a schema migration needs data volumes, compatibility constraints, and rollback requirements. One universal payload serves both poorly.
- A mandatory baseline prevents the common failure mode of “why are we doing this?” while remaining compact—typically under 10–20% of the usable context budget.
- Task-specific material keeps high-value evidence near the reasoning point, reducing rediscovery time without burying the next operator in irrelevant history.
- Selecting context at handoff forces an explicit next-step statement, which exposes ambiguity before it becomes a costly stalled session.
- Original systems remain the source of truth; the bundle should be an indexed operational briefing, not a duplicated archive.

## Trade-offs

- Creating a tailored section adds roughly 5–15 minutes of handoff effort versus a fully fixed template.
- A poor selection can omit a fact needed later; the baseline and links to source artifacts limit, but do not eliminate, that risk.
- Standardization is lower than with a fully fixed package, so a lightweight checklist is necessary.

## Risks

- **The next work is unclear.** Likelihood: medium; impact: medium. Mitigate by stating the top one to three candidate next actions and selecting context for each only when materially different.
- **Important context is omitted.** Likelihood: medium; impact: high for migrations, incidents, and security work. Mitigate with mandatory fields for decisions, risks, validation, rollback/recovery state, and source links.
- **Bundles grow into mini-archives.** Likelihood: high without limits; impact: medium. Mitigate with a size budget and a rule that raw logs and historical discussion are linked, not copied.
- **Handoffs become inconsistent across people.** Likelihood: medium; impact: medium. Mitigate with a standard baseline template plus a required “why this context is relevant next” line for selected material.

## Signals to Revisit

- Resume sessions routinely spend more than 20–30 minutes rediscovering prior state.
- More than 10% of handoffs cause rework due to missing decisions, constraints, or validation evidence.
- The same task-specific categories recur across most handoffs; promote them into the baseline.
- The next session’s task is frequently unknown at handoff time; use a stronger fixed baseline until planning discipline improves.


## Round 2: Critique

### Claude (claude-opus-4-8)

Both proposals arrive at the same structural answer — a fixed core plus a task-selected payload. My job is to find where each one's reasoning is load-bearing versus decorative, and where the shared consensus hides a common blind spot.

---

## Proposal A

**Steelman:** A is exactly right if handoffs are produced by disciplined humans who can reliably name a next action, and if the source systems (repo, tickets, dashboards) are genuinely available and stable at resume time. Under those conditions, the bundle *should* be a thin indexed briefing over durable sources, not a duplicated archive — and the two-layer split (mandatory baseline + work-selected context) captures exactly the right seam.

**Assessment: Partially agree.** The Position is correct. But A treats the bundle as a *pointer layer over authoritative sources* — "the bundle should be an indexed operational briefing, not a duplicated archive." That's the right instinct for a human ops team with Jira and Grafana. It is a dangerous default for an agent handoff, where the "source of truth" may be an ephemeral session state, a scrolled-off tool output, or a decision that was never written anywhere but the prior context window. A's model assumes the archive still exists to link to. Often it doesn't.

**Strongest point:** The concrete worked example — "a session fixing a failing deployment needs environment state and logs; a session designing a schema migration needs rollback requirements" — is the whole argument for selection in two sentences. One universal payload serves both poorly. That's irrefutable.

**Weakest assumption:** *"The handoff identifies an intended next action or a bounded set of likely next actions."* This is the same crack B correctly identifies: you hand off *because* work is unfinished and the path is uncertain. A half-acknowledges this in its risks ("the next work is unclear"), but its mitigation — "state the top one to three candidate next actions" — quietly assumes the ambiguity is enumerable. Sometimes the next session's first job is to *discover* what the next action is. If selection keys off a next-action that turns out wrong, the tailored payload was optimized for the wrong task, and A has no loud-failure mechanism to catch it.

**Hidden assumptions:**
1. **Source systems remain both available and legible to the next session.** A states availability as an assumption but not *retrievability* — that the next operator knows the link exists and has the access/tooling to follow it. A link to a dashboard the next session can't authenticate to is not a briefing, it's a taunt.
2. **The producing session knows what it knows.** A's baseline lists "key decisions and their rationale" as a fillable field, presuming decisions are consciously held and articulable. The expensive ones are often implicit — the thing the prior session stopped doing without noting why. A has no mechanism to surface un-flagged invariants.

**Overlooked risks:** A never addresses **silent omission as unrecoverable** — its own risk table rates "important context omitted" as medium-likelihood, mitigated by "mandatory fields and source links." But the victim of an omission doesn't know to follow a link for something they don't know is missing. This is precisely the failure B names and A doesn't. A also ignores **staleness of the baseline itself** — the baseline can be copied forward as boilerplate just as easily as any fixed template it criticizes.

---

## Proposal B

**Steelman:** B is exactly right if the dominant, most expensive failure mode is a cross-cutting invariant violation ("don't touch prod") that topical relevance-matching would never surface — and if the handing-off session can be trusted to keep the skeleton honest rather than copying it forward. Under those conditions, the skeleton-as-correctness-guarantee / payload-as-efficiency-optimization split is the cleanest possible ownership model.

**Assessment: Agree.** B's Position dominates A's on the one axis I care about most: **predictability of failure.** B's decision criterion 4 — "when the bundle is wrong, is it loud or silent?" — is the single most important sentence in either proposal, and A doesn't have its equivalent. The "dropped-for-space — retrievable here" line converts silent, unrecoverable loss into a cheap lookup. That is the 2am-engineer feature. A gives the on-call engineer a tailored briefing; B gives them a tailored briefing *plus a manifest of what was deliberately left out*. The manifest is worth more than the tailoring.

**Strongest point:** "The invariants are the load-bearing wall, and they're topically invisible." A security constraint won't be selected by relevance to "write the parser," but violating it is the expensive failure. This is the correct refutation of pure selection, and it's an architectural argument, not a preference.

**Weakest assumption:** *"The next session's general work area is knowable (the current goal), even if the exact next action isn't."* B leans on this to justify keeping the payload selection tractable. But goals drift across handoffs — the current goal at handoff N may be abandoned by handoff N+2, while the skeleton dutifully carries the stale goal forward. B partly catches this (its "skeleton becomes stale boilerplate" risk, rated impact: high), but its mitigation — "a skeleton that reads identically to last time is a smell" — is a *cultural* control, not a mechanical one. Smells don't wake anyone up at 2am.

**Hidden assumptions:**
1. **The handing-off session is competent to tag relevance and populate invariants honestly.** B explicitly rejects the ML ranker in favor of "author-tagged relevance" — which loads the entire correctness guarantee onto the discipline of the party least able to verify it (they're leaving; they won't feel the pain of a bad handoff). B assumes the author is a reliable narrator of their own state. The most dangerous omissions come from authors who don't know they're omitting.
2. **Five slots is the right cardinality, not just a round number.** B asserts ~5 slots and caps growth, but never derives why five. The number is presented as disciplined restraint; it's actually an unexamined guess. The real invariant is "cap it and force displacement," which is sound — but B dresses a guess as a principle.

**Overlooked risks:** B never addresses **verification of the invariant slot.** The skeleton *guarantees* invariants are carried, but guarantees nothing about their *correctness* — a wrongly-stated constraint ("safe to touch the cache") is worse than an absent one, because the skeleton lends it authority. B's whole value proposition is "the invariant is in slot 2," with no check that slot 2 is true. B also underweights **the trivial-handoff tax**: it admits the skeleton costs 200–500 tokens even on trivial tasks but treats this as acceptable; at high handoff frequency, a fixed per-handoff tax on structure people will start skipping — which is B's own criterion 5 turned against it.

---

## Where both proposals share a blind spot

Both A and B model the bundle as **content selected once, at handoff, by the departing session.** Neither treats the bundle as something the *arriving* session negotiates. The entire scheme is push, never pull. But the party who knows what's actually needed is the one resuming — and they have no channel to say "this baseline is stale, re-derive it" or "I need the thing you dropped." Both proposals optimize the author's guess; neither builds a cheap correction loop for the reader. B's "dropped-but-pointered" line is the closest thing, and it's the best idea in either document precisely because it's the one reader-facing affordance.

Both also assume **the source of truth outlives the bundle.** For agent handoffs specifically, the prior context window frequently *is* the only place a decision existed. "Link, don't copy" (A) and "pointers, not contents" (B, slot 5) both bet that the pointed-to thing survives. When it doesn't, the pointer is a dead link to the exact reasoning you needed.

---

## Revised recommendation

**I maintain my position — fixed core plus task-selected payload — and B's framing sharpened it. Two specific updates.**

What changed my mind, precisely: **B's "predictability of failure" criterion and the "dropped-but-pointered" line.** Before, I'd have accepted A's cleaner two-layer model. A's fatal gap is that its omission-recovery story is "we kept mandatory fields and links" — which does nothing for the omission the author didn't know they made. B's insight that the failure mode of pure selection is *silent and unrecoverable by the victim* is the decisive architectural point, and the manifest-of-omissions is the mechanism that fixes it. That is a genuine update, not a tie-break.

**My council position:**

1. **Fixed skeleton for invariants, selected payload for work — this is settled.** Both proposals are right that it's not one-or-the-other. B's ownership split (skeleton = correctness, payload = efficiency) is the reason to prefer it as the framing.

2. **Add what neither proposal has: verification of the invariant slot.** The skeleton must not just *carry* constraints; each one needs a source anchor or a "re-verify before relying" flag. An authoritative-looking-but-wrong invariant is the worst 2am outcome, and B's scheme actively manufactures that risk by lending the skeleton authority. Cheapest fix: every invariant slot entry is either (a) pointer-backed to where it's enforced, or (b) explicitly marked "asserted, unverified."

3. **Add a reader-side correction affordance.** Adopt B's dropped-but-pointered manifest, and go one step further: the arriving session's first cheap action is to challenge the skeleton — "is the stated goal still the goal?" A skeleton that survives contact with the resuming session is trustworthy; one that's never challenged is boilerplate. This directly attacks B's own highest-impact risk (stale boilerplate) with a mechanism instead of a "smell."

**Where I break from both:** I will not accept "link, don't copy" as an unconditional rule (A) or "pointers, not contents" (B, slot 5) for **irreversible or ephemeral** context. For invariants and the reasoning behind hard "do-not" decisions, **copy the load-bearing sentence inline, then link for depth.** The cost is a few hundred tokens; the alternative is a dead pointer to the one thing the next session couldn't afford to lose. Selection and linking are correct for the expensive *mass* of context (logs, code, history) — exactly where B says the budget is won. They are wrong for the small, irreplaceable core.


### Openai (gpt-5.6-terra)

## Proposal A

**Steelman first:** It is exactly right if the team can reliably name the next bounded task at handoff, source systems are accessible, and a lightweight template prevents cross-cutting constraints from being omitted.

**Your assessment:** Partially agree. The task should determine most carried context, but “select for the next session” is not enough to protect constraints that are not visibly relevant to that task.

**Strongest point:** The bundle should be an operational briefing with links, not a duplicate archive. That is the only approach likely to remain usable as work history grows.

**Weakest assumption:** That the next action is sufficiently known. In real interrupted work, ambiguity is often the reason for handoff; if that assumption fails, selection becomes guesswork and critical context is dropped.

**Hidden assumptions:**

- Everyone can access and interpret the referenced repository, tickets, dashboards, and documents.
- The producer has enough time and judgment to curate 5–15 minutes of task-specific material consistently.

**Overlooked risks:**

- The proposed baseline is already broad enough to become a long, stale status report unless it has hard size and freshness rules.
- Links alone are not recoverability: permissions, expired logs, changing dashboards, and deleted branches can make the “source of truth” unavailable.
- It does not define who owns correcting a bad handoff or how omissions are measured.

## Proposal B

**Steelman first:** It is exactly right if a small invariant core can be kept genuinely current, while authors can select payload using a declared next action and make omitted material discoverable through durable pointers.

**Your assessment:** Agree. This is the right operating model for a team that needs reliable handoffs without building expensive automation or carrying an ever-growing transcript.

**Strongest point:** It separates correctness from efficiency. Fixed invariants preserve safety and intent; selected payload preserves context budget. Those are different problems and should not be solved by one mechanism.

**Weakest assumption:** That a five-slot skeleton stays small and is rewritten rather than copied forward. If it becomes boilerplate, it creates false confidence: the next session believes it has current state when it has yesterday’s state.

**Hidden assumptions:**

- “Pointers” are durable, permissioned, and quick to retrieve in the next session.
- The team has enough planning discipline to state an open question or intended action meaningfully.

**Overlooked risks:**

- A “dropped for space” line can become performative unless it names concrete omitted categories and locations.
- The proposal does not distinguish high-risk handoffs—production incidents, migrations, security work—from ordinary feature work. Those need stronger evidence and rollback state than a generic skeleton.
- Manual author-tagging may fail under incident pressure, precisely when handoffs are most consequential.

## Revised recommendation

Adopt Proposal B’s model: a fixed, enforced handoff manifest plus a payload selected for the next declared work. This is not a compromise between fixed and selected bundles; it assigns each mechanism one job.

The fixed manifest must be short and mandatory:

1. Objective and definition of done  
2. Current verified state and validation evidence  
3. Binding constraints, irreversible decisions, and active risks  
4. Next action or explicit open question  
5. Durable pointers to changed artifacts and omitted material  

Everything else is selected only when it materially enables that next action. Raw logs, long discussions, and broad history remain linked.

Add one operational rule: use a higher-assurance variant for production, migration, and security handoffs, requiring environment state, rollback/recovery status, and ownership. Do not make every feature handoff pay that cost.

Proposal A reinforced the need for task-selected context and strict size limits. Proposal B changed the recommendation by making explicit that cross-cutting invariants must be fixed rather than entrusted to relevance selection.


## Synthesis (by gemini, non-participant)

## Consensus
Both participants independently arrived at the exact same structural solution: a **hybrid two-layer bundle** consisting of a small, fixed baseline (for invariants and state) and a task-selected payload (for specific work context). 

This consensus was driven by strong, shared independent reasoning rather than groupthink. Both recognized the catastrophic failure modes of the extremes: purely fixed bundles scale poorly and cause context bloat, while purely selected bundles inevitably drop critical, cross-cutting constraints that don't match the immediate topic. Both agreed on separating *correctness* (ensuring safety constraints survive) from *efficiency* (optimizing context budget).

## Unresolved Disagreements

**1. The Handling of Omitted Context and Source Durability**
* **The Crux:** Should the bundle rely on general links to external systems as an "indexed briefing" (Proposal A), or must it include an explicit manifest of exactly what was dropped, assuming source systems might be ephemeral (Proposal B)?
* **Stronger Argument:** Proposal B. As argued, silent omission is unrecoverable because the arriving session does not know what it doesn't know. Relying on external links assumes the source of truth outlives the handoff. In agent/LLM workflows, the prior context window often *is* the only record of a decision. A dead link to ephemeral session state is a fatal loss.

## Argument Quality Assessment
**Best Reasoned:** Proposal B (Claude) was the most rigorously argued position, ultimately reshaping the entire second round of the debate. It evaluated architectural failure modes (loud vs. silent failure) rather than just workflow preferences.

**Single Strongest Argument:** Proposal B's observation that *"invariants are the load-bearing wall, and they're topically invisible."* This dismantled the premise of pure relevance-selection by proving that critical negative constraints (e.g., "don't call the prod API") will never trigger a relevance match for a specific sub-task (e.g., "write a parser"). This single point mandates a fixed skeleton.

**Single Weakest Argument:** Proposal A's mitigation for when the next intended work is unclear: *"state the top one to three candidate next actions and select context for each."* This completely undermines the primary goal of the selected payload—saving context budget—and falsely assumes ambiguity is easily enumerable into neat, predictable buckets.

## Blind Spots
The debate surfaced several unaddressed dimensions:
1. **Push-Only Dynamics (Receiver Passivity):** Both initial proposals treated handoffs entirely as a "push" mechanism optimized for the departing author. Neither originally considered the arriving session's need for "pull" affordances—a way to challenge stale assumptions, ask clarifying questions, or dynamically fetch what was dropped.
2. **Verification of Constraints:** The fixed skeleton guarantees a constraint is carried, but gives no mechanism to verify if it is *true*. An authoritative-looking but outdated constraint copied forward as boilerplate is highly dangerous.
3. **The Trivial-Handoff Tax:** Imposing a structured skeleton and relevance-tagging burden on highly frequent, trivial handoffs was acknowledged as a cost but left unsolved. 

## Recommended Decision
**Implement a strict Two-Layer Handoff Bundle: A Fixed Skeleton + Task-Selected Payload.**

1. **The Fixed Skeleton (Correctness):** A strictly capped, mandatory manifest that appears every time. It must contain: Objective, verified state, binding constraints / "do not" decisions, the next open question, and an explicit *"dropped for space — retrievable here"* manifest.
2. **The Selected Payload (Efficiency):** Context (code snippets, logs, reasoning) pulled in *only* because it directly serves the next declared action.
3. **Inline Irreversible Context:** Do not blindly follow a "link, don't copy" rule. Irreversible decisions and load-bearing constraints must be copied inline. Use links for depth, not for baseline survival.

**Rationale:** This decision assigns one distinct job to each mechanism. The skeleton acts as a hedge against unpredictable handoff boundaries, guaranteeing that topically-invisible safety constraints are never silently dropped. The payload preserves the context window budget. Adopting the "dropped-for-space" manifest converts the worst failure mode of handoffs (silent omission) into a cheap, loud retrieval.

## Risks
* **Stale Boilerplate:** The skeleton gets mindlessly copied forward, drifting from reality and creating false confidence for the arriving session.
  * *Mitigation:* Require the arriving session to execute a cheap "challenge/verify" step on the skeleton before resuming work. The "current state" slot must be rewritten, not copied.
* **Dead Links to Ephemeral State:** External pointers fail due to expired session windows, deleted branches, or lack of permissions.
  * *Mitigation:* Enforce the rule that critical constraints and rationales are quoted inline.
* **Skeleton Bloat:** Over time, teams incrementally add "just one more mandatory field" until the fixed bundle consumes the entire context window.
  * *Mitigation:* Institute a hard cap (e.g., 5 slots) on the skeleton. Any new required field must displace an existing one.

## Action Items
1. **Draft the Skeleton Template:** Define the capped, 5-slot handoff baseline, explicitly building in the "dropped-for-space / retrievable here" field.
2. **Implement Receiver Verification:** Create a standard initialization step for the resuming session to actively challenge the received skeleton rather than accepting it blindly.
3. **Establish Link Guidelines:** Document clear standards defining what can be linked (heavy logs, raw code, historical discussion) versus what must be written inline (decisions, safety constraints).
4. **Defer Automation:** Explicitly halt any current efforts to build ML/automated relevance rankers for the payload. Rely on author-tagged context until handoff-failure data proves the need for automation.
