# AI Council Debate: How should a solo developer split work between a browser chat AI and a terminal-

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** judge
**Source:** cli

## Verdict Summary

**Decision:** Split work by side-effect authority—using the browser chat for stateless reasoning, architecture, and ambiguity resolution, and the terminal agent for stateful, repository-aware execution and testing—linked by a concise, structured markdown task brief that scales up to a committed decision record only for durable architectural choices.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 2/2
**Verdict author:** gemini (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

How should a solo developer split work between a browser chat AI and a terminal-based coding agent, and what handoff format minimizes friction transferring work between them?

## Round 1: Initial Responses

### Claude (claude-opus-4-8)

## Assessment
The split is real and worth formalizing: browser chat is a stateless reasoning surface, the terminal agent is a stateful execution surface with repo access. For a solo developer the binding constraint isn't capability — both can reason — it's **context fidelity across the handoff**, and that's where friction and 2am confusion actually live. The right design treats the handoff artifact as a durable contract checked into the repo, not a copy-paste of chat scrollback.

## Strengths
- **Clean division by side-effect authority.** Browser chat has no filesystem/exec access, so it's naturally the place for exploration, architecture debate, "should I even do this," and reviewing pasted diffs. The terminal agent is where irreversible things happen. Splitting on *who can touch state* is a more durable boundary than splitting on task difficulty, which drifts.
- **Cost/latency arbitrage is genuine.** Long open-ended reasoning (naming, tradeoffs, API design) is cheap and fast in a chat window; you don't want a repo-aware agent burning tool calls to think out loud. Reserve the expensive stateful loop for work that requires the repo.
- **The chat surface forces explicitness.** Because browser chat can't see your repo, you must state contracts — inputs, outputs, constraints — to get a useful answer. That artifact is exactly what the terminal agent needs anyway, so the split produces the handoff document as a byproduct.
- **Blast-radius containment.** Keeping the destructive-capable agent on a tight, well-specified task (rather than an open reasoning session) reduces the chance it wanders. Narrow scope in the surface that can `rm` things is good operational hygiene.

## Weaknesses / Gaps
- **Context loss is the dominant failure mode, and it's a *gap* in most workflows, not a bad choice.** The chat AI reasons over a mental model that never touches ground truth; the terminal agent sees ground truth but not the reasoning. Nobody owns the diff between "what we decided" and "what the code actually is." At 2am this is the killer: you inherit code whose rationale lives in an evaporated chat tab.
- **No single source of truth (bad, not just missing).** If decisions live in chat scrollback and implementation lives in the repo, the system has two authorities that silently diverge. Chat history is not durable, not searchable by the next engineer, and not versioned. This is technical debt by construction.
- **Handoff is manual and lossy by default.** Copy-pasting a plan drops the *why*, the rejected alternatives, and the constraints. The terminal agent then re-derives (or contradicts) decisions already made — the most expensive kind of rework.
- **Verification ownership is undefined.** Which surface confirms the work is correct? Chat can't run tests; the agent can but may mark its own homework. Without an explicit "agent runs tests and pastes real output back for chat-side review" loop, you get confident-but-unverified claims.

## Surprises
- **The highest-value handoff artifact is the rejected-alternatives list, not the plan.** Most people transfer *what to build*; the thing that actually prevents the terminal agent (and future-you) from thrashing is *what we already ruled out and why*. That's the context that's most expensive to reconstruct and most likely to be silently violated.
- **The handoff format should be optimized for the agent that receives it, but stored for the human who inherits it.** These pull in the same direction: a checked-in `docs/decisions/NNN-task.md` (a lightweight ADR) is both the cleanest agent prompt-context *and* the thing your on-call self reads in three years. The friction-minimizing format and the operability-maximizing format are the same file.

## Verdict
Split by side-effect authority (chat = stateless reasoning/review, terminal = stateful execution), and make the handoff a versioned, repo-committed decision record — a short ADR carrying goal, constraints, chosen approach, and rejected alternatives — rather than pasted chat text.

## Confidence
**Medium-high.** The side-effect-authority split and the "handoff must be durable and versioned" conclusion are robust regardless of tooling. What would change it: if your terminal agent gains reliable persistent project memory across sessions, the handoff-artifact burden drops and the split shifts toward the agent doing more of the reasoning in-place; and for genuinely throwaway/experimental work, a full ADR is overkill — a one-line intent comment suffices.


### Openai (gpt-5.6-terra)

## Assessment

A solo developer should use browser chat for thinking, decisions, and communication-heavy work; use the terminal agent for repository-aware implementation, tests, and verification. The lowest-friction handoff is a short, structured task brief stored with the code—not conversational transcripts.

## Strengths

- Browser chat is ideal for shaping requirements, comparing approaches, debugging concepts, and drafting user-facing text without granting filesystem access.
- Terminal agents are strongest when they can inspect the actual repository, make scoped changes, run tests, and report concrete file-level results.
- This split reduces context-switching: the browser handles ambiguity; the terminal handles execution.
- A repo-local Markdown handoff makes work durable, searchable, reviewable, and usable after interruptions.

## Weaknesses / Gaps

- Missing: a definition of “done.” Without acceptance criteria and verification commands, handoffs become vague and require repeated clarification.
- Missing: explicit scope boundaries—files or modules in scope, non-goals, and compatibility constraints.
- Bad: pasting long chat transcripts into the terminal agent; they hide the decision, waste context, and often include stale alternatives.
- Bad: asking either tool to own the whole lifecycle by default; that creates avoidable rework when requirements are still unsettled or implementation needs repository evidence.

## Surprises

- The best handoff is usually only 10–20 lines: goal, context, constraints, acceptance criteria, relevant files, and exact validation commands.
- The terminal agent’s final report should become the browser chat’s next input: changed files, behavior, tests run, remaining risks, and decisions needed.

## Verdict

Use browser chat as the product/architecture copilot and the terminal agent as the scoped implementer, exchanging concise repo-local task briefs and evidence-based completion reports.

## Confidence

High; this would change mainly if the terminal agent lacks safe repository access or the work is primarily research/design rather than code changes.


## Round 2: Critique

### Claude (claude-opus-4-8)

## 1. Agreement Map

**Strong agreement with both.** The split by *side-effect authority* (B) rather than task difficulty (A implies difficulty; B names the better axis) is correct — difficulty drifts, "who can touch state" doesn't. Both land on a **durable, repo-committed handoff** over chat scrollback. I fully endorse that.

**Where I sharpen B over A:** A's "10–20 line task brief" is the *forward* handoff. B's insight that the **rejected-alternatives list is the highest-value payload** is the one that survives to 2am. The on-call engineer doesn't fail from not knowing *what* was built — the code shows that. They fail from re-litigating a decision that was already closed, or "fixing" a deliberate constraint. Rationale is the expensive-to-reconstruct part.

**Where I disagree with both:** Both model the handoff as a **one-shot transfer**. Real solo-dev work is a *ping-pong loop* — chat → terminal → back to chat with a wall, → terminal again. Neither says who keeps the ADR honest across iterations. A committed decision record that isn't updated when the code contradicts it is **worse than no record** — it lies with authority. That's the failure mode a durable artifact *introduces*.

## 2. Missing Evidence

- **Does the solo dev actually maintain the ADR past round one?** The whole scheme rests on discipline sustained under time pressure. Instrument it: after a week, do the committed docs match the code, or are they stale at line 30? If they rot, the "single source of truth" is a fiction and pasted-chat was more honest.
- **Terminal agent's persistent memory across sessions** — B correctly flags this as the variable that collapses the whole handoff burden. That's testable *today* per tool; the answer reshapes the split.

## 3. Blind Spots (all of us, me included)

- **Trust-boundary leak.** Pasting repo context into browser chat crosses a *different* data boundary than the terminal agent. Nobody named it. For proprietary code this is a real contract, not hygiene.
- **The divergence-resolution rule is unwritten.** When chat's plan meets ground truth and loses, the ADR is now wrong. Needed contract: **repo is authority for facts; the decision record is authority for intent; any contradiction is a trigger to amend the record, not to ignore it.** Without that rule, two-authorities-diverging (B's own worry) is guaranteed, not merely risked.
- **Verification is asserted, not mechanized.** B says "agent may mark its own homework" but stops there. The fix is concrete: the completion report must carry **raw command output pasted verbatim**, not a summary sentence. "Tests pass" is a claim; a pasted exit-0 block is evidence. That distinction is the entire anti-hallucination control.
- **Ceremony tax.** A full ADR per task is overhead a solo dev abandons first under pressure. The format must degrade gracefully: one-line intent for throwaway, full record only when a decision closes an alternative someone would plausibly reopen.

## 4. Updated Verdict

Split by side-effect authority (chat = stateless reasoning/review, terminal = stateful execution) and carry a repo-committed decision record whose **most valuable field is rejected-alternatives** — but add the two contracts nobody wrote: **raw verbatim command output as the only accepted proof of "done," and an explicit rule that ground-truth/intent divergence must amend the record rather than silently fork it** — because a durable handoff artifact that goes stale is a more dangerous 2am liability than the chat tab it replaced.

**Confidence: High** on the contracts; **Medium** that a solo dev sustains the discipline — which is exactly why the format must degrade to one line, not demand a full ADR every time.


### Openai (gpt-5.6-terra)

## Agreement map

I agree with both: use browser chat for ambiguity, product tradeoffs, and drafting; use the terminal agent for repo-grounded implementation, testing, and evidence. The handoff should be concise, structured, and avoid raw transcripts.

I disagree with Proposal B’s default committed ADR. For a solo developer, creating and maintaining a decision record for each task is process overhead that will often be skipped or become stale. Commit durable decisions only when they affect architecture, operations, or future work; use a lightweight task brief for ordinary changes.

## Missing evidence

The choice depends on:

- How often work spans sessions or is interrupted.
- Whether the terminal agent has reliable project memory and safe test execution.
- Project longevity and likelihood of collaborators.
- Measured rework: how often unclear handoffs cause the agent to change the wrong thing or require another round.

A two-week trial tracking handoff length, follow-up prompts, and escaped defects would resolve whether task briefs or lightweight ADRs are earning their cost.

## Blind spots

Both proposals underweight operational discipline: the terminal agent needs explicit permission boundaries, a test-time budget, and a stopping rule for ambiguous or failing work. They also assume browser chat is inherently stateless—some tools preserve context, while terminal sessions can lose it—so classify work by repository access and side-effect risk, not surface alone.

Neither addresses the smallest useful loop: browser defines intent → terminal inspects and implements → terminal returns evidence → browser reviews only if a decision remains. Adding a formal review stage to every trivial change is unnecessary.

## Updated verdict

Use browser chat to clarify decisions and a terminal agent to execute narrowly scoped, repository-verified work; hand off a 10–20 line task brief with goal, constraints, scope, acceptance checks, and non-goals, escalating to a committed ADR only for durable decisions—because a solo developer needs reliable context without turning every change into documentation work.


## Synthesis (by gemini, non-participant)

## Overall Verdict
Split work by side-effect authority—using the browser chat for stateless reasoning, architecture, and ambiguity resolution, and the terminal agent for stateful, repository-aware execution and testing—linked by a concise, structured markdown task brief that scales up to a committed decision record only for durable architectural choices.

## Confidence Level
High. Both models strongly agree on the optimal functional split (stateless reasoning vs. stateful execution) and the format of the handoff (structured markdown over raw chat logs), differing only on the necessary level of documentation ceremony for routine tasks.

## Consensus Strengths
- **Clean division of labor:** Browser chat is isolated and ideal for exploration and debate; terminal agents possess filesystem access and are best suited for scoped, irreversible actions.
- **Structured handoffs:** Both agree that copy-pasting chat transcripts is an anti-pattern. Handoffs should be structured text containing goals, context, constraints, and scope.
- **Evidence-based verification:** The terminal agent must run actual commands and tests against the repository to ground the theoretical plan in reality.

## Consensus Weaknesses
- **Context loss and rework:** Default workflows lack explicit scope boundaries and acceptance criteria, causing agents to re-litigate decisions or thrash on execution.
- **Unverified claims:** Asking an agent to grade its own homework without demanding concrete proof often leads to confident but incorrect claims of completion.
- **Missing ping-pong dynamics:** Initial assumptions treat handoffs as a one-way street, ignoring the iterative loop where terminal agents hit roadblocks and must send evidence back to the browser for revised planning.

## Contested Points
- **Formality and Durability of the Handoff:** 
  - *Claude* argues the handoff should default to a version-controlled decision record (ADR) that highlights "rejected alternatives" to prevent future-you (or the agent) from repeating mistakes. 
  - *OpenAI* argues that mandating a committed ADR for every task is excessive overhead for a solo developer, which will inevitably rot and become stale. It advocates for ephemeral 10-20 line task briefs for routine work, reserving ADRs strictly for major architectural shifts.

## Blind Spots
- **Trust and Privacy Boundaries:** Pasting proprietary repository code into a SaaS browser chat crosses a different data and privacy boundary than using a local terminal agent, which neither model initially weighed.
- **Divergence Resolution Rule:** When the terminal agent encounters ground-truth reality that contradicts the browser's plan, there is no defined protocol for who updates the handoff document to prevent the code and the documentation from silently diverging.
- **Operational Discipline:** The lack of explicit budgets (e.g., max tool calls, strict stopping rules for failing tests) for the terminal agent to prevent runaway loops when a plan proves unworkable.

## Recommendations
1. **Enforce a Side-Effect Boundary:** Use browser chat exclusively to explore, debate, and draft plans. Use the terminal agent exclusively to execute, verify, and commit code.
2. **Standardize a Lightweight Handoff:** Pass work via a 10-20 line Markdown brief that includes the goal, strict scope, explicit non-goals, and exact validation commands. Do not paste chat scrollback.
3. **Include "Rejected Alternatives":** For non-trivial tasks, explicitly list what *not* to do and why in the brief, preventing the terminal agent from hallucinating its way into dead ends you already ruled out.
4. **Demand Verbatim Proof:** Require the terminal agent to pass back raw, verbatim command output (e.g., passing test logs) to the browser chat as the only acceptable proof of "done." 
5. **Scale Documentation Ceremony:** Degrade gracefully based on task size. Use disposable task briefs for everyday bug fixes and features, but commit the brief into the repository as a permanent ADR when the decision establishes a lasting pattern or constraint.
