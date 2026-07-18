# AI Council Debate: How should a solo developer split work between a browser chat AI and a terminal-

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** judge
**Source:** cli

## Verdict Summary

**Decision:** Split work by failure cost—using browser chat for divergent exploration and terminal agents for bounded repository execution—and hand off work via a lightweight, disposable intent note combined with an executable test, keeping the repository as the strict single source of truth.
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

The right split maps to where each tool's failure modes are cheapest: browser chat for divergent, exploratory thinking where being wrong costs a re-read; terminal agent for convergent, file-mutating work where being wrong costs a broken repo. The handoff friction problem is real but usually over-engineered — the actual bottleneck for a solo dev is not format elegance but reconstructing *context and intent* that lived in the chat but never made it into the repo. Treat that lost intent as your primary technical debt.

## Strengths

- **The division is genuinely operable.** Browser chat excels at breadth (comparing approaches, rubber-ducking, reading docs it can search) where output is disposable prose. Terminal agents excel at grounded work — they see your actual file tree, run tests, and get real error output. Keeping these separate means each operates in the mode where it's least likely to hallucinate confidently.
- **The handoff artifact can double as project documentation.** If you force the handoff into a durable file (not a pasted blob), the same act that transfers work leaves a record the future you — or the 2am you — can read. This is the rare case where friction reduction and maintainability point the same direction.
- **Terminal agents produce inspectable diffs.** Unlike chat, which produces suggestions you manually apply, a terminal agent's output is a git diff. That's an explicit contract: you review a diff, not a narrative claim about what changed. This is a strong operability property and should anchor your workflow.

## Weaknesses / Gaps

- **(Bad) Copy-paste handoff loses execution context.** Pasting a chat's code recommendation into a terminal agent strips the *why*. The terminal agent then re-derives intent from the code alone and often diverges. The paste feels frictionless but silently drops the highest-value information.
- **(Missing) No stated source of truth.** Your question assumes two tools; it doesn't say which one *owns* the current state. Without a designated authority (the repo, always the repo), you get two agents confidently working from stale mental models. This is the split-brain problem, and solo devs hit it constantly because there's no second person to notice the drift.
- **(Missing) No verification gate between handoffs.** The dangerous moment is trusting chat-generated code that the terminal agent then builds upon without running. Each unverified handoff compounds the last. There's no test-before-integrate step in the workflow as framed.
- **(Bad, if adopted) A rigid handoff schema is over-engineering for one person.** A solo dev doesn't need a formal ticket format. Enforcing YAML frontmatter and structured fields adds cognitive load you'll abandon in a week. The format should be the minimum that survives your own forgetting.

## Surprises

- **The best handoff format is often a failing test, not prose.** Instead of transferring "here's what to build" as text, have the browser chat help you write a test that encodes the desired behavior, then hand *that* to the terminal agent. The test is an executable contract — unambiguous, verifiable, and it survives context loss. This inverts the usual "spec-then-code" handoff and eliminates most re-derivation error.
- **The friction is asymmetric, and the expensive direction is terminal→chat.** Terminal→chat is where solo devs lose the most, because summarizing a sprawling repo state back into a chat window is lossy and tedious, so people skip it and the chat gives advice based on a fictional codebase. Most people optimize the chat→terminal direction (the easy one) and neglect the direction that actually causes bad advice.

## Verdict

Split by failure cost — chat for reversible exploration, terminal agent for repo mutation — and make the repo the sole source of truth with a lightweight handoff (ideally a failing test or a short intent-plus-diff note), not a formal schema.

## Confidence

**Medium-high.** The architectural principles (single source of truth, executable contracts, verify-before-integrate) are tool-independent and durable. What would change my answer: how autonomous the terminal agent is (a highly agentic one that reads the whole repo itself makes the terminal→chat handoff nearly irrelevant), and whether your work is greenfield (favors chat-heavy exploration) or maintenance on an existing large codebase (favors terminal-heavy, chat as a narrow consultant).


### Openai (gpt-5.6-terra)

## Assessment
A solo developer should use the browser chat AI for thinking-heavy, ambiguous, and communication-oriented work, while using the terminal coding agent for repository-grounded implementation, verification, and repetitive mechanical changes. The lowest-friction handoff is not conversational prose alone: it is a small, version-controlled task packet that names the goal, constraints, files, acceptance checks, and current state.

## Strengths
- **Natural division of labor:** Browser chat is better for architecture tradeoffs, debugging hypotheses, requirements clarification, and drafting plans where broad context matters more than direct repository access.
- **Terminal agents are execution-oriented:** A coding agent with shell and repository access can inspect actual files, make edits, run tests, check diffs, and avoid the copy/paste errors common when implementation is mediated through a browser.
- **Structured handoffs reduce context loss:** A concise Markdown task file or issue prevents the terminal agent from having to infer intent from a transcript, especially after interruptions or multiple iterations.
- **Version control becomes the shared memory:** Git commits, branches, diffs, test results, and issue references are more reliable handoff artifacts than chat history. They also make rollback and review feasible for a solo operator.
- **A solo developer can keep this lightweight:** One `TASK.md`, GitHub issue, or an entry in a `tasks/` directory is sufficient; this does not require a project-management system or elaborate agent orchestration.

## Weaknesses / Gaps
- **Missing: explicit acceptance criteria.** “Implement feature X” is not enough; the handoff should state observable outcomes, edge cases, and commands or tests that demonstrate completion.
- **Missing: a stop/escalation rule.** The terminal agent should know when to stop and return findings rather than making speculative architectural changes—for example, when tests fail for unrelated reasons or a data migration is required.
- **Bad: using the browser AI as a source of repository truth.** It may reason well, but it cannot reliably know the current code, dependency versions, uncommitted edits, or test state unless those are supplied.
- **Bad: giving the coding agent an unconstrained broad objective.** “Refactor authentication” or “fix the app” invites excessive scope and difficult-to-review changes; split work into independently verifiable slices.
- **Missing: security and operational boundaries.** Define which commands, environments, secrets, migrations, deployment actions, and destructive operations the agent may use. A solo developer still needs safeguards against accidental production impact.

## Surprises
- The most effective handoff is often a **Git commit plus a short next-step note**, not a detailed prompt: the commit establishes the exact starting state, and the note states what remains.
- Browser chat and terminal agents should not alternate on every small decision. Switching tools has cognitive overhead; use browser chat to produce a sufficiently stable plan, then let the terminal agent execute a complete vertical slice before returning to discussion.

## Verdict
Use browser chat for planning and judgment, terminal agents for bounded repository execution, and hand off through a version-controlled Markdown task packet tied to a branch, commit, and executable acceptance checks.

## Confidence
High; this would change with details about the specific agents’ repository access, persistence, test reliability, and the developer’s project complexity.


## Round 2: Critique

### Claude (claude-opus-4-8)

## 1. Agreement Map

**Where I agree with both:**

- **The repo is the single source of truth.** Both proposals land here, and it's correct. The browser chat cannot know your uncommitted edits, actual dependency versions, or test state. This is the load-bearing principle and everything else follows from it.
- **Split by nature of work: chat for divergent/judgment, terminal for convergent/mutation.** Both frame this well. Proposal A's "split by failure cost" is the sharper articulation — it tells you *why* the boundary sits where it does, not just where it sits.
- **Executable acceptance checks beat prose handoffs.** Proposal A's "failing test as handoff" and Proposal B's "acceptance criteria / commands that demonstrate completion" are the same instinct. This is the strongest shared insight and it maps directly to my lens: a test is an explicit contract the 2am engineer can *run*, not a narrative they have to trust.
- **Don't over-formalize for one person.** Both warn against heavyweight schemas. Agreed — a solo dev will abandon YAML frontmatter within a week, and abandoned process is worse than no process because it creates false confidence in stale artifacts.

**Where I lean toward one over the other:**

- Proposal A's **asymmetric friction insight (terminal→chat is the expensive direction)** is the most operationally important observation either made, and Proposal B misses it. The failure mode isn't losing a plan; it's the chat confidently advising against a *fictional codebase* because summarizing repo state back up is tedious and gets skipped.
- Proposal B's **stop/escalation rule and operational boundaries** are the most important thing Proposal A misses. "What commands may the agent run, and when must it stop and return findings" is precisely the 2am question. An unconstrained terminal agent that decides to run a migration or force-push is the disaster scenario, and A never names it.

**Where I mildly disagree with both:**

- Both treat the handoff artifact as roughly a solved problem once you pick a format. Neither reckons with the fact that the artifact *decays*. A `TASK.md` that's three handoffs stale is a liability, not documentation.

## 2. Missing Evidence That Would Resolve Disagreements

- **How agentic is the terminal agent?** Proposal A flags this correctly. If the agent reads the whole repo autonomously, the terminal→chat handoff and much of the "current state" section of the task packet become redundant. If it's a narrow apply-this-diff tool, the structured packet is essential. The entire answer bifurcates on this.
- **Is this greenfield or maintenance?** Greenfield favors chat-heavy exploration; maintenance on a large codebase favors terminal-first with chat as a narrow consultant. Both proposals gesture at this; neither commits.
- **What's the test suite's reliability and runtime?** The "failing test as contract" handoff only works if tests are fast and trustworthy. If the suite takes 20 minutes or is flaky, the executable-contract handoff quietly degrades back into prose, and nobody notices until it's failed three times.

## 3. Blind Spots All of Us Missed

- **The handoff artifact's own lifecycle.** We all recommended a durable file/note. None of us said **when to delete it.** A `TASK.md` or intent-note that outlives its task becomes exactly the kind of stale mental model we warned against — except now it's committed and looks authoritative. The rule must be: the artifact dies when the test goes green and the diff merges. Its half-life is one handoff.
- **The context window is a silent second source of truth.** The browser chat accumulates hundreds of messages of decisions that never entered the repo. That transcript *is* a source of truth in practice, competing with the repo, and it evaporates when you close the tab or the context gets truncated. The split-brain problem A named isn't just chat-vs-terminal — it's chat-transcript-vs-repo, and the transcript always loses eventually. The mitigation: decisions worth keeping must be flushed to a commit message or ADR *at the moment they're made*, not reconstructed later.
- **Nobody costed the human review step.** Both proposals lean on "you review the diff." For a solo dev, the diff-review is the only quality gate — there's no second reviewer. If the terminal agent produces a 900-line diff, review doesn't happen; it gets skimmed and rubber-stamped at 2am. The vertical-slice constraint (Proposal B) exists precisely to keep diffs reviewable, but neither of us stated the actual operability rule: **cap the diff size the agent may produce before returning for review.**
- **Provenance.** In three years, when something breaks, the engineer (probably still you) needs to know *which code was AI-generated and under what intent*. Neither proposal ties the handoff artifact to the commit as provenance. A commit trailer (`Generated-by: terminal-agent, intent: <link to task note>`) costs nothing and answers the archaeology question later.

## 4. Updated Verdict

Split by failure cost — browser chat for reversible exploration and judgment, terminal agent for bounded, verifiable repo mutation — with the repo as sole source of truth; hand off via a *disposable* task note plus an executable acceptance check, constrain the agent with an explicit stop-rule and a diff-size cap, and flush durable decisions into commit messages the moment they're made rather than reconstructing intent later.

**Justification:** The two enduring failure modes for a solo dev are split-brain (agents working from stale state) and unreviewed change (no second pair of eyes), so the workflow's job is to keep state authoritative in one place and keep every change small enough to actually inspect at 2am.

**Confidence:** High on the principles (single source of truth, executable contracts, bounded diffs, stop-rules are all tool-independent and durable); medium on the specific handoff mechanics, which collapse or expand depending on how autonomously the terminal agent reads the repo.


### Openai (gpt-5.6-terra)

## 1. Agreement map

### Agreement
- **Split by grounding and reversibility:** use browser chat for exploration, tradeoffs, requirements clarification, and drafting; use the terminal agent for repository-aware implementation, testing, and diffs.
- **The repository/Git is the source of truth:** neither chat transcript nor an agent’s claimed summary should outrank the checked-out commit, diff, and test state.
- **Handoffs need durable intent plus verification:** a task should carry enough context to avoid re-litigating decisions, and completion should be demonstrated with tests, commands, or observable checks.
- **Keep scope bounded:** a solo developer should hand a terminal agent a vertical slice or narrowly reviewable change, not a vague multi-system mandate.
- **Avoid process theater:** this should be a lightweight convention, not a ticket taxonomy or agent-orchestration platform.

### Disagreement / tension
- **Primary handoff artifact:** Proposal A favors a failing test or brief intent-and-diff note; Proposal B favors a structured Markdown task packet. Both are useful, but a mandatory structured packet is too heavy for every small task, while a failing test alone is insufficient for migrations, UX changes, operational constraints, or architectural decisions.
- **How much browser→terminal planning is needed:** A implies the terminal agent can often derive much of the implementation from the repo; B gives more weight to explicit task context. Pragmatically, the needed detail should scale with ambiguity and blast radius, not be fixed.
- **Importance of terminal→browser handoff:** A correctly highlights that this is often the more neglected direction. B’s commit-plus-note approach addresses it, but should explicitly include current state, failures, and decisions made—not merely “what changed.”

## 2. Missing evidence

The right workflow depends on evidence about:

- **Actual tool capability:** Can the browser chat read the repo, branch, diffs, logs, issue tracker, and docs? Can the terminal agent persist context, inspect history, and safely run commands?
- **Project health:** Are tests fast, deterministic, and representative? “Use a failing test as the contract” is much less useful when tests are flaky, slow, or absent.
- **Work type and risk:** Greenfield UI work, bug fixes, refactors, schema migrations, deployment changes, and incident response need different handoff depth and safety gates.
- **Developer operating pattern:** Frequency of interruptions, context switching, and how often work spans days matters more than team size alone. A persistent task note pays off when work is interrupted; it may be wasteful for a 20-minute fix.
- **Agent reliability and permissions:** Evidence from a week or two of usage—how often the terminal agent broadens scope, breaks tests, misses constraints, or needs re-prompting—should determine how prescriptive the packet and guardrails must be.

## 3. Blind spots

- **Do not make the handoff artifact another stale source of truth.** A `TASK.md` that is not updated becomes misleading documentation. It should point to a branch/commit and be deleted, closed, or updated when the task ends.
- **Separate “plan” from “authorization.”** A good plan does not grant permission to run migrations, delete data, change infrastructure, spend money, access secrets, or deploy. Explicit command/environment boundaries matter.
- **Optimize for reviewability, not agent autonomy.** The key solo-developer constraint is not whether the agent can finish a large task; it is whether the developer can understand and safely review the resulting diff in one sitting.
- **Use decision records only for irreversible choices.** Browser chat may surface architectural decisions worth preserving, but writing a record for every implementation detail is overhead. Capture only decisions that future work could reasonably reopen.
- **Cost and latency affect the split.** Long browser conversations and repeated full-repo terminal-agent passes can be expensive and slow. A compact repo status command, targeted file list, and bounded task reduce both.
- **Prompt injection and untrusted repository content deserve a boundary.** Terminal agents may read generated files, issues, logs, dependencies, or docs containing instructions. Treat repository text as data, not authority, especially before shell or deployment actions.

## 4. Updated verdict

Use browser chat to settle ambiguous decisions and produce a bounded intent, then give the terminal agent a branch/commit, a short Markdown note with constraints and acceptance checks, and—where practical—an executable test, because that is the smallest workflow a solo developer can reliably resume, verify, and review without turning handoffs into bureaucracy.


## Synthesis (by gemini, non-participant)

## Overall Verdict
Split work by failure cost—using browser chat for divergent exploration and terminal agents for bounded repository execution—and hand off work via a lightweight, disposable intent note combined with an executable test, keeping the repository as the strict single source of truth.

## Confidence Level
High — Both panelists fundamentally agreed on the core architectural principles (tool division by capability/risk, git/repo as the single source of truth, executable contracts over prose, and avoiding heavy schemas for solo developers), though minor mechanics flex depending on specific agent autonomy.

## Consensus Strengths
* **Clear Division of Labor:** Browser chat excels at ambiguous, exploratory thinking and architecture tradeoffs; terminal agents excel at convergent, grounded execution, file mutation, and test validation.
* **Repository as Source of Truth:** Relying on the actual git state, diffs, and test results prevents the "split-brain" failure mode where agents confidently operate on a fictional or stale mental model of the codebase.
* **Executable Contracts:** Verifying work via failing tests or explicit terminal commands is vastly superior to relying on prose, as it eliminates ambiguity and provides a concrete finish line.
* **Lightweight Process:** Both agree that formal schemas, Jira-like structures, or complex agent orchestration will be quickly abandoned by a solo developer; the handoff must remain frictionless. 

## Consensus Weaknesses
* **Unbounded Agent Scope:** Handing an agent a broad objective without explicit bounds invites massive, unreviewable diffs and scope creep.
* **Missing Stop/Escalation Rules:** Relying on autonomous execution without defining operational boundaries (e.g., forbidding data migrations, force-pushes, or deploying without permission) is highly dangerous.
* **Context Loss via Copy-Paste:** Blindly copying browser chat output into a terminal agent strips the *intent* (the "why"), forcing the terminal agent to re-derive context and increasing hallucination risk.
* **Stale Task Artifacts:** Using a durable file (like `TASK.md`) becomes a liability if the file is not actively maintained, as it quickly rots into misleading documentation.

## Contested Points
* **Primary Handoff Artifact Format:** 
  * *Claude's best argument:* The handoff should be a failing test (or a minimal intent-plus-diff note). It is an unambiguous, executable contract that perfectly survives context loss without the overhead of markdown structures.
  * *OpenAI's best argument:* A structured markdown task packet is necessary because failing tests alone cannot capture architectural bounds, UX changes, or operational constraints (like "don't modify the database schema").
* **Direction of Maximum Friction:** 
  * *Claude* argued that the hardest and most dangerous friction is *terminal-to-browser*—summarizing repo state back to the chat is tedious, leading devs to skip it and causing the chat AI to hallucinate based on outdated info.
  * *OpenAI* focused more heavily on optimizing the *browser-to-terminal* flow, emphasizing precise task packets and git commits to establish the starting state.

## Blind Spots
* **Artifact Lifecycle and Disposal:** Initially, neither addressed *when* to delete the handoff artifact. A `TASK.md` must be explicitly ephemeral—deleted or archived the moment the PR merges—so it doesn't become a competing source of truth.
* **The Context Window as a Competing Source of Truth:** The browser chat quietly accumulates decisions that never make it into the codebase. When the context window resets, that history evaporates. 
* **The Human Review Bottleneck:** A solo developer is the final and only QA. Neither panelist initially emphasized that agent output diffs must be strictly size-capped; otherwise, the developer will rubber-stamp 900-line diffs out of fatigue.
* **Provenance and Traceability:** Missing a mechanism for future-you to know what code was AI-generated. A simple git commit trailer (e.g., `Generated-by: terminal-agent`) costs nothing but saves massive debugging time later.

## Recommendations
1. **Establish the Boundary:** Use browser chat exclusively to brainstorm, plan, and define intent. Use the terminal agent exclusively for file mutation and testing.
2. **Implement Ephemeral Handoffs:** Bridge the two using a temporary markdown note combined with an executable test or script. State the goal, the boundaries, and the test command. 
3. **Delete Upon Merge:** Delete the handoff note the exact moment the task is complete and tests go green to prevent stale documentation.
4. **Set Explicit Stop Rules:** Give the terminal agent strict constraints in the handoff (e.g., "Do not run database migrations; stop and ask for review if tests fail twice").
5. **Flush Decisions to Git:** Force architectural decisions made in the browser chat into commit messages or lightweight Architecture Decision Records (ADRs) immediately. Do not leave them stranded in the chat context window.
6. **Cap Diff Sizes:** Instruct terminal agents to work in small, vertical slices to ensure the resulting diffs remain small enough for you to rigorously review in one sitting.
