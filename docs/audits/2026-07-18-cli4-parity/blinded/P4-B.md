# AI Council Debate: What mechanism should a handoff use to confirm a fresh chat has internalized the

**Panel:** claude-opus-4-8, gpt-5.6-terra
**Synthesizer:** gemini (non-participant)
**Rounds:** 2
**Panel Mode:** custom
**Debate Mode:** pick
**Source:** cli

## Verdict Summary

**Decision:** Implement a hybrid verification gate that combines binary falsifiability with behavioral pre-commitment, explicitly avoiding any corruption of the canonical bundle text.
**Dissent:** non-unanimous (see minority report)
**Panel seated:** 2/2
**Verdict author:** gemini (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

What mechanism should a handoff use to confirm a fresh chat has internalized the bundle - rather than only paraphrased it - before it begins substantive work?

## Round 1: Initial Responses

### Claude (claude-opus-4-8)

## Position

Gate substantive work behind a short **application-and-falsification quiz**: the fresh chat must answer 3-5 counterfactual/decision questions it cannot pass by paraphrase, and must catch at least one deliberately planted contradiction seeded into the bundle. Paraphrase is a readback, not a comprehension test — replace it with a test that only correct application can pass.

## Alternatives Considered

- **Structured paraphrase / readback** (have the chat summarize the bundle in its own words). Rejected: fluent restatement is exactly what a language model does cheaply and unreliably. It confirms the tokens were ingested, not that constraints will bind under pressure. It produces false confidence — the worst failure mode for a gate.
- **No gate; correct on the fly** (let it start and fix drift as it appears). Rejected: the failures a bundle exists to prevent (re-litigating a settled decision, violating an invisible constraint) are often irreversible or expensive by the time they surface. The cost of a 90-second gate is trivial against a half-day of work built on a misread premise.

## Decision Criteria

- **Discriminating power** — does passing actually correlate with correct downstream behavior, or can a bad session pass anyway?
- **Cost per handoff** — the gate must cost seconds-to-minutes, or people route around it.
- **Falsifiability** — the mechanism must be able to *fail* a genuinely confused session, not just always pass.
- **Author burden** — whoever writes the bundle must be able to produce the check cheaply, ideally as a byproduct of writing the bundle.
- **Operability at 2am** — an on-call human (or automated harness) must be able to read the result and know go/no-go without expert judgment.

## Assumptions

- The bundle contains at least a few **non-obvious decisions or constraints** (settled trade-offs, forbidden paths, invariants). If the bundle is trivial, any gate is overhead and paraphrase is fine.
- The fresh chat's failure mode of concern is **confident misapplication**, not refusal — i.e., it will act, and acting wrong is costly.
- The handoff author can invest ~5 minutes to author checks, and can encode a ground-truth answer key.

## Recommendation

**Now:**
- Every bundle ships with a **Verification Block**: 3-5 questions plus one planted contradiction, and a hidden answer key.
- Question types, in priority order: (1) **counterfactual** — "if constraint X were relaxed, what changes?"; (2) **application** — "here is a new sub-task; which bundle constraint governs it and why?"; (3) **negative** — "name something this design explicitly rejected and the reason." All three require reasoning from the bundle, not surface recall.
- Seed **one deliberate contradiction** (e.g., a constraint restated with a flipped value in a secondary section). A session that internalized the bundle flags it; one that paraphrased will smooth over it. This is the single highest-signal test.
- The chat must answer, then a check (human or scripted against the key) returns **go / no-go**. On no-go, re-inject the missed items and re-gate — do not proceed.

**Not now:**
- Do not rely on paraphrase, "confidence" self-reports, or "do you understand? (yes/no)."
- Do not build an elaborate scoring rubric or ML-graded evaluation — binary pass on the planted contradiction plus a human skim of the reasoning is enough.

**Defer:**
- Automated grading of free-text answers and a reusable question-bank generator — worth it only once handoff volume is high enough (say, >5/week) to amortize.

## Rationale

- **The planted contradiction is the load-bearing mechanism.** Paraphrase and application questions can occasionally be faked from context; silently reconciling a contradiction is something only a session that actually built a model of the constraints will do. It converts "did you read it" into a falsifiable behavior. One good planted contradiction outperforms ten paraphrase prompts.
- **Application questions test the transfer that matters.** The bundle exists so the next session acts correctly on *new* work. Testing on new mini-tasks measures exactly the thing you care about; testing on restatement measures a proxy.
- **Cost is bounded and front-loaded.** ~5 min author-time, ~60-90 sec runtime per handoff. Against the realistic downside — a session confidently reversing a settled decision — the expected value is strongly positive even if only 1 in 10 handoffs would have drifted.
- **Binary go/no-go is 2am-legible.** The on-call decision is "did it catch the trap? yes → proceed." No interpretation of prose quality required.
- **It hardens the bundle itself.** Writing falsifiable questions surfaces bundle ambiguities the author didn't know were there — the check doubles as a bundle-quality review.

## Trade-offs

- **~1-2 minutes of latency** on every handoff, and author time per bundle. You give up "just start."
- **Not exhaustive** — passing the gate proves the session grasped the *tested* constraints, not all of them. It's a spot-check, not a proof; you accept residual risk on unprobed items.
- **Some author skill required** to write a contradiction subtle enough to discriminate but not so buried it's unfair. Bad questions give false assurance.

## Risks

- **Teaching-to-the-test / gate theater** (likelihood: medium, impact: medium): questions become boilerplate the author copies, losing discriminating power. Mitigate by tying at least the contradiction and one application question to *this bundle's* specific decisions, not a template.
- **False fail on a competent session** (likelihood: low, impact: low): the session flags a real bundle inconsistency you thought was planted. Mitigate: treat any flagged contradiction as a signal to check the bundle, not just the session.
- **The contradiction leaks the answer** (likelihood: low, impact: medium): if it's too obvious, even a paraphrasing session catches it, and you learn nothing. Mitigate: place it in a secondary/detail section, phrased plausibly.
- **Author skips the gate under time pressure** (likelihood: medium, impact: high): the check only works if used. Mitigate: make the Verification Block a required, non-optional section of the bundle template so an empty one is visibly incomplete.

## Signals to Revisit

- **Handoff volume rises past ~5/week** → invest in automated grading and a generated question-bank (the deferred item).
- **Gate pass rate hits ~100% over many handoffs** → it's not discriminating; the contradictions are too easy or templated — make them harder or the gate is decorative.
- **Post-gate drift still occurs** (sessions pass then violate a constraint mid-work) → the failure is *retention over a long session*, not initial internalization; add periodic mid-session re-checks rather than a stronger entry gate.
- **Bundles turn out to be mostly trivial** in practice → drop the gate to a lightweight paraphrase; the machinery isn't earning its cost.


### Openai (gpt-5.6-terra)

## Position
Use a short, scored “proof of internalization” gate: require the fresh chat to produce a task-specific execution brief with constraints, decisions, open risks, and a concrete first action—then validate it against a small set of hidden or explicit acceptance checks.

## Alternatives Considered
- Free-form summary acknowledgment: rejected because fluent paraphrase does not demonstrate that constraints will guide subsequent decisions.
- Immediate substantive work with correction loops: rejected because misunderstandings surface after costly or irreversible work has begun.

## Decision Criteria
- Must distinguish operational understanding from textual recall.
- Must take under 5 minutes and require little maintainer effort.
- Must be reusable across handoffs without bespoke evaluation infrastructure.
- Must expose consequential misunderstandings before edits, external actions, or design decisions.
- Must work when the bundle contains ambiguous or conflicting information.

## Assumptions
- The handoff bundle contains enough concrete context to identify priorities, constraints, and expected outcomes.
- A human or orchestrator can compare a short response against a checklist.
- The cost of a brief pre-work gate is lower than rework from an incorrect start.

## Recommendation
What to do now: require a four-part readiness response before substantive work:

1. **Non-negotiables:** state the 3–5 constraints that must not be violated.
2. **Decision model:** identify the intended outcome, the chosen approach, and the most important rejected alternative.
3. **Applied scenario:** answer one bundle-derived “what would you do if…” question that requires applying a constraint, not repeating it.
4. **First move and stop condition:** name the first concrete action, what evidence it will produce, and when to pause for clarification.

Validate this against a compact checklist, ideally including one hidden counterfactual or conflict in the bundle. Approve only if the response is materially correct and actionably specific.

What not to do now: do not use a generic “summarize the context” prompt or require exhaustive recall.

What to defer until later: automate scoring only after repeated handoffs reveal stable failure modes and enough examples exist to define reliable checks.

## Rationale
- A summary tests retrieval; a scenario tests whether constraints affect choices under pressure.
- Four answers are usually 150–300 words and take roughly 2–4 minutes to assess.
- Naming a first action makes misunderstandings observable before the chat changes code, contacts users, or commits to an architecture.
- A hidden check prevents the chat from optimizing for a visible template while missing the bundle’s actual decision logic.
- This scales to a small team: one owner can maintain a 3–5 item checklist per handoff, rather than a formal evaluation system.

## Trade-offs
You add a small start-up delay and may reject a capable chat whose response is concise but insufficiently explicit. The gate verifies readiness for the stated task, not universal mastery of every detail in the bundle.

## Risks
- **Checklist becomes rote** — likelihood: medium; impact: medium. Mitigate by rotating the applied scenario and including one decision-specific hidden check.
- **Bundle is internally inconsistent** — likelihood: medium; impact: high. Treat correct identification of the conflict and a pause request as a passing result.
- **Evaluation becomes burdensome** — likelihood: low to medium; impact: medium. Cap the gate at four prompts and limit validation to high-consequence handoffs.
- **False confidence from a strong written response** — likelihood: low; impact: high. Require the first work artifact to be reversible and review it before broader execution.

## Signals to Revisit
- More than 10–15% of approved handoffs require material correction within their first substantive action.
- Reviewers consistently find the same category of missed constraint.
- The handoff bundle grows enough that manual checks exceed roughly 5 minutes.
- Work becomes high-risk or irreversible enough to justify automated evaluation, staged permissions, or a human approval checkpoint.


## Round 2: Critique

### Claude (claude-opus-4-8)

Both proposals land in the same neighborhood — kill paraphrase, test application, seed a contradiction — and both are basically right about the failure mode. My job is to find where each one breaks in year three, at 2am, in the hands of someone who didn't write it.

---

## Proposal A — Scored four-part readiness brief

**Steelman.** If handoffs are moderately frequent, owned by one maintainer, and the bundles carry real decision logic, then A is close to ideal: a fixed four-slot template (non-negotiables, decision model, applied scenario, first move + stop condition) gives the reviewer a *stable shape* to skim. The "first move and stop condition" slot is the part nobody else emphasizes and it's the most operationally valuable — it makes the misunderstanding observable *before* an irreversible action, and it names the pause point. That's genuinely 2am-legible: you're not grading prose, you're checking "is the first action safe and is the stop condition sane."

**Assessment: Partially agree.** The four-part structure is sound and the stop-condition slot is the best single idea in either proposal. Where I pull back: A hedges on the discriminating mechanism. It *optionally* includes "one hidden counterfactual or conflict" — "ideally including." That optionality is where the gate quietly decays into a template you fill out. A's own risk table names "checklist becomes rote," rates it medium/medium, and then makes the antidote optional. That's backwards.

**Strongest point.** "First move and stop condition." Requiring the chat to name what evidence its first action produces and when to pause converts internalization into an *observable pre-commitment*. This is the only mechanism in either proposal that keeps working after the entry gate — it constrains the first real action, not just the quiz.

**Weakest assumption.** "A human or orchestrator can compare a short response against a checklist" in under 5 minutes, materially and correctly. Judging whether a four-part free-text brief is "materially correct and actionably specific" *is* expert judgment. Unlike B's binary trap, A's pass criterion is a prose-quality call. At 2am the tired reviewer rubber-stamps a fluent brief — which is exactly the false-confidence failure the gate exists to prevent. A imports the paraphrase problem into the grading step.

**Hidden assumptions.** (1) That the four slots are the *right* four for every bundle — a fixed template assumes the important constraints are always expressible as "non-negotiables + one scenario," but some bundles' risk lives in sequencing or in what's deliberately *out* of scope, which the template has no slot for. (2) That the reviewer and the bundle author are competent at the same things — A needs the reviewer to already know the bundle well enough to catch a plausible-but-wrong brief, which means the gate provides the *least* value precisely when the reviewer is a fresh on-call who doesn't have that context.

**Overlooked risks.** A never addresses **grader drift** — over months, two different reviewers apply different bars, and "materially correct" means whatever the approver was willing to tolerate that day. There's no answer key, so there's no anchor. B has a key; A has a vibe. Also unaddressed: what happens when the brief is *right* but the bundle is *wrong* — A folds "identify the conflict" into a passing result but only if the optional conflict was planted.

---

## Proposal B — Application quiz + planted contradiction

**Steelman.** If you accept that the only *falsifiable* signal of a built mental model is silently-reconciled-vs-flagged contradiction, then B is the rigorous version of A. Binary go/no-go on the trap removes grader drift and expert judgment from the critical path — the very weakness I just charged against A. The answer key makes the gate reproducible across reviewers and across years, which is the property I care about most. And B's observation that the gate "hardens the bundle itself" is real: writing falsifiable questions surfaces ambiguities the author didn't know were there.

**Assessment: Partially agree — with one serious reservation that A doesn't share.** The planted contradiction is a genuinely clever falsification mechanism *and* an operability landmine, and B never sees the second half.

**Strongest point.** "The planted contradiction is the load-bearing mechanism… silently reconciling a contradiction is something only a session that actually built a model of the constraints will do." This is correct and it's the sharpest epistemics in either document. Paraphrase and even application questions can be faked from surface context; smoothing over a planted inconsistency cannot. It converts "did you read it" into a behavior.

**Weakest assumption.** That the handoff author can reliably author a contradiction "subtle enough to discriminate but not so buried it's unfair," *repeatedly, per bundle, under time pressure.* B rates the two adjacent risks (contradiction too obvious → learns nothing; teaching-to-the-test) as low/medium, but the real failure is that **contradiction authorship is a rare skill and the cost lands every single handoff.** A bundle author good enough to plant a fair trap is good enough to not need the gate; the median author will either plant an obvious one (no signal) or an unfair one (false fails). The mechanism's discriminating power is a function of author skill that B assumes is abundant.

**Hidden assumptions.** (1) **That corrupting the source of truth is free.** B treats the bundle as a disposable test fixture. But the bundle is also *documentation* — the on-call reads it to act. A planted contradiction means the canonical handoff now contains a deliberate falsehood in a "secondary/detail section." If the trap isn't stripped after the gate, someone downstream reads the flipped value and acts on it. B has no lifecycle for the contradiction: who removes it, when, in which copy? That's a maintainability defect the proposal never names. (2) **That "go/no-go on catching the trap" generalizes** — passing means the session caught *the one thing you thought to trap*, and B admits it's a spot-check, but then leans on it as *the* high-signal test. A single trap is a single sample; it's legible precisely because it's narrow, and narrow is why it under-covers.

**Overlooked risks.** The **stale-trap / divergent-copy problem** above is the big one. Second: **the trap trains the wrong reflex.** If sessions learn that bundles contain planted lies, a well-calibrated session starts *distrusting the bundle* — flagging real, load-bearing constraints as suspected traps and pausing on things it should have accepted. You've taught the fresh chat that its source of truth is adversarial. Third, B under-weights that its own mechanism has **no post-entry component** — it explicitly punts retention drift to "mid-session re-checks," but the entry quiz does nothing to constrain the *first action*, which is A's whole strength.

---

## Revised recommendation

**I update — but toward a synthesis of the two mechanisms, not toward either proposal as written, and not by splitting the difference on the thing they disagree about.**

They actually disagree on very little. The real disagreement is: **A's optional hidden conflict vs. B's mandatory planted contradiction as load-bearing.** On that specific axis, **B is right and A is wrong** — an optional discriminator decays to decoration, and B's falsifiability argument is airtight. That's my update: the contradiction check must be **mandatory, not "ideally."**

But B's contradiction mechanism is operationally reckless as specified, and A has the better *containment* mechanism. So:

**Adopt B's falsification requirement, reject B's method of delivering it.** Do not corrupt the bundle. A planted lie in the canonical handoff is a P0 operability hazard — it violates "explicit contracts over clever tricks," and it has no lifecycle. Instead, ship the contradiction as an **explicitly-labeled, out-of-band challenge item** that lives *outside* the bundle body — in a separate Verification Block that is stripped from the working copy. The falsification signal is identical (a comprehending session flags the conflict between challenge-item and bundle; a paraphrasing one won't), but the source of truth stays truthful and there's no stale-trap to leak. This costs a little discriminating power — the session knows *a* trap exists — so compensate by not telling it *which* item, and by phrasing the challenge as "reconcile these; one is wrong" rather than hiding it.

**Keep A's first-move-and-stop-condition slot as a required output**, because it's the only part of either proposal that governs behavior *after* the gate. The entry quiz proves initial internalization; the pre-committed first action + stop condition is what actually stands between a passed gate and an irreversible 2am mistake.

**Resolve the grading conflict in B's favor.** A's "materially correct, actionably specific" prose judgment reintroduces grader drift and imports the paraphrase problem into the reviewer. The go/no-go must hinge on the **binary, answer-keyed** items (did it catch the reconciliation, did it name the correct governing constraint), with the free-text brief as context for a human skim, not as the pass criterion.

**The concrete gate:** (1) 2-3 application/counterfactual questions with an answer key; (2) one out-of-band reconciliation challenge, mandatory, that never touches the bundle body; (3) a required first-move-and-stop-condition statement. Pass = keyed items correct AND first move is reversible/scoped. The bundle template makes an empty Verification Block visibly incomplete (B's anti-skip mechanism, which is good).

**What did *not* move me:** B's "just make it a required section" doesn't answer who has the skill to author a fair contradiction — I'm accepting that residual risk consciously, and the out-of-band framing at least makes a bad trap *visible* rather than silently poisoning the doc. And I'm rejecting A's claim that a single owner can carry prose-grading across years without drift — that's the assumption that fails first in the three-year test.


### Openai (gpt-5.6-terra)

## Proposal A

**Steelman:** This is exactly right if handoffs are consequential but varied, maintainers can spend 2–4 minutes checking a response, and the goal is to catch practical misunderstanding without creating a separate testing system.

**Assessment:** Agree. It tests whether the next chat can turn context into an executable plan, which is the relevant behavior—not merely recall it.

**Strongest point:** Requiring a first action, evidence, and stop condition exposes misunderstanding before irreversible work starts.

**Weakest assumption:** That someone can reliably judge the brief against a checklist. If ownership is unclear or reviewers are unavailable, the gate becomes ceremonial or delays work.

**Hidden assumptions:**

- The bundle clearly distinguishes settled decisions from background context.
- The task is important enough to justify a manual preflight.

**Overlooked risks:**

- A model can produce a highly plausible execution brief while still missing a crucial edge-case constraint.
- Hidden checks can become unfair if they test obscure details rather than material decisions.
- Four prompts may be too much for routine, low-risk handoffs; people will bypass it.

## Proposal B

**Steelman:** This is exactly right if the dominant failure mode is confident constraint violation, handoffs are frequent and costly, authors can consistently write good test cases, and the system can safely contain deliberately conflicting information.

**Assessment:** Partially agree. Application and negative questions are stronger than paraphrase. I disagree with planting contradictions in the operational bundle as the load-bearing control.

**Strongest point:** It correctly insists that the test must be falsifiable: a gate that cannot meaningfully fail is not a gate.

**Weakest assumption:** That a planted contradiction is a reliable proxy for internalization. A chat may catch it through local textual comparison without understanding the actual decision model; a competent chat may reasonably treat it as a source-document defect and halt.

**Hidden assumptions:**

- It is safe to deliberately corrupt the artifact that is meant to be the authoritative source of truth.
- Authors have the skill and time to create non-gameable, bundle-specific traps and answer keys.

**Overlooked risks:**

- Deliberate contradictions can leak into substantive work, incident response, future handoffs, or documentation reuse.
- The mechanism rewards “find the trick” behavior rather than safe operation under ambiguity.
- A binary pass based heavily on one trap is brittle: it can create false confidence after shallow success or false failures from wording ambiguity.
- Five minutes of author effort per handoff is unrealistic for many small teams; quality will decay first, then compliance.

## Revised recommendation

I maintain the pragmatic position closest to Proposal A: require a short operational readiness brief, with one bundle-specific application scenario and an explicit go/no-go reviewer check for high-consequence handoffs.

Use:

1. Non-negotiables: up to three.
2. Intended outcome and selected approach.
3. One novel “what would you do if…” scenario.
4. First reversible action, expected evidence, and clarification trigger.

Do **not** seed contradictions into the live bundle. If ambiguity detection is important, include a real ambiguous/conflicting case in the verification prompt or ask the chat to identify unresolved conflicts in the bundle. That tests the same safety behavior without compromising the source of truth.

Proposal B strengthens the case for an application-based test and a clear failure path, but its contradiction mechanism fails the operational test: a small team is unlikely to author and maintain it well, and deliberately tainting the handoff artifact creates avoidable failure modes.


## Synthesis (by gemini, non-participant)

## Consensus
The debate yielded strong consensus on several foundational points, driven by shared evidence regarding language model behavior and operational realities:
* **Paraphrase is a false signal:** Asking a model to summarize or read back a bundle tests retrieval, not comprehension or applied constraint-binding. Fluent restatements create false confidence.
* **The gate must be brief:** Verification must take minutes (both for the AI to generate and the reviewer to check), or users will route around it.
* **Application over recall:** "What would you do if..." scenarios are vastly superior to direct factual queries for testing operational readiness.
* **Pre-commitment is critical:** Requiring the AI to name its "first move and stop condition" before starting work is a highly valuable mechanism to catch misunderstandings before irreversible actions occur.
* **Corrupting the source of truth is dangerous:** While one participant initially proposed seeding deliberate contradictions directly into the canonical bundle, both participants ultimately agreed this is an unacceptable operational hazard that creates stale traps and poisons documentation.

## Unresolved Disagreements
**1. The Crux: Mechanism of Falsifiability and Grader Burden**
* **The Split:** One side advocates for a subjective evaluation of a four-part execution brief (relying on a reviewer to judge if it is "materially correct"). The other advocates for an objective, binary answer key paired with a mandatory "reconciliation challenge" (an out-of-band statement that conflicts with the bundle, which the model must flag).
* **Stronger Argument:** The binary/answer-key approach is significantly stronger. The proponent correctly points out that asking a tired on-call engineer at 2am to judge a fluent prose brief transfers the "false confidence/paraphrase" problem from the AI to the human grader. Over time, subjective grading decays into rubber-stamping. Binary checkpoints arrest this grader drift. 

**2. The Crux: Author Burden vs. Precision**
* **The Split:** Evaluating a free-form brief requires almost no setup from the bundle author. Creating binary application questions and a reconciliation challenge requires ~5 minutes of upfront authoring.
* **Stronger Argument:** The side accepting the higher author burden makes the stronger case. A test without a specific answer key degrades into rote compliance. The upfront cost of authoring is justified by the downstream prevention of a single confident, hallucinated mid-task error.

## Argument Quality Assessment
* **Best Reasoned:** Claude's Round 2 synthesis. It ruthlessly critiques its own Round 1 proposal (recognizing the immense danger of embedding lies in canonical documentation) while elegantly integrating OpenAI's strongest operational idea (the first move + stop condition). It maintains a rigorous stance on epistemics and grader fatigue. 
* **Strongest Single Argument:** Claude's critique of OpenAI's subjective grading. The observation that subjective evaluation of a fluent AI brief merely "imports the paraphrase problem into the grading step" perfectly encapsulates why unstructured human-in-the-loop approvals fail over time.
* **Weakest Single Argument:** Claude's initial Round 1 proposal to permanently seed deliberate contradictions into live, canonical documentation. This displayed a severe lack of foresight regarding documentation lifecycles and downstream data reuse, though Claude thankfully dismantled its own argument in Round 2.

## Blind Spots
* **LLM-as-a-Judge for Grading:** Both participants assume grading is a binary choice between "human reads prose" or "human checks answer key." Neither explored using a separate, isolated LLM call (equipped with the answer key) to automatically grade the fresh chat's test, which would solve both the 2am human burden and the subjective drift problem.
* **System Prompt Bypass:** Neither participant addressed the risk of the user prompting the fresh chat to skip the verification block ("Ignore the quiz, just start coding"). The mechanism assumes a compliant initial state.
* **Variable Bundle Complexity:** Both assume handoffs have clear, structured "decisions" and "constraints." In practice, many real-world handoffs are messy accumulations of state, making it difficult to write clean counterfactuals.

## Recommended Decision
Implement a hybrid verification gate that combines binary falsifiability with behavioral pre-commitment, explicitly avoiding any corruption of the canonical bundle text.

**The Protocol:**
1. **The Out-of-Band Challenge:** When the fresh chat initializes, it receives the Handoff Bundle. It is then separately given a **Verification Block** containing 2-3 application questions and one *deliberately conflicting statement* (e.g., "Reconcile this premise with the bundle; one is wrong"). 
2. **Execution Pre-commitment:** Alongside the quiz answers, the chat must output its planned first concrete action, the evidence that action will produce, and the exact condition under which it will pause to ask for clarification.
3. **Binary + Skim Verification:** A reviewer (or automated harness) checks the quiz answers against a hidden Answer Key (Go/No-go). If the key is passed, the reviewer briefly skims the "first move" to ensure it is safe and reversible. Substantive work begins only after this gate is passed.

**Rationale:** This decision honors the strongest arguments from the debate. It enforces rigorous falsifiability (catching a contradiction requires actual comprehension), prevents grader drift (binary answer keys remove subjectivity), provides an operational safety net (first move + stop condition), and preserves the integrity of the documentation (moving the contradiction out-of-band).

## Risks
* **Author Fatigue:** Writing good application questions and answer keys is a learned skill and takes time. Under pressure, authors may write trivial questions, reducing the gate to theater.
  * *Mitigation:* Make the Verification Block a required field in the bundle template. In the medium term, use an LLM tool to auto-generate candidate quiz questions and conflicts from the completed bundle for the author to approve.
* **Stale Answer Keys:** As the bundle is updated over a long project, the hidden verification block may not be updated, leading to false failures.
  * *Mitigation:* Store the Verification Block and Answer Key in the same document lifecycle as the bundle text to ensure they are versioned and updated together.

## Action Items
1. **Update Handoff Templates:** Refactor existing bundle templates to include two strict sections: the Core Bundle (clean, truthful context) and the Verification Block (out-of-band questions, the conflict challenge, and the hidden answer key).
2. **Update System Prompts:** Modify the fresh chat's system instructions so it refuses to execute substantive tools/actions until it has output the answers to the Verification Block and explicitly stated its "first move and stop condition."
3. **Define Reviewer Runbook:** Publish a standard operating procedure for on-call engineers: "1. Match quiz answers to the key (Binary Pass/Fail). 2. Check that the first move is reversible. 3. If fail, re-prompt with the missed constraints; do not allow work to proceed."
4. **Experiment with LLM Evaluation (Deferred):** Spin up a background spike to test if a lightweight, automated LLM evaluator can reliably perform the Go/No-go check against the answer key, fully removing the human from the 2am critical path.
