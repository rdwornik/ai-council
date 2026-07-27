# ADR-16: Evidence must be retrievable from the repository

**Date:** 2026-07-27
**Status:** Accepted (ratified 2026-07-27, operator)

## Context

The class recorded in LESSONS 2026-07-26 — *a value published without the predicate that produces
it cannot be checked* — has now recurred across three windows. The 2026-07-27 close-out audit
(`docs/audits/2026-07-27-close-out-audit-execution-window.md`) found its sharpest form.

The 2026-07-27 execution window's closure block (`BACKLOG.md`, Working-queue block) asserts
criterion 3 **MET** on the strength of *"13 findings across 9 passes: 11 accepted, 2 refuted with
reproductions."* **No record of those passes existed in the repository at the window's close.**

The evidence is **pinned to that commit**, deliberately: `git log 1dcee8b..17a239b -- docs/audits/`
returns zero commits, and `git ls-tree -r --name-only 17a239b -- docs/audits/` contains no
`2026-07-27` file. A working-tree `grep` would have been the wrong citation — **this ADR's own
arc makes it false**, because the audit lands in `docs/audits/` in the same commit that ratifies
the ADR. An evidence command whose answer changes when the repo changes cannot support a claim
about a past state; that is ADR-15's commit-tree ruling applied to prose rather than to a rule.

**This ADR will not repeat the mistake it is about, so the split is stated precisely.** What *is*
retrievable is the **effect**: commits `8215ada`, `c687e81`, `d06f9dd`, `a33db9e`, `6b8160e` and
`5757448` exist on `main`, and their messages describe findings being accepted and repaired. What
is **not** retrievable is the **cause** — the review outputs themselves, their count, and the
disposition of each. The figures "13 findings across 9 passes" and "11 accepted, 2 refuted" are
therefore reported here **as the closure block's claim**, quotable at its location, and **not as
established fact**: this ADR cannot verify them, and under clause 1 below they are a recollection.
That is not a slight on the session that wrote them; it is the whole point. A reader with a clone
can see that repairs happened and cannot see what prompted them.

The same window produced two self-reported instances of asserting a guarantee without the
mechanism that provides it, both in `scripts/check.ps1` and both recorded only in commit
messages: `6b8160e` (*"the try/finally form did NOT restore the caller's location on a failing
gate — and the commit that introduced it asserted that it did"*) and `5757448` (*"The previous
commit fixed the `exit` path and left the EXCEPTION path broken, then claimed 'every path'"*).
**Those commit messages record** that each was caught by adversarial review — attributed to the
messages, which are retrievable, rather than asserted as an established event, which is not. What
*is* independently checkable is the second half: **no gate in this repo could have caught either.**
`scripts/check.ps1` is a PowerShell script that no test in `tests/` executes, so its guarantees
live entirely in prose.

The common shape is **not** that the claims were false — nothing here establishes that they were,
and the repair commits above are consistent with them. It is that **nothing in a clone can check
them either way.** A record whose predicate is unreachable is indistinguishable, to its next
reader, from a record that was never true; and "the author believed it" is precisely the thing a
record cannot carry forward.

## Decision

### 1. A repo record must name a retrievable predicate

A claim written into a repo record — a closure status, a ticket status line, a `JOURNAL.md`
entry, an ADR — must name a **committed path, a commit sha, or a command that re-derives it from
the repository**. A claim whose only evidence is a chat transcript or an off-repo file is a
**recollection, not a record**.

### 2. Review outputs land on disk

Any adversarial or Codex review that a record cites as evidence is **written into `docs/audits/`
by the lane that ran it**. `codex exec` does not write a file; the lane must.

Routing through a review path that writes no file does not suspend this obligation — that is
precisely what caused the 2026-07-27 gap. The repo's own established practice was the opposite:
`docs/audits/` holds three committed Codex outputs from 2026-07-26 alone
(`2026-07-26-codex-97-registry-repair.md`, `2026-07-26-codex-crux-check-pass3.md`,
`2026-07-26-codex-116-resolution-model.md`). A tool choice silently suspended a convention that
had been honoured the day before.

### 3. Closure criteria are phrased so a clone can check them

A criterion reads *"a review record exists at `<path>`"*, not *"a review was run"*. **A criterion
that cannot be checked from a clone is not a criterion** — it is an assertion about a session that
outlives the session's ability to prove it.

### 4. A gating validator's success output states what it checked, and how many items

An exit code alone is a verdict without its predicate: exit 0 with zero bytes cannot be
distinguished from a run that examined nothing.

**This clause HOMES the #126 reporting contract**, which until now existed only inside
`BACKLOG.md` — an append-only work log whose items *leave* under ADR-65, so the contract's only
record was scheduled to disappear. It carries forward, unchanged, the **hub-supersession trigger**
#126 recorded: the 2026-07-26 fleet-intake commissions (A–J) own the fleet-level ruling on
validator reporting, and **when the hub rules, that ruling supersedes this clause and it expires
by reference** rather than lingering as a local divergence (the #111 precedent).

### Alternatives considered and rejected

**Leave it as convention.** Rejected on the evidence: this *was* the convention, and a routing
choice broke it silently within one day. A convention that a tool choice can suspend, with no
signal that it has been suspended, is not a mechanism.

**Ship a checker rule instead of a decision.** Rejected on the #125 precedent. A rule authored
without a stated decision *becomes* the spec, and whatever the rule does not read becomes
invisible — which is exactly how rule 4 spent its life reading one of the four surfaces its spec
named while reporting `pass`. The decision is recorded first, deliberately, so that any future
rule can be measured against it rather than substituting for it.

## Consequences

- **Precedent and extension.** Rule 8 (`sha-reachability`) already enforces the narrow form of
  this discipline: a sha cited in `JOURNAL.md` or `BACKLOG.md` must resolve and be reachable. This
  ADR extends the same reasoning from **shas** to **paths and review records**.
- **The mechanism is deliberately NOT ruled here.** A checker rule enforcing clauses 1–3 is a
  candidate for the Unit-2 stub set and is left unruled, per the rejected alternative above.
- **Cost, stated plainly.** Every review lane gains one extra write. That is the price of a
  criterion that survives a clone, and it is cheaper than the alternative demonstrated on
  2026-07-27: a closure claim that cannot be checked by anyone who was not in the room.
- **The motivating record is repaired retroactively.** The 2026-07-27 close-out audit is committed
  at `docs/audits/2026-07-27-close-out-audit-execution-window.md` in the same arc that ratifies
  this ADR — the minimum this decision requires of the record that motivated it.
- **Unrecoverable, recorded as a limitation.** The review outputs from that window — the closure
  block claims 13 findings across 9 passes — were never written to disk and **cannot be
  reconstructed**. What lands is the audit that discovered their absence, not the findings
  themselves. This ADR cannot repair the gap it was written about; it can only stop the next one.
  Its own Context section is written to that standard: the effect is cited by sha, the count is
  attributed to the claim that makes it.

## References

- `docs/audits/2026-07-27-close-out-audit-execution-window.md` — the close-out audit that found
  the gap (findings A5-a, A6-c, A2-a)
- LESSONS **2026-07-26** — the parent class this ADR is a structural response to; LESSONS
  **2026-07-27** records the two in-window instances
- Commits **`6b8160e`** and **`5757448`** — the two self-reported guarantee-without-mechanism
  instances, both caught by review rather than by a gate
- BACKLOG **#126** (the validator reporting contract this ADR's clause 4 homes, with its
  hub-supersession trigger), **#125** (the rule-4 narrowing that grounds the rejected
  "rule instead of a decision" alternative), **#111** (the expires-by-reference precedent)
- ADR-65 (done items leave the backlog — why a contract recorded only in `BACKLOG.md` had no
  durable home), ADR-15 (the sibling decision that made rule 2's verdict checkable across clones)
