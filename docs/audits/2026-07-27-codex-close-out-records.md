# Codex (terra) pre-merge review — 2026-07-27 close-out record turn

**Reviewer:** `gpt-5.6-terra` via `codex exec --sandbox read-only` (three passes)
**Subject:** the record turn that lands the close-out audit, ADR-16, the LESSONS entry, tickets
#130/#132/#133, three text repairs, and the ARCHITECTURE/CLAUDE reconciliation.
**Diff reviewed:** all authored changes. `docs/audits/2026-07-27-close-out-audit-execution-window.md`
was excluded from the diff as a byte-verbatim copy of a scratchpad file, sha256-verified identical
(`9e7628b06e420678a7c03ddad9231a6cda9ae21e27b9fb24c31cd25e19295827` on both sides); its *presence*
is the point, not its prose.
**Routing:** CC-produced diff → terra, per BACKLOG #73's routing posture.
**Verdict:** **safe to merge (third pass)**. Three HIGH findings raised across passes 1–2, **all
three accepted and fixed**; zero surviving.

**This file exists because ADR-16 clause 2 requires it.** The review that reviewed the ADR is
written to disk by the lane that ran it — the first record turn after the decision is not the one
that breaks it. That obligation was itself the reviewed subject, which is why it is discharged
here rather than promised.

---

## Pass 1 — one HIGH: ADR-16 violated its own clause 1 on its first page

> **[HIGH] `docs/decisions/ADR-16-retrievable-predicates.md:15` — ADR-16 violates its own
> retrievability rule.**
> **What:** ADR-16 calls the unrecorded reviews and their outcomes "real," "acted on," and true,
> while explicitly stating no repository record can confirm them; LESSONS repeats "Every word …
> was true" at `LESSONS.md:13`.
> **Why:** Clause 1 defines claims supported only by chat/off-repo evidence as recollections, not
> records. These are precisely such assertions, creating the ADR's central self-contradiction.
> **Fix direction:** Describe these historical assertions as unreconstructable recollections, or
> cite committed evidence sufficient to verify them.

**ACCEPTED.** This was the sharpest possible finding: the ADR about unretrievable claims made one,
as established fact, in its own Context. The fix splits what the repository can and cannot support:

- **Retrievable — the EFFECT.** Commits `8215ada`, `c687e81`, `d06f9dd`, `a33db9e`, `6b8160e`,
  `5757448` are on `main` and their messages describe findings being accepted and repaired.
- **Not retrievable — the CAUSE.** The review outputs, their count, and each disposition.
- The figures *"13 findings across 9 passes"* are now reported **as the closure block's claim**,
  quotable at its location in `BACKLOG.md`, and explicitly **not as established fact**.
- The claims are stated as **neither shown true nor shown false — only uncheckable**, which is the
  honest position and the entire reason the ADR exists.

## Pass 2 — two HIGH

> **[HIGH] `ADR-16:15` — historical evidence command is mutable.**
> **What:** `HEAD` is not pinned; after merge, the new audit itself makes the `grep` assertion
> false. LESSONS repeats it at line 13.
> **Why:** The ADR's evidence cannot be re-derived and contradicts its own rule.
> **Fix direction:** Pin the historical commit (`17a239b`) and use commit-tree commands.

**ACCEPTED, and confirmed already false before merge.** `grep -rl "2026-07-27" docs/audits/` run in
the working tree returned `docs/audits/2026-07-27-close-out-audit-execution-window.md` — the ADR's
own arc had already falsified its citation. Replaced with commit-pinned, immutable forms, both
verified to return what is claimed:

| pinned command | result |
|---|---|
| `git log 1dcee8b..17a239b -- docs/audits/` | 0 commits |
| `git ls-tree -r --name-only 17a239b -- docs/audits/` | no `2026-07-27` file |

This is ADR-15's commit-tree ruling applied to **prose** rather than to a rule, and the ADR now
says so explicitly.

> **[HIGH] `ADR-16:33` — review occurrence still asserted as fact.**
> **What:** "Both were caught by adversarial review" is unretrievable; LESSONS line 15 repeats it.
> **Why:** It asserts the off-repo review event despite correctly classifying those outputs as
> unavailable.
> **Fix direction:** Attribute it to the cited commit messages, rather than assert it as
> established fact.

**ACCEPTED.** Now reads *"**Those commit messages record** that each was caught by adversarial
review"* — attributed to a retrievable artifact. The independently checkable half is kept separate
and retained: **no gate in this repo could have caught either**, because `scripts/check.ps1` is
executed by no test in `tests/`. LESSONS mirrors both changes, and its limitation bullet now
attributes the 13-findings count rather than asserting it.

## Pass 3 — merge decision

> **(a)** No. The count and review events are consistently attributed; independently checkable
> claims are separated.
> **(b)** Yes. Both pinned commands are correct and returned the stated historical result.
> **(c)** No new defect found in ADR-16 or the new LESSONS entry.
> **Safe to merge: yes.**

Pass 2 also independently confirmed the six cited repair commits: *"all six cited commits are
reachable from `main`; their messages accurately describe repairs/findings. That proves the
recorded effect, not the reviews' occurrence"* — which is precisely the distinction the fix draws.

---

## Assessment

All three findings were the same defect at descending depth: **asserting, in a repo record, a
thing the repo cannot check.** Pass 1 caught it in the ADR's claims, pass 2 in its *evidence
commands* and in a residual verb, and the third pass confirmed none survived. That the ADR about
retrievable predicates needed three passes to stop making unretrievable claims is the strongest
available argument for the decision it records — and it is why the review is on disk instead of in
a transcript.

**Not claimed:** this review covers the authored record changes. It does not re-verify the
close-out audit's own findings, which were derived and reported in the preceding audit turn.
