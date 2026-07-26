# ADR-15: Path-claim resolution model for the claim-vs-reality checker

**Date:** 2026-07-26
**Status:** Accepted (2026-07-26)

## Context

`scripts/validate_claims.py` rule 2 asserts that every backtick-quoted repo path in a canonical
doc is a true claim about this repository. The question "is this path real?" turns out to have
three defensible answers, and the first two were both shipped and both wrong.

The rule originally answered it with **`Path.exists()` plus a repo-rooted guard**: a token was
skipped unless its first segment was an existing top-level directory, then tested against the
disk. That guard was dispositioned in the 2026-07-24 review as a deliberate
precision-over-recall tradeoff, with a revisit deferred to gating promotion. The revisit fired
early, for two independent reasons: rule 8 reached zero findings, so rule 2 stopped being one
contributor among several and became the entire report; and a defect arrived by a route the
tradeoff had never considered.

That defect (BACKLOG #116) is the reason this ADR exists. `logs/` is gitignored. On the primary
checkout it existed as untracked debris, so the guard passed and the finding fired; on a fresh
clone it was absent, so the guard suppressed and no finding fired. **The same commit reported
two different ways depending on which checkout you ran it in.** A rule whose verdict depends on
untracked local debris cannot gate anything, independently of any precision argument.

A one-run instrumented measurement of the guard (2026-07-26, guard unchanged) made the second
problem concrete: of 110 tokens reaching the guard, **25 were dropped and every one of the 25
failed existence** — the guard suppressed 25 potential findings and zero harmless ones. Those 25
were not one population but three: 16 base-relative shorthand for paths that genuinely exist
(`research/merger.py` meaning `src/ai_council/research/merger.py`; `decisions/` meaning
`docs/decisions/`), 6 ecosystem paths above this repo (`Dev/`, `ai-council/output/`), and 3
GitHub org/repo slugs that are not filesystem paths at all (`astral-sh/ruff-pre-commit`). One
heuristic was silently doing three different jobs, correctly for two of them.

## Decision

### The three candidate models, and why the first two failed

**Model 1 — disk existence with a repo-rooted guard (shipped, withdrawn).** Rejected on
determinism, not precision: `Path.exists()` answers a question about the *working directory*, so
its verdict is a function of the checkout rather than the commit. It also conflated the three
populations above, excluding ecosystem paths only as a side effect of a first-segment test that
simultaneously suppressed 16 legitimate claims without saying so.

**Model 2 — disk existence with the guard removed.** Rejected because it fixes the recall
problem and keeps the determinism problem. Every ecosystem path and org/repo slug becomes a
finding, and the verdict still moves with untracked debris. Strictly worse in report noise, no
better in the property that actually blocks gating.

**Model 3 — resolution against git state under declared bases (ADOPTED).** A candidate token
resolves if it names a path in **git's record of the commit**, under one of an ordered list of
declared bases. A finding fires only if it resolves under none of them. Exclusions become
**declarations** rather than side effects.

### Git state means the COMMIT TREE, not the index

The implementation must read `git ls-tree -r --name-only HEAD`, **not** `git ls-files`.

`git ls-files` reports the **index**. Reading it leaves a staged-but-uncommitted path able to
satisfy a claim, so the verdict still depends on working state — a weaker version of the very
defect being repaired.

**This distinction was invisible to the acceptance test, and that is the durable lesson.** The
acceptance ran the checker on the primary checkout and on a fresh clone at the identical commit,
with `output/` and `logs/` present in one and absent in the other, and required identical
output. It passed — while the implementation still read the index. Index and commit tree are
**equal on a clean tree**, and a clean working tree is what this repo's own policy requires at
every commit and every session end (CLAUDE-FLOOR "working tree must be clean at session end";
the `git status` step in the verify trio). The acceptance therefore compared two things policy
guarantees are identical, and no run of it could have gone red. It was caught afterwards by
adversarial review, and is now pinned by a test that stages a path **without** committing it —
the input the acceptance could not reach.

### The four declared bases

Canonical docs routinely write a path relative to a base rather than to the repo root. The
declared bases, in order:

| base | why |
|---|---|
| `` (repo root) | the default reading |
| `src/ai_council/` | ARCHITECTURE writes module paths relative to the package |
| `docs/` | the ADR-60 docs taxonomy is written base-relative |
| `docs/decisions/` | **required**, see below |

**Why the fourth was required.** The frozen acceptance demanded that all 16 base-relative tokens
produce no finding. Three bases satisfy 14 of them. The remaining two — `handoffs/` and
`transcripts/`, both written base-relative at `ARCHITECTURE.md:368` — name subtrees that an ADR
routes to the hub (ADR-42, ADR-43) and that therefore do not exist locally at all. They are
already carried in the rule's allowlist in their **full** form (`docs/handoffs/`,
`docs/decisions/transcripts/`), so the fix is to match the allowlist against **base expansions**
as well as the raw token — and reaching `docs/decisions/transcripts/` requires `docs/decisions/`
as a base. The fourth base is thus an allowlist-reachability requirement, not a widening of what
resolves on disk.

**This is recorded as a cost, not a virtue.** Each additional base widens what silently
resolves, and because a token carries no source-base context, a path is accepted if it resolves
under *any* base — so a genuinely-missing root-relative `research/merger.py` passes on the
strength of `src/ai_council/research/merger.py`. That collision is the inherent price of the
model. **Do not resolve future cases by adding bases.** The correct repair is for
`ARCHITECTURE.md` to declare which sections are relative to which base, so bases can be bound to
context — tracked as BACKLOG #118, which carries a required collision test.

### `_R2_RUNTIME_PATHS`, and why no measurement predicted it

Two tokens resolve under no base and are still true claims: `output/` and
`council_inbox/archive/`. Both are gitignored by design (`.gitignore:38`, `:41`) — runtime
artifact directories the tool creates. They are correctly absent from the commit tree while the
docs naming them are telling the truth.

**These were not predictable from the guard measurement.** They never reached the guard: they
exist on disk, so the old disk check passed them, and they never entered the drop-set. The
drop-set was a complete and accurate measurement — of the wrong thing. Changing a guard surfaces
a population that no measurement of the *old* guard can show, because the old guard's own
behaviour determined which tokens were observable. Any future guard change should expect a new
population rather than trusting the prior measurement to be exhaustive.

They are handled by a **declared list**, not by calling `git check-ignore`. `check-ignore` also
consults `.git/info/exclude` and the user's global excludesfile, both untracked — using it would
reintroduce exactly the checkout-dependence this ADR removes.

### Self-validating declarations

Every declared list carries a test that proves the declaration true, so none of them can decay
into folklore:

- **every declared base must exist in the tracked tree** — a base that does not exist silently
  suppresses every token written relative to it, which is the original failure one level up;
- **every declared runtime path must be genuinely gitignored AND genuinely uncommitted**,
  validated against the **tracked** `.gitignore` rather than `git check-ignore`, for the same
  determinism reason as above.

A declaration without a test that can falsify it is precisely the defect class recorded in
LESSONS 2026-07-26: a value published without the predicate that produces it.

### Evidence commands

A finding's evidence must probe what the rule probes: `git cat-file -e HEAD:<path>`. A
`Path.exists()` probe would contradict the rule and reproduce the nondeterminism inside the
evidence itself; `git ls-files --error-unmatch` would carry the index dependence.

## Consequences

- Rule 2's verdict is a function of the commit, so it can be compared across checkouts, clones
  and worktrees. This satisfies BACKLOG #104's promotion criterion (ii).
- **Rule 2's clean-window clock restarts at this ADR's implementing commit** — its behaviour
  changed, so prior clean runs are evidence about a different rule. The same reasoning restarts
  rule 8's clock at the #108 merge.
- Report noise falls to zero on the live repo without a single path being added to the allowlist.
- The base-collision weakness (#118) and the hand-maintained-spec weakness (#114) are both open
  and both disclosed in the checker's own KNOWN LIMITATIONS output.

## References

- BACKLOG **#116** (the nondeterminism defect, closed), **#104** (promotion criteria),
  **#118** (undeclared base convention), **#114** (hand-maintained spec set), **#108** (the
  shared context predicate that shares this rule's scan)
- `scripts/validate_claims.py` — `_R2_BASES`, `_R2_EXTERNAL_PREFIXES`, `_R2_RUNTIME_PATHS`,
  `_committed_paths`, `_resolves_under_a_base`
- `docs/audits/2026-07-26-codex-116-resolution-model.md` — the pre-merge review that found the
  index-vs-commit-tree defect after the acceptance had passed
- LESSONS 2026-07-26 — the general class this ADR is one instance of
- ADR-42 / ADR-43 (hub-routed subtrees), ADR-60 (docs taxonomy)
