# Lessons Learned — Append-Only Log
<!-- scope: hybrid -->

> **Format:** `### YYYY-MM-DD | source | lesson | category | [scope: X] | action taken`
> New entries go at the top of the Entries section. Never edit old entries. Never delete.
> Grandfathered: earlier entries below use a `### YYYY-MM-DD | title` header + `CONTEXT / MISTAKE / RULE` body (never rewritten to the newer schema — append-only per ADR-29).
> Last updated: 2026-07-26

---

### 2026-07-26 | #97 checker arc + terra pre-merge review | a value published without the predicate that produces it cannot be checked | process | [scope: hybrid] | filed ADR-15, BACKLOG #118, amended #104 criterion (iii)

- RULE (publish the predicate with the value): **a count, verdict, or status published without the predicate that produced it cannot be checked.** A reader sees that the value moved, but not whether it moved for the right reason; and no test can separate *correct* from *correct-looking*, because both produce the same number. Five instances, all live in this repo, none caught by the check that existed to catch them:
  1. **A registry test parametrized over its own registry.** `@pytest.mark.parametrize("leg", vc.RULES)` published "every rule is covered" while its actual predicate was "every rule *present* is covered". 41 tests were green while two of the fourteen spec rules were absent from `RULES` entirely — and would have stayed green forever, because the omission removed its own test case. Repaired only by an **external literal expected set** that the registry cannot edit.
  2. **ARCHITECTURE's edge count.** `ARCHITECTURE.md:167` states "14 real internal module edges" and `:151` "the only edge in the codemap that skips a layer", with no stated rule for what counts as an edge or which base a written path is relative to. The number is therefore unfalsifiable: any recount that disagrees is equally defensible, so the claim can never be wrong and never be verified. (Base half filed as #118.)
  3. **R8's finding count with no defect unit.** `FINDINGS n` counted citation SITES, so one dangling SHA cited twice was two findings and the number moved when a *document* was re-cited rather than when the *repo* changed. Without a published defect unit the count could not be gated to zero — it grew by being written about. (#105.)
  4. **OVERRIDES.md counting events with no cause distinction.** The ADR-85 session gate derives "this session's work" from commits-ahead-of-base in a **shared** checkout, so it cannot distinguish authors; the hub's `logs/OVERRIDES.md` records it firing twice on 2026-07-25 against a concurrent session's commits. The log exists as override-rate telemetry, but with no cause field the rate cannot separate *"operator bypassed a correct gate"* from *"gate misattributed authorship"* — and those two readings demand opposite responses. A rate without a cause field is a number that cannot inform the decision it was collected for.
  5. **An acceptance contract that could not fail.** The #116 acceptance compared the checker's verdict on the primary checkout against a fresh clone at the same commit, to prove the verdict was a function of the commit. It passed. The implementation was still reading `git ls-files` — the **INDEX** — rather than `git ls-tree -r HEAD`, the **COMMIT TREE**. Index and tree are equal on a clean tree, and a clean tree is what repo policy *requires* at every commit and every session end. So the acceptance compared two things policy guarantees are identical, and no run of it could ever have gone red. It was found afterwards by review, not by the test written to prove it.
- COROLLARY (the repair has one shape): publish the predicate beside the value, or make the value checkable by an **external** denominator the thing being measured cannot edit. `_SPEC_RULE_IDS`, `_R2_BASES` and `_R2_RUNTIME_PATHS` are each a *declaration paired with a self-validating test* — that pairing is the transferable pattern, not the particular lists.
- RULE (an acceptance test must be able to fail): before accepting a green acceptance run, state the input that would turn it RED. If producing that input requires violating a policy the repo enforces everywhere — a dirty tree, an unpushed main, a missing gate — the test is measuring a tautology and its green tells you nothing. Instance 5 is the worked example: "stage a file without committing it" was the missing input, and it was unreachable because the whole workflow forbids it.

### 2026-07-20 | three-lane reintegration | file-disjointness is not contract-disjointness; a summary line is not evidence | process | [scope: hybrid] | filed BACKLOG #75-#79 + scripts/verify_output_contract_e2e.py

- RULE (lane splitting): **file-disjointness is not contract-disjointness.** Three lanes ran in parallel with ZERO shared files. Two were individually green and `main` went RED on merge: Lane A1 changed `OutputRoutingError`'s constructor from `str` to `list[RoutingFailure]` with no type guard, and Lane A2's fixtures hardcoded the old signature. The coupling was a public signature, not a path. MISSING CHECK, now mandatory: before splitting lanes, list each lane's changed **public signatures** (constructors, function params, exception shapes) and grep every other lane for them. A file-overlap check would have passed this cleanly and did.
- RULE (per-lane gates are structurally blind to composition): two per-lane checkers were both green while the composed guarantee was broken — each half sound, the seam broken. A cross-lane claim needs its own checker at the composition level (`scripts/verify_output_contract_e2e.py`), or nothing tests it. Corollary: a per-lane green gate is evidence about the lane, never about the merge.
- RULE (a summary line is not evidence — three instances in ONE session): (1) a Codex console summary printed `High 0` over 8 real findings, because its regex expected `High:` while Codex now emits `### [HIGH] file:line`; measured across 68 audits in 4 repos, 8 understated, 26 findings invisible. (2) Two stale fixtures PASSED while exercising a message shredded into one "deliverable" per character — they asserted on a label phrase that survived the mangling. (3) A closure gate returned "close both" while citing the commits that PROPOSED the guards rather than the commits that BUILT them. In all three the summary was green and the substance was not. Assert on something the corruption would BREAK — a count, a path, a structured attribute — never on a phrase that survives it.
- RULE (proof-by-violation beats proof-by-green): both new pre-commit guards were trivially bypassable on first implementation (unicode path, `git mv` into a tracked path, four registry bypasses) — every unit test was green throughout. A guard is only evidenced by a witnessed FAILURE on the thing it claims to stop.
- RULE (safe prove-by-reversion): **COMMIT THE FIX FIRST**, then revert-edit → observe FAIL → `git checkout -- <file>` → observe PASS. Reverting an UNCOMMITTED fix and undoing it with a whole-file checkout restores HEAD, which does not contain the fix — it silently destroys the work. Cost this session: both regression fixes lost and re-applied. If the fix cannot be committed first, undo the reversion with the surgical inverse edit, never a whole-file checkout. A throwaway detached worktree is for when the control is a DIFFERENT COMMIT, not for reverting your own working-tree change.
- RULE (under-match toward the loud failure): fixing a too-broad match, prefer under-matching. A bare `"considered"` marker matched `## Risks Considered` and served risks to a caller as the decision's options. `[]` is honestly empty; `['Risk one']` is plausibly wrong and consumed silently. Same principle that WIDENS a secret-leak guard: move toward whichever side fails LOUDLY.

### 2026-07-19 | night-consolidation verification | verification discipline + worktree safety | process | [scope: hybrid] | filed docs/audits/2026-07-19-night-consolidation-verification.md + BACKLOG #59-#68
- RULE (worktree isolation): a parallel worktree is warranted by SIDE EFFECTS (writes to canonical `output/`, provider calls, git ops), not by task size. A "small" leg that writes artifacts still needs isolation; a large read-only sweep does not. This session ran 8 legs + 4 evidence probes read-only in one worktree because none mutated tracked state.
- RULE (parameter-space matrix): a trial's parameters (model pins, seats, modes, sealed keys) must be enumerated as ONE matrix up front, or the gates arrive serially and each late-discovered dimension silently reopens a "closed" trial.
- RULE (path-naming in orders): any instruction that COULD create a path must NAME the path. "Put the report under `docs/audits/`" is safe; "file it appropriately" invites an unsanctioned new folder. The prompt named the target dir + cited the sibling files as its authority — that is the pattern.
- RULE (git-add safety near an exclusion zone): `git add -A` nearly staged a `SEALED-KEY*.json` during the 2026-07-18 smoke-pair merge. Never blanket-add near an exclusion zone; stage explicit paths. Proposed a pre-commit guard that rejects staged sealed artifacts (#67).
- RULE (verification-as-code): verification PROSE rots and cannot be re-run. Codify each leg as a script that prints PASS/FAIL (`scripts/verify_night_consolidation.py`, 8/8) — the report cites it, the script re-proves it. A report without a re-runnable checker is a claim, not a witness.
- RULE (mock-witness reconciliation): when a "live witness" would cost money or need a gated config flip (here the ADR-12 §5 `backend: cli` flip, gated on #27 scoring), exercise the SHIPPED code with `MockProvider`/canned inputs at $0 and record the un-exercisable remainder as an explicit GAP. Running real shipped code on controlled inputs IS empirical proof; reading the diff is not.

### 2026-05-12 | Architect failure mode — defending local config as "by-design"
- CONTEXT: Scrum-master addendum (2026-05-12) — strażnik caught that `tasks/lessons.md` location was non-canonical after main review implementation
- MISTAKE: Accepted `tasks/lessons.md` as intentional per CLAUDE.md Lessons Discovery section ("by-design") when the ecosystem convention is `LESSONS.md` at root. Local config can be wrong relative to ecosystem baseline; defending it as intentional blocks the cross-repo audit from working.
- RULE: When a cross-repo audit flags a convention divergence, default response is "evaluate against ecosystem baseline" — NOT "intentional per local config." Local config documents what exists; ecosystem convention determines what should exist. If they conflict, the convention wins unless explicitly overridden by an ADR. This failure mode applies symmetrically to the audit consumer, not only the audit producer.

### 2026-05-11 | Target resolver fail-loud pattern (cross-project routing)
- CONTEXT: ADR-43 transcript routing — `target-project` frontmatter + `--target-project` CLI flag
- MISTAKE: Early design considered silently falling back to canonical-only when an unknown target name was given. This hides config typos.
- RULE: When introducing optional config-driven routing, fail loudly at parse time on unknown targets rather than silently routing to canonical only. Silent fallback hides config typos; loud failure surfaces them at the boundary. Pattern applies broadly to any optional routing mechanism.

### 2026-05-11 | Inbox/CLI code-path parity (recurring blind spot, 3rd instance — structural fix needed)
- CONTEXT: Transcript routing feature added to CLI direct path; inbox path needed explicit wiring
- MISTAKE: This is the third occurrence of the same pattern (--full, --mode, now target-project routing). Each instance costs a follow-up commit.
- RULE: Investigate whether the two paths (CLI direct + inbox processor) can share a common processor function rather than duplicating logic. If not addressable structurally, add a parity-check test that exercises both paths for any new feature. The pattern has repeated 3x — structural change is warranted.

### 2026-05-11 | ADR-43 amendment cycle 1 — lift repeated path prefix to root field
- CONTEXT: Original `target_projects: dict[name, full_path]` schema repeated `<dev_root>/<name>/docs/decisions/transcripts/` prefix per entry
- MISTAKE: Path prefix duplication in config — each entry had to repeat the shared root
- RULE: When config has multiple entries that repeat a path prefix, lift the prefix to a root field and compute the suffix. Refactored to `dev_root: str + target_projects: list[str]` with computed paths. Reduces migration error if root path ever moves, cuts noise.

### 2026-05-11 | Observability field design — avoid redundant signal
- CONTEXT: Codex review caught `synth_timeout_flag` as dead in observability schema — timeout cases already captured via `error_class="timeout"`
- MISTAKE: Boolean flag carried the same signal already in `error_class`. Dead field added noise and false coverage impression.
- RULE: When designing observability schema, avoid carrying the same signal in two fields. One canonical field (e.g., `error_class`) is sufficient; boolean flag mirrors create dead-field risk. If a flag is truly needed, ensure it captures something the primary field cannot.

### 2026-04-30 | mock.patch string literals are invisible to import refactoring
- CONTEXT: ADR-38 migration renamed all src.X imports to ai_council.X
- MISTAKE: 56 mock.patch("src.debate.X") string literals in tests/ were NOT caught by import-only find-replace. Caused 30 test failures.
- RULE: After any package rename, do a SECOND pass specifically for mock.patch() string literals. Pattern: `grep -r 'mock.patch.*"old_name\.' tests/`

### 2026-04-27 | Inbox path must mirror interactive path
- CONTEXT: Research mode worked in interactive CLI but not via --inbox
- MISTAKE: Third time this pattern appeared (--full, --mode, now research routing). Inbox loop is a separate code path that doesn't automatically inherit interactive features.
- RULE: After adding ANY new feature to the interactive CLI path, immediately check: does the inbox loop handle this too? If not, add it. This is a recurring blind spot.
