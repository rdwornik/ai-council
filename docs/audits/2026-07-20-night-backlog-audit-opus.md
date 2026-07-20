# Night backlog trust audit — BACKLOG.md vs git+code witness

**Model:** Opus 4.8 (1M context), xhigh effort · **Mode:** unattended night batch, auto-accept
**Scope:** read-only trust audit of every open item in `BACKLOG.md`. No writes to `BACKLOG.md` or `src/`. Nothing struck. Nothing billed. Recommendations only — the operator executes.
**Worktree:** `night-backlog-audit` @ `f758fa6` (verified identical to `main` HEAD)
**Report date:** 2026-07-21 (filename retains the operator-specified `2026-07-20` night-batch stamp)
**Method:** every status claim cross-checked against `git log --first-parent main` + `JOURNAL.md` + the actual source on main. BACKLOG.md treated as the CLAIM; git+code as the WITNESS. Where they disagree the witness wins.

---

## HEADLINE

**50 open claimed / 36 genuinely live / 2 kill-candidates / 5 anomalies**

> **Trust verdict: the backlog EARNS trust. 48 of 50 classifications survive the witness.**
> Zero fabricated items, zero id-space gaps, zero internal contradictions, zero cross-references to struck items masquerading as open. The only drift is two items (`#2`/`#3`) whose done-when was satisfied across *two separate merges*, neither of which struck them — a known, narrow, non-systemic mechanism. This is a high-integrity backlog and the operator's distrust, while well-founded in method, is not borne out in fact.

| Class | Count | Ids |
|---|---|---|
| **LIVE** — genuinely open, not built | 36 | 5, 7, 10, 11, 17, 19, 21, 32, 34, 36, 37, 38, 43, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 61, 64, 69, 70, 72, 75, 76, 78, 79, 82, 83, 110, 128 |
| **DONE-BUT-LISTED** — kill candidates | 2 | 2, 3 |
| **MERGED-NOT-CLOSED** — correctly open, gate named | 4 | 18, 20, 27, 66 |
| **AWAITING-RULING** — blocked on operator, not engineering | 8 | 4, 6, 8, 9, 55, 73, 80, 81 |
| **SUPERSEDED** | 0 | — |
| **ORPHAN/ANOMALY** (orthogonal axis) | 2 items + 3 structural | 110, 128 + A3/A4/A5 below |

Verification depth: all 50 items were checked against source or git. 25 items had their done-when clause traced to specific current `file:line` evidence.

---

## KILL CANDIDATES

Both are the same defect, from the same arc. Neither is a judgement call — the done-when text is satisfied verbatim.

| id | Why it should die | Evidence SHA | Recommended action |
|---|---|---|---|
| **#2** | Done-when = *"ADR-01 amended + the chosen branch shipped"*. **Both halves landed.** Branch A shipped: `config/settings.yaml:12` = `synthesizer: "openai"` (commit `ca7e85c`, merged `6e83e41`). Amendment written: `docs/decisions/ADR-01-synthesizer-selection.md:16` carries the dated `Revised 2026-07-18 (Epic B — BACKLOG #2/#3)` paragraph (commit `e3bdcc8`, merged `a854bd3`). Status line `:7` = `Revised (2026-07-18)`; `docs/decisions/README.md:11` index row agrees. | `ca7e85c` → merge `6e83e41`<br>`e3bdcc8` → merge `a854bd3` | **STRIKE** |
| **#3** | Done-when = *"the principle is written into the ADR-01 amendment"*. The amendment contains it **verbatim and id-tagged**: *"**Cost-optimization principle (#3):** the default is chosen by balancing measured synthesis quality against per-run cost, not by assumption."* (`ADR-01:16`) | `e3bdcc8` → merge `a854bd3` | **STRIKE** |

### Why these two survived — the drift mechanism, precisely

The close was **split across two merges, and each merge disclaimed the other's half**:

1. `6e83e41` (2026-07-18) shipped Branch A. Its own subject says **"ADR-01 amendment text deferred"** — correctly leaving #2/#3 open.
2. `a854bd3` (2026-07-18) then wrote the amendment, but its subject frames it as **"ADR-01 text-drift fix"** — a hygiene rider inside a *filing* pass (`file audit-remainder as #45–#55`). A filing pass does not strike items, so nobody looked back at #2/#3.

Neither pass was wrong; the handoff between them dropped the closure. **This is not evidence of systemic backlog rot** — it is one two-merge seam, and it is the only instance found in 50 items.

**Corroborating tell (self-contradicting artifact):** ADR-01's own Deployment-Status stamp at `:3` still reads *"residual: #2/#3 (amendment codification)"* — while the amendment sits thirteen lines below it in the same file. The stamp was written in the same pass that discharged it. Recommend refreshing that stamp when #2/#3 are struck.

**Knock-on:** #4 is written as *"conditional on #2 Branch A"*. Branch A shipped, so **#4's condition has FIRED** — and its own escape hatch (*"or is closed as not-needed if Gemini retained"*) is now void, because Gemini was not retained. #4 is no longer conditional; it is an unblocked, required ADR-02 amendment. See AWAITING-RULING below.

---

## MERGED-NOT-CLOSED

Correctly open. **Not kill candidates.** Each row names what actually gates the close.

| id | Merge SHA | What shipped | What gates the real close |
|---|---|---|---|
| **#18** | `c91ce26` | Substantial code: `crux_check.py` (+313), `research/headless.py` (+87), `debate.py` (+63), plus `orchestrator.py`/`synthesis.py`/`models.py` wiring and 507 lines of tests | **A live end-to-end witness.** Every crux path is mock-tested only. The merge subject itself says *"merged, not closed; e2e witness pending #83"*; `JOURNAL.md:258` states *"No live end-to-end witness was run"*. A real run bills research + synthesizer calls — so this is **cost-gated, not engineering-gated**. |
| **#20** | `3e1005f` (stopgap), `2fc5978` (PyYAML) | The widened-scope half landed: `types-PyYAML>=6.0` declared at `pyproject.toml:28` | **The stopgap is still in place.** 5 `# type: ignore` comments remain — `grok_research.py:67,:154`, `openai_deep_research.py:103,:157`, `openai_mini_research.py:156` — and the pin is still unbounded (`pyproject.toml:16`: `"openai>=1.50"`). Close needs the 2.x typing migration **or** a deliberately bounded pin, then removal of the ignores. |
| **#27** | `d6cbbd6` (Phases 1+2), `4495dfd` (witness) | Frozen corpus + 24 blinded transcripts + scoring sheet | **Phases 3–4 (scoring + unseal) never ran.** `JOURNAL.md:856` lists them as *"Not done (handed forward)"*. No parity report at the audits root; DRAFT-CLI-3 is still a pre-draft with an **empty evidence slot** (`docs/intake/2026-07-06-lane-cli-functional-design.md:181`) — neither ratified nor retired. The corpus is still live on disk and still a row in `docs/audits/README.md:39`. **The retire-the-stale-row-at-unseal obligation in #27's body is correct and must not be struck** — verified: `validate_docs_registry` catches corpus-without-row, not row-without-corpus. |
| **#66** | `3f9f332` (leg filed), `2492371` (lane A2) | The discharge leg exists and is honest: `scripts/verify_cli_output_contract.py:496` L11, gating at `:522`/`:530`, state printed at `:615-619` | **A witnessed run that cannot currently be made free.** Verified the blocking premise first-hand: `grep -n "backend" config/settings.yaml` returns **zero matches** — no `backend: cli` seat is declared. So the leg correctly reports GAP rather than falsely PASSing. Unblocks via **#27** (the ADR-12 §5 backend=cli flip) or an explicitly authorized billed run. |

---

## AWAITING-RULING

Blocked on an operator/Council decision, **not** on engineering. Filing these separately matters: they will never close by writing code, so they should not be read as engineering debt.

| id | The decision that is owed |
|---|---|
| **#4** | ADR-02 panelist/synthesizer overlap amendment. **Condition has FIRED** (see #2 knock-on). `ADR-02-default-panel.md:6` still reads `Revised (2026-05-11)`; no overlap text exists (`grep overlap docs/decisions/` → empty). Its Deployment-Status stamp claiming *"No open remainder"* (`ADR-02:3`) is now stale. |
| **#6** | DeepSeek replace/keep/demote. **A recommendation already exists** — `docs/audits/2026-07-17-night-batch-empirical-e2e-audit.md:118` says *"**DEMOTE** from default"* — but it was UC3 *test fodder* in a pipeline-quality trial, and nothing ratified it: `config/settings.yaml:14` still lists deepseek in `full_panel`. Re-scope from "evaluate" to "ratify or reject the existing DEMOTE verdict". |
| **#8** | ADR-34 ISO-timestamp exemption. Note the upstream ADR is **self-contradictory on this exact point**: `.dev-knowledge/.../ADR-34:73` says `YYYYMMDD_HHMMSS` (underscore) while `:62` says `council-out-YYYYMMDD-HHMMSS-topic.md` (hyphen). Live emitter `output.py:31` uses underscore. Also sequenced behind #7. |
| **#9**, **#55** | Both explicitly baseline-gated on the **T1 baseline-planning session** (an operator-scheduled event), mirroring `.dev-knowledge` #70. Correctly deferred; `.claude/commands/` contains only `override.md`. |
| **#73** | Review-runner convention. **First done-when clause already discharged** (the global script ran end-to-end on codex-cli 0.144.5). What remains is purely a decision: point at `/codex-review` + `~/.claude/bin/`, or direct `codex exec`, or re-home a repo-local wrapper. Note its embedded severity-summary defect is correctly marked operator-owned (core-invariant #6 forbids unilateral `~/.claude/` edits). |
| **#80**, **#81** | Both done-whens read *"a ruling picks a rule"*. **Substantial evidence was produced and then stranded — see ANOMALY A3.** A markdown-it-py spike proved #80's rule authority exists in CommonMark (§5.2/§4.8: continuation = `softbreak` inside one inline token; annotation = nested `bullet_list`) and proved #81's predicted **inversion is real** (fenced whole list → scanner correct, library returns `[]` = total option loss). Recommendation was KEEP-SCANNER. `JOURNAL.md:200` is explicit: *"this spike is evidence, not a closure"*. |

---

## ANOMALIES

### A1 — #110 and #128: foreign hub numbering, legitimately re-filed (suspicion 1: RESOLVED, not strays)

**The provenance claim checks out completely.** Hub commit `ea6217a` (2026-07-07 23:56) is real:

> `chore(groom): leg-c MOVE -- relocate 2 ai-council tasks out of the hub backlog`
> *"Per ADR-41 (child-repo work runs in its dedicated chat), #110 (round-2 isolation audit) and #128 (prompt caching) leave the hub backlog and re-file to ai-council's backlog at its Wave-1 onboarding arc… Leg-c ruling 2026-07-08."*

Corroborated three ways: the hub's own grooming log records *"#110 #128 moved-to: ai-council (Wave-1)"*; the local re-file commit is `2511329`, merged `1bdc2ea`; and `git show ea6217a -- BACKLOG.md` shows the two task lines being deleted hub-side with text matching what landed locally.

**Both are also genuinely LIVE, not stale strays** — verified against source:
- **#128** — `grep -rn "cache_control|prompt_caching|ephemeral" src/` → **zero hits.** No prompt caching applied anywhere. (`cli_base.py:279-284` *reads* `cache_read_input_tokens` from the claude CLI's own implicit caching — that is the CLI's behavior, not `cache_control` we set.)
- **#110** — no isolation audit exists, and the code answer is that **round-2 debaters DO see round-1 peer arguments**: `debate.py:234` builds `anon_block` via `_anonymize_responses`, injected at `:154` (pick) and `:173` (ideas/judge) under the header *"Below are anonymized contributions from other council members"*. `_anonymize_responses` (`:28-34`) **shuffles and relabels only — it strips attribution, not content.** The property actually held is **anonymity, not isolation**; the 58%-convergence risk the item cites is not mitigated by anonymization alone. This item is well-founded and arguably under-prioritized at P3.

**The real anomaly is the id-space, not the provenance — two defects:**

1. **Collision hazard.** Local ids run `1..83` with next-free at **#84**. The local counter is on a collision course with the two foreign ids: **26 more filings reach #110, 44 more reach #128.** At that point two distinct items share one id in one file. Distant but certain if left alone.
2. **Dangling cross-reference (confirmed).** Both items carry `refs #96` — **`#96` is unresolvable in either id space.** Verified: zero occurrences in ai-council git history; **zero occurrences in the current hub `BACKLOG.md`** (it was closed hub-side). It survives only in hub git archaeology (`cb62fe36`, `caf2ddc3`, `a4d081bd`). The re-file carried the text verbatim, including a hub-relative ref that stopped resolving when #96 closed upstream.

**Recommendation: RENUMBER `#110 → #84`, `#128 → #85`** (or the next free pair at execution time), and in the same edit **resolve or drop the `#96` ref** — either inline the one-line substance from hub `cb62fe36` or delete the pointer. This removes both defects at once. Do **not** close them: both are live, verified-unbuilt work.

### A2 — id-space integrity: **PASS** (suspicion 4)

Ran the full sequence `#1..#83`. **There are no gaps and no out-of-sequence local ids.**

- 48 local open + 35 closed = **exactly 83**, complete coverage of `1..83`.
- All 35 closed ids have closing evidence on main. 32 carry an `[#id]` tag in a first-parent merge subject.
- **The 3 that do not — #13, #14, #15 — were chased and cleared.** The grooming log cites bare SHAs; all three verified real, all 2026-07-02, all thematically correct: `bfc268f` (`feat(output): route output via return_dir`), `53ad525` (`fix(inbox): strip leading 'council'… double-council`), `f1a4b74` (`feat(output): emit minority report`). These predate the `backlog-id-on-close` commit-msg hook, which explains the missing tags.
- Story ids **`[S1]..[S16]`: no gaps** (14 live `###` headers + 2 collapsed delivered-story notes, `[S12]`/`[S14]`).
- Theme ids **`[E1]..[E7]`: no gaps.**
- Next-free pointer (`#84`) is consistent with `max(local open) = 83`.

Only out-of-range ids present are the two foreign task ids (A1) and legitimate hub-scoped prose citations (`#281`/`#286` = ADR story-id provenance, `#326` = a hub arc, `#96` = the dangling ref in A1).

### A3 — #80/#81 ruling evidence is on **unreachable commits** (new finding — highest-value action in this report)

The markdown-it-py spike that produced the decisive evidence for both open forks was committed as a throwaway and its branch deleted. **The commits are now reachable from no ref at all:**

```
1eb4ecb : refs_containing=0  NOT-on-main
a38f699 : refs_containing=0  NOT-on-main
b6c10af : refs_containing=0  NOT-on-main
```

`git fsck --unreachable` lists `b6c10af` explicitly as an unreachable commit. Its tree holds `spike/FINDINGS.md`, `spike/evidence.py`, `spike/md_it_options.py`, `spike/plugin_base.py`, `spike/plugin_swap.py`, `spike/worktree_path.py` — **none of which exist on main** (`git ls-tree -r main | grep '^spike/'` → empty).

What survived is only the *narrative*: `3bae0dd` is on main's first-parent spine but is **JOURNAL-only (1 file, +39 lines)**, whereas the dangling `b6c10af` is the same commit message with **3 files including the 50-line `FINDINGS.md` rewrite** carrying the inversion table.

**Risk:** these objects are subject to `git gc` pruning. Default `gc.pruneExpire` is 90 days, but any `git gc --prune=now`, an aggressive gc, or a repo re-clone drops them immediately and **the evidence base for two open rulings is gone** — leaving #80/#81 requiring a decision whose supporting analysis no longer exists.

**Recommendation (do this first — it is the only time-sensitive item in this report):** rescue the artifact before deciding anything. Either `git tag spike/md-parser-evidence b6c10af` to make it reachable, or cherry-pick `spike/FINDINGS.md` onto a `docs/` path as a dated audit artifact. Low cost, prevents an irreversible loss. Note this is also a **§5.9 "No leftovers" edge case the rule does not currently cover** — teardown removed the working tree correctly, but the evidence the arc was *for* went with the branch.

### A4 — merge `3c0541f`'s `[#56]` tag is a FILING reference, not a closure (suspicion 2: caught)

This was the one merge subject that read like a closure but is not. `3c0541f` is tagged `[#56]` and would be easy to misread as discharging it.

**Verified it did not.** `git show 3c0541f --stat` touches **61 files, none of them under `docs/archive/`**. What it actually created were the *other two* archives (`docs/audits/archive/README.md`, `docs/intake/archive/README.md`) — precisely the preservation archives that #56 exists to disambiguate `docs/archive/` *from*. `git log -- docs/archive/README.md` returns **a single commit, `82528c5` (2026-05-27 seeding)** — the file has never been modified since. `JOURNAL.md:901` confirms the merge's own intent: it *"Filed #56"* as rider (c).

**#56 is correctly LIVE.** Its done-when (rename to `docs/triage/`, or a prominent "NOT a preservation archive — deletion-tracked" banner) is unmet: the phrase "preservation archive" appears nowhere in the README and the deletion-default policy is buried mid-file. **Recommendation: leave open; no action beyond awareness that the `[#56]` tag is a filing marker.**

### A5 — two BACKLOG bodies are stale against work that has since happened

Not status errors — the items are correctly open — but the bodies no longer describe the state of play, which is how a reader loses trust in an otherwise sound record:

- **#80/#81** carry no mention of the spike at all. A reader deciding these forks would not learn that the rule authority was already established, that the inversion was already tested and confirmed real, or that a KEEP-SCANNER recommendation with an open question already exists. Compounded by A3 — the referenced evidence is both uncited *and* unreachable.
- **#82's premise is overstated.** It asserts the crux-check *"fires a headless retrieval on **every** debate"*. Verified false: `crux_check.py:245-250` reaches retrieval **only when parsing yields a real crux**; `ParseState.NO_CRUX` returns early at `:235-241` and MALFORMED at `:229-234`. The *extraction* call at `:214` is unconditional — but that one **is** metered (`build_call_metrics(..., round_number=-1)` at `:220`). The done-when remains unmet either way, but the invisible spend is **conditional, not per-debate**, which lowers the item's urgency.
- **#58's framing is slightly off.** It reads as though dissent-gating must be built. Gating already exists — `output.py:620` `extract_dissent` gates the emit via `save_minority_report:670-672`. The real work is **sharpening the existing extractor** (`_is_genuine_dissent:604-617` only rejects empty bodies, `_NO_DISSENT_PREFIXES` openers, and bodies under 12 chars). Confirmed nothing landed post-filing: `git log -S"_NO_DISSENT_PREFIXES"` returns exactly one commit, `f1a4b74` (2026-07-02, the original #15 feature).

---

## Cleared suspicions

| # | Suspicion | Verdict |
|---|---|---|
| 1 | #110/#128 are stale strays | **CLEARED on provenance, UPHELD on id-space.** Legitimately moved (hub `ea6217a`, ADR-41 leg-c, operator-ratified) and both verified genuinely unbuilt. The defects are the foreign numbering and the dangling `#96` ref — see A1. |
| 2 | Done-when already satisfied but still listed (#41-class drift) | **UPHELD — exactly 2 instances: #2 and #3.** Found by tracing every done-when to source, not by reading claims. One near-miss caught and cleared (#56 / merge `3c0541f`, see A4). |
| 3 | #77 cross-refs in #80/#81 masquerading as open | **CLEARED — fully.** #77 was struck at `94421c2` (`BACKLOG.md | 1 -`). Its only surviving mentions are prose (`BACKLOG.md:30`, `:31` — *"terra pass 6 on the #77 arc"* — and the grooming log at `:159`). **No struck id appears as an open task line anywhere.** |
| 4 | id-space gaps / out-of-sequence ids | **CLEARED — see A2.** No gaps in `#1..#83`, `[S1]..[S16]`, or `[E1]..[E7]`. |
| 5 | Validator counts vs real open work | **CLEARED — see below.** |

### Suspicion 5 — count reconciliation

`py scripts/validate_backlog.py BACKLOG.md` → `OK (7 themes, 14 stories, 50 tasks, 0 warning(s))`, exit 0.

Independently reproduced by hand: 7 `## [E<n>]` headers, 14 `### [S<n>]` headers, 50 `- [#id]` task lines. **The validator's counts are accurate.**

Two things it structurally cannot see, both worth knowing rather than fixing:

1. **It counts 14 stories; 16 exist.** `[S12]` and `[S14]` are delivered and collapsed to italic one-line notes (`:114`, `:141`) per ADR-65, so they are not `###` headers. Not a defect — the ADR-65 collapse is working as designed — but "14 stories" understates the story-id space, and the id-space audit must use all 16 to conclude no gaps.
2. **It validates structure, not truth.** It cannot know that 2 of its 50 tasks are already done. That gap is exactly what this audit exists to fill, and is the argument for repeating it periodically rather than trusting the green check.

**Internal-consistency check (run separately): PASS.** No open task line's id appears anywhere in the grooming log as `struck` or `CLOSED`. BACKLOG.md does not contradict itself.

---

## Recommended actions, in priority order

**All of these are recommendations. Nothing was struck, edited, or executed. The operator owns every one.**

| # | Action | Why this priority |
|---|---|---|
| 1 | **Rescue the spike evidence** — `git tag` `b6c10af` or cherry-pick `spike/FINDINGS.md` to a dated `docs/audits/` artifact | **Only time-sensitive item.** GC-able; loss is irreversible and would strand two open rulings (A3) |
| 2 | **Strike #2 and #3** | Done-when verified satisfied verbatim; cite `ca7e85c`/`6e83e41` + `e3bdcc8`/`a854bd3` |
| 3 | Refresh ADR-01's Deployment-Status stamp (`:3` still claims #2/#3 residual) | Same edit-window as action 2; the file currently contradicts itself |
| 4 | **Re-read #4** — its condition has fired and its escape hatch is void | Silently changed status; also refresh ADR-02's stale *"No open remainder"* stamp |
| 5 | **Renumber #110 → #84, #128 → #85**; resolve or drop the `#96` ref in the same edit | Closes both A1 defects at once, before the id counter gets closer |
| 6 | Refresh the #80/#81 bodies with the spike findings (after action 1) | Restores the decision context a ruling needs (A5) |
| 7 | Correct #82's *"every debate"* premise; re-frame #58 as *sharpen*, not *build* | Body accuracy; #82's urgency drops once corrected (A5) |
| 8 | Re-scope #5 (document the existing probe) and #6 (ratify the existing DEMOTE verdict) | Both are further along than their text implies; neither is unstarted |

**Deliberately NOT recommended:**
- **Do not strike #56** — merge `3c0541f`'s `[#56]` is a filing tag; `docs/archive/README.md` is untouched since `82528c5` (A4).
- **Do not strike #27's retire-the-stale-row-at-unseal clause** — independently re-verified this session: `validate_docs_registry` catches corpus-without-row, **not** row-without-corpus. The item's own warning *"Do not strike this clause believing #68 covers it"* is correct.
- **Do not treat #7 as partially covered by `validate_audit_casing.py`** — that validator is scoped to added `docs/audits/*.md` at exactly 3 path parts, prospective-only, and its header records that hub Rule A was **deliberately not carried** because it would block every new file under `src/`. It covers roughly one-ninth of #7's scope; `src/` is entirely uncovered.

---

## Appendix — per-open-item classification (all 50)

Verdict = whether the **done-when clause** is satisfied on main. Line numbers are the **current** ones; where BACKLOG's cited lines had drifted, both are shown.

### [E1] Invocation surface & delegation-readiness

| id | Class | Evidence |
|---|---|---|
| #34 | LIVE | `research/output.py` defines only `save_research_to_file:36` / `print_research_summary:123`. `save_verdict_package` (`output.py:1258`) has exactly **one** call site repo-wide — `orchestrator.py:255`. `runner.py:170`/`:222` are the only saves in `run_research`, both debate-free. Research commissions still emit zero verdict packages. |
| #75 | LIVE | `output.py:265-268` — secondary write is bare, outside any `try`; contrast the `target_paths` loop at `:305-313` which **is** wrapped (`except Exception … logger.warning`). The `.exists()` guard at `:266` only covers a *missing* dir. **Bonus finding:** the docstring at `:255-256` asserts *"Best-effort destinations (`secondary_dir`, `target_paths`) … only warn"* — false for `secondary_dir` on a raising write. Tests pin old behaviour (`test_dual_output.py:63-66`, `:88-92`, `:125-130` — missing-dir only). |
| #76 | LIVE | *(BACKLOG cites `790-791`; now `1284-1291`)* `:1285` builds the payload from `guaranteed_dirs`, `:1291` performs the write. `_build_verdict_payload:1210` emits `"paths": [str(d / filename) …]` from that pre-write list. The `p.exists()` filter at `:1195` protects other `written` kinds but by construction cannot cover the verdict's own entry. |
| #78 | LIVE | *(`283` → `305`)* `output.py:305`: `for target_dir in target_paths or []:` — no isinstance, no materialization. `try/except Exception` at `:306`/`:312` swallows per-item errors. Unvalidated upstream too (`cli.py:815`, `:685`); `list[Path] | None` at `:229` is an unenforced annotation. |
| #79 | LIVE | *(`orchestrator.py:224` → `:250-252`; `output.py:899` → `:1195`)* Both sites still existence-driven. `_save_metrics_json` (`output.py:1306`) returns `None`; its success signal is never plumbed to `:251`. |
| #80 | AWAITING-RULING | `output.py:1018-1019` still drops every indented line unconditionally (`if raw[:1] in (" ", "\t"): continue`). No continuation-vs-annotation rule in `_top_level_bullets:1001-1039`. Guard test `test_output.py:1435-1439` pins current rule. Spike established the CommonMark authority — see A3/A5. |
| #81 | AWAITING-RULING | `output.py:1017-1039` carries no fence state — no `in_fence` flag, no ``` detection. Guards are thematic-break (`:1025`,`:1034`), indent (`:1018`), bullet-grammar (`:1027`) only. Backtick handling at `:960-998` is inline code-span logic applied per-item at `:1036`, never block fences. Inversion confirmed real by spike — see A3. |
| #69 | LIVE | Both halves confirmed; **the two paths still disagree.** (a) `cli.py:811-814`: `eff_full = (use_full_panel or not lite) …` with `--lite` `default=False` (`:490-492`) ⇒ `not eff_full` at `:813` discards frontmatter `models:` on every non-`--lite` run, while the comment at `:804` promises "CLI flag > frontmatter > config default". (b) `cli.py:687` guards on `use_full_panel` instead (a declared **no-op** flag, `:485-489`), so `--inbox` *does* honour it. |
| #64 | LIVE | (a) `inbox.py:129` `frontmatter.load` unguarded in `parse_file`; call sites catch only `RoutingError` (`cli.py:679-683`, `:789-795`). Note `inbox.py:97` *does* guard it in the downloads scanner. (b) `cli.py:821-829` → `detect_mode`, whose every failure branch returns hardcoded `"pick"` (`mode_detector.py:39,:51,:53,:57`); bare `else` at `cli.py:839-840` also hardcodes `"pick"`. **The correct implementation already exists** at `cli.py:700-702` (inbox path) — a second `--file`/`--inbox` divergence, reinforcing #69's shared-helper remedy. |
| #36 / #37 / #38 | LIVE | `protocols/` holds exactly 4 files (`COUNCIL_INVOCATION_CONTRACT.md`, `COUNCIL_QUESTION_GUIDE.md`, `README.md`, `SYNTHESIS_QUALITY_RUBRIC.md`). No authoring template, decomposition advisory, or verdict→ADR read-back guide. Hub `templates/` likewise. |

> **Cross-cutting note for [E1]:** #69 and #64 share one root cause — two hand-maintained copies of frontmatter resolution — and #64(b)'s correct implementation already exists on the inbox path. Strong candidates to fix together behind one helper, as #69's done-when already suggests. This is the CLAUDE.md §10 inbox-parity anti-pattern manifesting twice.

### [E2] Synthesizer refresh

| id | Class | Evidence |
|---|---|---|
| #2 | **DONE-BUT-LISTED** | See kill-candidates table. `ADR-01:16` + `settings.yaml:12`. |
| #3 | **DONE-BUT-LISTED** | See kill-candidates table. `ADR-01:16`, principle present verbatim and id-tagged. |
| #4 | AWAITING-RULING | `ADR-02:6` = `Revised (2026-05-11)`; body ends `:13`; `grep overlap docs/decisions/` → empty. **Condition fired** via #2 Branch A; escape hatch void. |

### [E3] Provider reliability & CLI engine

| id | Class | Evidence |
|---|---|---|
| #5 | LIVE (re-scope) | `scripts/verify_openai_deep.py` exists and runs, but is a **different probe** — its docstring says *"Uses the migrated gpt-5.5 path; NOT the deprecated o3-deep-research"* — and is undocumented (zero refs in `CLAUDE.md`/`CONTRIBUTING.md`/`README.md`/`ADR-05`). No `@pytest.mark.integration` deep-research test. |
| #6 | AWAITING-RULING | DEMOTE recommendation at night-batch audit `:118`, unratified. `settings.yaml:14` still lists deepseek in `full_panel`. |
| #20 | MERGED-NOT-CLOSED | PyYAML half done (`pyproject.toml:28`). 5 ignores remain (`grok_research.py:67,:154`; `openai_deep_research.py:103,:157`; `openai_mini_research.py:156`); pin unbounded (`pyproject.toml:16`). |
| #21 | LIVE | `tests/test_integration.py:38-41` still imports `_build_all_providers` from `ai_council.cli`; symbol does not exist. Real function is `build_all_providers` (no underscore) in `runner.py:15`, called at `cli.py:635` **with a second arg**. Both name and arity changed. |
| #27 | MERGED-NOT-CLOSED | See table above. Corpus live on disk; row live at `docs/audits/README.md:39`; DRAFT-CLI-3 evidence slot empty (`lane-cli-functional-design.md:181`). |
| #43 | LIVE | `cli.py:48-55` `PROVIDER_CLASSES` has no `codex`. `seat_router.py:32-35` `CLI_PROVIDER_CLASSES` **does**. `runner.py:19-20` skips names absent from `provider_classes` and `seat_router.py:131` drops seats with no API twin — so the design fork (no-API-twin fallback semantics) is untouched and a truthful `codex` seat still cannot be built. |
| #32 | LIVE | `doctor.py` has only `check_keys:73`, `check_seats:115`, `validate_config:150` — no auth-lane probe; its docstring `:19-21` still lists the CLI-fleet re-probe as deferred. `metrics.py:39-40` zeroes unconditionally (`if response.backend == "cli": cost = 0.0`), with the comment at `:31-37` naming #32 as the unbuilt fix. |
| #72 | LIVE | *(`275-277` → `282-283`)* Still `sorted(p for p in health_dir.glob("doctor-*.json") if p.name != "doctor-latest.json")`. No anchored regex. |
| #52 | LIVE | No `first_seen`, no `held(...)` anywhere in `doctor.py`. `build_record:254` writes a fresh snapshot; `_prune_health_records:278` only deletes. Docstring `:21` names advisory aging as deferred. |
| #53 | LIVE | No auto-mode-detection guidance for directive-shaped research briefs; GUIDE unamended. |
| #54 | LIVE | `gemini_research.py:23` `_TERMINAL_STATUSES = frozenset({"completed","failed","cancelled","incomplete"})`; at `:101-107` only `"completed"` reaches extraction, every other terminal raises `ResearchProviderError`. No partial path. |
| #59 | LIVE | `research/output.py:18` `_LEADING_RESEARCH_RE = re.compile(r"^research[-_ ]+(.+)$", re.I)`. Bare case traced: `"research!"` → `_slug` strips to `"research"` → regex needs ≥1 separator, hits end-of-string → no match → `council-out-{ts}-research-research.md`. The comment at `:16-17` acknowledges it. |
| #61 | LIVE | `cli_base.py:283-292`: `input_tokens` is an `or 0` sum guarded only by `if usage else None`, so a *present but partial* usage yields `0`, not `None`; `token_count` at `:289-292` then becomes a definitive `0`. `metrics.py:28` (`response.input_tokens or 0`) cannot distinguish it. Only *fully absent* usage degrades to None. |
| #70 | LIVE | `research/output.py:58` second-resolution `ts`, `:59` slug at `max_len=50` (`:21`), `:63` filename, unconditional `_write_routed` at `:114`. No sub-second precision, hash, or exists-check. |
| #66 | MERGED-NOT-CLOSED | See table above. `verify_cli_output_contract.py:496`/`:522`/`:530`/`:615-619`; `grep -n backend config/settings.yaml` → **zero matches**. |
| #82 | LIVE (premise overstated) | Ledgers separate: `research/models.py:22`,`:36` vs `models.py:95`,`:109`. `print_cost_summary:108-143` prints no research line. `metrics.py:70-72` self-documents the hole naming #82. **Correction:** retrieval is **conditional**, not per-debate — `crux_check.py:245-250` vs early returns at `:235-241`/`:229-234`; the unconditional extraction call at `:214` **is** metered at `:220`. |
| #83 | LIVE (date-gated) | `ls docs/audits/ | grep -i crux` → exactly 2 files (pass 1, pass 2). No pass-3 artifact. Gate 2026-07-25 has not elapsed (today 2026-07-21). Claim accurate. |
| #128 | LIVE + ANOMALY (A1) | `grep -rn "cache_control|prompt_caching|ephemeral" src/` → zero hits. |

### [E4] Model currency

| id | Class | Evidence |
|---|---|---|
| #17 | LIVE | No online version check. `doctor.py:192-214` config checks cover only research roster resolution and `min_successful`. Docstring `:20-21` lists the pin-currency sweep as future v2. No refresh process documented. |
| #57 | LIVE | `config/settings.yaml:66` still `model: "claude-opus-4-7"`. Not bumped. |

### [E5] Naming & quality automation

| id | Class | Evidence |
|---|---|---|
| #7 | LIVE | No such hook in `.pre-commit-config.yaml`. `validate_audit_casing.py` is far narrower — docstring `:41-43` scopes it to added `docs/audits/*.md` at exactly 3 path parts, prospective-only (`--diff-filter=A`); header records hub Rule A **deliberately not carried** (its `SANCTIONED_TIER1_DIRS` omit `src/`, so carrying it would block every new `src/ai_council/` file). Casing regex `:39` is right; coverage is ~1/9 of scope. |
| #8 | AWAITING-RULING | No in-repo record. Upstream ADR-34 self-contradictory (`:73` underscore vs `:62` hyphen); emitter `output.py:31` uses underscore. |

### [E6] Council process & epistemic quality

| id | Class | Evidence |
|---|---|---|
| #9 | AWAITING-RULING | `.claude/commands/` contains only `override.md`. Explicitly baseline-gated in its own text. |
| #10 | LIVE | `SYNTHESIS_QUALITY_RUBRIC.md` touched **exactly once** since creation (`0966c2a`). Faithfulness still the unrefined single line at `:13`. |
| #11 | LIVE | Hub `LESSONS.md` (mtime 2026-07-16) has no "bilateral handshake = 1 round trip" entry; closest is `:272` (2026-05-12), which contrasts handshake vs scrum-master review without round-trip codification. |
| #55 | AWAITING-RULING | Baseline-gated on the T1 planning session; no design delivered. |
| #18 | MERGED-NOT-CLOSED | See table above. `c91ce26`; `JOURNAL.md:258`. |
| #19 | LIVE | `grep -rn "framing|false_consensus|false-consensus" src/` → **one hit, unrelated**: `output.py:384`, a docstring about *"controls document framing"* (markdown layout). No framing role, no alarm. |
| #110 | LIVE + ANOMALY (A1) | No isolation audit exists. Code: round-2 debaters **do** see round-1 peer arguments — `debate.py:234` → `:154`/`:173`; `_anonymize_responses:28-34` shuffles/relabels only. Property held is anonymity, not isolation. |
| #49 | LIVE | `grep -rn "intent_match|intended_author|intent_source" src/ tests/` → zero hits. |
| #58 | LIVE (re-frame) | Gating **already exists**: `output.py:620` `extract_dissent` gates `save_minority_report:670-672`. Weak discriminator is `_is_genuine_dissent:604-617` + `_NO_DISSENT_PREFIXES:570-583` + a 12-char floor. `git log -S"_NO_DISSENT_PREFIXES"` → one commit, `f1a4b74` (2026-07-02). Nothing landed post-filing. |

### [E7] Record & governance hygiene

| id | Class | Evidence |
|---|---|---|
| #50 | LIVE | `grep "## Watches" BACKLOG.md` matches only `:145` (the task text) and `:159` (grooming narration). No section; W1–W3 unseeded. |
| #51 | LIVE | No ADR-status↔index check among the 8 local pre-commit hooks; `scripts/check.ps1` is 3 steps (pytest, mypy, ruff); `canonical_freshness_gate.py` has zero ADR/index references. |
| #73 | AWAITING-RULING | No `scripts/codex-review.ps1` in-repo. `CLAUDE.md:141` still says *"Codex via `/codex-review`"*, unreconciled. First done-when clause discharged; decision outstanding. |
| #56 | LIVE | See A4. `docs/archive/README.md` unmodified since `82528c5`; merge `3c0541f` touched 61 files, none under `docs/archive/`. |

---

## Method notes and limits

**What was done:** every one of the 50 open items was checked against main. 25 had their done-when traced to specific current `file:line` evidence; the remainder were verified by targeted grep/git for the absence of the named artifact. Every closed id in `1..83` was checked for closing evidence. Suspicions 1–5 were each chased to a verdict. All strike-bearing findings (#2/#3) were re-verified first-hand rather than accepted from delegated search.

**Limits, stated honestly:**
- **No tests were run and nothing was executed against a provider** — per the unattended/read-only/no-billing constraint. Verdicts are source-inspection verdicts. In particular the *behavioural* claims in the appendix (e.g. #59's bare-`"research"` trace, #61's partial-usage collapse) are traced by reading the code path, not by executing it.
- `validate_backlog.py` was run read-only against `BACKLOG.md` (exit 0) — the sole command executed.
- The hub (`.dev-knowledge`) was read read-only for provenance (`ea6217a`, `#96`, `LESSONS.md`). No hub writes.
- Line numbers are accurate as of `f758fa6` and will drift.
- **Nothing was struck, edited, renumbered, or closed.** `BACKLOG.md` and `src/` are untouched. Every recommendation above awaits operator execution.
</content>
