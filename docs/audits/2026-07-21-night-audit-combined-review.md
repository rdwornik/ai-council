# Combined review — the 2026-07-20/21 night audit set (four reports)

**Type:** synthesis across four merged night-batch audits. **No decision ratified, nothing struck, nothing filed** (moratorium held).
**Written for:** a fresh chat + the architect. Insight-first, ranked, deduped against `BACKLOG.md` @ `116eb16`.
**Date:** 2026-07-21.
**Inputs (all now on `main`):**

| Report | Model | Merged as |
|---|---|---|
| `2026-07-20-night-vision-audit-fable.md` | Fable 5 | `worktree-night-vision-audit` |
| `2026-07-21-night-input-layer-audit-fable.md` | Fable 5 | same branch (the `night-input-audit` worktree never existed) |
| `2026-07-20-night-code-audit-opus.md` | Opus 4.8 | `worktree-night-code-audit` |
| `2026-07-20-night-backlog-audit-opus.md` | Opus 4.8 | `worktree-night-backlog-audit` |

---

## 1. Per-report value verdict

Value test applied: **does it change a decision the operator now faces?** Not length, not erudition.

### Code audit (Opus) — **VALUABLE** (highest raw yield in the set)

~10 genuinely new P1 defects with `file:line` citations, four confirmed by *executing* the logic rather than reading it. The new set is not cosmetic: a synthesis hiccup discards an entire paid-for run (P1-9), one seat's non-`ProviderError` cancels the whole round (P1-2), degraded research reports are frozen into a 7-day cache so the condition cannot self-heal (P1-6), and mid-debate seat loss is invisible in *every field* of the contract-1.0 delegation surface (P1-8). It also retracted its own headline mid-session after finding the `markdown-it-py` spike evidence — the retraction is itself a reliability signal, and the corrected finding (that #81's ruling is the real blocker) is more useful than the recommendation it replaced.

### Backlog audit (Opus) — **VALUABLE**, for one finding the others could not have produced

The headline ("the backlog earns trust — 48 of 50 survive") is a *negative* result and will regenerate identically. Its value is concentrated in four places: **A3** (the spike evidence backing two open rulings sits on unreachable, GC-able commits — the only time-sensitive item in the whole night set), the two verified kill candidates (#2/#3), **#4's condition having silently fired**, and **A5's correction of #82's own premise**. That last one matters disproportionately: it lowers a ticket's urgency *and* it caught an error the vision audit had already propagated (see §3).

### Vision audit (Fable) — **VALUABLE in its second half, LOW-VALUE in its first**

§2's holes are the payload. **H7 is the strongest single new finding in the report** — the #18 Phase-A freeze deliberately excludes the crux artifact from the verdict package, so a Lane-A caller cannot distinguish a *grounded* verdict from an *ungrounded* one without parsing the transcript the package exists to spare it from. That is concrete, unfiled, and a natural rider on the #34/#76 Contract-1.1 batch. **H1** (creativity vs decision engine) forces a vision-record decision. **H2** (research mode *bypasses* debate, so "debate that researches" ships at ~5% of its stated ambition) reframes the flagship use case honestly.

**§1 is where the low value sits.** The for/against direction survey is a literature review, and the report names its own adjudicator: **#55, the matched-compute baseline, is unbuilt.** Until it runs, §1 is an argument that regenerates verbatim next week, and re-running the audit will not settle it. Read §2; treat §1 as background.

### Input-layer audit (Fable) — **PARTIAL**

A well-built ADR seed: three boosting architectures with costs and owners, three routing options, a clean split of the 599-line GUIDE into linter-codifiable vs judge-required halves, and a genuinely useful honesty-loop design (provenance block + edit-distance as free supervision on boost quality). But the great majority of its "findings" restate already-filed tickets — #36, #37, #38, #9, #53, #64 — with a 2026 literature layer on top, and it resolves nothing by design.

Two things in it *are* new and earn the report its keep:
- **`detect_mode()` cannot emit `research`.** Its prompt offers only pick/ideas/judge (`mode_detector.py:14-22`), so `research` is structurally unreachable by auto-detection and must be forced via `-M`/frontmatter. Not in `BACKLOG.md`; adjacent to #64 but a distinct defect.
- **ADR-95 lane discipline is in direct tension with boosting itself.** A boost that improves a weak question *is* substance-shaping, which ADR-95 reserves for the architect. The report's proposed boundary rule — *the boost may restructure, interrogate, and flag; it may never assert a fact the caller didn't give* — is the useful contribution, and whether it suffices or ADR-95 needs amending is correctly left open.

---

## 2. Cadence verdict — which of these is a nightly routine?

**Answer: none of them, as run. Three are triggered audits; one is triggered-but-frequent.**

| Axis | Verdict | Justification |
|---|---|---|
| **Vision** | **TRIGGERED** | Subject is `VISION.md` (`last_reviewed` 2026-07-19), the ADR set, and the published literature — none change on a nightly clock. Re-running tonight reproduces §1 verbatim. **Triggers:** H1 ruled · #55 baseline result lands · an ADR changes the direction. |
| **Input layer** | **TRIGGERED — and should not re-run at all until a ruling lands** | Purest case in the set. It maps a decision space; the space is unchanged until the operator picks a §1 mechanism and rules H1. A second run produces the same six open questions. |
| **Backlog** | **TRIGGERED, at merge-arc cadence** | Subject *does* change — but on filing/closing passes, not overnight. Yield rate argues the point: **one drift seam in 50 items**, and it was a two-merge handoff. Nightly would burn a full sweep to re-report "earns trust". Run it after each multi-item filing or closure arc. |
| **Code** | **TRIGGERED at full-sweep scale; the nightly-shaped version is diff-scoped and already exists** | `src/` changes every coding session, so the *axis* is the closest to nightly-eligible in the set. But findings persist until fixed — night 2 re-reports the same 16 P1s at full cost. Re-run the full sweep after the P1 set is absorbed, or quarterly. The genuinely nightly-shaped review is per-diff, which `/codex-review` already covers. |

**The pattern worth naming:** a *finding* audit becomes nightly-valuable only once the backlog absorbs its previous output. Until the P1 set is triaged, a second code sweep is mostly a re-print. The moratorium is therefore the binding constraint on this whole cadence question — see §5.

---

## 3. Cross-cutting findings — where two or more reports converge

Ranked by strength of signal.

**C1 · The crux-check surface is criticized from three independent angles.** Vision **H7** (artifact excluded from the verdict package by the Phase-A freeze). Backlog **#82 + A5** (its cost is invisible to `print_cost_summary`, and the ticket's own premise is overstated). Code **P1-1** (`_parse()` sits outside `generate()`'s guard, so `CruxCheckService.check`'s *"Never raises."* docstring at `crux_check.py:203` is a claim the code cannot honour). Three reports, three lanes — contract, cost, correctness — all landing on the newest subsystem in the repo. It is the single most-flagged surface in the night set.

**C2 · The input/authoring layer is the largest unbuilt gap, and all three reports that look at it agree.** Vision break **B1** ("authoring is unassisted" — the advisor the vision calls *the window on the world* is the least-built part of the chain). Input audit **§0.3** (everything the operator calls the core is filed and unbuilt). Backlog appendix (`protocols/` holds exactly four files; #36/#37/#38 all LIVE, all P3). No contradiction anywhere — the disagreement is only about *who should own the fix*, which is the input audit's §5.

**C3 · The research/debate fork is the central architectural crack, and two reports independently propose the same cheapest remedy.** Vision **H2/B2/Swing A** (the archetypal "which library?" question is simultaneously a research and a pick question; the mode system forces a fork, so it is served whole by neither lane). Input audit **§2 R1** (hybrid as composition — the boost splits the ask into a research sub-commission feeding a pick sub-commission). The vision audit's remedy (b) and the input audit's R1 are **the same design arrived at from opposite ends**, and both note the CONTRACT already supports sequencing, so R1 needs no architecture change. That convergence is the strongest evidence in the set for what to build first if H1 resolves toward composition.

**C4 · Prose asserts guarantees the implementation does not honour — at every layer.** The code audit makes this its theme 4 and its closing paragraph: *"a confident comment is now the least reliable signal in this repo"* (P2-19..P2-26, plus the phantom-path filter at P2-4 that exempts its own artifact). The backlog audit finds the **same failure mode one level up**: #82's premise is verifiably false, #58 reads as "build gating" when gating already exists, ADR-01's Deployment-Status stamp claims a residual discharged thirteen lines below it in the same file. Two independent reports, two different artifact classes, one mechanism. **This is the most transferable finding in the night set** and neither report frames it as cross-cutting because neither could see the other.

**C5 · Cost ledgers are fragmented and the cost story contradicts itself.** Vision **H6** (`settings.yaml` declares zero `backend: cli` seats, so every live run bills API while ADR-12's headline is $0 seats; default flip evidence-gated on the unrun #27). Backlog **#66/#27** (independently verified: `grep -n backend config/settings.yaml` → zero matches). Code **P2-1/P2-30** (crux extraction renders as "Round -1" in the operator's cost summary; gemini research cost is structurally always $0). All three agree the accounting is not trustworthy today.

**C6 · The inbox/interactive parity split is live, not hypothetical.** Code **P1-4** with an executed truth table. Backlog **cross-cutting note for [E1]** (#69 and #64 share one root cause — two hand-maintained copies of frontmatter resolution — and #64(b)'s correct implementation *already exists* on the inbox path). Both prescribe the same fix: one shared resolver. Repo doctrine already warned about this in `CLAUDE.md` §10; it is now measured.

### Where the reports contradict each other

**X1 · The vision audit propagated an error the backlog audit disproved the same night.** Vision H6 states the crux-check *"now spends research money on every debate."* Backlog A5 verified this false against source: retrieval is reached only when parsing yields a real crux (`crux_check.py:245-250`; `NO_CRUX` returns early at `:235-241`, `MALFORMED` at `:229-234`). The vision audit inherited the claim from #82's own ticket text rather than from the code. **Consequence: #82's urgency is lower than two documents currently imply, and H6's cost argument is weaker than stated.** Correcting #82's body fixes both at once.

**X2 · Severity disagreement on #69, stated openly and worth adjudicating.** `BACKLOG.md` files it P2; the code audit argues P1 and gives its reasoning: the defect is silent, operator-facing, and *unfalsifiable from the artifacts* — the transcript records the panel that ran, never the panel requested, so a run that quietly used the default 5-model panel instead of the operator's chosen 2 is indistinguishable after the fact, **including in the verdict package downstream repos consume as a binding input**. That argument is sound and is not addressed in #69's text.

**X3 · The two Fable reports sit on opposite sides of the H1 fork without either resolving it.** The vision audit argues creativity is *unbuilt and overstated* (H1). The input audit *takes the boosting frame as given* and flags H1-sensitivity only in §2 (under H1-creativity the boost should default to proposing an ideas→pick two-step; under H1-decision the identical mechanism is just R1 composition with `ideas` in the first slot). Not a contradiction in fact — a shared dependency on an unmade decision. See §6.

---

## 4. Dedup against BACKLOG

The code audit did its own reconciliation honestly and it holds up. Consolidated:

**Already tracked — treat these as independent confirmations, do not re-file:**
#69 (P1-4, plus a new `--full`-is-not-a-no-op half absent from #69's text) · #75 (P1-5) · #76 (P2-4) · #79 · #80 · #81 · #70 (sibling: P2-3 is the *debate transcript* emitter, #70 is the *research report* emitter — likely a genuine second instance) · #82 (P2-30) · #21 (partial — same file, different defect) · #20 · #61 · #72 · #58 · #54 · #59 · #27/#43 (P2-32 is a gated feature, correctly not a defect) · #36/#37/#38 · #9 · #53 · #64 · #110 · #128 · #55 · #19.

**Genuinely new, not in `BACKLOG.md` @ `116eb16`:**

| Source | New items |
|---|---|
| Code audit | **P1-1** `_parse()` outside the guard (falsifies the *"Never raises"* contract two callers depend on) · **P1-2** `gather` without `return_exceptions=True` · **P1-3** SDK clients built in `__init__` and reused across per-file `asyncio.run()` loops — *the documented gemini gotcha, unfixed in the other four providers* · **P1-6** degraded research cached 7 days, cannot self-heal · **P1-7** `classify_error` substring-matches naked HTTP digits; `auth` outranks `server_error` · **P1-8** `provider_statuses` means "ever succeeded" · **P1-9** synthesis failure discards the paid-for run · **P1-10** summarizer has no timeout (~30 min worst case) · **P1-11** timeouts detected by a substring the providers never emit · **P1-12/13** the two god-modules · **P1-14/15/16** the test-integrity set · **P2-1** "Round -1" · **P2-5/6/7** · **P2-27/28** dead config keys · the P2-19..26 doc-divergence set |
| Backlog audit | **A3** spike evidence on unreachable commits · **A1** id collision (#110/#128) + dangling `#96` ref · **#4's condition has fired** (escape hatch void) · **#2/#3 verifiably done** |
| Vision audit | **H7** crux artifact excluded from the verdict package · **H1** the vision-record fork |
| Input audit | **`detect_mode()` cannot emit `research`** · **ADR-95 vs boosting** substance-shaping tension |

**Calibration note the code audit recorded against itself, worth preserving:** it did *not* independently find **#64**, in a file it claimed to have audited in full. Coverage claims from these sweeps should be read with that in mind.

---

## 5. Top 5 ranked actions

Each with a recommended next step, not a menu.

**1 · Rescue the spike evidence — tonight.** `git tag spike/md-parser-evidence b6c10af`. This is the only time-sensitive item across all four reports: `1eb4ecb` / `a38f699` / `b6c10af` are reachable from **no ref at all**, `git fsck --unreachable` lists `b6c10af` explicitly, and the tree holds the `spike/FINDINGS.md` inversion table that is the entire evidence base for two open rulings (#80/#81). Any `git gc --prune=now`, aggressive gc, or re-clone destroys it irreversibly.
**Moratorium: NOT blocked** — a git tag is neither a strike nor a `BACKLOG.md` filing.

**2 · Rule #81's preferred-failure question: fabrication vs total option loss.** Both Opus reports independently name this the highest-leverage outstanding decision, and it is upstream of everything else on that surface — the parser choice follows from the ruling, not the reverse. The evidence is already gathered: the hand-rolled scanner fabricates options from a fenced diff (#81) but is *correct* on a wholly-fenced options list where `markdown-it-py` returns `[]`; the scanner wins that row **by accident**, because the line-level fence-blindness causing the fabrication is what saves the payload. Neither implementation satisfies both halves of #81's done-when. Two sessions have now spent effort downstream of this unmade ruling.
**Moratorium: NOT blocked** — a ruling, not a filing. Do action 1 first so the evidence survives to be read.

**3 · Rule H1 — the vision fork.** It gates more than its own report: the input audit's §2 routing default flips on it, its §6 swing ordering depends on it, and the vision audit's §3 proposals cannot be prioritized without it. Recommended next step: amend `VISION.md` in one direction and let the downstream designs settle — see §6 for the framing, which this review deliberately does not resolve.
**Moratorium: NOT blocked.**

**4 · Triage the code audit's new P1 set into tickets.** This is the largest block of new, verified, actionable value in the night set and the one thing the moratorium actually blocks. Recommended next step: a single filing pass over **P1-1, P1-2, P1-3, P1-6, P1-7, P1-8, P1-9, P1-10, P1-11** plus the test-integrity trio **P1-14/15/16**; the reconciliation table already separates new from tracked, so the pass should not need to re-derive anything. Prioritize **P1-3** inside that set — it is the repo's own documented gemini gotcha (`CLAUDE.md` §10) unfixed in the other four providers, and `--inbox` with ≥2 files is the live trigger.
**Moratorium: BLOCKED.** Nothing else in this review is worth more per token spent, so this is the strongest argument for lifting it.

**5 · Close the verified record drift in one edit-window.** Strike **#2/#3** (done-when satisfied verbatim; cite `ca7e85c`/`6e83e41` + `e3bdcc8`/`a854bd3`), refresh ADR-01's Deployment-Status stamp — which currently claims a residual discharged thirteen lines below it in the same file — re-read **#4** whose condition has fired and whose escape hatch is void, and correct **#82's** false "every debate" premise (which also repairs vision H6, per X1). Renumber `#110 → #84` / `#128 → #85` and resolve the dangling `#96` ref in the same pass.
**Moratorium: BLOCKED** (striking + filing). Low risk, high record-integrity return, and every item is verified against source.

**Deliberately not in the top 5, recorded so it is not re-litigated:** #55 (the matched-compute baseline) is the adjudicator for the entire vision question and the vision audit is right that the direction is an unfalsified bet without it — but it is correctly baseline-gated on the operator-scheduled T1 planning session, so it is not an action available now.

---

## 6. Open vision decision — the H1 fork (surfaced, not resolved)

Three of the four reports touch it; none can resolve it.

**The record says decision engine.** `VISION.md:14` — *"Multi-model AI debate and research tool for architectural decision-making."* No ADR mentions creativity. The differentiation the vision audit argues is genuinely defensible (§1 case FOR) is the **decision/governance layer**: blind voting, non-participating synthesizer, minority reports, degradation alarms with exit-code semantics, transcript-free verdict packages, a binding ADR pipeline. No peer framework ships that.

**The operator's frame says question-boosting / creativity engine** — the input audit's premise, and the GUIDE's own doctrine supports the leverage claim: *"question framing is the only bias-control point with no safety net… the highest-leverage step in running a debate."* That doctrine exists in prose with **zero mechanism behind it**.

**What each side costs, from the reports:**

- **If decision engine:** `ideas` mode stays the least-developed mode (1 round, no divergence support, no creativity metric, and `SYNTHESIS_QUALITY_RUBRIC` scores synthesis *fidelity*, not idea novelty). The honest move is to strike "creativity engine" from the self-description. Input-layer routing then defaults to R1 composition with `ideas` merely available in the first slot.
- **If creativity / boosting engine:** `VISION.md` must be amended to claim it, and `ideas` mode needs real mechanics — divergence rounds, semantic-diversity scoring. The input layer's default flips: the boost should *propose* an ideas→pick two-step whenever a raw ask arrives option-less, making option-set generation part of the input stage.

**What makes the fork urgent rather than philosophical:** the mechanism is shared either way — only the default flips — but the *owner* is not. The input audit's §1 shows each candidate architecture anchors a different owner (A anchors caller-side, B anchors an interaction channel, C anchors a council-side entry stage), and §5 shows only Design B (the interactive clarify-loop) genuinely reopens ADR-11. Choosing a first build before ruling H1 risks anchoring the wrong owner for the wrong engine.

**Not resolved here by design.** The operator owns it.

---

**End of combined review.** Synthesis only — read-only across four merged reports. Nothing struck, nothing filed, no decision ratified. Source reports on `main` at `docs/audits/`.
