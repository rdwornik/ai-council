# Plan of Record — ai-council build program

**Date:** 2026-07-16 · **Status:** committed record (operator's frozen phase plan, materialized) · **Authored by:** operator (phase plan, verbatim) + Claude Code (Fable) (task mapping, this arc) · **Branch:** `docs/plan-of-record`

**Sources:** `docs/intake/2026-07-06-technical-architect-intake.md` (entry hub; §3 seam contracts, §5 rulings register) · the five lane functional designs `docs/intake/2026-07-06-lane-{cli,doc,epi,gov,int}-functional-design.md` · `docs/audits/2026-07-05-fleet-recon-liveness-and-process-design.md` (§8 fork list F1–F12) · `docs/audits/2026-07-06-code-refactoring-guide.md` (A1–A5/B1–B7) · ADR-11, ADR-12 · `protocols/COUNCIL_INVOCATION_CONTRACT.md` §7 (Known deviations) · `BACKLOG.md` (reconciled this arc: #22–#31, [E7], [S10]–[S12]).

---

## 1. Phase plan (frozen — embedded verbatim from the operator prompt)

> G1 operator accepts 15 rulings -> G2 consolidation+GOV-1 lifts pause -> G3 EPI-1 ruling = Epic B event.
> P0 this arc | P1 consolidation (G1) | P2 EPI-1 archaeology, zero code, pause-independent |
> P3 quick unlocks: F3 grok login, F12 stale pin, DOC-3 post-pause | P4 build wave 1: doctor ->
> CLI seats claude+codex -> INT verdict package; sidecar seam rule: first lane defines the
> extension mechanism, never built concurrently | P5 evidence-gated: CLI-4 parity (n=12) ->
> default-flip; #18/#19/#9 after G3 | P6 window completion: ADR-11 deviation closure; hub-side
> arc is a separate hub session.

## 2. Gates

| Gate | Event | Effect |
|---|---|---|
| **G1** | Operator accepts the 15-item rulings register (intake §5; source of record = each lane doc's §6) | Consolidation session becomes runnable |
| **G2** | Consolidation session + GOV-1 execution (**#31**: rulings → RULED; ADR-09/10 → Accepted; CLAUDE.md §11 through ADR-11/12; VISION reconcile; CONTRIBUTING re-stamp; push `main`) | **Feature-work pause lifts** |
| **G3** | Operator's ruling on the EPI-1 archaeology report (**#24**) | **Epic B event** — un-gates #18/#19/#9, D12/D13, and the v2 crux-resolver ranking (ADR-13) |

G2 gates the build wave (P4) and the post-pause quick unlock (DOC-3). G3 gates only the epistemic mission drafts (P5 second row). P2 (EPI-1) is pause-independent and can run before, during, or after G1/G2.

## 3. Seam rules

**Sidecar seam rule (frozen, verbatim):** *first lane defines the extension mechanism, never built concurrently.* Concretely: `seats[]` (L-CLI) and `synthesis` (L-EPI) both extend the `_metrics.json` sidecar — whichever lands first defines how the sidecar is extended; the second conforms. Serialize; never build both extensions in parallel sessions.

The five cross-lane seam contracts are canonical in the intake doc §3 (`docs/intake/2026-07-06-technical-architect-intake.md`) — **design against, never redesign**: (1) metrics sidecar namespacing, (2) doctor ownership (L-DOC owns; L-INT consumes as optional pre-flight; L-CLI contributes exactly the identity re-probe), (3) Epic B gate (L-EPI owns), (4) parity evidence (L-CLI owns; only CLI-4 results ratify the ADR-12 §5 flip), (5) enforcement (L-GOV owns; hub-carrier work is a hub arc).

Additional seam note (L-INT): the verdict package is **not** a sidecar extension — it is a separate caller-facing artifact (the sidecar is telemetry; the package is the deliverable). It consumes `seats[]`/`synthesis` facts by reference and designs neither.

## 4. Phase → task map (closure, as amended 2026-07-16)

Closure rule (operator's amendment, this arc): *every phase row maps to either a task ID or an explicitly named session-event/gate; P0 maps to this arc's plan doc cited by merge SHA.*

| Phase | Row item | Maps to |
|---|---|---|
| P0 | this arc | session-event: this doc, cited by its merge SHA |
| P1 | consolidation (G1) | **#31** |
| P2 | EPI-1 archaeology | **#24** |
| P3 | F3 grok login | **#28** |
| P3 | F12 stale pin | **#29** |
| P3 | DOC-3 post-pause | **#30** |
| P4 | doctor | **#25** |
| P4 | CLI seats claude+codex | **#16** (existing) |
| P4 | INT verdict package | **#26** |
| P5 | CLI-4 parity (n=12) → default-flip | **#27** (depends-on #16) |
| P5 | #18/#19/#9 after G3 | **#18, #19, #9** (existing; baseline-gated) |
| P6 | ADR-11 deviation closure | **#22, #23** |
| P6 | hub-side arc | session-event: separate hub session (requirements = L-GOV §3(Q3) R1–R8; no local task) |

## 5. P4 pre-work map (refactoring guide → build stories)

Pre-work notes are attached on the three P4 tasks only (#25, #16, #26); this table is the rationale. Guide framing: Part A structural (unblocks the build wave), Part B mechanical (do anytime); *"A1, A2, and A3 are the load-bearing three."*

| P4 item | Load-bearing pre-work | Same-module adjacencies |
|---|---|---|
| doctor (#25) | **A2** — decompose `cli.py:main` → `@click.group` with `run`/`doctor` subcommands | A5, B5, B7 touch `cli.py`; land first |
| CLI seats (#16) | **A1 → A3** — template-method provider base, then the one error classifier + timeout/retry contract (five-token cause vocabulary = `seats[].fallback_events[]`) | A4/B3/B7 touch `output.py`/`policy.py` |
| verdict package (#26) | **A4** — decompose `save_to_file`; `save_verdict_package` lands as a sibling | B3 (tz-aware timestamp helper for the deterministic `<ts>`) |

**Contention:** `output.py` is the highest-contention module of the wave (A4 + B3 + `seats[]` sidecar + verdict package) — serialize work touching it; `cli.py` is second (A2/A5/B5/B7 + doctor). **Do-anytime, no dependencies:** B2, B3, B4, B5, B6, A5. **Do-not-touch reference module:** `healthcheck.py` — doctor consumes it, never rewrites it.

## 6. Notes

- **#24 vs #1 method overlap:** the EPI-1 full-corpus archaeology protocol supersedes #1's ~15-transcript sampling as the evidence method for the Branch A/B trigger; #1 was left untouched this arc — reconcile the pair at the consolidation session (#31).
- **Residual stale mention, deliberately left:** `docs/audits/2026-07-06-code-quality-audit.md:5` references the intake by bare filename (no path claim); audits are immutable — documented here instead of edited.
- **No ADR status flips this arc:** ADR-09/10 remain `Proposed`; the flip is #31's (consolidation session), per DRAFT-GOV-1 (header + index row in the same commit).
- **Label collision (carried from the intake §5):** "CLI-4" names the parity-run *process* (L-CLI §3 Q2, the gate on DRAFT-CLI-3); DRAFT-CLI-**4** is the separate v2-resolver annotation. This document's P5 row means the former.
