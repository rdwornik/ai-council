# ADR-11: Delegated Invocation Contract — two lanes, one machine-readable surface

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — CONTRACT v1.0 (`5dd4782`); `--file` #22 + research #23 parity; open: #34, #35. _(Additive inventory stamp; body below unchanged.)_
> **Deployment-Status refresh (2026-07-23 amendment, additive):** #35 was struck 2026-07-19 (three-lane reintegration; discharged with checker-leg evidence) — the open remainder is **#34 only** (research-path verdict parity, the earmarked Contract-Version 1.1 bump alongside #76). The 2026-07-22 boost amendment below (§Amendment) post-dates the inventory stamp above; its caller-facing invocation note is tracked as #88. _(Marker only; the 2026-07-18 stamp and the body are unchanged per CLAUDE §5.3.)_

**Date:** 2026-07-05
**Status:** Accepted (ratified 2026-07-05 — operator ratification decision; drafted 2026-07-04 in the Fable architecture audit)

## Context

ai-council is commissioned by external repos (VISION: "Called by other repos … via the
`council` CLI"). Faza 0 shipped the plumbing (protocols/, `--return-dir`, minority report),
but the machine-facing contract is implicit: a delegating agent must reverse-engineer
flags, exit codes, and artifact names from code. Two verified lane-parity defects make the
implicit contract inconsistent: (1) `--file` mode does not parse frontmatter (verified in
`cli.py` — raw `read_text`) — a guide-conformant brief behaves differently per lane, and its
frontmatter leaks into the question text; (2) research mode ignores `--return-dir`
(`run_research` has no such parameter).

## Decision

1. **Two invocation lanes, both first-class:**
   - **Lane A (delegated, synchronous):** `council --file <brief.md> --return-dir <dir>
     [--format json] [overrides]`, runnable from any cwd (config already resolves paths
     against repo root by design). The lane for agent callers.
   - **Lane B (inbox, batch):** unchanged operator-mediated fire-and-forget
     (`council --inbox`). The interaction model stays batch — no context-pull /
     interactive concepts (consolidation-brief §5 cuts upheld).
2. **One brief format across lanes:** `--file` parses YAML frontmatter through the same
   `parse_file()` path as the inbox, with identical precedence: CLI flag > frontmatter >
   config default. Frontmatter is stripped from the question text.
3. **Research parity:** `run_research()` gains `return_dir`, routed through the research
   output writer, same semantics as debate outputs (canonical `./output/` always; return-dir
   is an additional copy; hub never a default).
4. **The contract is documented as `protocols/COUNCIL_INVOCATION_CONTRACT.md`** covering:
   both lanes; full flag set; frontmatter keys + precedence; exit codes 0/1/2/3 (ADR-08);
   artifact naming (`council-out-*`, `council-minority-*`, `*_metrics.json`); `--format
   json` stdout payload (DebateResult schema); degradation semantics; RoutingError
   fail-loud; ownership boundary (hub owns WHEN/WHY — convene rules, loop, caps; this
   repo's protocols/ owns HOW — formats, mechanics, outputs).
5. **CLI-as-ABI reaffirmed:** no Python library API, no server/MCP. The CLI surface named
   in the contract doc becomes a compatibility surface — breaking changes require an ADR.

## Considered and rejected

- Python library API / MCP server — contradicts VISION ("not embedded as a library"),
  adds a stateful surface to a deliberately stateless tool, and duplicates what exit
  codes + JSON + known paths already provide.
- Single-lane (inbox-only) delegation — leaves the delegating agent unable to run
  synchronously and consume results programmatically; contradicts ADR-08's own rationale
  (automation harness reading exit codes).

## Reconciliation with hub ADRs (no edits, no supersession)

- **ADR-67:** implements its step-4/6 "known-path I/O" assignment to ai-council; the
  `~/.claude` `council.return_dir` reader remains ADR-10's reserved seam (legal future
  setter; implementing it needs no new ADR). No divergence.
- **ADR-95:** the contract is exactly the surface "CC mechanically expands" against. No divergence.
- **ADR-43:** target-project routing semantics untouched. Courtesy design-stage flag-back
  to the hub anyway (see flags), extending ADR-43's feedback edge to invocation semantics.
- **Hub flags (hub-side session, not this repo):** stale guide path in
  AI_COUNCIL_PROCESS.md §authoritative-sources; Stage-3 "Command:" wording now that this ADR
  is ratified.

## Amendment (2026-07-22)

**(a) What is added.** A council-side entry stage: `council boost` — raw question in
(file or arg), boosted brief out (file). Owner ruled **C — council-side entry stage** by
the operator (executing the EXTEND position of
`docs/audits/2026-07-21-night-input-layer-audit-fable.md` §5).

**(b) Why this is a CLARIFICATION of decision 5, not a reversal.** The object this ADR
rejected is **statefulness** — a Python library API or an MCP/server surface (see
`## Considered and rejected`). `council boost` is file-in / file-out, stateless, one
invocation, exit-code-carrying: it is a **CLI-surface EXTENSION** under decision 5's
CLI-as-ABI ("no Python library API, no server/MCP. The CLI surface named in the contract
doc becomes a compatibility surface"), not a new surface class.

**(c) What is NOT admitted — the interactive rider, deferred.** A bounded clarify-loop
that asks the caller questions mid-flight (MCP elicitation) **would** reopen this ADR:
it is named in `## Considered and rejected`, and it additionally collides with
decision 1's explicit no-interactive-concepts cut (Lane B: "no context-pull /
interactive concepts"). It is deferred as a **separable rider requiring its own ADR**.
Information gaps are therefore handled by **advisory annotation in the emitted brief**,
never by asking the caller.

**(d) Contract impact.** Additive subcommand → backward compatible, not a breaking
change; recorded here as a CLI-surface extension per decision 5's "breaking changes
require an ADR" discipline.

**(e) The ADR-95 boundary rule (folded here deliberately — no net-new document).**
Because substance-shaping now moves council-side, the boundary is stated as a rule:

> The boost **may** restructure, interrogate, classify, decompose (into at most three
> linked sub-briefs), and **flag** gaps. The boost **may never assert a fact the caller
> did not supply.** A gap is annotated, never filled.

Hybrid asks are handled by **decomposition into linked sub-briefs** (a research
sub-commission may feed a decision sub-brief). `research` mode is reached by the boost
**emitting a research sub-commission**, not by any change to `detect_mode()` — which
structurally cannot emit that mode.

## Consequences

- External repos can commission the Council today, without waiting for Epic B or #9;
  #9's template/gate, when built, targets this contract instead of an implicit one.
- Baseline-INDEPENDENT (same separability argument as ADR-10/#13).
- Cost: one protocol doc + two small parity fixes + tests; the CLI surface becomes a
  versioned commitment.

## Related

- ADR-09 (protocols/ surface), ADR-10 (return-dir), ADR-08 (exit codes)
- Hub (immutable): ADR-67 (gated loop), ADR-95 (lane split), ADR-43 (routing)
- Drafted in `docs/audits/2026-07-04-fable-architecture-audit.md` §3 (D1/D2/D8)
- Ratified after live reconciliation — 2026-07-05 fleet-recon report §5.2 (D1–D3 HOLD):
  `docs/audits/2026-07-05-fleet-recon-liveness-and-process-design.md`
