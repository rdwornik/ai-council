# ADR-13: Invocation-contract versioning

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — `Contract-Version: 1.0` stamped in `protocols/COUNCIL_INVOCATION_CONTRACT.md` and echoed by the verdict package (`output.py` `contract_version: "1.0"`), witnessed live; commit `5dd4782`. Open remainder: #34 (research-path verdict parity) is the earmarked first `1.1` additive bump. _(Additive inventory stamp; decision content below authored fresh this session.)_

**Date:** 2026-07-18
**Status:** Accepted (2026-07-18)
**Decision:** The delegated-invocation CONTRACT carries a `Contract-Version: MAJOR.MINOR` line; additive changes bump MINOR by doc revision, breaking changes bump MAJOR and require an ADR.

## Context

`docs/decisions/ADR-11-delegated-invocation-contract.md` §5 committed the CONTRACT to a
compatibility promise but left the mechanism informal — the F8 fork (fable audit
`docs/audits/2026-07-04-fable-architecture-audit.md`) flagged that the first breaking change would
otherwise become a precedent-setting improvisation. The lane-int functional design
(`docs/intake/2026-07-06-lane-int-functional-design.md` §4) drafted the resolution as **DRAFT-INT-2**;
the night-batch candidate-ADR review (`docs/audits/2026-07-17-night-batch-candidate-adr-from-verdict-uc1.md`)
carried it forward with `contract_version` deliberately `null` until the D2 Known-deviations emptied.
This ADR ratifies DRAFT-INT-2 into the ADR-13 slot per the operator's 2026-07-18 consolidation ruling.

> **Reservation reconciliation:** earlier notes (fable audit; fleet-recon) informally floated "ADR-13"
> for a bounded crux-check idea. That idea is **not** ADR-13 — it lives on as BACKLOG **#18** (tool-grounded
> crux resolution, baseline-gated). The ADR-13 number is assigned here to invocation-contract versioning.

## Decision

- **`Contract-Version: MAJOR.MINOR`** line in `protocols/COUNCIL_INVOCATION_CONTRACT.md`; the verdict
  package echoes it (`contract_version`).
- **MAJOR (breaking)** — any change that invalidates a conforming caller (removed/renamed field, changed
  exit semantics, tightened obligation). A MAJOR bump **requires an ADR**, making ADR-11 §5's compatibility
  commitment mechanical rather than aspirational.
- **MINOR (additive)** — new optional fields / new destinations / relaxed obligations. A MINOR bump is a
  documented CONTRACT revision; no ADR required.
- **`1.0` stamping moment** — `1.0` stamps when §7 Known-deviations empties (the D2 parity fixes land: `--file`
  frontmatter parse #22, research `return_dir` #23). A `1.0` shipped with known deviations would make the
  first version a lie, so until then the CONTRACT stayed version-line-less (the deviations section *was* the
  version statement). Stamped at commit `5dd4782` on 2026-07-18, witnessed by a live `council` run echoing
  `"contract_version": "1.0"`.

## Consequences

- F8 (contract versioning) resolves at the shape level: the fork's two options become tiers (doc-revision for
  additive, ADR for breaking), not alternatives.
- The first breaking flag change has a rail to run on instead of becoming precedent-setting improvisation.
- **#34** (research-path verdict-package parity) is the earmarked first **1.1** additive bump — a research Lane A
  commission currently emits no verdict package, so closing #34 adds a destination without breaking any caller.
