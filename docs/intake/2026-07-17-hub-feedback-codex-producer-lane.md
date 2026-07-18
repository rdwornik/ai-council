# Hub feedback — Codex producer-lane vs the global read-only policy (NEEDS-RULING)

> **Deployment-Status (2026-07-18 inventory):** HUB-OWNED — NEEDS-RULING filed to hub (`321fa76`); interim operator ruling in force; no local mechanism. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-17 · **From:** ai-council (consumer) · **To:** hub (`.dev-knowledge`), Codex-lane (EPIC-H)
· **Channel:** consumer `docs/intake/` NEEDS-RULING note; operator carries to the hub (precedent:
`docs/intake/2026-07-08-runbook-gap-notes.md`). Do **not** edit hub/global infra from a consumer
(core-invariant #6).

## The conflict (witnessed 2026-07-17, BACKLOG #30)

A day-session plan tasked **Codex as the PRODUCER** for a bounded code change (#30 DOC-3), CC as
verifier. Two independent machine-level facts block Codex-as-producer here:

1. **Global read-only policy (primary).** `~/.codex/AGENTS.md` is a *"Global Codex Reviewer
   Configuration"*: *"Codex is a read-only code reviewer across all repos. It does not build, fix, or
   modify."* `codex exec` refuses to write even with `-s danger-full-access`, citing this policy.
   `--ignore-rules` does not bypass it (it only skips `.rules` execpolicy files).
2. **Windows write-sandbox (secondary).** `codex exec -s workspace-write` cannot apply patches on this
   Windows box ("restricted-token sandbox cannot enforce split writable root sets; refusing to run
   unsandboxed").

So the **Codex-PRODUCER doctrine** (as it appears in build-lane planning) is currently unrealizable on
this machine without editing hub-owned global infra — which a consumer must not do.

## Interim fallback (operator ruling 2026-07-17, in force now)

Until the hub reconciles this: **bounded build tasks run as CC-implements-Codex's-design + terra
read-only review** (`codex exec review`) pre-merge. Codex/terra stays in its configured reviewer role.
This shipped #30 cleanly (terra review: no Critical/High). Recorded as a machine-level gotcha in
`~/.claude/skills/gotchas/gotchas.md` ("Codex is a READ-ONLY reviewer …") so no future session re-asks.

## The ruling the hub owes

Reconcile the **Codex-producer doctrine** with the **global read-only `AGENTS.md` policy** (EPIC-H):
either (a) formally adopt "Codex is review-only; CC/other agents produce" as doctrine (retire
Codex-as-producer from build-lane plans), or (b) define a sanctioned producer-lane exception (a
scoped profile / per-invocation policy) that a consumer may invoke without editing global infra — and,
if (b), fix the Windows workspace-write sandbox too. Consumer requirement: the mechanism must not
require a consumer to edit `~/.codex/AGENTS.md` or any hub/global file.
