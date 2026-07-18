# Hub feedback — session-close gate must mechanically block handoff regen; consumer hub-write guard (NEEDS-RULING)

> **Deployment-Status (2026-07-18 inventory):** HUB-OWNED — NEEDS-RULING filed to hub (`3d39db9`); no local mechanism. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-17 · **From:** ai-council (consumer) · **To:** hub (`.dev-knowledge`), governance/lifecycle lane
· **Channel:** consumer `docs/intake/` NEEDS-RULING note; operator carries to the hub (precedent:
`docs/intake/2026-07-17-hub-feedback-codex-producer-lane.md`, `docs/intake/2026-07-08-runbook-gap-notes.md`).
Do **not** edit hub/global infra from a consumer (core-invariant #6). Filing only.

## Ask 1 — the Stop-gate must mechanically gate handoff-bundle generation

Today the ADR-85 Stop-gate (`session_end_backpressure.py`) hard-blocks on a JOURNAL SHA-anchor, and the
`tier1-lifecycle` Stop hook proposes closures. Neither **mechanically blocks handoff-bundle
generation/regeneration** on the criteria that make a bundle trustworthy. The handoff bundle is what the
NEXT session boots from; a bundle regenerated before session-close criteria hold ships a stale or
unverified boot state by design.

**Requested mechanism:** the Stop-gate (or a dedicated pre-handoff gate) must refuse to
generate/regenerate a handoff bundle until ALL hold:

- **(a)** an **audits-class session-audit artifact exists** for the session (a dated `docs/audits/…md`
  covering the session's work) — not just a JOURNAL entry;
- **(b)** the **doc-currency legs are green** for the session's merges — `BACKLOG.md` structural gate
  passes AND `ARCHITECTURE.md` / `CLAUDE.md` are current vs what merged this session (the
  `canonical_freshness` A2 check already exists for the latter; wire it into the handoff precondition);
- **(c)** an **explicit operator "go" is recorded** (a HEAD-bound token, like the existing `/override`
  mechanism, but asserting close-readiness rather than bypassing the gate).

Rationale, witnessed tonight: this night batch deliberately does NOT regenerate the handoff bundle — the
mission scoped it to "regenerated tomorrow, after the operator's morning decisions … tonight's runs would
make a fresh bundle stale by design." That discipline is correct but currently lives in **prose/operator
memory**, not in a mechanism. A future session without that prose would regenerate a stale bundle and
nothing would stop it. The hub owes the mechanism.

## Ask 2 — a consumer-session guard against hub writes

**Boundary class witnessed:** a browser-side architect can instruct a consumer Claude Code session to
write directly into the hub (`.dev-knowledge`) or other global infra. That is the **same boundary
violation class** as the Codex producer-lane case (`…-hub-feedback-codex-producer-lane.md`): a
consumer must not edit hub/global infra unilaterally (core-invariant #6; global-infra edits are
exception-with-ruling, never from an instruction that originates outside a hub ruling). Today the only
defense is the consumer session *recognizing* the boundary and refusing — prose/judgment, not a guard.

**Requested mechanism:** a consumer-side guard analogous to the `block-onedrive` PreToolUse hook —
a pre-write check that **blocks Write/Edit/NotebookEdit whose `file_path` resolves under a hub/global
path** (`.dev-knowledge/`, `~/.claude/`) from a consumer session, allowing it ONLY when an explicit hub
ruling token authorizes that specific edit. Consumer requirement: the guard must be hub-owned/fleet-level
(the OneDrive-guard precedent, `.dev-knowledge` BACKLOG #289), not per-machine ad-hoc, and must not
require a consumer to hand-maintain it.

## Note on the boundary in THIS filing
This note is authored **locally** in the consumer's `docs/intake/` and proposes hub rulings; it does not
itself write the hub. The operator carries it across the boundary. That is the sanctioned consumer→hub
feedback path — exactly the pattern Ask 2 asks the hub to make mechanical rather than prose-enforced.
