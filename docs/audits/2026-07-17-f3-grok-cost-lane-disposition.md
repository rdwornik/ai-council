# F3 grok cost lane — re-probe disposition (#28)

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — #28 closed; grok OAuth subscription lane witnessed (`~/.grok/auth.json`); open: #27 (grok seat deferred to parity). _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-17 · **Class:** audits (immutable) · **Task:** BACKLOG #28 (F3 grok cost lane)
**Refs:** fleet-recon `docs/audits/2026-07-05-fleet-recon-liveness-and-process-design.md` §8 F3;
ADR-12 §Decision 2/§4 (cost lanes, quota gradient); L-CLI functional design
`docs/intake/2026-07-06-lane-cli-functional-design.md` §5 F3.

> Immutable disposition record. Filing only — no seat/build work performed this session
> (grok seat remains deferred; ADR-12 v1 adapter set stays claude+codex).

## Why this record exists

#28 asks the operator to run `grok login` (subscription OAuth) + env-key shielding, then a
repo re-probe to witness whether grok enters the subscription **cost lane** or stays API-billed.
The operator verified the machine's CLI fleet live this session; this note records the repo
re-probe and the resulting disposition.

## Re-probe evidence (witnessed 2026-07-17, this checkout)

| Check | Recon baseline (2026-07-05) | Re-probe 2026-07-17 | Verdict |
|---|---|---|---|
| grok CLI version | 0.2.82 | **0.2.102** | drifted (F5 class) |
| `~/.grok/auth.json` (subscription OAuth marker) | absent (Test-Path) | **absent** | no subscription lane |
| `grok models` billing header | "You are using XAI_API_KEY" | **"You are using XAI_API_KEY."** | **API-billed** |
| `XAI_API_KEY` in env | set | **set** | API-billing source present |

**Load-bearing finding:** grok is still **API-billed**, not subscription. `grok login` was not
performed (auth.json absent), so the F3 subscription lane is **not configured** on this machine.
grok is "authenticated and working" via the env API key — not via a subscription OAuth lane.

### Model currency intel (parked — #17, not actioned here)
grok CLI DEFAULT model is stale **"Grok 4.2.0 non-reasoning"**. Available variants witnessed/
reported: **grok-4.5** (flagship reasoning/SWE), **grok-build-0.1** (terminal-agent driver),
**grok-code-fast-1** (cheap autocomplete). The configured debate/research pins (`grok-4.3` /
`grok-4.20-0309-reasoning`) trail grok-4.5 — currency evidence for #17, not this task.

## Disposition

- **F3 outcome: grok STAYS API — cost lane NOT entered.** An API-billed grok CLI seat is
  strictly dominated by the existing grok API seat (same billing + subprocess/parse surface +
  harness tax — L-CLI F3), so no grok CLI adapter is built. ADR-12 v1 adapter set remains
  **claude + codex only**.
- **Architect ruling recorded (this session):** when the grok debate seat is eventually built,
  it **pins `grok-4.5` explicitly** — never the stale CLI default (Grok 4.2.0). This binds the
  future seat per ADR-12's per-call pin rule; it is a forward ruling, applied at seat-build time,
  not now.
- **Seat status:** grok seat work remains deferred — gated on BOTH F3 subscription OAuth (still
  unconfigured) AND #27 CLI-4 parity evidence. No seat work this session.

## Adjacent fleet re-probe (parked evidence — not this task's dispositions)

| CLI | Witnessed 2026-07-17 | Operator note | Parked for |
|---|---|---|---|
| deepcode | **0.1.33** present | working (deepseek-v4-pro, thinking max) | #6 (DeepSeek eval). Recon: no headless mode (TTY-required, DC-1) — capability question, not billing |
| antigravity (`agy`) | **1.1.3** present (operator reported 1.0.16 — drift) | working (Gemini 3.5 Flash, Google AI Pro) | remains ADR-12-EXCLUDED until a probe clears the AG-2 identity-roulette finding; parked |

## Closure verdict — #28 does NOT close (gap)

#28 done-when: *"the re-probe witnesses subscription-billed grok identity and the disposition is
recorded (enters cost lane or stays API)."* The re-probe witnessed **API-billed** identity — the
opposite of the required subscription-billed identity — because `grok login` (the operator action
#28 names) was not performed and `~/.grok/auth.json` is absent. The subscription-identity clause
is therefore **unmet**, so #28 remains **OPEN** pending the actual `grok login` + subscription
re-probe. The disposition above is recorded; the cost-lane question is answered "stays API for
now." (If the operator elects to retire F3 as "stays API / OAuth deferred indefinitely" rather
than pursue `grok login`, that is an operator ruling that would close #28 — not taken here.)

---

## AMENDMENT 2026-07-17 (post `grok login`) — #28 CLOSED: subscription lane witnessed

> In-file amendment marker per the immutability rule (CLAUDE.md §5.3). Supersedes the
> "does NOT close (gap)" verdict above, which reflected the pre-`grok login` state.

The operator completed `grok login` (subscription OAuth, browser) after the initial re-probe.
The subscription re-probe was repeated with **scoped env-key shielding** (subshell-local
`unset XAI_API_KEY` — the central `~/Documents/.secrets/.env` store is **untouched**; other
tools still use the key; shielding applies to grok invocations only):

| Check | Result 2026-07-17 (post-login) |
|---|---|
| `~/.grok/auth.json` (subscription OAuth marker) | **PRESENT** (created 17:48) |
| `grok models` UNSHIELDED (`XAI_API_KEY` in env) | "You are using XAI_API_KEY." — env key still shadows |
| `grok models` SHIELDED (scoped `unset XAI_API_KEY`) | **"You are logged in with grok.com."** — subscription lane active |

**Subscription-billed grok identity WITNESSED** (shielded). Key finding: the env-key shielding
is a **per-invocation requirement** — unshielded, `XAI_API_KEY` shadows the subscription lane
(API-billed), so a future grok CLI seat must strip `XAI_API_KEY` per grok call (mirror the
claude key-strip pattern), never by editing the central store.

**Revised disposition:** the grok **cost lane is now AVAILABLE** — the F3 subscription blocker is
cleared. The grok seat itself remains **deferred** (gated on #27 CLI-4 parity; no seat work this
session) and, when built, **pins `grok-4.5`** (ruling above) and **scope-shields `XAI_API_KEY`**
per invocation.

**#28 done-when MET** — subscription-billed identity witnessed + disposition recorded (enters
cost lane) → **#28 CLOSED**.
