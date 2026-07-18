# ADR-12: Provider Backend Engine — CLI-subscription seats and two-lane cost policy

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — `seat_router.py` + `cli_base.py` live (`0cab825`/`39b3941`); §5 default-flip evidence-gated; open: #27, #32, #43. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-05
**Status:** Accepted (ratified 2026-07-05 — operator ratification decision; drafted 2026-07-04 in the Fable architecture audit; fleet-recon markup of 2026-07-05 applied). The §5 default-flip clause remains evidence-gated.

## Context

All providers are API-SDK based (`PROVIDER_CLASSES`, cli.py; zero subprocess in src/ —
verified). The operator pays for Claude Code / Codex subscriptions the council never uses.
Intent: CLI-subscription lane for most debates; API reserved for important debates + research.

The CLI surface was live-probed on 2026-07-05 (fleet-recon report §2 — witnessed, replacing
the earlier "re-verify at build start" clause). Witnessed working invocations:

- **claude** (2.1.200): `claude -p "<q>" --output-format json --tools "" --model <id>` —
  **stdin closed**; answer in `.result`; **identity in-band via `.modelUsage`**; pin honored.
- **codex** (0.141.0): `codex exec --sandbox read-only --skip-git-repo-check [--json] -m <id>
  "<q>"` — **stdin closed or it hangs**; answer in JSONL `agent_message`; `--json` carries NO
  identity — **identity via the plain-mode stderr banner** (`model:`, `sandbox:` echoed);
  pin honored.
- **grok** (0.2.82): `grok -p "<q>" --output-format json -m <id>` — answer in `.text`;
  identity NOT in stdout — **authoritative `model_id` in
  `~/.grok/sessions/<cwd-enc>/<sessionId>/events.jsonl`** (sessionId from stdout JSON);
  pin honored.
- `gemini -p -o json` is **struck as a seat path**: the legacy gemini CLI is auth-dead on
  consumer tiers (witnessed `IneligibleTierError` → Antigravity migration notice).

## Decision

1. **Backend axis:** each `models:` block in settings.yaml gains `backend: api | cli`
   (default `api`). Config-only — no hardcoding (invariant 3 upheld).
2. **CliProvider adapters** behind the existing `AIProvider` ABC — separate classes,
   `asyncio.create_subprocess_exec`, structured stdout, `timeout_sec` reused as hard kill.
   **v1 adapter set = claude + codex only** (witnessed seat-grade). Coverage per fleet
   reality:
   - **deepseek remains API-only** — its CLI (Deep Code 0.1.33) has no headless mode
     (TTY-required; witnessed DC-1).
   - **grok is seat-capable** (pin honored, identity recoverable) **but joins the cost lane
     only after subscription OAuth** (`grok login`) plus env-key shielding — as configured it
     bills `XAI_API_KEY`, which defeats the lane's purpose.
   - **agy is EXCLUDED.** Re-admission evidence: a version that both honors `--model` and
     reports the served model identity (witnessed AG-2: pin silently rewritten to default,
     no identity channel).
   The no-merge rule (CLAUDE.md §5.7, #16) is upheld.
3. **Safety invariants (non-negotiable, all witnessed 2026-07-05):**
   - **cwd = a scratch dir, never a repo — this is the PRIMARY isolation.** Tools-off does
     not block cwd context ingestion: `claude -p --tools ""` ingests cwd `CLAUDE.md` (CL-3);
     grok ingests `CLAUDE.md`/`AGENTS.md` (GR-3).
   - **stdin must be closed** on every invocation (`codex exec` hangs on open stdin — CX-1).
   - **Identity-logged-or-no-seat:** every CLI-seat response must record the actual served
     model identity into metrics, via the per-CLI channels named in §Context. A seat whose
     identity cannot be read does not enter the panel.
   - **The synthesizer seat is never a CLI seat** (highest epistemic load → pinned API
     model identity required).
   - Additionally: read-only/tools-disabled flags on every invocation; personas + mode
     directives injected via the prompt; provider output enters the normal pipeline so
     ADR-03 anonymization (shuffle + relabel, our side) applies unchanged. Residual risks
     accepted and logged per-run: CLI system-prompt contamination, style leakage
     (pre-existing across API models), non-determinism.
4. **Routing policy — profiles, not heuristics:**
   - `standard` (default *after* the flip clause): CLI backend where available, API elsewhere.
   - `important` (CLI flag / frontmatter key): all-API.
   - research mode: always API (research providers are API-only products).
   - Fallback: CLI failure or quota error (extend classify_error's categories) retries
     the same seat via API — quota exhaustion degrades cost, never the debate.
   - **Quota gradient (probed 2026-07-05): codex > claude > grok (post-OAuth only) > agy
     (excluded).** codex idles on a ChatGPT subscription; claude competes with the
     operator's own CC dev work; grok has no cost advantage until OAuth; agy is out.
   - **Per-call pin rule:** the seat's settings.yaml model MUST be pinned explicitly on
     every CLI call — codex otherwise runs its own config default (`gpt-5.5`), not the
     seat pin.
   The cost gate stays in operator judgment (hub doctrine unchanged); v1 does no dynamic
   quota accounting — profile choice is the quota control.
5. **Default-flip clause (evidence-gated — unchanged):** `standard` defaults to the CLI lane
   only after a parity run — N paired debates (same briefs, CLI-backed vs API-backed), scored
   with protocols/SYNTHESIS_QUALITY_RUBRIC.md, no material quality regression. Until
   then `backend: api` remains the default everywhere. This is the parity-evidence gate,
   NOT the Epic B baseline gate — it reuses the rubric, not the synthesizer decision.

## Considered and rejected

- Judge-centric cost shape (cheap CLI panel + one expensive API judge) as the *default* —
  concentrates epistemic authority (brief #6's own tension); remains expressible via
  config once the axis exists, so it needs no decision now.
- Dynamic per-call quota allocator — three-axis accounting complexity without evidence it
  is needed; profiles + fallback cover the failure mode.

## Reconciliation

- ADR-02 panel composition untouched (seats change transport, not membership).
- ADR-03 unaffected — anonymization is applied by the debate engine, not the provider.
- ADR-06 extended, not superseded: this is the cost-optimization lane it anticipated.
- Brief §6.2 fork resolved as: subscription-first is a *profile*, not a posture change;
  API remains the no-ceiling fallback.

## Consequences

- Rama-1 bonus unlocked: a CLI agent seat is a candidate crux RESOLVER (ADR-13 v2 —
  candidates ranked by witnessed sandbox posture: codex > claude > grok).
- New failure surface (subprocess, JSON/banner/session-file parsing, CLI version drift) —
  contained by the API fallback, by `important`/research staying API-only, and by the
  identity-or-no-seat invariant degrading loudly.
- BACKLOG #16 done-when is satisfied by the first CLI backend running a debate turn
  through the provider protocol.

## Related

- BACKLOG #16 [Rama 2]; ADR-02 (panel), ADR-03 (anonymization), ADR-06 (cost)
- Hub (immutable): AI_COUNCIL_PROCESS (cost gate = operator judgment — unchanged)
- Drafted in `docs/audits/2026-07-04-fable-architecture-audit.md` §3 (D6/D7)
- Fleet-reality markup applied verbatim from
  `docs/audits/2026-07-05-fleet-recon-liveness-and-process-design.md` §7 (witnessed matrix §2;
  probe IDs CL/CX/AG/GR/DC/GM-*)
