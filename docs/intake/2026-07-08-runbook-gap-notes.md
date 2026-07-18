# Wave-1 onboarding — runbook gap-notes (ai-council, n=1)

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — local fixes landed (G8 pin, G3 handoffs, G4 codemap `d5c4e25`); hub gaps G1/G2/G6/G7/G9 forwarded (hub-scoped, untracked locally by design). _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-08 · **Arc:** Wave-1 onboarding pilot (hub #131 pilot / #215 conformance) ·
**Produced by:** ai-council's dedicated chat (ADR-41) · **Routing:** text artifact for the operator
to file into `.dev-knowledge` — **no hub write was made from this consumer chat.**

This is the #131 n=1 evidence: every gap hit while EXECUTING `docs/runbooks/repo-onboarding.md`
(post-A-S3 version) is a deliverable, not a failure. Gaps already covered by the A-S3 amendment
(hub-runnable/consumer-only markings; `--fire` requires `--run-date`) are **excluded** — none of
the below are A-S3-covered.

Attestation was pinned to hub `main` @ **`9e6ceb6`** (clean; session #164 had returned to main).
Per the rider-4 ratification, hub-sourced tooling was byte-identity-checked against `main` before
running (the codemap generator earlier; the attestation organs ran with the hub actually on main).

---

## Gaps discovered

### G1 — `validate_backlog` has no deploy-carrier (NAMED)
ADR-99 clause A and the runbook's per-repo checklist both say "carry the ADR-66 story-map +
`validate_backlog` as part of that deploy," but the v1.2.0 manifest ships **no `validate_backlog`
component** — nothing delivers the validator to a consumer (confirmed: hub #281 says "the deploy
manifest ships no such component"). ai-council therefore has no in-repo validator.
- **Impact:** `#281`'s "validation green in-repo" cannot be mechanically satisfied by a deployed
  artifact. This session verified green via the hub validator against a **scratch copy** (Q2), NOT
  an in-repo file (no hand-copied hub scripts into the consumer — that drift class is what the
  methodology kills).
- **Candidate hub enhancement:** fold a `validate_backlog` carrier into the **post-Wave-1
  mesh-portability epic** (alongside the enforcement-mesh port, #237-class). **Kill-candidate per
  backpressure:** if the story-map is only ever authored/checked hub-side, a per-consumer validator
  may be unnecessary — decide carrier-vs-hub-only before building.

### G2 — `validate_backlog.py` hardcodes its target path
`BACKLOG = Path(__file__).resolve().parent.parent / "BACKLOG.md"` — no CLI arg, so the validator
can only check the `BACKLOG.md` two dirs above the script; it cannot be pointed at a consumer by
argument. This is why G1's scratch-copy workaround was needed. Pairs with G1: a consumer-runnable
validator needs a `--path`/`--repo` argument.

### G3 — `docs/handoffs` census target conflicts with ADR-60/42 (NEEDS-RULING)
Census f.6 names ai-council's docs/ target as `docs/intake` **+ `docs/handoffs`**. But ADR-60/ADR-42
centralize handoffs in `.dev-knowledge`; child repos have **no local `docs/handoffs/`**
(ARCHITECTURE.md L254 states this). The census itself flagged this "intended divergence vs drift".
- **Resolution taken this session:** created `docs/intake/` only (operator GO); did **not** create
  `docs/handoffs/`. **Route the census `docs/handoffs` line back to the hub as a NEEDS-RULING** —
  either amend the census target to intake-only for child repos, or amend ADR-60/42.

### G4 — Codemap CLI incompatible with flat single-package layouts (UPGRADED)
This is bigger than the anticipated "CLI cannot reach unmarked blocks." The codemap generator is
**package/`tach.toml`-based**: with no `tach.toml` and a flat single-package layout
(`src/ai_council/*.py`, 14 flat modules), `codemap generate` degenerates to a **2-orphan-module
stub** (`providers`, `research`, no edges), discarding the real 16-node / 14-edge structure.
`--write` would have regressed the doc. `#262`'s generator-MANAGED intent is therefore **unreachable**
for ai-council without adopting Tach (explicitly rejected, ARCHITECTURE.md L130).
- **Handled:** both blocks (codemap L23 + layer-boundary L109) hand-converted to compact-text,
  marked HAND-AUTHORED, LF (commit `d5c4e25`). The separate "CLI cannot reach *unmarked* blocks"
  gap (L109 has no CODEMAP markers) also holds and will recur at **corp-monorepo L72/L275**.
- **Recurrence:** the flat-layout stub likely recurs at **demo-prep / life-architect in Wave-2**,
  not just corp-monorepo.
- **Candidate hub enhancement (NAMED):** codemap **flat-layout support** — derive module edges from
  the import graph (AST walker already exists) instead of requiring `tach.toml`. **Kill-candidate
  per backpressure:** if only tach-bearing repos are expected to adopt generator-management, close
  this and let flat-layout repos stay HAND-AUTHORED by policy (which ai-council now does).

### G5 — pre-commit hub-hook source: doc vs config drift
`ARCHITECTURE.md` L306 and `CONTRIBUTING.md` describe the hub-sourced pre-commit hooks as
`repo: ../.dev-knowledge`, but `.pre-commit-config.yaml` pins
`https://github.com/rdwornik/dev-knowledge` @ `rev: v1.2.0` (the URL pin landed in `71e1307`,
2026-07-06; the two prose docs were not updated). Minor doc reconcile. (Not fixed this arc — out of
the frozen scope; noted for a hygiene pass.)

### G6 — `audit.py repo <name> --repo-path` prints a "Report:" path but does not persist it
`python scripts/audit.py repo ai-council --repo-path <consumer>` exits 0 and prints
`Report: ...docs/audits/2026-07-08-ai-council-audit.md`, but the file is **not created** and the hub
tree stays clean (no `state.yaml`/history/report write). The `--help` claims "Same state.yaml /
history / report writes as `run`." So the hub-side per-repo `floor_integrity` verdict could not be
captured to disk this run. (floor_integrity was instead attested consumer-side:
`check_floor_hash.py --require-present` exit 0 + no orphaned root floor + `floor_conformance` 9/9.)
Possibly the `--repo-path` override path suppresses the report write, or an exception is swallowed.

### G7 — `observe-arc` n-of-6 needs a billed authenticated child; not runnable read-only/cheaply
`lived_sandbox.cli observe-arc` requires `ANTHROPIC_API_KEY` in the env (the isolated child
authenticates) and, once given it via `DEV_SECRETS_ENV`, spawns a **long-running billed child arc**
(timed out at **4m40s**, killed). The `#215` "coverage n-of-6" leg therefore **cannot be produced
read-only/cheaply from a headless harness**. The mesh FIRING was still proven (see below) — but the
specific n-of-6 number is env-constrained.
- **Candidate hub enhancement:** a lightweight/dry `observe-arc` coverage mode that reports per-organ
  FIRED/ARMED/SILENT **without** a live billed child (static inspection of armed hooks + config).

### G8 — Layer-6 verify doesn't exercise the hook's actual runtime interpreter
Runbook layer-6 verify (`INSTALL_PYTHON resolves` + a bare-`python -c "import pre_commit"` shim
import check) does **not** exercise the SessionStart hook's actual runtime interpreter — **the
verify passed while the SessionStart hook later failed** (`.venv\Scripts\python.exe: No module named
pre_commit`, non-blocking, on `SessionStart:resume`). Root cause: both the layer-6 verify and the
hook (`python -m pre_commit install`, `settings.json`) use **bare `python`**, which is
context-dependent — Python312 (has `pre_commit`) in the original session, the repo `.venv` (lacked
it) under an active venv on resume. The shim's hardcoded `INSTALL_PYTHON` (the only deterministic
path) was exercised by neither the verify nor the hook.
- **Runbook fix (the deliverable):** the layer-6 verify must **invoke the hook's own command line
  verbatim** (`python -m pre_commit install`) under the same shell the hook runs in — not a proxy
  `import` check that can pass on a different interpreter than the hook uses.
- **Fleet note:** corp-monorepo shares the latent fragility — `pre_commit` is present in its `.venv`
  but **undeclared in pyproject**, so a fresh re-clone would not repopulate it (same failure would
  surface there once the venv is rebuilt).
- **Fix applied here:** `pre-commit>=4.5` added to `pyproject.toml [dev]` (matches the hub floor) —
  durable: a fresh clone's `pip install -e ".[dev]"` populates the `.venv`. The immediate `.venv`
  population was **reverted** after it surfaced **G9** (a deeper WMI CLI hang), so daily SessionStart
  stays a fast non-blocking fail, not a 30s hang. Runtime verify is **BLOCKED-by-WMI** — see G9.

### G9 — pre_commit CLI import hang: `platform._wmi_query()` at golang import (MACHINE-level, NOT ai-council)
Discovered while verifying the G8 fix: with `pre_commit` importable, `python -m pre_commit install`
(and even `pre_commit --version`) **hangs**. faulthandler pinpoints it — `pre_commit/languages/golang.py:42`
calls `platform.machine()` at import → Python 3.12 `platform.uname()/_win32_ver()/_wmi_query()`, which
**stalls on this machine's WMI service**. pre_commit imports all languages eagerly (`all_languages.py:11`
→ `golang`), so every `pre_commit` CLI invocation trips it.
- **SCOPE — machine-level, not ai-council:** the stall is in **Python312's `platform.py`** (the `.venv`
  shares that stdlib), so it hits **any repo on this machine** intermittently — nothing in ai-council's
  code causes or can fix it. **Intermittency evidence:** this same session landed **6 ai-council commits
  + multiple hub merges today** with the **commit-time** pre-commit/commit-msg hooks firing normally
  (and it worked 2026-07-07) — the WMI stall is transient/state-dependent, not a hard break.
- **REMEDIATION CANDIDATES (to verify, not asserted):** (a) **CPython 3.12.x point-release upgrade** —
  `platform.machine()`/`_wmi_query()` hangs are a known fixed class in later 3.12 patch releases;
  verify against the CPython changelog before acting. (b) **Windows WMI service repair** (`winmgmt` /
  WMI repository rebuild). Route to the hub as an **ENVIRONMENT gotcha (machine-scoped)** — a candidate
  for the hub gotcha registry, not an ai-council BACKLOG item.
- **RETRY PATH (step-3 runtime verify = BLOCKED-by-WMI):** the G8 fix is import-verified only
  (`import pre_commit` succeeds in the `.venv`). To close the runtime verify once WMI clears, install
  pre-commit into the `.venv` (`pip install -e ".[dev]"`) then re-run **verbatim** from the consumer root:
  `python -m pre_commit install` → expect exit 0 (no ModuleNotFoundError, no WMI hang). One paste, not an
  archaeology dig.
- **ATTESTATION INTEGRITY (scopes G8 + the 2026-07-08 #215 attestation):** today's #215 attestation
  covered the **commit-time** armed mesh — demonstrably **FIRED** (6 commits landed today through the
  armed hooks; `enforcement_coverage --fire` + `floor_conformance` 9/9). The **SessionStart:resume**
  `pre_commit install` auto-arm leg is the degraded one (WMI). The attestation **stands with that stated
  scope**: commit-time enforcement proven; the SessionStart auto-arm convenience is WMI-blocked.

---

## Honest-scope closures recorded (not silent passes)

- **#281:** story-map + `[S<n>]` convergence DONE (commit `2511329`); zero structural `Track` strings;
  hub validator `OK (6 themes, 9 stories, 18 tasks, 0 warnings)`. "Green **in-repo**" is attested via
  **hub-validator-against-copy, PENDING the G1 carrier** — not a silent pass. ai-council is the first
  consumer to adopt `[S<n>]`.
- **#262:** compact-text ACHIEVED, **HAND-AUTHORED**; generator-MANAGED **NOT met** for this repo —
  blocked by design (Tach rejected, ARCHITECTURE.md L130). Tool-based n>=1 evidence for hub #262 must
  come from a **tach-bearing repo (corp-monorepo, B-S2)**, not ai-council.

---

## #215 attestation result (fired, not asserted from presence)

Pinned to hub `main` @ `9e6ceb6`. Organs run read-only (hub tree stayed clean; only gitignored logs
`ENFORCEMENT-COVERAGE.md` / `FLEET-HEALTH.md` written).

| organ | result |
|---|---|
| `check_floor_hash --require-present` (consumer) | **PASS** (exit 0) |
| `audit.py health` (hub self-health) | **OK** (hooks_armed OK, reconciled_versions OK) |
| `enforcement_coverage --fire --run-date 2026-07-08` | **FIRED** (exit 0): `session_end_backpressure` + `canonical_freshness` **enforcing-local**; floor/tier1-plugin/global-config **present-and-wired**; `doc_claims`/`git_backlog_drift` **hub-scoped** (reported, not faked) |
| `floor_conformance` | **CONFORMANCE PASS (9/9)** incl. poisoned+deleted floor caught, hook auto-arms, real task branch->commit->gate->merge --no-ff |
| `fleet_health` | exit 0, ai-council in the 5-repo roll-up |
| `observe-arc` (n-of-6) | **env-constrained** (G7) — not produced; mesh firing proven by the two rows above |
| `audit.py repo` (floor_integrity report) | report not persisted (G6); floor attested consumer-side instead |

`precommit present-not-wired` in `enforcement_coverage` is the **fresh-clone** measurement artifact
(hooks don't travel with a clone — the #275 per-clone gotcha); the **live** consumer has all three
stages armed (Phase-1 verify + this session's commits ran the hooks).
