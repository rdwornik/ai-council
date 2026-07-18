# ai-council — Methodology-Adoption & Hygiene Audit

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — §4 fixes direct (`8361660`; INSTALL resync `288f256`, #315); residual: egg-info dup (harmless) + `.env` verify untracked nits. _(Additive inventory stamp; body below unchanged.)_

> **Date:** 2026-07-02
> **Auditor:** Claude Code (Opus), read-live audit — no files changed.
> **Audience:** the incoming ai-council architect (post-handoff).
> **Purpose:** an honest map of (a) how much of the `.dev-knowledge` methodology is actually in place, and (b) what's internally messy. Not pass/fail — a prioritization aid. Nothing was deleted or restructured; every finding is the architect's + operator's call.

---

## TL;DR

- **Methodology floor: armed and intact.** All six parts present and the guard exits 0. (The raw `sha256sum` mismatch you might see is a CRLF artifact only — the guard normalizes line endings and the normalized hash matches the sidecar exactly.)
- **`tier1-lifecycle` plugin: installed, enabled, and used — the deploy assess (`present_correct`) was right.** The reason the operator "sees no plugin folder" is that project-scope plugins live in the **user cache** (`~/.claude/plugins/cache/…`), not in the repo tree. Contradiction resolved.
- **CLAUDE.md conforms** to the ADR-53 canonical agent-contract shape. Local ADRs, story-map BACKLOG, and docs taxonomy are all present. ai-council correctly relies on the hub for PLAYBOOK/ESSENTIALS rather than duplicating them.
- **The real mess is `.claude/settings.local.json`** — it's full of **stale paths** (`Documents/Scripts/ai-council`, `venv/`, `requirements.txt`, `src.cli`) that no longer match this repo. This is the single highest-value cleanup.
- **Minor cruft:** duplicate `*.egg-info` dirs, an `INSTALL.md` that arguably belongs to the hub, and a spurious plugin-install record. All low-stakes.

---

## §1 — Root-file inventory + naming

**Git-tracked root files (12) — all legitimate, none dead:**

| File | Role | Verdict |
|---|---|---|
| `.ai-council.code-workspace` | VS Code workspace (dot-prefixed per ADR-59) | good |
| `.gitignore` | ignore rules w/ floor negations | good |
| `.pre-commit-config.yaml` | pre-commit gate | good |
| `ARCHITECTURE.md` | structural model (ADR-51) | good |
| `BACKLOG.md` | ADR-66 story-map | good |
| `CLAUDE.md` | agent contract (ADR-53) | good |
| `CONTRIBUTING.md` | contribution guide | good |
| `INSTALL.md` | tier1-lifecycle plugin install doc | **see note** |
| `JOURNAL.md` | dated log (newest 2026-06-03) | good |
| `LESSONS.md` | append-only lessons (ADR-29) | good |
| `pyproject.toml` | single source of truth | good |
| `VISION.md` | vision/values | good |

**The `install`-type file the operator flagged = `INSTALL.md`.** It is **not dead** — it documents how to install the `tier1-lifecycle` plugin and the ruff gate into a host repo. But it reads as a **hub concern that leaked into a consumer repo**: ai-council is Layer-3 (the consumer), and generic "how to install the plugin" instructions arguably belong in `.dev-knowledge`, not duplicated in every repo that adopts it. **Surface for the architect to decide** — keep (repo-local convenience) vs. relocate to hub. Do not delete without that call.

**Untracked root noise (gitignored, not committed — informational):**
- `ai_council.egg-info/` **and** `src/ai_council.egg-info/` — **two** editable-install metadata dirs. Both gitignored, but the duplication is a smell (see §4).
- `.env` (100 bytes, gitignored) — confirm it holds no API keys (repo rule: keys live in global `Documents/.secrets/.env`). Could not/should not inspect contents in this audit; **architect should verify**.
- `.venv/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `logs/`, `output/`, `council_inbox/` — all expected, all gitignored.

**Naming:** consistent. ADRs are `ADR-NN-topic.md` (hyphen, per ADR-34); audits are `YYYY-MM-DD-*.md`; markdown is kebab-case. No inconsistencies found at root.

**§1 verdict: GOOD.** One item to adjudicate (`INSTALL.md` ownership); no dead files.

---

## §2 — Methodology adoption state

### Floor — **ARMED & INTACT** ✅

All six required parts present:

1. **`.claude/CLAUDE-FLOOR.md`** — present (3177 bytes, the methodology baseline).
2. **`.claude/CLAUDE-FLOOR.md.sha256`** sidecar — present (`4d268f32…`).
3. **`.claude/check_floor_hash.py`** guard — present; ADR-78/ADR-93 carrier; two legs (permissive pre-commit + `--require-present` session-start).
4. **`@`-include** — `CLAUDE.md` line 1 is `@.claude/CLAUDE-FLOOR.md`, so the floor auto-loads with the session contract.
5. **`settings.json` SessionStart hook** — runs `python .claude/check_floor_hash.py --require-present` (timeout 10) on every session start; a deleted-but-tracked floor fails loud.
6. **`.gitignore` negations** — `.claude/*` is ignored, then `!CLAUDE-FLOOR.md`, `!CLAUDE-FLOOR.md.sha256`, `!check_floor_hash.py` re-include the three floor artifacts so they're tracked despite the blanket `.claude/` ignore.

**Verification run live:** `python .claude/check_floor_hash.py --require-present` → **exit 0**. Normalized SHA-256 of the floor = `4d268f32…` = sidecar exactly. **The floor is genuinely intact** — any `sha256sum` mismatch is a CRLF-vs-LF artifact the guard deliberately normalizes away.

### Plugin (`tier1-lifecycle`) — **INSTALLED, ENABLED, USED** ✅ (contradiction resolved)

The operator's "I don't see a plugin folder" and the deploy assess's `present_correct` are **both correct** — they're describing different locations:

- **Enabled in the repo:** `.claude/settings.json` → `"enabledPlugins": { "tier1-lifecycle@dev-knowledge-methodology": true }`, with the `.dev-knowledge` directory registered as a marketplace under `extraKnownMarketplaces`. This config is **force-added** (repo gitignores `.claude/`) so the install is tracked/reproducible.
- **Code lives in the user cache, not the repo:** `~/.claude/plugins/cache/dev-knowledge-methodology/tier1-lifecycle/0.1.10/`. Project-scope plugins never materialize a folder inside the repo tree — that's why there's nothing to see under `ai-council/`.
- **Install record:** `~/.claude/plugins/installed_plugins.json` shows `tier1-lifecycle@dev-knowledge-methodology` v0.1.10, scope `project`, `projectPath = …/Dev/ai-council`, installed 2026-06-30. ✔

**Is it actually used?** Yes — it wires three things into this repo:
- **Stop hook** → `propose_closures` (writes `logs/PROPOSALS-*.md`; `logs/` is gitignored).
- **SessionStart hook** → surfaces pending closure proposals.
- **`/review-closures`** skill → human-gated backlog close (ADR-70 Tier-1).
The ruff lint gate ships **separately** (a plugin can't carry a `.pre-commit-config.yaml`) — merged from `assets/ruff-pre-commit.yaml` into `.pre-commit-config.yaml` (present, pinned `v0.15.5`).

> **One wrinkle (minor):** `installed_plugins.json` contains a **second, spurious record** for the same plugin keyed to `projectPath = …/Dev/.dev-knowledge` (installed 2026-06-29). That's a user-level cache artifact, not a repo problem, but worth a cleanup pass if the operator wants the plugin registry tidy. See §4.

### Skills

- **Repo-level: NONE.** `.claude/` holds only `rules/` (`code-standards.md`, `python-env.md`, `testing.md`), `settings*.json`, and the floor artifacts. No `.claude/skills/` and no `.claude/commands/` — consistent with what CLAUDE.md §7–§8 already claim.
- **User-level (in use):** `gotchas`, `verify`, plus the plugin-provided `tier1-lifecycle:review-closures` / `:ship`. These are the skills that actually apply here.

### Hooks

- **Local pre-commit framework** (`.pre-commit-config.yaml`) — four gates:
  - `normalize-headers` (local) — normalizes `LESSONS`/`JOURNAL` headers.
  - `floor-hash-verify` (local) — the floor guard at commit time.
  - hub `toc-freshness` + `toc-generate` (`repo: ../.dev-knowledge`, **rev-pinned `v1.0.0`**) — scoped to `docs/council-question-guide.md`.
  - `ruff` (`astral-sh/ruff-pre-commit`, **rev-pinned `v0.15.5`**) — Tier-1 lint gate, blocks on violations.
- **`settings.json` SessionStart hooks** — floor guard (`--require-present`) + `python -m pre_commit install` (auto-arms the git hook each session).
- **Plugin hooks** — Stop (`propose_closures`) + SessionStart (surface), as above.
- **No custom `~/.claude`-global or `.git/hooks/` scripts** beyond the pre-commit-installed shim. Clean.

### CLAUDE.md — **CONFORMS** ✅

Matches the ADR-53 canonical shape: `@`-include of the floor on line 1, then 12 numbered sections (first-read order, repo identity, architecture, conventions, critical rules, session-start protocol, slash commands, skills, hooks, anti-patterns, binding ADRs, section history), `last_updated` + maintainer footer. Substantive, ≤200-line target, single canonical agent-instruction file. No drift from the methodology's agent-contract template.

### Methodology docs — **local where it should be local, hub-reliant where it should be**

| Artifact | State |
|---|---|
| Local ADRs | ✅ `docs/decisions/ADR-01…08` + `README.md` (hyphen-named, ADR-34) |
| BACKLOG | ✅ `BACKLOG.md` migrated to ADR-66 story-map (themes + `Done when:` form) |
| VISION / ARCHITECTURE / CONTRIBUTING | ✅ present, canonical spine (ADR-38 A6) |
| LESSONS / JOURNAL | ✅ present; LESSONS append-only (ADR-29); JOURNAL dated, newest 2026-06-03 |
| docs taxonomy | ✅ `docs/{decisions,audits,archive}` each README-seeded (ADR-60) |
| PLAYBOOK / ESSENTIALS | ⛔ **none local — by design.** Relies entirely on the hub (`../.dev-knowledge/protocols/`), which is present as a sibling. Correct: universal protocols are not duplicated per repo. |
| Session handoffs | ⛔ **none local — by design.** Per ADR-42, handoffs are centralized in `.dev-knowledge`. (Note: the many `*handoff*` files under `council_inbox/archive/` and `output/` are **Council I/O artifacts**, not session handoffs — don't conflate.) |

**§2 verdict: GOOD.** Floor armed, plugin resolved, CLAUDE.md conforms, docs correctly split local-vs-hub. Only follow-ups are hygiene (§4), not adoption gaps.

---

## §3 — Conformance to the `.dev-knowledge` way-of-working

| Discipline | Present? | Evidence |
|---|---|---|
| **Methodology floor** (re-anchor before structural change) | ✅ | armed + intact (§2) |
| **Lint/format gate** | ✅ | ruff pinned `v0.15.5` in pre-commit + `scripts/check.ps1` pre-merge trio (pytest + mypy + ruff) |
| **Floor integrity gate** | ✅ | `floor-hash-verify` pre-commit + session-start `--require-present` |
| **Backlog discipline** (ADR-66 story-map) | ✅ | themes/backbone + `[#id] … Done when:` items |
| **Tier-1 closure loop** (ADR-70) | ✅ | plugin Stop→propose, SessionStart→surface, `/review-closures` |
| **Docs taxonomy** (ADR-60) | ✅ | `decisions/` + `audits/` + `archive/`, each README-seeded |
| **ADR immutability + append-only LESSONS** | ✅ | conventions in CLAUDE.md §5; 8 immutable local ADRs |
| **Conventional Commits + branch→merge --no-ff** | ✅ | CLAUDE-FLOOR ship rule; recent history shows `--no-ff` merges |
| **PLAYBOOK/ESSENTIALS reliance** | ✅ (hub) | no local copies; sibling hub present |
| **Session-handoff infra** | ✅ (hub, ADR-42) | centralized in `.dev-knowledge`, not local |

**§3 verdict: GOOD.** No missing methodology discipline detected. ai-council is a well-behaved Layer-3 consumer: it carries what a consumer must carry (floor, plugin config, local ADRs/backlog, docs taxonomy) and defers to the hub for what the hub owns (protocols, handoffs).

---

## §4 — Internal hygiene (independent of methodology)

Ranked by cleanup value.

1. **`.claude/settings.local.json` is stale — highest-value fix.** Its permission allowlist is pinned to a **path that no longer exists**: `C:\Users\1028120\Documents\Scripts\ai-council` (repo is now `…/Documents/Dev/ai-council` — verified the `Scripts/` path is gone). It also references **`venv/`** (repo uses `.venv/`), **`requirements.txt`** (repo uses `pyproject.toml` editable install), and **`python -m src.cli`** (namespace is `src.ai_council`). Every entry after the first three is dead/wrong. This file is gitignored (local-only), so it's low-risk to regenerate, but it's actively misleading. **Recommend: rewrite the allowlist against current paths.**

2. **Duplicate `*.egg-info`.** Both `ai_council.egg-info/` (root) and `src/ai_council.egg-info/` exist. Editable installs normally produce one; two suggests an install was run from two layouts over time. Both gitignored → harmless, but a `pip install -e .` refresh (or deleting the stray root one) removes the noise.

3. **Spurious plugin-install record.** `~/.claude/plugins/installed_plugins.json` lists the plugin twice — once correctly under `…/Dev/ai-council`, once under `…/Dev/.dev-knowledge`. User-level, not repo, but tidy-able with a reinstall/prune if the operator cares.

4. **`INSTALL.md` ownership** (also §1). Hub-generic content sitting in a consumer repo. Relocate-to-hub vs. keep-local is an architect call.

5. **`.env` at root.** Gitignored, 100 bytes. Repo policy is "no keys in repo-local `.env`." **Verify contents** hold no secrets (likely a placeholder/pointer, given the size).

**§4 verdict: NEEDS-WORK (low-stakes).** Nothing broken; several stale/duplicate artifacts. The `settings.local.json` staleness is the one worth doing first.

---

## §5 — Verdict per area

| Area | Verdict | The specific gap / action |
|---|---|---|
| **§1 Root inventory & naming** | **GOOD** | Only decision pending: `INSTALL.md` — keep (repo-local) or relocate to hub. No dead files. |
| **§2 Methodology adoption** | **GOOD** | Floor armed+intact; plugin installed/enabled/used (lives in user cache, not repo — that's normal); CLAUDE.md conforms; docs correctly split local vs hub. No adoption gaps. |
| **§3 Conformance** | **GOOD** | All gates, backlog discipline, closure loop, docs taxonomy, and hub-reliance present. Nothing missing. |
| **§4 Internal hygiene** | **NEEDS-WORK** | (1) `settings.local.json` stale paths — **fix first**; (2) duplicate egg-info; (3) spurious plugin record; (4) `INSTALL.md` ownership; (5) confirm `.env` holds no keys. All low-stakes. |

### Suggested priority order for the incoming architect
1. **Rewrite `.claude/settings.local.json`** against current paths (`…/Dev/ai-council`, `.venv/`, `pyproject`, `src.ai_council`). — *messy + actively misleading, cheap fix.*
2. **Decide `INSTALL.md` ownership** (keep vs. hub). — *the "redundant install file" the operator flagged; surfaced, not deleted.*
3. **Prune the duplicate `egg-info` + spurious plugin record.** — *cosmetic.*
4. **Confirm `.env` contents.** — *policy check.*

**Nothing above blocks work.** The methodology adoption is genuinely in good shape — the floor is real, the plugin is real, the docs conform. The open items are hygiene, and none of them touch Council runtime behavior.

---

*Read-only audit. No files were created, deleted, or restructured beyond this document. All paths cited were verified live on 2026-07-02.*
