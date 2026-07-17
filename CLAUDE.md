---
last_reviewed: 2026-07-17
status: active
owner: Rob
---

@.claude/CLAUDE-FLOOR.md

# CLAUDE.md — AI Council
> **Session contract for Claude Code in this repo.** Read on every session start (auto). Single canonical agent-instruction file (≤200 lines). Per ADR-53.
>
> **For universal rules:** read `../.dev-knowledge/protocols/ESSENTIALS.md` and `../.dev-knowledge/protocols/PLAYBOOK.md`.

> **Section ownership (methodology boundary):** sections marked `owner=hub` are fleet methodology tracked from the `.dev-knowledge` hub; `owner=repo` sections are project-local. The machine-readable owner map is the `<!-- methodology:… owner=… -->` markers on each section below; sanctioned divergences from hub-generic expectations are recorded in `.methodology.yaml`.

## 1. First read (session start)
<!-- methodology:start id=first-read owner=hub -->

In order, read:
1. This file (you're here)
2. The hub methodology protocol `ESSENTIALS.md` — Rob's universal working style (read at the hub `.dev-knowledge/protocols/` set; hub-pointer, never copied into a consumer)
3. The hub methodology protocol `PLAYBOOK.md` — universal protocols (only sections relevant to the current task; same hub `.dev-knowledge/protocols/` location)
4. Most recent `docs/handoffs/*/` bundle — start with its `HANDOFF_BOOT.md` (v5 bundles' operator session entry: slug · purpose · mode; older bundles use `README.md`), then the canonical operator runbook `docs/handoffs/README.md` — if continuing prior session
5. Last 5 entries of `JOURNAL.md`

If ESSENTIALS or PLAYBOOK are unavailable, proceed with this file alone but flag it.
<!-- methodology:end id=first-read -->
<!-- methodology:start id=first-read-local owner=repo -->
> **Repo note (ai-council):** this repo carries no local `docs/handoffs/` — handoffs centralize in `../.dev-knowledge/docs/handoffs/` (ADR-42); read the most recent bundle there. The hub protocol set (ESSENTIALS/PLAYBOOK/handoff docs in item 4) is the sibling `../.dev-knowledge/protocols/` + `../.dev-knowledge/docs/handoffs/`.
<!-- methodology:end id=first-read-local -->

## 2. Repo identity
<!-- methodology:start id=repo-identity owner=repo -->

- **Name:** `ai-council`
- **Status:** `active`
- **Purpose:** Multi-model AI debate and research CLI tool; produces binding ADRs governing the `Dev/` ecosystem.
- **Owner:** Rob
- **Critical paths:** `src/ai_council/`, `tests/`, `docs/decisions/`, `config/settings.yaml`
<!-- methodology:end id=repo-identity -->

## 3. Architecture
<!-- methodology:start id=repo-architecture owner=repo -->

See `ARCHITECTURE.md` for the structural model; read it before structural changes (required per ADR-51 — mandatory for every repo).
<!-- methodology:end id=repo-architecture -->

## 4. Conventions

<!-- methodology:start id=conventions-naming-local owner=repo -->
- **Naming:** snake_case Python; kebab-case markdown; `ADR-NN-topic.md` future ADRs (existing ADRs hyphen-named per ADR-34)
<!-- methodology:end id=conventions-naming-local -->
<!-- methodology:start id=conventions-commit-branch owner=hub -->
- **Commits & branches:** Branch prefixes are `feat/ fix/ docs/ chore/` (these four only). Commit **types** follow Conventional Commits and additionally include `refactor` and `test` — commit types are **not** branch prefixes. Never commit directly to `main`: branch → `--no-ff` merge.
<!-- methodology:end id=conventions-commit-branch -->
<!-- methodology:start id=conventions-testing-local owner=repo -->
- **Testing:** `pytest tests/ -m "not integration and not envcheck" -v` (unit suite, no API keys); `pytest -x --tb=short` (quick); `asyncio_mode = auto` in `pyproject.toml`
- **Linting:** `ruff check src/ tests/ --fix`; pre-merge: `.\scripts\check.ps1` (pytest + mypy + ruff)

**Out of scope for this repo:**
- Client/pre-sales data → Obsidian vault
- Cross-ecosystem lessons → `.dev-knowledge/LESSONS.md`
- Curated Council transcripts → `.dev-knowledge/docs/decisions/transcripts/`
<!-- methodology:end id=conventions-testing-local -->

<!-- methodology:start id=conventions-output-formatting owner=hub -->
- **Output formatting (render-layer):** Claude does **not** emit box-drawing glyphs — the Claude Code TUI *paints* plain markdown pipe-tables (`| col | col |`) as Unicode borders (`┌─┬─┐ │ └─┴─┘`) **client-side at render time**. So a bare table looks clean in the terminal but copies into browser chat as costly border glyphs (~3× the tokens), and a rule that merely bans Claude from *writing* box-drawing is a no-op (Claude already doesn't). The working fix is at the render layer: any report the operator copies out must be (1) **flat** — plain markdown or `key: value` / bullet lists, no column-padding spaces — **and** (2) **wrapped in a triple-backtick code fence**, which makes the TUI render it raw/un-painted so the copied text carries no borders. Same fenced-block discipline already used for Scale-S snippets (ESSENTIALS) and downloadable prompts (§2). Persistent diagrams live on the separate human-facing visualization surface (ADR-59; the ADR-51 amendment 2026-07-05 moved Mermaid out of canonical `ARCHITECTURE.md` — its codemap is now compact text), out of scope. Full rationale + `/session-summary` reconciliation: PLAYBOOK §8 "Output the operator copies into browser chat".
<!-- methodology:end id=conventions-output-formatting -->

## 5. Critical rules

<!-- methodology:start id=critical-rules-records owner=hub -->
1. **`LESSONS.md` and `logs/TOKEN-LOG.md` are append-only** — never edit old entries; only append (ADR-29, ADR-39)
2. **`JOURNAL.md` is append-only newest-first** — prepend at session wrap or workday close
3. **ADRs, transcripts, handoffs, and audits are immutable** — supersede with a new file or an in-file amendment marker; never edit in place. **ADR ratification exception (ADR-94):** an ADR's *status line* MAY be edited in place on ratification (e.g. Proposed → Accepted) — the status line is metadata, not decision content. This exception is ADR-specific and covers the status line only; ADR decision content, and transcripts / handoffs / audits in full, remain immutable.
<!-- methodology:end id=critical-rules-records -->
<!-- methodology:start id=critical-rules-local-a owner=repo -->
4. Read `.claude/rules/` before making code changes: `code-standards.md`, `python-env.md`, `testing.md`
5. API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `PERPLEXITY_API_KEY`) live in `C:\Users\1028120\Documents\.secrets\.env` — never add keys to a repo-local `.env`
<!-- methodology:end id=critical-rules-local-a -->
<!-- methodology:start id=critical-rules-consistency owner=hub -->
6. **Keep files consistent** — ESSENTIALS summarizes PLAYBOOK, not copies it; divergence causes drift
<!-- methodology:end id=critical-rules-consistency -->
<!-- methodology:start id=critical-rules-local-b owner=repo -->
7. Run `.\scripts\check.ps1` (pytest + mypy + ruff) before every merge
8. Config strings (models, prompts, personas, timeouts) live in `config/settings.yaml` — never hardcode
<!-- methodology:end id=critical-rules-local-b -->
<!-- methodology:start id=critical-rules-no-leftovers owner=hub -->
9. **No leftovers** — any automated or scratch-creating process (parallel-session worktree, temp file, scratch dir) removes **and verifies removal of** everything it created before it counts as done; cleanup fires even on abort. The provision→cleanup round-trip must leave the tree identical. See PLAYBOOK §Session-boundaries "No leftovers"
<!-- methodology:end id=critical-rules-no-leftovers -->
<!-- methodology:start id=critical-rules-local-c owner=repo -->
10. Do NOT merge `xai.py` and `deepseek.py` into a single provider — keep separate
<!-- methodology:end id=critical-rules-local-c -->

## 6. Session start protocol
<!-- methodology:start id=session-start-protocol owner=hub -->

1. `git status` — clean working tree?
2. `git log --oneline -5` — recent context
3. Read most recent handoff if continuing prior session
4. Check `BACKLOG.md` for in-progress items
5. `pytest --collect-only` — test discovery sanity check
6. Wait for Rob's prompt — never improvise

If any check fails → stop and ask Rob before proceeding.

Verify after updates: ESSENTIALS ↔ PLAYBOOK alignment; ENVIRONMENT ↔ `~/.claude/` state; SESSION_SETUP ↔ PLAYBOOK process changes; JOURNAL reflects last session.
<!-- methodology:end id=session-start-protocol -->
<!-- methodology:start id=session-start-protocol-local owner=repo -->
> **Repo note (ai-council):** the collect-only sanity check (step 5) is `pytest --collect-only -q`; handoffs are read from `../.dev-knowledge/docs/handoffs/` (ADR-42). Run the full pre-merge gate `.\scripts\check.ps1` (pytest + mypy + ruff) when ready to merge (§5 item 7).
<!-- methodology:end id=session-start-protocol-local -->

## 7. Slash commands available
<!-- methodology:start id=commands-repo-roster owner=repo -->

User-level (`~/.claude/commands/`):
- `/session-summary` — generate token-efficient session summary
- `/codex-review` — invoke Codex review on a staged code diff

Repo-level (`./.claude/commands/`):
- `/override` — bypass the ADR-85 session-end gate for this HEAD (logged, HEAD-bound)

Plugin (`tier1-lifecycle@dev-knowledge-methodology`, enabled in `.claude/settings.json`):
- `/review-closures` — review + execute ONLY operator-approved closures (ADR-70 Tier-1)
- `/ship` — merge the current branch to main via `--no-ff`, push, delete the branch
<!-- methodology:end id=commands-repo-roster -->

## 8. Skills active
<!-- methodology:start id=skills-repo-roster owner=repo -->

**User skills** (`~/.claude/skills/`):
- `gotchas` — universal dev gotchas (encoding, shell safety, test pitfalls)

(`session-summary`/`codex-review` are **commands**, not skills — see §7. This repo has **no** repo-level `.claude/skills/` directory; a repo-specific gotchas skill, if added, would go under `.claude/skills/gotchas/`.)

**Repo rules** (`./.claude/rules/` — read before code changes; these are rules, not skills):
- `code-standards.md` — ecosystem code standards
- `python-env.md` — venv, install, async-first guidance
- `testing.md` — pytest + pytest-asyncio standards

Code review: Codex via `/codex-review`; threshold 3+ files for a full review.
<!-- methodology:end id=skills-repo-roster -->

## 9. Hooks active
<!-- methodology:start id=hooks-repo-roster owner=repo -->

Pre-commit (`.pre-commit-config.yaml`) — hub hook-source pinned `rev: v1.3.1` (methodology corpus v1.3.1):
- `normalize-headers` — normalizes dated-log headers in `LESSONS.md`/`JOURNAL.md`
- `floor-hash-verify` — verifies `.claude/CLAUDE-FLOOR.md` matches its `.sha256` sidecar
- `canonical_freshness` — `last_reviewed` A2 gate; FAIL blocks the commit on a canonical doc edited since its last review
- `validate-audit-casing` (consumer-local, ADR-101 R4) — audit-filename casing gate (fleet ruling d1; casing-only carry; `scripts/validate_audit_casing.py`, `always_run`)
- `validate-backlog` (consumer-local, ADR-66) — `BACKLOG.md` story-map structural gate (content-parity D1); carried as the ADR-78 floor twin of the hub validator (`scripts/validate_backlog.py`) — hand-synced, NO automated carrier yet (intake G1/G2); floor twin omits `[S<n>]` enforcement (Wave-1 doctrine)
- `toc-freshness` / `toc-generate` (hub-sourced) — TOC freshness for `protocols/COUNCIL_QUESTION_GUIDE.md`
- `backlog-id-on-close` (hub-sourced, commit-msg) — requires `[#id]` in the commit message when a `BACKLOG.md` task line is removed
- `block-ff-push` (hub-sourced, pre-push) — refuses a direct-to-main / FF push to `main` (core-invariant #5 prevent organ, #302); a `--no-ff` merge passes. Added at the v1.3.1 carrier bump
- (`codemap-freshness` from the hub set is intentionally NOT consumed — ai-council's codemap is hand-authored, so `codemap check` always diffs; sanctioned in `.methodology.yaml`, `hub-codemap-hooks` is waivable per manifest-v1.3.x; same exclusion as corp-monorepo)
- `ruff` (consumer-owned, `astral-sh/ruff-pre-commit` mirror pinned `v0.15.5`, gate mode `args: []`; NOT from the hub hook-source above) — blocks a commit on any E/F/I/W lint violation (config in `pyproject.toml`). Deliberate bare `id: ruff` (no `name:`) for prune-safety, so a future hub remove-leg targeting the canonical-named stanza cannot silently delete this consumer gate. Pruned 2026-07-04 ([#244] deploy `31e785d`), then **RE-ACTIVATED 2026-07-12** by fleet ruling overriding that prune; declared in `.methodology.yaml` (`ruff-gate`). Authority = the fleet ruling; the prompt's cited "divergence-register item 9" pointer was hub-side (the hub's divergence register), mis-addressed as local. `.\scripts\check.ps1` still runs the full pytest+mypy+ruff trio pre-merge

Session hooks (`.claude/settings.json`):
- SessionStart: `check_floor_hash.py --require-present` (floor guard) + `python -m pre_commit install` (arms the commit hooks)
- Stop: `session_end_backpressure.py` — deterministic session-end gate (JOURNAL SHA-anchor hard block)
- The enabled `tier1-lifecycle` plugin also fires a Stop (`propose_closures`) + a SessionStart surface hook

Manual pre-merge gate:
- `.\scripts\check.ps1` — pytest + mypy + ruff (run before every merge; not wired to pre-commit)
<!-- methodology:end id=hooks-repo-roster -->

## 10. Anti-patterns specific to Claude Code in this repo
<!-- methodology:start id=antipatterns-universal owner=hub -->

- **Editing old LESSONS.md or logs/TOKEN-LOG.md entries** — append-only; editing corrupts the institutional record
- **Adding orchestration scripts** — Layer 2 invariant: validators only, no scripts that drive state in child repos
- **Narrating or managing AGENTS.md** — AGENTS.md is retired (ADR-53); CLAUDE.md is the single instruction file
- **Duplicating content between files** — ESSENTIALS summarizes PLAYBOOK, not copies; drift is the failure mode
- **Putting executable rules in this repo** — those belong in `~/.claude/` with `verify:` lines
- **Running validators with no args** — vacuous pass; always pass `--all` or specific paths
<!-- methodology:end id=antipatterns-universal -->
<!-- methodology:start id=antipatterns-repo owner=repo -->

- **Windows cp1252**: Do not print Unicode chars in Rich progress callbacks — ASCII only
- **google-genai event loop**: `genai.Client(api_key=...)` must be created INSIDE the async method, NOT in `__init__`
- **Interactions API warnings**: suppress `UserWarning` from `client.aio.interactions` at the call site
- **MockProvider ABC**: `async def generate` must exist in class body AND be shadowed by `AsyncMock` in `__init__`
- **pytest-asyncio**: `asyncio_mode = auto` required in `pyproject.toml`
- **Critique template**: Uses `{previous_responses_anonymized}`, not `{previous_responses}`
- **Inbox loop parity**: Features added to interactive CLI must be explicitly mirrored into inbox loop
- **`_anonymize_responses()` shuffle**: Part of blind-voting contract — do not change without an ADR
- **`make_cache_key()` location**: In `src/ai_council/research/merger.py`, NOT `src/ai_council/research/cache.py`
- **Windows /dev/null**: Use `io.StringIO()` for Console mocking in tests, not `open("/dev/null", "w")`

Do NOT:
- Re-add scope-tag enforcement — withdrawn under ADR-46; `validate_scope_tags.py` deleted
- Recreate `CHANGELOG.md` or `BACKLOG_ARCHIVE.md` — removed per ADR-49
- Edit existing `LESSONS.md` entries — append-only per ADR-29
- Add API keys to a repo-local `.env` — global secrets only
- Change Council runtime behavior to fix question-quality problems — fix in `protocols/COUNCIL_QUESTION_GUIDE.md`
<!-- methodology:end id=antipatterns-repo -->

## 11. Recent ADRs binding here
<!-- methodology:start id=recent-adrs-roster owner=repo -->

**Local (`docs/decisions/`):**
- ADR-01: Synthesizer Selection — non-participating model synthesizes; default gemini (Revised 2026-04-30)
- ADR-02: Default Panel Composition — full 5-model default; `--lite` for 3-model (Revised 2026-05-11)
- ADR-03: Blind Voting in Round 2 — `_anonymize_responses()` shuffles; hides provider identity
- ADR-04: Mode System — pick/ideas/judge/research with aliases and auto-detection
- ADR-05: Research Mode Integration — parallel-research code path, file cache, `--deep` opt-in
- ADR-06: Cost Optimization — per-provider tracking; Qwen trial deferred (Revised 2026-05-11)
- ADR-07: Dual Output Paths — superseded by ADR-43 (opt-in target-project routing)
- ADR-08: Research Degradation Alarm — <3 research providers succeed → exit code 3 + alarm banner
- ADR-09: protocols/ as the invocation surface — GUIDE + RUBRIC live under `protocols/` (Accepted 2026-07-17)
- ADR-10: Output routing — local `./output/` default + `--return-dir` override; hub never a silent default (Accepted 2026-07-17)
- ADR-11: Delegated Invocation Contract — two lanes, one machine-readable surface (`COUNCIL_INVOCATION_CONTRACT.md`) (Accepted 2026-07-05)
- ADR-12: Provider Backend Engine — CLI-subscription seats (v1 = claude+codex) + two-lane cost policy; §5 default-flip evidence-gated (Accepted 2026-07-05)
- ADR-14: ADR lifecycle states — Proposed/Accepted/Revised(dated)/Superseded + header↔index sync; ratifies DRAFT-GOV-1 (Accepted 2026-07-17)

**Ecosystem (`.dev-knowledge/docs/decisions/`) binding here:**
- ADR-29: append-only LESSONS; ADR-34: filename conventions; ADR-38: `src/ai_council/` namespace
- ADR-42: handoffs centralized in `.dev-knowledge`; ADR-43: cross-project transcript routing
- ADR-48/49: no CHANGELOG/BACKLOG_ARCHIVE; Conventional Commits; JOURNAL/LESSONS structure
- ADR-51: ARCHITECTURE.md convention (universal); ADR-53: CLAUDE.md as single canonical instruction file
- ADR-59: universal visual pattern (dot-prefix configs, ALL-CAPS canonical, `.code-workspace` sort) — repo conforms; ADR-60: docs/ folder taxonomy (decisions/ + audits/ + archive/, README-seeded)
- ADR-67: AI-Council process operationalization — six-step gated loop; downstream `/council-question` template + gate + `council.return_dir` are ai-council's to implement (not yet built)

> **BACKLOG form (resolved 2026-06-02):** the ADR-66 story-map binds all repos with **proportional depth** (`.dev-knowledge` ADR-38 A6; BACKLOG #20 closed). This repo's `BACKLOG.md` was migrated from the ADR-41/47 stream schema to the story-map on 2026-06-02 (all items preserved).
<!-- methodology:end id=recent-adrs-roster -->

## 12. Section history
<!-- methodology:start id=section-history owner=repo -->

- v1.0 (pre-ADR-53) — technical reference document (architecture, commands, design decisions)
- v2.1 (2026-05-19) — ADR-53: retire AGENTS.md; CLAUDE.md becomes substantive single canonical agent-instruction file; technical depth moved to ARCHITECTURE.md
- v2.2 (2026-06-02) — universalization conformance audit: add `last_reviewed` frontmatter (resolves audit.py check #10 WARN); fix §header PLAYBOOK path; reconcile §7/§8 to actual `~/.claude/` + `.claude/` state (`/save` repo-command and `handoff`/`save` skills do not exist; +`/evolve`/`/codex-review`; +`verify` skill; `/review`→`/codex-review`); §10 namespace path `src/research/`→`src/ai_council/research/`; §11 +local ADR-08, +ecosystem ADR-59/60/67, note unresolved backlog-schema scope
- v2.3 (2026-06-02) — ecosystem-unify to the canonical standard (ADR-38 A6): added `CONTRIBUTING.md`; normalized VISION (Mission→Vision, +Values/References) and ARCHITECTURE (+Key conventions/Authority/Validators/Governing ADRs) to the canonical spine; migrated `BACKLOG.md` to the ADR-66 story-map (11 items preserved); `LESSONS.md` H1 → canonical title; §11 backlog-schema note resolved (#20 closed)
- v2.4 (2026-07-05) — currency re-review (first consumer-measurement session; hub [#252] Phase 0.5): §9's `ruff` pre-commit bullet reconciled to the v1.2.0 prune (`31e785d` updated the config but not this file — the exact A2 staleness `canonical_freshness` had flagged since 2026-07-03); §7 gains the two enabled-plugin commands (`/review-closures`/`/ship`). Genuine end-to-end re-read verified §1–§11 against live state (commands dir, pre-commit config, session hooks); `last_reviewed` re-stamped 2026-07-05.
- v2.5 (2026-07-06) — Arc 3 conformance residuals (Track rename + hygiene sweep): §9 gains the `backlog-id-on-close` hub hook (closes the v1.2.0 manifest gap — `.pre-commit-config.yaml` was missing it alongside the existing `toc-freshness`/`toc-generate` pull from `../.dev-knowledge`); `last_reviewed` re-stamped 2026-07-06.
- v2.6 (2026-07-11) — methodology **v1.3.1** rollout (Wave-1 first fleet consumer, hub ADR-101 hermetization): the precommit carrier's hub hook-source rev bumped `v1.2.0 → v1.3.1` and the **`block-ff-push`** pre-push gate added (core-invariant #5 prevent organ, #302) — §9 reconciled to state the rev and the new gate. `codemap-freshness` from the fleet-generic install set is intentionally NOT consumed (hand-authored codemap ⇒ `codemap check` always diffs; same exclusion as corp-monorepo), recorded machine-readably in the new root `.methodology.yaml` waiver (`hub-codemap-hooks` waivable). v1.3.1 (not v1.3.0) was armed because the v1.3.0 tag predated the #318/#319 block-ff-push range-reconstruction fix. **Form-A boundary markers** (hub #312 / ADR-101; design `.dev-knowledge/docs/audits/2026-07-11-technical-fleet-boundary-marker-design.md`) grandfathered onto §1–§12 as additive `<!-- methodology:start/end id=… owner=hub|repo -->` HTML comments — **body prose byte-identical** (marker-only; divergent/gap owner=hub regions left for the read-only boundary reporter to surface, reconciled in a later pass). `owner=hub` = `first-read` (§1), `conventions-commit-branch` (§4 sub-span), `critical-rules-records` (§5 items 4–5), `session-start-protocol` (§6); the rest `owner=repo`. Genuine end-to-end re-read to place every region + reconcile §9; `last_reviewed` re-stamped 2026-07-11.
- v2.7 (2026-07-11) — ADR-101 root-parity backfill (parity with corp-monorepo's 2026-07-11 rollout): added the **human-visible methodology-boundary note** above §1 (owner=hub = fleet methodology tracked from `.dev-knowledge`; owner=repo = project-local; the `<!-- methodology:… owner=… -->` markers ARE the machine map; sanctioned divergences in `.methodology.yaml`) — corp carried it, ai-council did not. **Marker regions untouched; body prose otherwise byte-identical.** Same arc: the **#326 consumer leg** was verified a no-op (ARCHITECTURE.md already ToC/Mermaid-free since #262; ASCII Data-Flow retained, out of ruled scope) and re-review-stamped; and a new `docs/audits/2026-07-11-technical-root-parity-disposition.md` disposition table (ai-council root surface vs hub + corp) was committed. `last_reviewed` stays 2026-07-11 (same-day re-review).
- v2.8 (2026-07-12) — ruff-gate re-activation (fleet ruling) + hub `.vscode/settings.json` carry: §9's `ruff` bullet flipped from PRUNED to a live consumer-owned gate (`astral-sh/ruff-pre-commit` mirror `v0.15.5`, gate mode `args: []`, prune-safe bare `id: ruff`), re-activated 2026-07-12 overriding the [#244] 2026-07-04 prune; declared in `.methodology.yaml` (`ruff-gate` divergence, since the fleet-generic manifest carries no ruff). **Wording correction:** the triggering prompt's "divergence-register item 9" authority was a hub-side pointer (the hub's divergence register), mis-addressed as local — recorded truthfully here rather than as "item 9 did not exist." Also carried `.vscode/settings.json` byte-identical to the hub (77-byte `files.watcherExclude` for `.claude/worktrees/**`; `.gitignore` narrowed `.vscode/` → `.vscode/*` + `!.vscode/settings.json`). `last_reviewed` re-stamped 2026-07-12.
- v2.9 (2026-07-13) — content-parity T1 canonical baseline (consumer half; hub audit `2026-07-13-technical-content-parity-inventory.md` @ `2b21cb26`, rows A1–A8/B1/B2/B4/B5/C/D1/E3/F1). §1/§4/§5/§6 hub regions **expanded from 4 → 8** and materialized **byte-verbatim** from `templates/claude-regions/`: `first-read` (A1, +adjacent repo handoff note), `conventions-commit-branch` (A2), **+`conventions-output-formatting`** (A3), `critical-rules-records` (A4, now 3 canonical items), **+`critical-rules-consistency`** (A5, slot 6), **+`critical-rules-no-leftovers`** (A6, slot 9), `session-start-protocol` (A7, +adjacent repo merge-command note), **+`antipatterns-universal`** (A8, §10). §5 renumbered to the hub 1–10 skeleton (hub regions at 1-3/6/9; all 5 local rules preserved at 4/5/7/8/10). B2: contiguous §4/§5 project prose wrapped `owner=repo`. B4: §8 relabels the three `.claude/rules/` files as **rules, not skills**. B5: §9 gains the two missing live local ids (`validate-audit-casing`, `validate-backlog`). §11 title kept **without** "(last 5)" per operator ruling (a false-claim over a ~20-item curated list; declared in `.methodology.yaml` `claude-md-section-11-title`). `last_reviewed` re-stamped 2026-07-13.
- v2.10 (2026-07-17) — GOV-1 consolidation currency pass (#31, gate G1→G2): §11 Local ADR list extended from ADR-08 through **ADR-09/10/11/12 + ADR-14** (the roster stopped at ADR-08 while ADR-09..12 were live and ADR-14 ratified this session — the exact A2 staleness `canonical_freshness` guards). ADR-09/10 flipped `Proposed`→`Accepted` and DRAFT-GOV-1 ratified as ADR-14 (see the ADR index + `docs/intake/2026-07-17-gov1-rulings-register.md`). Genuine end-to-end re-read verified §1–§10 unchanged against live state (hooks, commands, conventions all current since the 2026-07-13 stamp); only §11 was stale. `last_reviewed` re-stamped 2026-07-17.
<!-- methodology:end id=section-history -->

---

**Last updated:** 2026-07-13
**Maintained by:** Rob
