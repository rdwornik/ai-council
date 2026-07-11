# Root-surface parity disposition — ai-council

**Date:** 2026-07-11
**Status:** applied (session-ruled rows) + proposals (all others)
**Owner:** Rob

**Scope:** every root file/folder of `ai-council` assessed against the fleet methodology baseline (`.dev-knowledge` hub) and against `corp-monorepo`'s 2026-07-11 parity rollout (`docs/audits/2026-07-11_AUDIT_root-parity-disposition.md`). Produced as the ADR-101 root-parity backfill; mirrors corp's artifact structure. **The table is a set of proposals** — only two verdict classes were *applied* this session: (i) **IGNORE-verification** (gitignore/untracked checks, no file changes) and (ii) rows **ruled this session** (the #326 re-review stamp; the CLAUDE.md boundary note). Every row whose action needs an operator verb remains a **proposal**.

**Verdicts:**
- **CARRIED** — a hub/fleet-canonical artifact present here and kept in sync.
- **LOCAL (declared)** — a project-local artifact or a sanctioned divergence from hub-generic expectations; parity is *met by declaration*, not by carrying the hub-generic form.
- **IGNORE** — gitignored / ephemeral; nothing tracked, no parity obligation. Verified, not assumed.

## Disposition table

| # | Root entry | ai-council state | hub / corp expectation | Verdict | Action taken / proposal |
|---|---|---|---|---|---|
| 1 | `INSTALL.md` | hub-canonical, resynced 2026-07-11 | hub-canonical, deploy-carried (#315) | **CARRIED** | none — already current (commit `288f256`, #315). |
| 2 | `.pre-commit-config.yaml` hub block | GitHub `rdwornik/dev-knowledge` @ `v1.3.1` + `block-ff-push` + `backlog-id-on-close` | same rev + enforcement gates | **CARRIED** | none — armed at the v1.3.1 rollout (CLAUDE §9 v2.6). |
| 3 | `CLAUDE.md` Form-A markers + boundary note | 11 marker regions (§12 v2.6); **boundary note added this session** | inline `owner=hub/repo` markers + human-visible boundary note | **CARRIED** | **applied (ruled)** — boundary note inserted above §1; marker regions untouched, prose otherwise byte-identical. owner=hub = first-read (§1), conventions-commit-branch (§4 sub-span), critical-rules-records (§5 items 4–5), session-start-protocol (§6); rest owner=repo. |
| 4 | `ARCHITECTURE.md` ToC / Mermaid | no ToC, no Mermaid; ASCII Data-Flow block only | CC-facing, ToC/Mermaid-free (operator ruling) | **CARRIED (parity met)** | **applied (ruled)** — #326 consumer leg verified no-op: diagrams already converted under #262 (2026-07-08); ASCII Data-Flow retained (out of ruled scope); re-review stamped after a genuine end-to-end re-read (A2). |
| 5 | `.methodology.yaml` — `hub-codemap-hooks` | present (codemap-freshness excluded; `review_date: 2026-10-11`) | ai-council declares the codemap exclusion machine-readably | **LOCAL (declared)** | none — present; the reference declaration corp itself mirrored. |
| 6 | `assets/ruff-pre-commit.yaml` | tracked (single file) | ai-council is the origin repo for the INSTALL.md §2 reference file | **LOCAL (declared)** | none — kept; corp deliberately does **not** carry it (runs its own ruff gate). |
| 7 | dot-caches `.mypy_cache/` `.pytest_cache/` `.ruff_cache/` | present on disk, **gitignored + untracked** | gitignore policy uniform; cache presence project-specific | **IGNORE** | **verified** untracked (`git check-ignore` = yes for each; `git ls-files` = none). No removal needed. |
| 8 | `.venv/` (+ `venv/` `env/`) | gitignored ephemeral | project-local ephemeral | **IGNORE** | none — already gitignored. |
| 9 | `.env` | **gitignored** (`git check-ignore .env` → exit 0); present as a CWD fallback for `cli.py` `load_dotenv(override=False)`, with global `~/Documents/.secrets/.env` winning; not tracked | project-local secret; never committed | **IGNORE** | **verified** gitignored + untracked. Content never read or printed. Present-but-optional: the canonical key source is the global secrets file; the repo-local `.env` is a convenience CWD fallback only. |
| 10 | `ai_council.egg-info/` `output/` `logs/` `council_inbox/` (archive + `*.md`) | gitignored build/ephemeral | uniform gitignore policy | **IGNORE** | **verified** untracked. No action. |
| 11 | `.vscode/` | gitignore pattern present; **directory absent** | gitignored if present | **IGNORE** | none — nothing on disk to track. |
| 12 | `.gitattributes` | present (`* text=auto eol=lf`) | uniform fleet-wide | **LOCAL (parity met)** | none — conforms. |
| 13 | `protocols/` | present; project-local Council-domain (#314, `153b6b2`) | #314 ruled a methodology-mandated genre; ai-council owns a project-local instance | **LOCAL** | none — corp references the hub `protocols/`; ai-council owns its own. **Genre may be re-classified by #327 (protocols-as-interface ruling).** |
| 14 | UPPERCASE living docs (`VISION` `BACKLOG` `JOURNAL` `LESSONS` `CONTRIBUTING`) + `pyproject.toml` + `.gitignore` + `.ai-council.code-workspace` + `config/` `src/` `tests/` `scripts/` `docs/` `.claude/` | present, conform (`.claude/` gitignored except the allow-listed floor + sidecar + `check_floor_hash.py` + `commands/override.md`) | methodology-generic core set | **LOCAL (parity met)** | none — already conform. |
| 15 | **This artifact's filename** | `docs/audits/2026-07-11-technical-root-parity-disposition.md` (hub lowercase class-token form) | — | **LOCAL (decision recorded)** | audit filename convention: hub form used pending the fleet ruling (register item: corp uppercase `_AUDIT_` vs hub lowercase class-token). Corp's `2026-07-11_AUDIT_…` form deliberately **not** adopted — propagating one side to a third repo would widen the open register item. |

## Applied this session (branch `docs/adr-101-root-parity`)

| commit | change |
|---|---|
| `30e4dce` | `docs(architecture)`: #326 consumer-leg re-review stamp (row 4) — verified ToC/Mermaid-free no-op |
| `1859d59` | `docs(claude)`: human-visible methodology-boundary note + §12 v2.7 (row 3) |
| *(this commit)* | `docs(audits)`: this disposition table |

All other rows are **proposals** — no operator verb was applied to them (rows 1–2, 5–14 record existing state; row 15 records a naming decision only).

## Locked operator decisions (2026-07-11)

1. `ARCHITECTURE.md` is CC-facing; the #326 leg strips ToC/Mermaid (verified no-op here — already converted under #262). ASCII Data-Flow retained: the ruling covers ToC + Mermaid only.
2. The CLAUDE.md human-visible boundary note is backfilled (corp had it; ai-council did not). Marker regions untouched.
3. This audit uses the **hub lowercase class-token** filename form, **not** corp's `_AUDIT_` uppercase form — pending the fleet ruling on the open naming register item (see row 15).

## Open register items (reported, not resolved here — core-invariant #6)

- **Audit filename convention** — corp ADR-14 uppercase `_AUDIT_` class-token vs hub lowercase kebab class-token. Unresolved fleet-parity register item awaiting an operator ruling; not propagated further by this artifact.
- **`protocols/` genre** — may be re-classified by **#327 (protocols-as-interface ruling)**; row 13 is provisional pending that ruling.
