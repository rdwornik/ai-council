# Lessons Learned — Append-Only Log
<!-- scope: hybrid -->

> **Format:** `### YYYY-MM-DD | source | lesson | category | [scope: X] | action taken`
> New entries go at the top of the Entries section. Never edit old entries. Never delete.
> Grandfathered: earlier entries below use a `### YYYY-MM-DD | title` header + `CONTEXT / MISTAKE / RULE` body (never rewritten to the newer schema — append-only per ADR-29).
> Last updated: 2026-07-13

---

### 2026-05-12 | Architect failure mode — defending local config as "by-design"
- CONTEXT: Scrum-master addendum (2026-05-12) — strażnik caught that `tasks/lessons.md` location was non-canonical after main review implementation
- MISTAKE: Accepted `tasks/lessons.md` as intentional per CLAUDE.md Lessons Discovery section ("by-design") when the ecosystem convention is `LESSONS.md` at root. Local config can be wrong relative to ecosystem baseline; defending it as intentional blocks the cross-repo audit from working.
- RULE: When a cross-repo audit flags a convention divergence, default response is "evaluate against ecosystem baseline" — NOT "intentional per local config." Local config documents what exists; ecosystem convention determines what should exist. If they conflict, the convention wins unless explicitly overridden by an ADR. This failure mode applies symmetrically to the audit consumer, not only the audit producer.

### 2026-05-11 | Target resolver fail-loud pattern (cross-project routing)
- CONTEXT: ADR-43 transcript routing — `target-project` frontmatter + `--target-project` CLI flag
- MISTAKE: Early design considered silently falling back to canonical-only when an unknown target name was given. This hides config typos.
- RULE: When introducing optional config-driven routing, fail loudly at parse time on unknown targets rather than silently routing to canonical only. Silent fallback hides config typos; loud failure surfaces them at the boundary. Pattern applies broadly to any optional routing mechanism.

### 2026-05-11 | Inbox/CLI code-path parity (recurring blind spot, 3rd instance — structural fix needed)
- CONTEXT: Transcript routing feature added to CLI direct path; inbox path needed explicit wiring
- MISTAKE: This is the third occurrence of the same pattern (--full, --mode, now target-project routing). Each instance costs a follow-up commit.
- RULE: Investigate whether the two paths (CLI direct + inbox processor) can share a common processor function rather than duplicating logic. If not addressable structurally, add a parity-check test that exercises both paths for any new feature. The pattern has repeated 3x — structural change is warranted.

### 2026-05-11 | ADR-43 amendment cycle 1 — lift repeated path prefix to root field
- CONTEXT: Original `target_projects: dict[name, full_path]` schema repeated `<dev_root>/<name>/docs/decisions/transcripts/` prefix per entry
- MISTAKE: Path prefix duplication in config — each entry had to repeat the shared root
- RULE: When config has multiple entries that repeat a path prefix, lift the prefix to a root field and compute the suffix. Refactored to `dev_root: str + target_projects: list[str]` with computed paths. Reduces migration error if root path ever moves, cuts noise.

### 2026-05-11 | Observability field design — avoid redundant signal
- CONTEXT: Codex review caught `synth_timeout_flag` as dead in observability schema — timeout cases already captured via `error_class="timeout"`
- MISTAKE: Boolean flag carried the same signal already in `error_class`. Dead field added noise and false coverage impression.
- RULE: When designing observability schema, avoid carrying the same signal in two fields. One canonical field (e.g., `error_class`) is sufficient; boolean flag mirrors create dead-field risk. If a flag is truly needed, ensure it captures something the primary field cannot.

### 2026-04-30 | mock.patch string literals are invisible to import refactoring
- CONTEXT: ADR-38 migration renamed all src.X imports to ai_council.X
- MISTAKE: 56 mock.patch("src.debate.X") string literals in tests/ were NOT caught by import-only find-replace. Caused 30 test failures.
- RULE: After any package rename, do a SECOND pass specifically for mock.patch() string literals. Pattern: `grep -r 'mock.patch.*"old_name\.' tests/`

### 2026-04-27 | Inbox path must mirror interactive path
- CONTEXT: Research mode worked in interactive CLI but not via --inbox
- MISTAKE: Third time this pattern appeared (--full, --mode, now research routing). Inbox loop is a separate code path that doesn't automatically inherit interactive features.
- RULE: After adding ANY new feature to the interactive CLI path, immediately check: does the inbox loop handle this too? If not, add it. This is a recurring blind spot.
