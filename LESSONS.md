# AI Council — Lessons Learned

## Session: Phase 1 Foundation (2026-02-21)

### Rules to Follow

1. **google-genai async**: Use `client.aio.models.generate_content()` — NOT `asyncio.to_thread`. Package is `google-genai`, not deprecated `google-generativeai`.

2. **Click + asyncio**: Use `asyncio.run()` inside the sync Click handler — don't add `asyncclick` dependency.

3. **pytest-asyncio 0.24+**: Requires `asyncio_mode = auto` in `pytest.ini`. Without this, async tests silently skip or fail.

4. **`config/__init__.py`**: Required for `from config.config_loader import ...` to work as a package import.

5. **`asyncio.wait_for`**: Takes the coroutine object directly — `asyncio.wait_for(coro, timeout=n)` — not a lambda.

6. **Provider isolation**: Providers must NOT import each other per CLAUDE.md spec. XAI and OpenAI have near-identical code — that's intentional, no shared base class.

7. **`output_dir.mkdir`**: Called lazily in `output.py` when saving, not at startup.

8. **Synthesizer also debates**: Claude instance participates in debate rounds AND synthesizes. Same provider instance, by design.

9. **No bare except**: Always catch specific exceptions. Log with `logging`, never `print()`.

10. **Type hints**: Use `X | None` not `Optional[X]`. Use `Path` objects, not raw strings.

---

### 2026-04-30 | mock.patch string literals are invisible to import refactoring
- CONTEXT: ADR-38 migration renamed all src.X imports to ai_council.X
- MISTAKE: 56 mock.patch("src.debate.X") string literals in tests/ were NOT caught by import-only find-replace. Caused 30 test failures.
- RULE: After any package rename, do a SECOND pass specifically for mock.patch() string literals. Pattern: `grep -r 'mock.patch.*"old_name\.' tests/`

### 2026-04-27 | Inbox path must mirror interactive path
- CONTEXT: Research mode worked in interactive CLI but not via --inbox
- MISTAKE: Third time this pattern appeared (--full, --mode, now research routing). Inbox loop is a separate code path that doesn't automatically inherit interactive features.
- RULE: After adding ANY new feature to the interactive CLI path, immediately check: does the inbox loop handle this too? If not, add it. This is a recurring blind spot.

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

### 2026-05-12 | Architect failure mode — defending local config as "by-design"
- CONTEXT: Scrum-master addendum (2026-05-12) — strażnik caught that `tasks/lessons.md` location was non-canonical after main review implementation
- MISTAKE: Accepted `tasks/lessons.md` as intentional per CLAUDE.md Lessons Discovery section ("by-design") when the ecosystem convention is `LESSONS.md` at root. Local config can be wrong relative to ecosystem baseline; defending it as intentional blocks the cross-repo audit from working.
- RULE: When a cross-repo audit flags a convention divergence, default response is "evaluate against ecosystem baseline" — NOT "intentional per local config." Local config documents what exists; ecosystem convention determines what should exist. If they conflict, the convention wins unless explicitly overridden by an ADR. This failure mode applies symmetrically to the audit consumer, not only the audit producer.
