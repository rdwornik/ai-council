# Installing the `tier1-lifecycle` plugin

ADR-70 Tier-1 lifecycle for a host repo: session-end **propose-closures** (plugin
Stop hook), proposal **surfacing** (global SessionStart hook — see note below), the
**`/review-closures`** human-gated close command, and a portable **ruff lint gate**.

The plugin operates on the host repo's `BACKLOG.md` and `logs/` via
`$CLAUDE_PROJECT_DIR`; the scripts themselves live under `${CLAUDE_PLUGIN_ROOT}`.

> **Surfacing is NOT a plugin hook.** Plugin `hooks.json` SessionStart hooks register
> too late for the one-shot SessionStart init event and never fire (verified
> 2026-06-02; the Stop hook is per-turn so it survives). Proposal surfacing is therefore
> handled by a **global** `~/.claude` SessionStart hook (`hooks/surface-closures.ps1`),
> which is self-contained (reads `$CLAUDE_PROJECT_DIR/logs/PROPOSALS-*.md`; no plugin-cache
> dependency) and covers every repo. Install it once per machine into `~/.claude/settings.json`
> alongside any existing SessionStart hooks.

## Prerequisites

- The host repo has a `BACKLOG.md` whose tasks use the `- [#id] ... · Done when: ...`
  form (ADR-66). The proposer degrades gracefully on a smaller/simpler backlog —
  it proposes only what matches and writes "no closures detected" otherwise.
- **REQUIRED — gitignore the loop's ephemeral output.** The Stop hook writes
  `logs/PROPOSALS-<date>.md` into the host repo on first run. Before installing, add
  to the host `.gitignore`:
  ```
  logs/PROPOSALS-*.md
  ```
  Skipping this leaves the proposals file as an untracked working-tree artifact that
  can be accidentally committed (the gap that surfaced on corp-sca). If the host repo
  keeps **nothing** tracked under `logs/`, ignoring the whole `logs/` directory is
  simplest — but do NOT blanket-ignore `logs/` in a repo that tracks files there (e.g.
  `.dev-knowledge` tracks `logs/TOKEN-LOG.md`); use the `logs/PROPOSALS-*.md` pattern
  there. (Add `logs/FLEET-HEALTH.md` too if the repo also runs the Tier-2 fleet audit.)

## 1. Install the CC plugin (commands + hooks)

The `.dev-knowledge` hub is a plugin **marketplace** (`.claude-plugin/marketplace.json`).
Add it and install at **project scope** (config committed into the host repo so the
install is reproducible):

```bash
# from the host repo root
claude plugin marketplace add /path/to/.dev-knowledge --scope project
claude plugin install tier1-lifecycle@dev-knowledge-methodology --scope project
/reload-plugins        # in an interactive session, to activate
```

This writes to the host repo's `.claude/settings.json`:

```json
{
  "extraKnownMarketplaces": {
    "dev-knowledge-methodology": {
      "source": { "source": "directory", "path": "/path/to/.dev-knowledge" }
    }
  },
  "enabledPlugins": { "tier1-lifecycle@dev-knowledge-methodology": true }
}
```

> If the host repo gitignores `.claude/` (e.g. `ai-council` does), this config is
> not committed by default. Either force-add it (`git add -f .claude/settings.json`)
> or keep the install local — decide per repo; do not silently override the repo's
> ignore intent.

## 2. Install the ruff lint gate (separate — pre-commit framework)

A CC plugin **cannot** ship a `.pre-commit-config.yaml`. Merge the pinned-rev stanza
in `assets/ruff-pre-commit.yaml` into the host repo's `.pre-commit-config.yaml`
(create the file with a `repos:` list if absent), then:

```bash
pre-commit install
```

## Verify

- `python "$CLAUDE_PLUGIN_ROOT/scripts/review_closures.py" surface` (with
  `CLAUDE_PROJECT_DIR` = host root) prints a summary when proposals exist, else silent.
- Stage a lint error and attempt a commit → the ruff gate blocks it.
- Make a `closes [#N]` commit for an open host item → the Stop hook proposes it next stop.
- `/review-closures` → human-gated close on the host backlog.

## Keeping the plugin up to date

When the plugin source changes in `.dev-knowledge` (new script version, hook tweak, command
update), installed repos run a **stale cached copy** until the cache is refreshed. The
plugin cache lives at `~/.claude/plugins/cache/dev-knowledge-methodology/tier1-lifecycle/`.

### Workflow reference

| Situation | Command(s) | Notes |
|---|---|---|
| **Plugin source changed** in `.dev-knowledge` (you committed to `plugins/tier1-lifecycle/`) | 1. **Bump `version` in `plugin.json`** · 2. `claude plugin marketplace update dev-knowledge-methodology` · 3. **per already-installed repo:** `claude plugin update tier1-lifecycle@dev-knowledge-methodology --scope project` · 4. restart the session | **The cache is keyed by version** (`~/.claude/plugins/cache/.../<version>/`). Three gotchas, all verified 2026-06-02: (a) `marketplace update` alone does NOT refresh files at an unchanged version; (b) `plugin install` **no-ops** on an already-installed repo ("already installed") — it will NOT upgrade; (c) the upgrade command is `plugin **update** --scope project` (defaults to user scope — pass `--scope project`), which fetches the new `<version>/` into the cache. Restart applies it. |
| **New machine or cache wiped** (no prior `marketplace add` on this machine) | `claude plugin marketplace add "C:\Users\1028120\Documents\Dev\.dev-knowledge" --scope project` then `claude plugin install tier1-lifecycle@dev-knowledge-methodology --scope project` | The `marketplace add` step populates the cache; `install` records the enablement. Without `marketplace add` first, `install` fails with "Plugin not found in marketplace" even when settings.json is correct (the cache-population step is required). |
| **New repo** receiving the plugin | Copy the `enabledPlugins` + `extraKnownMarketplaces` block from an existing repo's `.claude/settings.json` (force-add with `git add -f` if `.claude/` is gitignored), then `claude plugin install tier1-lifecycle@dev-knowledge-methodology --scope project` | The cache is already populated from the earlier `marketplace add` on this machine, so install resolves immediately. |

> **Scope note:** `--scope project` writes to `.claude/settings.json` (committed, shared).
> Omit `--scope` for user-scope (`~/.claude/settings.json`, personal only).
