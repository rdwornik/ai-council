# Codex Review — dev-root-schema-refactor

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — review `6b87a7d`; dev-root schema refactor applied. No open remainder. [ARCHIVED 2026-07-18] _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-05-11
**Branch:** `refactor/dev-root-schema`
**HEAD:** `43608e4`
**Diff range:** `main..refactor/dev-root-schema`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

- config/config_loader.py: new dev_root + target_projects list validation logic
- src/ai_council/routing.py: TargetResolver signature change (dict -> dev_root + list)
- src/ai_council/cli.py: TargetResolver construction call update
- tests/test_config.py: new validation tests (dev_root required, dir check, dict migration error, duplicates)
- tests/test_routing.py: fixture and path assertion updates
- tests/test_cli.py: _make_test_config signature change and call site updates

---

## Findings
`AGENTS.md` is not present in this checkout, so I used the requested `Critical / High / Medium / Low` bands directly.

**Critical**
- `(none)`

**High**
- `(none)`

**Medium**
- Severity: `Medium`  
  File: `config/config_loader.py:328-333`  
  What: `dev_root` is validated and stored as `Path(raw_dev_root).expanduser()` without anchoring or resolving relative paths.  
  Why: this makes `dev_root` interpretation depend on the process working directory, unlike other config paths in this loader such as `output_dir` and inbox paths. A relative `dev_root` can therefore pass/fail unpredictably or mirror transcripts into the wrong tree when `council` is launched outside the repo root.  
  Fix direction: normalize `dev_root` deterministically before validation, e.g. if it is relative, join it to `_REPO_ROOT` (or the settings file parent), then call `.resolve()` and validate/store that resolved path.

**Low**
- Severity: `Low`  
  File: `tests/test_cli.py:320-326`  
  What: `test_cli_multiple_target_projects` still constructs `_make_test_config(...)` with the old `dict`-shaped `target_projects` and does not set `dev_root`.  
  Why: after the schema change, this test no longer exercises the real contract. It passes only because `TargetResolver` ends up iterating dict keys and the assertion checks only `len(target_paths) == 2`, so it would miss regressions in actual path computation under the new `dev_root + list[str]` design.  
  Fix direction: update the fixture call to `dev_root=tmp_path, target_projects=[".dev-knowledge", "foo"]` and assert the exact resolved paths, not just the count.

- Severity: `Low`  
  File: `src/ai_council/cli.py:245-246`  
  What: the `--target-project` help text still says names must match `target_projects map`.  
  Why: that is stale after the migration to `dev_root` plus a `target_projects` list, and will mislead users during config setup or migration.  
  Fix direction: update the help text to refer to the `target_projects` list, ideally also mentioning that paths resolve under `dev_root`.

I couldn’t run the test suite in this environment because command execution is sandbox-blocked here, so the review is from diff and source inspection only.
