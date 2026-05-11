# Codex Review — transcript-routing

**Date:** 2026-05-11
**Branch:** `feat/transcript-routing`
**HEAD:** `d2d20dc`
**Diff range:** `main..feat/transcript-routing`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

(none specified)

---

## Findings
**Critical**

(none)

**High**

(none)

**Medium**

- Severity: `Medium`  
  File:line: [src/ai_council/inbox.py:121](C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/inbox.py:121)  
  What: `target-project` frontmatter is passed straight into the resolver with no shape validation, while the inbox loop only handles `RoutingError`.  
  Why: a malformed value like `target-project: 123` or a mixed-type list reaches `list(target_project)` in [src/ai_council/routing.py:42](C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/routing.py:42), raises `TypeError`, and aborts the entire `--inbox` batch instead of marking just that file failed and continuing.  
  Fix direction: validate `target-project` as `str | list[str]` before resolving, convert invalid shapes into `RoutingError`, and keep the per-file skip path for all routing parse failures.

- Severity: `Medium`  
  File:line: [config/config_loader.py:290](C:/Users/1028120/Documents/Dev/ai-council/config/config_loader.py:290)  
  What: `target_projects` paths are stored verbatim and never `expanduser()`/`resolve()`ed, despite the config comment saying they are resolved at load time.  
  Why: relative or `~`-prefixed target roots will mirror into the caller’s current working directory (or a literal `~` path) rather than a stable absolute location, which silently routes transcripts to the wrong place because mirror writes are best-effort.  
  Fix direction: normalize each configured target path during config load, the same way other filesystem settings are normalized, and fail early on invalid path values.

**Low**

- Severity: `Low`  
  File:line: [config/settings.yaml:24](C:/Users/1028120/Documents/Dev/ai-council/config/settings.yaml:24)  
  What: the Downloads auto-detection key list was not updated to include `target-project`.  
  Why: a markdown file saved to `~/Downloads` that uses the new routing frontmatter but none of the older keys (`mode`, `rounds`, `models`, `synthesizer`, `full`) is silently skipped by `--inbox`, so the new frontmatter-based workflow does not work consistently across inbox sources.  
  Fix direction: add `target-project` to the default `council_frontmatter_keys` and add a regression test for Downloads detection with only that key present.
