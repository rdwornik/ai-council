# Codex Review — audit-addendum-i7-i8

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — `f094d08`/`baeb6bc`; I8 rename `2986ac0`. No open remainder. [ARCHIVED 2026-07-18] _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-05-12
**Branch:** `chore/audit-addendum-i7-i8`
**HEAD:** `baeb6bc`
**Diff range:** `main..chore/audit-addendum-i7-i8`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

- I7: tasks/lessons.md → LESSONS.md root move + tasks/ folder retirement
- I8: docs/handoffs/_archive/ → docs/handoffs/archive/ rename
- CLAUDE.md Folder Governance + Lessons Discovery section updates
- VISION.md lessons path reference update
- LESSONS.md architect lesson append
- CHANGELOG + JOURNAL addendum entries
- Verify no stale _archive or tasks/lessons refs in active files

---

## Findings
**Critical**
- (none)

**High**
- (none)

**Medium**
- (none)

**Low**
- `docs/handoffs/archive/2026-05-11_dev-knowledge-cycle-closure.md:64`  
  What: The newly added handoff artifact still tells readers to archive the file under `ai-council/docs/handoffs/_archive/...`.  
  Why: This branch’s stated goal is to rename `docs/handoffs/_archive/` to `docs/handoffs/archive/` and eliminate stale `_archive` references. Leaving the old destination in a newly-added document propagates the deprecated path and undermines the rename cleanup.  
  Fix direction: Update the suggested ai-council archival path in the document body to `ai-council/docs/handoffs/archive/2026-05-11_dev-knowledge-cycle-closure.md`.
