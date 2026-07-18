# audits/archive/ — Preservation archive

Preservation zone for **completed** audit artifacts whose findings are fully DEPLOYED
with no open remainder (per the 2026-07-18 deployment-status inventory). Distinct from:

- `docs/archive/` — the ADR-60 *pending-classification triage queue* (default-to-deletion after two reviews). This folder is **not** that; nothing here is on a deletion track.
- `docs/audits/archive/legacy/` — pre-convention legacy code-review reports.

Files land here when their audit's asks are in effect in code/process and no BACKLOG
item remains open against them. They are retained as the institutional record; each keeps
its original filename and its in-file `Deployment-Status` stamp. Moved 2026-07-18 in the
consolidation archival pass.
