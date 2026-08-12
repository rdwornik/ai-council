# Ecosystem Code Standards

- Python 3.11+, Windows-first (pathlib, `py -m`)
- pyproject.toml as single source of truth
- ruff for lint/format, pytest for testing
- Type hints everywhere, no bare except
- Logging not print (except CLI Rich output)
- Dataclasses or Pydantic for data, not raw dicts
- Config via YAML + .env, never hardcode paths
- Feature branches, never commit to main
- API keys via global Documents/.secrets/.env
- Comments explain WHY, not WHAT
- **"OneDrive - Blue Yonder" = exclusion zone, three tiers** — the fleet-global form. The canonical
  text and the live grant list are at `~/.claude/rules/core-invariants.md` §1; this row **points
  rather than restates**, so the two cannot drift apart:
  - **T2 — absolutely denied, no grant mechanism exists in code:** write / delete / move / rename /
    copy-**INTO** / new-item.
  - **T1 — grant-gated:** content reads and hydration. A grant is dated, scoped to one literal
    subtree, binds the `Read` tool only, and is added only by a ruling that edits the global rule
    **and** `~/.claude/hooks/block-onedrive.ps1` in the same commit.
  - **T0 — allowed and logged:** hydration-free name/metadata enumeration.

  *Recorded 2026-08-12.* Hub ruling `N1-D03` (2026-08-10, landed together with `A3`; hub register
  `.dev-knowledge/protocols/STANDING_RULINGS.md` L-2) unified the fleet on the tiered form. This row
  previously read *"NEVER touch \"OneDrive - Blue Yonder\" paths"* — **stricter than the fleet rule,
  and wrong in the direction that reads as safe**: it forbids the T0 enumeration and the grant-gated
  T1 read the global rule permits, so a satellite obeying it refuses work the fleet allows while
  learning nothing about the T2 boundary that is the one carrying the real hazard (past cleanup
  scripts deleted personal files alongside SharePoint copies — destructive writes, not reads). The
  ruling's own condition was that the unification be **documented rather than silent**, which is why
  the reason is recorded here and not only in the hub register.
