# ADR-07: Dual Output Paths

**Date:** 2026-04-28
**Status:** Accepted

## Decision

All debate transcripts and research reports save to two locations:

1. `<repo-root>/output/` — operational archive, always written
2. `~/Documents/Dev/.dev-knowledge/docs/decisions/transcripts/` — curated knowledge base, written only if the directory exists on disk

## Context

Running `council` from any directory other than the repo root caused transcripts to land in `<cwd>/output/` — scattering files across the filesystem. Two users of these transcripts need different guarantees:

- **Operational archive**: every run produces a durable record, regardless of where the command was invoked
- **Dev knowledge base**: Council decisions accumulate alongside other architecture notes in `.dev-knowledge/`

## Solution

- `output_dir` in `DefaultsConfig` is now resolved to an absolute path anchored at `_REPO_ROOT` (`Path(__file__).parent.parent` in `config_loader.py`), eliminating cwd-relative drift
- `save_to_file` and `save_research_to_file` accept `secondary_dir: Path | None`; if the directory exists at write time, they write there too and return both paths
- Orchestrator and research runner extract `secondary_dir` from `config.defaults.secondary_output_dir` (gated by `secondary_output_enabled`) and pass it down
- `_save_metrics_json` also writes to secondary when present (metrics and transcript stay paired)

## Fallback

If the secondary directory does not exist: log a `WARNING`, write primary only. The secondary is never auto-created — its absence is intentional when running outside the dev ecosystem.

## Configuration

```yaml
defaults:
  output_dir: "./output"                   # resolved to repo root
  secondary_output_dir: "~/Documents/Dev/.dev-knowledge/docs/decisions/transcripts"
  secondary_output_enabled: true
```

Set `secondary_output_enabled: false` to disable dual-write entirely.

## What does NOT go to secondary

- `--format json` stdout dumps (not a file write, unaffected)
- JSON metrics files follow transcripts — if transcript goes to secondary, metrics do too
