# ADR-10: Output routing — local default + return-dir override

**Date:** 2026-07-02
**Status:** Accepted (2026-07-17)

## Context

ADR-67 (hub) formalized the six-step Council loop, whose final step is a **deterministic
return**: the ADR/verdict is written to an operator-commissioned return directory held in
`~/.claude` global config (`council.return_dir`), so a delegating agent picks the output up
from a known path instead of guessing. ai-council has no `return_dir` plumbing yet — this ADR
records the design for it.

**Current output paths — verified 2026-07-02, none default to the hub:**

1. **Canonical write** → `defaults.output_dir` = `./output/` (resolved to repo root). Local;
   always written; the hard requirement.
2. **Legacy secondary mirror** → `defaults.secondary_output_dir` points at the hub transcripts
   dir but `secondary_output_enabled: false` — **disabled by default** (ADR-07, superseded by
   ADR-43).
3. **ADR-43 per-invocation mirror** → mirrors to `<dev_root>/<name>/docs/decisions/transcripts/`
   **only** when a run supplies `target-project:` / `--target-project`. `TargetResolver` returns
   an empty list otherwise. `.dev-knowledge` sits on the `target_projects` allow-list but that
   makes it a *permitted* target, not a *default* one.

So the hub is never a silent default today. What is missing is a home for ADR-67's step-6
deterministic return, and a sane default + override for it. Two independent needs:

- A **default** when nobody has commissioned a return directory — predictable, local, and
  **never** the methodology hub.
- An **override** so a commissioning agent/operator can redirect the deterministic return per
  ADR-67.

## Decision

1. **Default:** when `return_dir` is unset, the deterministic return defaults to the repo-local
   `./output/` directory (the existing canonical location). The methodology hub
   (`.dev-knowledge`) is **never** a default target — for the return path or any other.
2. **Override:** when a return directory is set, it **overrides** the default and receives the
   deterministic return. Override sources, in precedence:
   - `--return-dir <path>` CLI flag — implemented now.
   - `~/.claude` global config `council.return_dir` (per ADR-67) — reserved as a valid future
     source; not implemented in this pass but explicitly a legal setter.
3. Canonical `./output/` write remains a hard requirement regardless of return-dir (the return
   is a copy/route to the commissioned path, not a replacement — consistent with ADR-43's
   "canonical always written first" rule).
4. The two existing hub-capable paths are **unchanged** and remain opt-in: the disabled
   `secondary_output_dir` mirror stays disabled; ADR-43's per-invocation routing keeps requiring
   an explicit `target-project`. This ADR touches neither — verification confirmed neither
   defaults to the hub, so there is nothing to correct there.

## Reconciliation with ADR-67 (hub, immutable)

This ADR does **not** contradict ADR-67. `~/.claude`'s `council.return_dir` remains a legal way
to set the return directory — this ADR *adds* (a) a sane local default for when it is unset and
(b) a CLI override for per-invocation control. It records ai-council's concrete implementation
of ADR-67's step-6 deterministic return, which ADR-67 explicitly assigned to this repo (its
"Where each piece lives" table → known-path I/O is `ai-council`'s downstream work). ADR-67 is
untouched. ADR-43 (per-invocation mirror) is a distinct mechanism and is likewise unchanged.

## Alternatives considered

- **Default to the hub / a fixed shared path.** Rejected: makes the hub an implicit sink,
  couples every uncommissioned run to `.dev-knowledge`, and violates "never silently default to
  the hub".
- **CLI flag only, no `~/.claude` source ever.** Rejected: contradicts ADR-67, which designates
  `~/.claude` `council.return_dir` as the return-path config home.
- **`~/.claude` only, no CLI flag.** Rejected: no per-invocation override; every redirect would
  require editing global config.

## Consequences

- Uncommissioned runs behave exactly as today (`./output/`); no regression.
- A commissioning agent can redirect output per-invocation via `--return-dir` without global
  config edits.
- The `~/.claude` `council.return_dir` reader is deferred (reserved), so ADR-67's config key is
  honored as a design commitment without being built in this pass — tracked in BACKLOG #13.
- Return-dir is I/O plumbing only and is **baseline-independent** — it does not depend on the
  synthesizer-refresh baseline (Epic B), which is why #13 is separable from the deferred
  question-quality work in #9.
- No change to ADR-43 routing or the secondary mirror is required: the 2026-07-02 verification
  confirmed neither defaults to the hub, so ADR-10 is scoped to the new return path alone.

## Implementation note (for the #13 build)

Closure on #13 is an **empirical run**, not "code compiles":

- a real debate with no flags writes to `./output/` and **not** to `.dev-knowledge/`;
- a run with `--return-dir <path>` writes the deterministic return to `<path>`.

These two observations confirm the "nothing defaults to the hub" finding at build time.

## Related

- ADR-67 (hub) — Council process; step-6 deterministic return (unchanged by this ADR)
- ADR-43 (hub) — cross-project transcript routing (opt-in secondary mirror; unchanged)
- ADR-07 (local, superseded by ADR-43) — dual output paths history (always-on hub write, now off)
- BACKLOG #13 — the implementation task
