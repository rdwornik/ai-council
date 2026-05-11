# Codex Review — docs-hygiene-sweep

**Date:** 2026-05-11
**Branch:** `chore/docs-hygiene`
**HEAD:** `d74dc8b`
**Diff range:** `main..chore/docs-hygiene`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

(none specified)

---

## Findings
`AGENTS.md` is not present in this checkout, so I used the requested `Critical / High / Medium / Low` bands directly.

**Critical**
(none)

**High**
(none)

**Medium**
- Severity: `Medium` | File: `docs/COUNCIL_QUESTION_GUIDE.md:47` | What: the guide still documents the wrong default panel shape and default usage model. It says the default is “all 4” models, later says “Default 3: claude, gemini, openai”, and recommends `synthesizer: openai` in the template. | Why: current runtime/README behavior is a full 5-model default panel, with `--lite` as the 3-model override; this branch touched the guide but left core invocation guidance internally inconsistent and out of sync with the actual CLI contract. | Fix direction: rewrite the panel/default sections and template so they match `src/ai_council/cli.py` and README exactly, or clearly label any “recommended” setup as distinct from the runtime default.
- Severity: `Medium` | File: `CHANGELOG.md:26` | What: the 2026-05-11 changelog still says `target_projects` is a “map for target name → path resolution” and `AppConfig.target_projects: dict[str, str]`. | Why: that directly contradicts the same file’s newer 2026-05-11 entry at line 13 and the current implementation, which uses `dev_root` plus `target_projects: list[str]`. The result is a single release note with two incompatible schemas for the same feature. | Fix direction: update or remove the stale bullets in the older 2026-05-11 routing entry so the changelog reflects only the post-amendment schema.
- Severity: `Medium` | File: `docs/decisions/README.md:17` | What: the ADR index presents `ADR-07` as an accepted current decision for output-path behavior, even though `ADR-07` still describes the old always-on secondary write model and `secondary_output_enabled: true`. | Why: this docs-hygiene branch explicitly adds ADR-43 routing references, but the main ADR index still points readers to an obsolete local ADR without marking it superseded or amended. That leaves the architectural record contradictory at the exact place users are told to consult it. | Fix direction: mark `ADR-07` as superseded/partially superseded by ADR-43, or amend ADR-07 so its status and content reflect the current opt-in `target-project` routing model.

**Low**
- Severity: `Low` | File: `docs/HANDOFF.md:3` | What: the repo-local handoff document now contains only pointers to `.dev-knowledge/...` paths outside this repository. | Why: anyone reading this checkout without the sibling `.dev-knowledge` repo gets no usable local handoff instructions, only external references that may not exist in their workspace. That is a regression in documentation self-sufficiency. | Fix direction: keep a minimal local summary of the required handoff flow and artifact locations, and treat the `.dev-knowledge` docs as the canonical deep reference rather than the only instructions.
