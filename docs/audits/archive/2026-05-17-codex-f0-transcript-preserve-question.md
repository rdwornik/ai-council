# Codex Review — f0-transcript-preserve-question

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — merge `b9b30b5` (fix `d00989f`); Source<Question<Round-1 asserted by test. No open remainder. [ARCHIVED 2026-07-18] _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-05-17
**Branch:** `fix/transcript-preserve-full-question`
**HEAD:** `e2b5f7b`
**Diff range:** `main..fix/transcript-preserve-full-question`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

- Verify the `## Question` section is positioned correctly in the transcript (after metadata, before Round 1) and nothing else in the writer depends on the exact line offsets we shifted.
- Confirm parity with research-mode Query block. Is a `## Question` heading the right shape, or should it be inline like research?
- Check that target_paths mirroring and the metrics JSON sidecar still work correctly with the new section.
- Verify no existing transcript-parsing code (ADR-07 dual-output routing, downstream tooling, inbox archive) regex-matches against the old format and would break.
- Test coverage for pick + judge with long questions exists; should ideas-mode also be asserted?

---

## Findings
**Critical**
- (none)

**High**
- (none)

**Medium**
- `tests/test_output.py:128` — The new regression tests only assert that the full question text appears somewhere in the transcript, not that the new `## Question` block stays in the intended position after metadata and before `## Round 1`. Why: the main contract introduced by this patch is structural placement; with the current assertions, a future refactor could move the block below the rounds or into synthesis and these tests would still pass. Fix direction: make this a parametrized transcript-shape test that asserts ordering (`**Source:**` < `## Question` < `## Round 1`) and include `ideas` if you want the contract stated explicitly for every debate mode sharing `save_to_file()`.

**Low**
- (none)

No parser-coupling breakage showed up in-repo: I didn’t find transcript readers or regexes that depend on the old line offsets/header layout, and `target_paths` mirroring plus `_save_metrics_json()` still operate on the same primary filename/content flow. Research mode is not shape-identical today (`**Query:**` inline vs `## Question` section), but for debate transcripts the section heading is the safer form because it preserves multiline question bodies cleanly. I couldn’t run `pytest` here because command execution was blocked by policy.
