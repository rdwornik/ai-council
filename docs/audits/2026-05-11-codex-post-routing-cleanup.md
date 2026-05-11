# Codex Review — post-routing-cleanup

**Date:** 2026-05-11
**Branch:** `chore/post-routing-cleanup`
**HEAD:** `0416a3f`
**Diff range:** `main..chore/post-routing-cleanup`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

- secondary_output_enabled flipped to false in settings.yaml — verify no code path is silently broken
- README Transcript Routing section accuracy vs actual routing.py behavior
- CLAUDE.md test count correction (349 → 354)
- CHANGELOG and JOURNAL entries completeness

---

## Findings
I couldn’t locate `AGENTS.md` in this repo, so I used the requested `Critical / High / Medium / Low` bands directly.

**Critical**
- (none)

**High**
- (none)

**Medium**
- `config/settings.yaml:8`  
  What: `secondary_output_enabled` now defaults to `false`, which disables the legacy global mirror for both debate and research runs.  
  Why: The code paths in [src/ai_council/orchestrator.py](/C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/orchestrator.py:49) and [src/ai_council/research/runner.py](/C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/research/runner.py:152) simply skip `secondary_dir` when this flag is off, so existing operators who relied on the default `.dev-knowledge` mirror will stop getting those writes with no runtime signal. The path isn’t broken internally, but the default behavior change is silent and can break downstream ingestion.  
  Fix direction: Either keep the legacy default on for one deprecation cycle, or add an explicit migration signal: a startup warning when `secondary_output_dir` is configured but disabled, plus a prominent upgrade note in user-facing docs.

- `README.md:266`  
  What: The new Transcript Routing section says unknown target names “exit with `RoutingError` ... no debate runs,” but that is only true for direct CLI usage.  
  Why: In inbox mode, [src/ai_council/cli.py](/C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/cli.py:338) catches `RoutingError`, archives the bad file as failed, and continues processing the rest of the batch; it does not abort the overall run. That mismatch can lead operators to expect batch atomicity that the implementation does not provide.  
  Fix direction: Split the behavior note by invocation path: CLI should say it exits before running, inbox mode should say the offending file is skipped/failed and the batch continues.

**Low**
- `CHANGELOG.md:18`  
  What: The changelog still says the routing change brought the suite to “349 total,” while this cleanup branch separately records a correction to `354`.  
  Why: That leaves the release notes internally inconsistent on the same feature area, which weakens the audit trail you’re trying to clean up.  
  Fix direction: Update the stale total in `CHANGELOG.md`, or avoid hard-coded totals unless they were re-verified for that entry.

- `README.md:220`  
  What: The README still says “255 unit tests,” even though `CLAUDE.md` was just corrected to `354`.  
  Why: This cleanup fixes one test-count drift but leaves another stale count in the same edited document set, so the docs remain inconsistent.  
  Fix direction: Update the README test count to the verified current number, or remove the hard-coded total and describe coverage more generally.
