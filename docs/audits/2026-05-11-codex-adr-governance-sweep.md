# Codex Review — adr-governance-sweep

**Date:** 2026-05-11
**Branch:** `chore/adr-sweep`
**HEAD:** `284d566`
**Diff range:** `main..chore/adr-sweep`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

- ADR status headers match index (especially ADR-07 supersede context)
- ADR-02 revision accuracy (5-model default, --lite/--full flags)
- ADR-06 Qwen deferred language
- HANDOFF.md deletion rationale
- CHANGELOG/JOURNAL completeness

---

## Findings
**Critical**
- (none)

**High**
- (none)

**Medium**
- Severity: `Medium` | File: `docs/decisions/ADR-02-default-panel.md:5` | What: ADR-02 now says the implementation is a 5-model default with `--lite` for 3-model and `--full` as a no-op, but the underlying implementation details cited in the ADR do not actually match that story. | Why: `config/settings.yaml:10-11` still defines `default_panel` as 3 models and `full_panel` as 5, and `src/ai_council/runner.py:35-41` still implements “`--full` wins over default”. The current 5-model CLI default is being achieved indirectly by `src/ai_council/cli.py:484` passing `use_full_panel or not lite`, not by `default_panel` itself. That makes the revised ADR/CHANGELOG/JOURNAL explanation materially misleading for maintainers. | Fix direction: either make the code/config match the ADR literally (`default_panel` = 5, explicit lite path), or rewrite ADR-02/CHANGELOG/JOURNAL to describe the real implementation mechanism instead of claiming `default_panel` is the 5-model default.

- Severity: `Medium` | File: `JOURNAL.md:16` | What: the new governance-sweep entry claims the docs are now “internally consistent”, but the branch still has stale default-synthesizer guidance outside the ADR set. | Why: ADR-01 now correctly says Gemini is the default (`docs/decisions/ADR-01-synthesizer-selection.md:5`) and config agrees (`config/settings.yaml:9`), but user-facing docs still say Claude/Sonnet is default in `README.md:169`, and CLI help still says “Defaults to claude” in `src/ai_council/cli.py:217`. The sweep therefore did not actually make the repo docs consistent. | Fix direction: update README and CLI help to Gemini, and only keep the “internally consistent” claim once the non-ADR surfaces are aligned.

- Severity: `Medium` | File: `docs/decisions/ADR-07-dual-output-paths.md:10` | What: ADR-07 is marked superseded, but most of the body still reads as current truth, including a config example that says `secondary_output_enabled: true`. | Why: line 6 says the old design was replaced and defaulted off, yet line 10 still states “All debate transcripts and research reports save to two locations”, and line 39 shows the old enabled-by-default config. That is especially risky because the new journal entry says the ADR file is now the source of truth (`JOURNAL.md:7`). | Fix direction: clearly mark the remaining sections as historical (“Original decision at adoption time”), or add inline notes on each stale section/config snippet pointing readers to ADR-43/current config.

**Low**
- Severity: `Low` | File: `JOURNAL.md:14` | What: the branch deletes `docs/HANDOFF.md` as “noise”, but repo guidance still points readers to that file. | Why: `CLAUDE.md:281` still lists `HANDOFF.md` as part of the `docs/` contract, so removing the file without updating those references leaves a dead path and weakens the deletion rationale. | Fix direction: either update the remaining repo-map references to remove `HANDOFF.md`, or keep a minimal stub file that redirects to the external handoff authority.

I couldn’t find a repo-local `AGENTS.md` in this checkout, so I used the requested `Critical / High / Medium / Low` bands directly.
