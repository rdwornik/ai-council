# Synthesis Quality Rubric

Operator-applicable checklist for evaluating synthesis output quality. Used during smoke tests
and ongoing operations to flag regressions in synthesizer behavior.

## Checklist

For each synthesis output, score yes/no:

1. **Position representation** — Does the summary represent all major positions from the debate?
2. **No hallucinated consensus** — Does the synthesis avoid claiming agreement that wasn't actually reached?
3. **Scannability** — Is the structure clear enough that an operator can extract verdict + key trade-offs without re-reading the full debate?
4. **Faithfulness** — Does the synthesis accurately reflect what each panelist actually said, without distortion or invented claims?
5. **Verbosity proportionality** — Is output length appropriate to debate substance (not bloated, not truncated)?

## Use cases

- **Smoke tests** — score 25-50 sample synthesis outputs from candidate synthesizer (e.g., Opus 4.7 vs current Gemini)
- **Regression detection** — flag synthesis output that fails any item; two consecutive failures triggers rollback consideration per ADR-01 amendment

## Origin

Established 2026-05-12 per AI Council debate on synthesizer/panel refresh
(transcript: `.dev-knowledge/docs/decisions/transcripts/council-out-2026-05-11-*-synthesizer-panel-*.md`).

## Operating principle

Negative answer to any item is the signal — investigate the failure mode (timeout, model regression, prompt issue) before flipping defaults or rolling back.
