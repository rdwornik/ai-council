# ADR-02: Default Panel Composition

**Date:** 2026-03-20
**Status:** Revised (2026-05-11)
**Decision:** Default 5-model panel: Claude + Gemini + OpenAI + Grok + DeepSeek.

**Context:**
Original panel was Claude + Gemini + DeepSeek. Switched to OpenAI as third member for stronger reasoning diversity.
DeepSeek retained as available model but not default — API reliability concerns and key availability.

Revised 2026-05-11: default changed to full 5-model panel. `--lite` flag selects 3-model panel (Claude + Gemini + OpenAI); `--full` is now a no-op kept for backward compatibility.

**Implementation:** `determine_panel()` in `src/runner.py`. The 5-model effective default is achieved via `cli.py`: when `--lite` is not passed, `use_full_panel or not lite` evaluates to `True`, selecting `full_panel` from `config/settings.yaml`. `default_panel` in config remains the 3-model lite set.

**Overrides:** `--models` flag > `--lite`/`--full` flags > default 5-model panel.
