# ADR-02: Default Panel Composition

**Date:** 2026-03-20
**Status:** Accepted
**Decision:** Default 3-model panel: Claude + Gemini + OpenAI.

**Context:**
Original panel was Claude + Gemini + DeepSeek. Switched to OpenAI as third member for stronger reasoning diversity.
DeepSeek retained as available model but not default — API reliability concerns and key availability.

**Implementation:** `determine_panel()` in `src/runner.py`; `default_panel` in `config/settings.yaml`.

**Overrides:** `--models` flag > `--full` flag (5-model) > default 3-model panel.
