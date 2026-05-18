# Codex Review â€” openai-research-migration

**Date:** 2026-05-18
**Branch:** `fix/openai-research-provider-migration`
**HEAD:** `c1c1212`
**Diff range:** `main..fix/openai-research-provider-migration`
**Codex version:** codex-cli 0.122.0
**Mode:** diff-review

---

## Focus

- Migrated openai_mini/openai_deep providers use gpt-5.4-mini / gpt-5.5 with web_search tool on Responses API
- Dropped background=True + poll loop vs sync responses.create path
- Annotation-based parsers in _extract_sources for web_search citation shape
- Single-shot retry on transient APIError/APITimeoutError
- Cost rates in settings.yaml (mini 0.75/4.50, deep 5.00/30.00 per 1M)
- openai_deep stays gated behind --deep (config default_providers excludes it)
- Test coverage in tests/test_research.py

---

## Findings
**Critical**
- `(none)`

**High**
- `High` â€” [openai_deep_research.py](/C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/research/providers/openai_deep_research.py:58) (same issue also in [openai_mini_research.py](/C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/research/providers/openai_mini_research.py:56)); what: the migrated sync `responses.create` path does not pass the configured `timeout_sec` into `AsyncOpenAI`; why: after removing background polling, the whole research run now lives inside one HTTP request, but the OpenAI Python client has its own default request timeout and retry behavior, so `settings.yaml` no longer actually controls long-run request lifetime, especially for `openai_deep`â€™s 1800s setting; fix direction: instantiate the client or per-request call with explicit `timeout=self._timeout_sec` and explicit `max_retries`, then keep `asyncio.wait_for` only as the outer cancellation guard.

**Medium**
- `Medium` â€” [openai_mini_research.py](/C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/research/providers/openai_mini_research.py:94) and [openai_deep_research.py](/C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/research/providers/openai_deep_research.py:96); what: the new single-shot retry wraps `APIError`/`APITimeoutError` even though the OpenAI SDK already retries those classes by default; why: a transient failure can now fan out into multiple full research attempts, which is hard to reason about for latency and can duplicate expensive web-search work/cost on timeout edges; fix direction: either remove `_call_with_retry` and rely on SDK retries, or set `max_retries=0/1` on the client and own one clearly budgeted retry policy with tests.

**Low**
- `Low` â€” [openai_deep_research.py](/C:/Users/1028120/Documents/Dev/ai-council/src/ai_council/research/providers/openai_deep_research.py:32); what: `reasoning_effort` is newly hard-coded in the provider implementation instead of coming from `settings.yaml`; why: this repoâ€™s config rule is that provider behavior knobs live in config, and this one now requires a code change to tune cost/latency or roll back from `high`; fix direction: add `reasoning_effort` to `ResearchProviderConfig`, wire it through `config_loader.py` and `research/runner.py`, and cover it in `tests/test_research.py`.

Reference: OpenAIâ€™s official Python SDK docs note the default client timeout and built-in retry behavior, and the web-search docs describe the sync `responses.create` shape for `web_search`: https://github.com/openai/openai-python https://developers.openai.com/api/docs/guides/tools-web-search
