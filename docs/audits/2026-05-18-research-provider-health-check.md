# Audit — Research-Provider Health Check

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — openai web_search migration + grok pin 0309; open: #54 (`"incomplete"`-terminal nuance). _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-05-18
**Branch:** `docs/research-provider-health-check`
**HEAD (audit-time):** `bba023c18fe4674468dcb2cf8a9151ef0078bcd3`
**Scope:** read-only diagnosis of the five research providers under `src/ai_council/research/providers/` plus shared runner/display/merger code. No `src/` changes. No live API calls.
**Trigger:** operator reports the OpenAI mini deep-research provider fails frequently; xAI deprecated the Live Search API on 2026-01-12, raising a concern that the Grok provider may be on a dead endpoint.

---

## Purpose and boundaries

This report is **evidence**. It identifies the most probable failure modes for two specific providers, lightly scans the other three, and answers a systemic question about how research-mode runs degrade when a provider fails. It does **not** patch anything. Fix scope and ordering are an operator decision.

Findings are marked **[verified]** when traceable to a quoted line of code in this repo, and **[inferred]** when they depend on external API behaviour or model availability that this audit could not exercise without paid calls.

---

## Per-provider status

| Provider | File | Configured model | Status | One-line reason |
|---|---|---|---|---|
| `perplexity` | `perplexity.py` | `sonar-pro` | **OK** | Chat completions on `api.perplexity.ai`; model and endpoint are current. |
| `openai_mini` | `openai_mini_research.py` | `o4-mini-deep-research` | **at-risk** | Uses deprecated `web_search_preview` tool name **[inferred]** and bare model alias **[inferred]**; no retry on transient failures **[verified]**. |
| `openai_deep` | `openai_deep_research.py` | `o3-deep-research` | **at-risk** | Background poll submits **with no search tool at all** — deep-research models require one (the mini provider passes `web_search_preview`) **[verified]**. Unused in default panel. |
| `gemini` | `gemini_research.py` | `deep-research-preview-04-2026` | **at-risk** | Configured agent ID differs from the only ID the google-genai SDK type hint knows (`deep-research-pro-preview-12-2025`) **[verified]**; whether the runtime accepts the new ID is **[inferred]**. |
| `grok` | `grok_research.py` | `grok-4.20-reasoning` | **at-risk** | Endpoint and tool shape are correct (Responses API + `x_search`/`web_search`) **[verified]** — **not** on the deprecated Live Search API. But the configured model string `grok-4.20-reasoning` is unusual; CLAUDE.md and the prompt expected `grok-3` **[verified mismatch; validity inferred]**. |

No provider is conclusively `broken` from code alone — all failures inferred here would manifest as runtime errors against the live APIs.

---

## Provider details

### `openai_mini` — most probable cause(s), ranked

**File:** `src/ai_council/research/providers/openai_mini_research.py`

Ranked from most to least likely, each tied to specific code:

**1. Tool name `web_search_preview` is the deprecated/renamed variant. [inferred]**

```python
# openai_mini_research.py:75-83
response = await client.responses.create(
    model=self._model,
    input=[
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ],
    tools=[{"type": "web_search_preview"}],
    background=True,
)
```

OpenAI's deep-research models *require* at least one search tool (the inline comment on line 74 notes this). The tool name `web_search_preview` was the early-access alias; the current production tool type on the Responses API is `web_search`. If OpenAI has retired the preview alias, every `responses.create` call from this provider returns a 400-class error and is surfaced by the `APIError` branch (line 63) as `[openai_mini] API error: ...`. This matches "fails frequently" — it would in fact fail every time the alias is rejected. Note the sibling `openai_deep` provider passes **no tools at all** (see that section); the two diverge here.

**2. Bare model alias `o4-mini-deep-research` may not resolve. [inferred]**

```python
# openai_mini_research.py:30
model: str = "o4-mini-deep-research",
```

The configured value (also in `settings.yaml:404`) is the unversioned alias. If OpenAI has rotated the canonical pin to a dated suffix (e.g. `o4-mini-deep-research-2025-06-26`) and dropped the bare alias, every call returns a model-not-found error. Less likely than the tool issue because aliases are normally kept; included because it would produce the same symptom.

**3. No retry on transient failures. [verified]**

```python
# openai_mini_research.py:57-64
except asyncio.TimeoutError as exc:
    raise ResearchProviderError("openai_mini", f"Timed out after {self._timeout_sec}s") from exc
except APITimeoutError as exc:
    raise ResearchProviderError("openai_mini", f"API timeout: {exc}") from exc
except APIError as exc:
    raise ResearchProviderError("openai_mini", f"API error: {exc}") from exc
```

A single 5xx, rate-limit, or transient network blip during the background poll loop (`responses.retrieve`, line 90) raises through the loop and fails the run. Deep-research jobs are long-running (timeout 1200s); the probability that *some* poll fails over a 20-minute window is non-trivial. This compounds whichever first-order issue is in play but is unlikely to be the sole cause of *frequent* failure.

**Not a likely cause** (ruled out from code):
- Sync-vs-async mismatch — the provider does use background + polling correctly (lines 75-92).
- Response parsing — the `_extract_content`/`_extract_sources` methods are defensive (use `getattr` throughout); a shape mismatch would produce empty content/no sources, not an exception. Empty content would still be merged silently (see "Systemic finding" below).
- Error surfacing — failures *are* logged (`display.py:120,126`) and shown in the status table (`display.py:75-80`); they are not swallowed at provider boundary.

**Recommended fix direction (operator decides):**
- Replace `web_search_preview` with `web_search`. Cross-check against current OpenAI Responses API docs.
- Consider pinning the model to a dated suffix once verified.
- Optionally wrap `responses.retrieve` poll in a small retry-on-transient.

---

### `grok` — diagnosis

**File:** `src/ai_council/research/providers/grok_research.py`

**Endpoint and tools are on the current API. [verified]** The provider posts to the Responses API (`client.responses.create`, line 58) at `https://api.x.ai/v1` (line 15) and passes server-side tools `x_search` and `web_search` (lines 64-67). This is the post-deprecation shape described in the prompt. The provider does **not** use the deprecated chat-completions `search_parameters` field. **The "deprecated Live Search API" concern does not apply to this code path.**

**However — the model string is suspicious. [verified mismatch; validity inferred]**

```python
# grok_research.py:32
model: str = "grok-4.20-reasoning",
```

```yaml
# config/settings.yaml:417-418
grok:
  model: "grok-4.20-reasoning"
```

`CLAUDE.md` (Research-providers table, line "grok ... grok-3") and the operator's prompt both expected `grok-3`. The value in code/config is `grok-4.20-reasoning`, which does not match any standard xAI model naming this audit can confirm (typical aliases are `grok-4`, `grok-4-fast-reasoning`, `grok-3`). If `grok-4.20-reasoning` is not a real model identifier, every call returns a model-not-found error.

Two possibilities, both possible from code alone:
- The model was renamed in advance of an xAI release and the rename has not landed (or has been rolled back) — calls would 404/400.
- The string is correct against an internal/preview channel — calls succeed.

This audit cannot distinguish without a live call.

**Recommended fix direction (operator decides):**
- Verify the configured model identifier against current xAI docs. If wrong, restore `grok-3` (or pin to a confirmed `grok-4-*` identifier).
- Reconcile `CLAUDE.md` provider table with whatever is chosen, so the documented and configured strings agree.

---

## Light scan — remaining three providers

### `perplexity` — OK

`api.perplexity.ai` chat-completions with `sonar-pro` is current. 60-second timeout, no retry, no streaming. Sources are read from `response.citations` as a flat URL list (lines 77-80) — matches Perplexity's documented schema. No issue flagged.

### `openai_deep` — at-risk

**`openai_deep_research.py:74-81`:**

```python
response = await client.responses.create(
    model=self._model,
    input=[
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ],
    background=True,
)
```

This provider passes **no `tools=` argument at all**. The sibling mini provider's comment on line 74 says "Deep research models require at least one search tool"; that requirement applies to `o3-deep-research` as well. **[inferred]** Every call from this provider should be rejected for missing a search tool. Symptom would mirror `openai_mini`'s tool-name issue: APIError on `responses.create`.

This provider is gated behind `--deep` and is rarely invoked, which is consistent with the "Known issues" line in `CLAUDE.md` ("o3-deep-research integration test not run — blocked $10+ per run"). Worth flagging because cost-blocked code paths drift undetected.

**Recommended fix direction:** add `tools=[{"type": "web_search"}]` (or whichever tool name `openai_mini` lands on) before `--deep` is exercised in production.

### `gemini` — at-risk

**`gemini_research.py:34` vs `config/settings.yaml:426`:**

- Code default: `agent: str = "deep-research-pro-preview-12-2025"`
- Configured (effective): `"deep-research-preview-04-2026"`

`CLAUDE.md` documents the SDK type hint as only recognising `deep-research-pro-preview-12-2025` (google-genai 1.73). The configured value is a *newer* agent ID. If the runtime accepts unknown agent IDs (type hint is non-enforcing), this works; if it validates against a known list, calls fail. **[inferred]** The provider's broad `except Exception` (line 68) wraps any failure as `[gemini] API error: ...`, so a runtime rejection would be reported but not distinguishable from other failures without inspection of the warning message.

The `_TERMINAL_STATUSES` set treats `"incomplete"` as terminal (line 24) and raises `ResearchProviderError` for any non-`completed` terminal (lines 104-107). That's correct behaviour but means an autonomous-agent run that ends "incomplete" — possibly a legitimate partial — counts as a hard failure.

**Recommended fix direction:** verify the configured agent ID against the current google-genai runtime; if it works, update the CLAUDE.md note documenting the SDK-vs-runtime gap.

---

## Systemic finding — loud or silent

**Q: When one research provider fails, does the run fail loud or continue silently?**

**A: It continues, but it is *not* silent.** Failures surface in three places:

1. **Status table during the run.** `display.py:74-80` renders each failed/timed-out provider with a red `✗ failed` or `✗ timeout` row showing the truncated error message. The user sees this live.
2. **Console summary.** `output.py:125-133` prints `Providers: N succeeded, M failed | ... sources total` and lists each failed provider with its error in red.
3. **Saved transcript.** `output.py:61` records per-provider `status` (`ok`/`error`/`timeout`) in the written `_research.md` file.

What it does **not** do: raise an exception or alter the exit code. The merger silently filters errored results out of the merged document — `merger.py:91` (`successful = [r for r in results if not r.error and r.content]`) — and the summarizer runs on whatever's left. If 4 of 5 providers fail, the run produces a one-provider report that on its face looks fine; the failures are visible only by reading the status block above the report.

**Single-provider hard-fail mode (separate concern):** `runner.py:178-182` raises `RuntimeError` **only if zero providers can be built** (e.g. all API keys missing). It does not raise if 4 of 4 built providers then fail at the API call — that case returns a near-empty merged report instead.

**Implication.** If `openai_mini` fails on every call (Tool 1) and `grok` fails on every call (Model 1), the default 4-provider research panel has been silently running on `perplexity` + `gemini` only — degraded by half, still producing a clean-looking report. The degradation is **logged and displayed**, not hidden, but there is no aggregate alarm (e.g. "≥ N failures" warning, no metric, no exit code change). Operators relying on the summary section without reading the status section would not notice.

---

## Note on next step

This report is **evidence**. It identifies most-probable causes for two providers and adds a third (`openai_deep`) and fourth (`gemini` agent ID) to the watch-list, plus a systemic observation about how degraded runs present.

Fix scope, ordering, and whether to bundle (e.g. a single Responses-API tool-name PR covering mini + deep) versus split — these are operator decisions. This audit does not specify them. A confirmed migration of any of the at-risk providers would be a separate change; an aggregate health alert (raise / exit-code / metric) when a configurable fraction of providers fail would be a separate change again.

Two items flagged here as evidence, not recommendations:
- **`CLAUDE.md` drift.** The provider table lists `grok-3` but `settings.yaml`/code use `grok-4.20-reasoning`. Whatever is correct, one of the two is wrong.
- **No live verification was performed.** Every `[inferred]` finding above can be turned into `[verified]` or `[ruled out]` with a single paid call per provider; this audit declined to incur that cost without instruction.
