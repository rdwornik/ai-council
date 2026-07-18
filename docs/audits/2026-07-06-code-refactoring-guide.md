# Technical Refactoring Guide — ai-council `src/` (the how, at code level)

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — A1 `fa4c1b5`, A3 `3583ae5`, A4/B3 `0ae7429` shipped; open: #45–#48 (A5/B2/B4/B7), #20. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-06 · **Companion to:** `docs/audits/2026-07-06-code-quality-audit.md` · **Model:** Fable · **Mode:** READ-ONLY (this document is the only artifact — **no code was changed**) · **Verified against:** live `main` @ `ee8a323`

> **Why this exists.** The code-quality audit answers *is the code clean?* at verdict altitude (SOLID / WORKABLE / REFACTOR-FIRST per surface). It is deliberately functional. **This document is the engineer's companion:** for each thing the audit flagged, the exact file, the exact current code, the exact target code, and the steps to get there. Every `Now`/`Target` block is faithful to live `main`.
>
> **Nothing here is applied.** These are concrete proposals. Each `R#` is implementable as its own `branch → --no-ff → push` session. `Maps to` ties each item back to the audit's seed (SEED-1..5) or Wave-C build. Ordering: **Part A** = structural (unblocks Wave-C); **Part B** = mechanical (fast, low-risk, do anytime).

## Index

| # | Refactor | File(s) | Effort | Risk | Maps to |
|---|---|---|---|---|---|
| **A1** | Template-method provider base | `providers/*.py` | M | Med (ADR-gated) | SEED-2 · CliProvider #16 |
| **A2** | Decompose `cli.py:main` | `cli.py` | L | Med | SEED-1 · D2 parity · doctor |
| **A3** | One error classifier + wire the dead policy + kill the `_config` reach-through | `debate.py` `policy.py` `providers/base.py` | M | Med | SEED-3 · CliProvider |
| **A4** | Decompose `save_to_file` | `output.py` | S | Low | SEED-4 · verdict package |
| **A5** | Break the `runner`↔`orchestrator` re-export | `runner.py` `orchestrator.py` `cli.py` | S | Low | SEED-5 |
| **B1** | Hoist the triplicated research-provider helpers | `research/providers/*.py` `research/provider.py` | M | Low | §2.4 audit |
| **B2** | `datetime.utcnow()` → tz-aware (×5) | `research/providers/*.py` | XS | Low | §2.9 · known sweep |
| **B3** | Naive `datetime.now()` timestamps (×7) | `output.py` `inbox.py` `research/output.py` | S | Low | §2.9 |
| **B4** | Remove genuinely dead code | `routing.py` `research/display.py` `research/models.py` | XS | Low | §2.5 |
| **B5** | Clear the ~13 non-#20 `mypy --strict` errors | 8 files | S | Low | §2.8 |
| **B6** | Isolate the W1 flake (real `~/Downloads` scan) | `tests/test_research.py` | S | Low | §6 · W1 |
| **B7** | Load `RunPolicy` from `settings.yaml` | `policy.py` `config/` `settings.yaml` `cli.py` | S | Low | §2.10 |

**Reference module — do not touch:** `healthcheck.py` (68 LOC, MI 79.51, the highest in the repo) is the pattern the rest should emulate — one job, parallel `gather`, a clean `dict[name → (ok, msg)]` contract, `classify_error` reused not reinvented. Its only shared dependency is the implicit `provider._config.timeout_sec` probe at `healthcheck.py:34`, which **A3** formalizes. Leave the file as-is; copy its shape.

---

# Part A — Structural refactors

## A1 · Template-method base for the provider families

**Files:** `providers/base.py` (+ all five `providers/*.py`) · **Effort:** M · **Risk:** Medium — sits on the "keep providers separate" ADR, so the **architect ratifies before build** (SEED-2). A shared base is *not* a provider merge: each provider stays its own class in its own file.

**Symptom.** ~85% of each debate provider is copy-paste. The timing, timeout guard, two error wrappers, empty-check, logging, and 9-field `ModelResponse` construction are identical across all five; only the SDK call and the token/text parse differ. `xai.py` and `deepseek.py` differ by 12 non-logic lines.

**Now** — `xai.py:35-84` (the four siblings are identical modulo the call + parse):

```python
async def generate(self, prompt: str, round_number: int) -> ModelResponse:
    start = time.monotonic()
    try:
        response = await asyncio.wait_for(
            self._client.chat.completions.create(                      # <-- the ONLY per-provider line pair
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens,
            ),
            timeout=self._config.timeout_sec,
        )
    except TimeoutError as exc:
        raise ProviderError(self._config.name, f"Request timed out after {self._config.timeout_sec}s") from exc
    except Exception as exc:
        raise ProviderError(self._config.name, f"API call failed: {exc}") from exc
    latency = time.monotonic() - start
    choice = response.choices[0] if response.choices else None          # <-- and the parse block
    if not choice or not choice.message.content:
        raise ProviderError(self._config.name, "Empty response content")
    input_tokens = output_tokens = token_count = None
    if response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        token_count = response.usage.total_tokens
    logger.info("xAI round %d: %.2fs, %s tokens", round_number, latency, token_count)
    return ModelResponse(provider=self._config.name, model=self._config.model, round_number=round_number,
                         content=choice.message.content, latency_sec=latency, token_count=token_count,
                         input_tokens=input_tokens, output_tokens=output_tokens)
```

**Target** — the skeleton moves into `AIProvider`; each provider implements two small hooks. `base.py`:

```python
@dataclass
class _Parsed:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_count: int | None = None

class AIProvider(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._api_key = os.environ.get(config.api_key_env, "").strip()
        if not self._api_key:
            raise ProviderError(config.name, f"Missing API key: {config.api_key_env}")
        self._configure()                       # subclass builds its client here

    def name(self) -> str:            return self._config.name          # concrete now — kills the dead abstract
    def model_string(self) -> str:    return self._config.model         # model_string (audit §2.5)

    async def generate(self, prompt: str, round_number: int, *, timeout: float | None = None) -> ModelResponse:
        start = time.monotonic()
        try:
            raw = await asyncio.wait_for(self._invoke(prompt), timeout=timeout or self._config.timeout_sec)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise ProviderError(self._config.name, f"Request timed out after {timeout or self._config.timeout_sec}s") from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self._config.name, f"API call failed: {exc}") from exc
        parsed = self._parse(raw)
        if not parsed.content:
            raise ProviderError(self._config.name, "Empty response content")
        latency = time.monotonic() - start
        logger.info("%s round %d: %.2fs, %s tokens", self._config.name, round_number, latency, parsed.token_count)
        return ModelResponse(provider=self._config.name, model=self._config.model, round_number=round_number,
                             content=parsed.content, latency_sec=latency, token_count=parsed.token_count,
                             input_tokens=parsed.input_tokens, output_tokens=parsed.output_tokens)

    def _configure(self) -> None: ...                    # optional override (xai/deepseek need base_url)
    @abstractmethod
    async def _invoke(self, prompt: str): ...            # the SDK call
    @abstractmethod
    def _parse(self, raw) -> _Parsed: ...                # SDK response -> _Parsed
```

`xai.py` collapses to ~18 lines:

```python
class XAIProvider(AIProvider):
    def _configure(self) -> None:
        if not self._config.base_url:
            raise ProviderError(self._config.name, "base_url is required for xAI provider")
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._config.base_url)

    async def _invoke(self, prompt: str):
        return await self._client.chat.completions.create(
            model=self._config.model, messages=[{"role": "user", "content": prompt}],
            max_tokens=self._config.max_tokens)

    def _parse(self, raw) -> _Parsed:
        choice = raw.choices[0] if raw.choices else None
        if not choice or not choice.message.content:
            return _Parsed("")
        u = raw.usage
        return _Parsed(choice.message.content, u.prompt_tokens if u else None,
                       u.completion_tokens if u else None, u.total_tokens if u else None)
```

`anthropic._parse` keeps its genuinely-divergent text-block reassembly (`anthropic.py:57-61`); `gemini` keeps its per-call client (`_invoke` builds `genai.Client` each call — the event-loop rationale survives) and its token-split derivation. **CliProvider v1 (#16) then implements only `_configure`/`_invoke`/`_parse`** — no copy of the skeleton.

**Steps.** 1) Add `_Parsed` + the template `generate`/`__init__` to `base.py`; keep the old `generate` abstract signature satisfied. 2) Convert one provider (openai — the de-facto core) and run `pytest tests/test_providers.py`. 3) Convert the other four. 4) Delete the now-duplicate `name`/`model_string`/`__init__` bodies.

**Done-when.** A new OpenAI-compatible provider is < 20 LOC; adding a field to `ModelResponse` touches exactly one file (`base.py`); `radon raw` SLOC of `providers/` drops ≥ 40%; `pytest tests/test_providers.py` green unchanged.

---

## A2 · Decompose `cli.py:main` (277 LOC / F-72)

**Files:** `cli.py` · **Effort:** L · **Risk:** Medium (it is the entry point) · **Maps to:** SEED-1; unblocks D2 parity (R1) and doctor v1.

**Symptom.** `main` (`cli.py:359-635`) inlines two near-parallel request-building paths — the inbox loop (`445-562`) and the single-question path (`565-635`) — which have **already drifted**: inbox parses frontmatter (`parse_file`, `:468`), the single `--file` path does a raw `read_text` (`:565-566`). That drift *is* the known D2 `--file` gap. Fix the structure and the gap closes for free.

**Now** — the two paths duplicate this shape:

```python
# inbox path (cli.py:466-549), per file:
question_text, meta = parse_file(file_path, resolver=resolver)          # parses frontmatter
fm_synthesizer = synthesizer if synthesizer is not None else meta.get("synthesizer", ...) or default
fm_mode = resolve_mode(mode_arg, ...) if mode_arg else resolve_mode(meta["mode"], ...) if "mode" in meta else default
... if fm_mode == "research": asyncio.run(run_research(query=question_text, ...))   # research branch (500-514)
... else: request = RunRequest(...); asyncio.run(runner.run(request, ...))          # debate branch (537-551)

# single path (cli.py:565-635):
if question_file:  question_text = Path(question_file).read_text(...).strip()        # <-- RAW read, no parse_file (D2 gap)
effective_mode = resolve_mode(mode_arg, ...) if mode_arg else detect_mode(...)       # different resolution
if effective_mode == "research": asyncio.run(run_research(query=question_text, ...)) # research branch (596-608) — dup
request = RunRequest(...); asyncio.run(runner.run(request, ...))                      # debate branch (623-635) — dup
```

**Target** — three extracted functions, both paths share them:

```python
def load_question(source: Path | str, resolver: TargetResolver) -> tuple[str, dict]:
    """One frontmatter-aware reader for BOTH --file and inbox. Closes the D2 --file gap."""
    return parse_file(Path(source), resolver=resolver)   # --file now parses too; inline arg -> ("", {})

def build_request(question_text: str, source: str, meta: dict, config: AppConfig,
                  cli: CliOverrides, mode: str) -> RunRequest:
    """The single precedence resolver: CLI flag > frontmatter > config default. Used by both paths."""
    ...

async def dispatch(question_text, meta, config, cli, runner, mode) -> int:
    """Research-vs-debate branch, in ONE place. Returns the exit code (0/3)."""
    if mode == "research":
        report = await run_research(query=question_text, config=config, return_dir=cli.return_dir, ...)  # +return_dir (D2 #2)
        return 3 if report and report.degraded else 0
    await runner.run(build_request(question_text, ..., mode), ...)
    return 0
```

`main` becomes: parse args → setup (dotenv/logging/config/resolver/providers/health) → build the work-list (inbox files **or** `[single]`) → `for item: dispatch(...)` → aggregate exit code. For **doctor v1**, promote `main` to a `@click.group` with `run` (current behavior) and `doctor` subcommands.

**Steps.** 1) Extract `build_request` first (pure, unit-testable) and route both existing paths through it — behavior-preserving. 2) Extract `dispatch`; collapse the two research call-sites into one (thread `return_dir` here → D2 #2). 3) Route `--file` through `load_question`/`parse_file` → D2 #1. 4) Only then convert to `@click.group` for doctor.

**Done-when.** `radon cc` rank of `main` ≤ C (≤ 15); exactly **one** `run_research(` call-site and **one** `parse_file(` call-site feed request-building (grep); a parity test asserts identical `RunRequest` fields from `--file X` and an inbox `X` with the same frontmatter.

---

## A3 · One error classifier · wire the dead policy · kill the `_config` reach-through

**Files:** `debate.py` `policy.py` `providers/base.py` · **Effort:** M · **Risk:** Medium · **Maps to:** SEED-3; the ABC contract CliProvider inherits.

**Symptom (three coupled defects).**
1. **Two classifiers.** The live retry decision uses `policy.should_retry()` (raw substring match, `policy.py:35-41`) via `debate.py:50`. A richer, canonical `classify_error()` → `is_retryable()` (9 categories, `base.py:16-69`) exists, is used by `healthcheck.py:52` + `synthesis.py:101`, but **`is_retryable` is never called in the retry path**.
2. **Dead knob.** `max_retries_per_provider` (`policy.py:12`) is never read — `_call_provider` hardcodes exactly one retry (`debate.py:47-94`). `should_abort` (`policy.py:23`) is never called either.
3. **Reach-through.** The retry mutates a private attribute in place:

**Now** — `debate.py:50-56, 92-94`:

```python
if policy.should_retry(str(exc)):                              # (1) second classifier
    cfg = getattr(provider, "_config", None)                  # (3) reaches through the ABC
    if cfg is not None and hasattr(cfg, "timeout_sec"):
        original_timeout = cfg.timeout_sec
        cfg.timeout_sec = int(original_timeout * 1.5)          # (3) mutates shared provider state
        ...
    finally:
        if cfg is not None and original_timeout is not None:
            cfg.timeout_sec = original_timeout                 # (3) ... and restores it
```

**Target** — one classifier, a real retry loop, no mutation (uses A1's `timeout=` kwarg):

```python
async def _call_provider(provider, prompt, round_number, policy) -> ModelResponse | ProviderError:
    last: ProviderError | None = None
    for attempt in range(policy.max_retries_per_provider + 1):     # (2) knob now honored
        timeout = provider._config.timeout_sec * (1.5 ** attempt)  # grows per attempt, no mutation
        try:
            result = await provider.generate(prompt, round_number, timeout=timeout)  # (3) A1's kwarg
            result.was_retry = attempt > 0
            return result
        except ProviderError as exc:
            last = exc
            if not is_retryable(classify_error(exc)):              # (1) THE canonical classifier
                break
            logger.warning("Provider %s attempt %d failed: %s", provider.name(), attempt + 1, exc)
    return last
```

Then **delete** `policy.should_retry`, `policy.retryable_errors`, `policy.non_retryable_errors` (superseded by `base.classify_error`/`is_retryable`), and **wire** `should_abort` into `run_debate` (`debate.py:253-280`) — replace the ad-hoc `_MIN_QUALITY_RESPONSES` raise with `if policy.should_abort(len(responses), round_num): ...`.

**Done-when.** `grep should_retry src/` = 0; `vulture` reports no unused symbol in `policy.py`/`base.py`; no `getattr(provider, "_config"` in `debate.py`; `pytest tests/test_debate.py tests/test_base_provider.py tests/test_policy.py` green (the retry/classify tests will need their assertions retargeted to `classify_error`).

---

## A4 · Decompose `save_to_file` (E-33, 138 LOC)

**Files:** `output.py` · **Effort:** S · **Risk:** Low · **Maps to:** SEED-4; makes room for the verdict package (DRAFT-INT-1).

**Symptom.** `save_to_file` (`output.py:179-316`) mixes four jobs: filename derivation, header-metadata assembly (`:196-271`), round-by-round body assembly (`:282-306`), and routing + metrics trigger. The verdict package adds a fifth emission on top.

**Target** — two pure builders; `save_to_file` shrinks to orchestration:

```python
def _build_header(result: DebateResult) -> list[str]:
    """All the **Panel/Synthesizer/Rounds/Cost/Status/Provider-Notes** lines (was :196-271)."""
    ...

def _build_body(result: DebateResult) -> list[str]:
    """The '## Round N' transcript + '## Synthesis' block (was :282-306)."""
    ...

def save_to_file(result, output_dir, *, slug_override=None, secondary_dir=None,
                 target_paths=None, return_dir=None) -> list[Path]:
    filename = f"council-out-{_ts()}-{result.mode}-{slug_override or _slug(result.question.text)}.md"
    content = "\n".join([f"# AI Council Debate: {result.question.text[:80]}", "",
                         *_build_header(result), "", "---", "", *_build_body(result)])
    saved = _write_routed(content, filename, output_dir, secondary_dir, target_paths, return_dir)
    if result.metrics:
        _save_metrics_json(result, saved[0])
    return saved

def save_verdict_package(result, output_dir, **routes) -> list[Path]:   # verdict package = a SIBLING
    """DRAFT-INT-1: consumes seats[]/synthesis, emits via _write_routed. NOT more lines in save_to_file."""
    ...
```

**Done-when.** `radon cc` rank of `save_to_file` ≤ B; `_write_routed` remains the single write primitive (one definition); the verdict emitter is a separate function; `pytest tests/test_output.py tests/test_dual_output.py` green unchanged.

---

## A5 · Break the `runner`↔`orchestrator` re-export shim

**Files:** `runner.py` `orchestrator.py` `cli.py` · **Effort:** S · **Risk:** Low · **Maps to:** SEED-5.

**Symptom.** `runner.py:78-79` re-exports `CouncilRunner` from `orchestrator` (`# noqa: E402, F401`) while `orchestrator.py:20` imports two utilities *from* `runner` — a soft cycle. `cli.py:26` still imports `CouncilRunner` through the deprecated path.

**Now:**
```python
# runner.py:78-79
# Backward-compat re-export — new code should import from ai_council.orchestrator directly
from ai_council.orchestrator import CouncilRunner as CouncilRunner  # noqa: E402, F401
# cli.py:26
from ai_council.runner import CouncilRunner, build_all_providers, determine_panel
```

**Target:**
```python
# runner.py — delete lines 78-79 entirely. runner.py holds ONLY the four panel utilities.
# cli.py:26 — split the import by true origin:
from ai_council.orchestrator import CouncilRunner
from ai_council.runner import build_all_providers, determine_panel
```

**Done-when.** `grep "from ai_council.orchestrator" src/ai_council/runner.py` = 0; `CouncilRunner` has exactly one definition source (grep); an `import-linter` layered contract (`cli → orchestrator → {debate,runner,output,synthesis}`) passes; full suite green.

---

# Part B — Mechanical fixes (fast, low-risk)

## B1 · Hoist the triplicated research-provider helpers

**Files:** `research/provider.py` (+ `grok_research.py`, `openai_deep_research.py`, `openai_mini_research.py`) · **Effort:** M · **Risk:** Low.

**Symptom.** `_extract_content` (grok `:105-125` ≡ mini `:111-131`, byte-identical), `_collect_annotations` (grok `:146-156` ≡ mini `:151-161`, byte-identical), and `_extract_sources` (one-line difference: grok skips `("x_search_call","web_search_call")`, mini skips `"web_search_call"`) are copied verbatim across the three Responses-API providers — ~150 LOC. Plus `datetime.utcnow()` and the `1_000_000` cost formula pasted into all five.

**Target** — an intermediate base the three OpenAI-shaped providers extend; `perplexity` (chat.completions) and `gemini_research` (Interactions API) stay direct `ResearchProvider`:

```python
# research/provider.py
class ResponsesAPIResearchProvider(ResearchProvider):
    _SKIP_ITEM_TYPES: tuple[str, ...] = ("web_search_call",)     # grok overrides -> (+ "x_search_call")

    def _extract_content(self, response) -> str: ...             # the one shared copy
    def _extract_sources(self, response) -> list[Source]:        # parametrized by _SKIP_ITEM_TYPES
        ...
    def _collect_annotations(self, annotations, sources, seen) -> None: ...

    def _finish(self, response, query, start, timestamp) -> ResearchResult:
        """Shared token/cost/ResearchResult construction (was duplicated 5x)."""
        in_t  = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
        out_t = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
        cost = in_t / 1_000_000 * self._cost_per_1m_input + out_t / 1_000_000 * self._cost_per_1m_output
        return ResearchResult(provider=self.name(), query=query, content=self._extract_content(response),
                              sources=self._extract_sources(response), token_count=in_t + out_t,
                              cost_usd=cost, duration_sec=time.monotonic() - start, timestamp=timestamp)
```

`grok_research` keeps only `name`, `model_string`, the `_SYSTEM_PROMPT`, `_SKIP_ITEM_TYPES = ("x_search_call","web_search_call")`, and the `responses.create(... tools=[x_search, web_search])` call — everything else is inherited.

**Done-when.** `_extract_content`/`_collect_annotations` each have **one** definition (grep); `research/providers/` SLOC drops ≥ 30%; `pytest tests/test_research.py` green unchanged. *(B2 then becomes a one-line change on the base.)*

## B2 · `datetime.utcnow()` → tz-aware (×5)

**Files:** all five `research/providers/*.py` · **Effort:** XS · **Risk:** Low · known deferred sweep (audit §2.9 — note: **undocumented anywhere but the code**; consider filing it).

`datetime.utcnow()` is deprecated for removal (repo pins Python ≥ 3.12; the warning fires live in every test run). Fix each site — `grok_research.py:53`, `gemini_research.py:55`, `openai_deep_research.py:56`, `openai_mini_research.py:54`, `perplexity.py:48`:

```python
- timestamp = datetime.utcnow().isoformat()
+ timestamp = datetime.now(timezone.utc).isoformat()      # + `from datetime import timezone`
```

If **B1** lands first, this is a single line in `ResponsesAPIResearchProvider._finish` (+ perplexity/gemini). **Done-when.** `grep utcnow src/` = 0; zero `DeprecationWarning` in `pytest -W error::DeprecationWarning tests/test_research.py`.

## B3 · Naive `datetime.now()` timestamps (×7)

**Files:** `output.py:196,239,421,429` · `inbox.py:153` · `research/output.py:39,47` · **Effort:** S · **Risk:** Low.

These build filenames/headers with wall-clock-local `datetime.now()` (no tz). Consistency-only — the values are display/filename strings, not compared. Centralize into one helper and make intent explicit:

```python
def _ts(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(fmt)      # keep local-time intent, but ONE place, documented
```

**Done-when.** One timestamp helper per module boundary; `git grep 'datetime.now().strftime'` count drops to the helper definitions.

## B4 · Remove genuinely dead code

**Files/lines** (all cross-checked: no production reader) · **Effort:** XS · **Risk:** Low — **confirm no external importer before deleting** (per repo rule: ask before removing).

| Symbol | Site | Action |
|---|---|---|
| `_target_projects` | `routing.py:26` (`self._target_projects = list(target_projects)`) | Assigned, never read → delete the assignment (the ctor param is used directly). |
| `done_tasks` | `research/display.py:135,149` | Unused local → delete the two assignments. |
| `snippet` | `research/models.py:12` | Unused field/var → remove or wire. |
| `model_string`, `should_abort`, `is_retryable`, `max_retries_per_provider` | see A1/A3 | **Not deletions** — A1 makes `model_string` concrete; A3 *wires* the policy trio. |
| `cache_invalidate` | `research/cache.py:133` | Public util, tests-only — **keep** (library surface). |

**Done-when.** `vulture src/ai_council/ --min-confidence 80` returns only intentional API surface; full suite green.

## B5 · Clear the ~13 non-#20 `mypy --strict` errors

**Files:** `runner.py` `cli.py` `output.py` `inbox.py` `research/{cache,merger,display,runner}.py` `orchestrator.py` · **Effort:** S · **Risk:** Low. *(The #20 six SDK-typing errors in the three research providers are out of scope — BACKLOG #20.)*

Two mechanical classes:

| Error | Sites | Fix |
|---|---|---|
| `type-arg` (bare `dict`/`list`/`Task`) | `runner.py:15` `cli.py:169` `output.py:88` `inbox.py:115` `cache.py:107` `merger.py:138,170` `display.py:135` | add params, e.g. `provider_classes: dict` → `dict[str, type[AIProvider]]`; `modes: dict` → `dict[str, ModeConfig]`; `by_round: dict[int, list]` → `dict[int, list[ProviderCallMetrics]]` |
| `no-untyped-def` | `orchestrator.py:34,115` `research/runner.py:68` | annotate params, e.g. `run(self, request, output_dir=None, ...)` → `output_dir: Path \| None = None`; the `on_round_complete(rnd)` inner → `rnd: Round` |

**Done-when.** `mypy --strict src/ai_council/` reports only the six `#20` research-SDK errors; no `type-arg`/`no-untyped-def` remain.

## B6 · Isolate the W1 flake — it scans the real `~/Downloads`

**Files:** `tests/test_research.py:1632` (+ sibling `:1692`) · **Effort:** S · **Risk:** Low.

**Symptom (verified, audit §6).** `test_inbox_exits_3_when_any_batch_run_degraded` invokes real `cli.main` but patches only `run_research` + `_check_and_filter_providers`. It never disables `scan_downloads` (`settings.yaml:31 = true`) or overrides `downloads_dir` (`~/Downloads`), and `load_config` is uncached — so the run scans the operator's real Downloads. Any council-tagged `.md` there makes `call_count > 2` → the assertion fails; processed files get archived (moved). That is the n=1 order-dependence *and* a live side effect.

**Target** — make the test hermetic and drop the count coupling:

```python
with patch("ai_council.research.runner.run_research", side_effect=fake_run_research), \
     patch("ai_council.cli.run_research", side_effect=fake_run_research, create=True), \
     patch("ai_council.cli._check_and_filter_providers", side_effect=lambda p: p), \
     patch("ai_council.cli.scan_downloads_folder", return_value=[]):        # <-- isolate the Downloads seam
    result = CliRunner().invoke(cli_root, ["--inbox", "--inbox-dir", str(inbox_dir),
                                           "--output", str(output_dir), "--skip-health-check"])
assert result.exit_code == 3                                                # drop the brittle call_count == 2
```

Apply the same `scan_downloads_folder` patch to `test_inbox_exits_0_when_all_batch_runs_healthy:1692`.

**Done-when.** Both inbox tests pass in isolation **and** under `pytest -p no:randomly` **and** `pytest -p randomly` across 5 seeds; neither reads `Path.home()/"Downloads"` (assert via a `patch` that raises if called with the real path).

## B7 · Load `RunPolicy` from `settings.yaml`

**Files:** `policy.py` `config/config_loader.py` `settings.yaml` `cli.py` · **Effort:** S · **Risk:** Low · **Maps to:** audit §2.10 (config discipline).

**Symptom.** `RunPolicy` is always `.default()` (`cli.py:443`, `debate.py:214`) — its thresholds are code-only and unconfigurable, and `_MIN_QUALITY_RESPONSES = 3` (`debate.py:16`) duplicates the intent of `settings.yaml`'s research `min_successful_providers: 3` in a different home.

**Target** — a `policy:` block + a loader:

```yaml
# settings.yaml
policy:
  min_panel_size: 2
  abort_if_round1_below: 2
  max_retries_per_provider: 1
  min_quality_responses: 3        # absorbs debate.py's _MIN_QUALITY_RESPONSES
```
```python
# policy.py
@classmethod
def from_config(cls, raw: dict | None) -> "RunPolicy":
    return cls(**raw) if raw else cls()
# cli.py:443
policy = RunPolicy.from_config(config.policy)
```

**Done-when.** `grep 'RunPolicy.default()' src/` = 0 outside the fallback; `_MIN_QUALITY_RESPONSES` removed from `debate.py`; a test sets `policy.max_retries_per_provider` via YAML and observes the retry count change.

---

## Suggested sequencing

**Do-anytime (no dependencies, land first for quick wins):** B2, B3, B4, B5, B6, A5.
**Before / with CliProvider v1 (#16):** A1 (template base) → A3 (the ABC's error+timeout contract CliProvider inherits).
**Before / with D2 parity + doctor:** A2 (the `--file` gap and the `@click.group` both fall out of it).
**With the verdict package (DRAFT-INT-1):** A4.
**Independent cleanup:** B1 (then B2 collapses to one line), B7.

**A1, A2, and A3 are the load-bearing three** — they remove ~40% of provider SLOC, collapse the F-72 entry point, and give the provider ABC an honest contract, which is exactly the ground CliProvider / D2 / doctor build on. Everything in Part B is a safe afternoon each.

*Nothing in this document has been applied. Each `R#` is its own `branch → --no-ff → push` session.*
