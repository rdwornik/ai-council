# ai-council — Fleet Recon, Liveness, Audit Reconciliation & Process Design

**Date:** 2026-07-05 · **Mode:** Recon + synthesis — repo read-only, zero commits (this file is the only repo artifact; left UNCOMMITTED for operator review)
**Auditor:** Fable 5 · **Operator approval:** probe matrix approved with 4 amendments (canary-prompt fix, non-default pins, GM-1 included, verbatim settings.yaml pins — all applied)
**Evidence base:** live probes on THIS machine (probe IDs `CL/CX/AG/GR/DC/GM-*`, raw captures in scratch — see §9 manifest), live API calls (all keys from `~/Documents/.secrets/.env`, names only), live repo/hub file reads. **Zero press/browser claims presented as fact** — every fleet cell is `witnessed(probe-id)` or `SKIPPED(reason)`.
**Primary reference reconciled:** `docs/audits/2026-07-04-fable-architecture-audit.md` (Fable audit) + the architect's 2026-07-05 browser research (operator-supplied, re-verified below).

---

## 1. Executive synthesis

**The fleet is real but the browser research got the load-bearing details wrong in both directions.** All five CLIs exist and respond. Two are council-seat-grade today: **codex** (0.141.0 — headless JSONL, `--sandbox read-only`, model pin honored, ChatGPT-subscription lane, identity in the plain-mode banner) and **claude** (2.1.200 — headless JSON, pin honored, identity machine-readable in `.modelUsage`, subscription lane via the key-strip guard). **grok** (0.2.82) is mechanically seat-grade (headless JSON, pin honored, identity recoverable from session records) but is **currently API-billed** — `grok models` prints "You are using XAI_API_KEY"; no subscription OAuth is active on this machine, so the "grok = idle subscription quota" gradient premise is false as configured. **agy** (1.0.16) is disqualified by witnessed identity roulette: the `--model` flag exists (press said it didn't) but a pin is **silently rewritten to default** ("Model ID … not in local config, defaulting to CCPA") with contradictory log lines and "not logged into Antigravity" errors while still serving. **deepcode** (0.1.33) has **no headless mode at all** ("requires an interactive terminal (TTY)") — the DeepSeek CLI seat question is moot, not merely API-billed. Legacy `gemini` is dead for consumer tiers (witnessed `IneligibleTierError` pointing at Antigravity), confirming the shutdown claim.

**Liveness is green across the board:** 6/6 keys present, 9/9 provider pings PASS (Anthropic credit state healthy), and — against the #17 suspicion — **all six debate-seat model pins are current and listed live**, including `deepseek-v4-pro`. The single stale pin is research `grok-4.20-reasoning` (serves via unlisted alias; listed successor `grok-4.20-0309-reasoning`).

**The Fable audit survives contact with reality almost intact:** all 14 decisions and all 5 corrections HOLD, with one embedded factual premise INVALIDATED — ADR-12's "xai/deepseek stay API-only (no CLI exists)". Both CLIs exist; the conclusion survives for deepseek (no headless — witnessed, different reason) and must be revised for grok (CLI exists and pins honorably; the blocker is billing lane, not capability).

**Two safety facts every future CliProvider must inherit:** `claude -p --tools ""` **still ingests cwd `CLAUDE.md`** (CL-3: "Hello! CANARY" — the architect predicted otherwise; empty-tools ≠ context isolation; scratch-cwd is the only real defense, and grok ingests too per GR-3), and `codex exec` **hangs on open stdin** (must be launched with stdin closed).

---

## 2. Witnessed fleet matrix

Prompt for all headless probes: `Reply with exactly: OK` (canary probes: `Reply with a one-word greeting.` per amendment 2). All probes ran with cwd in the scratch dir, never a repo.

| Capability | **claude** 2.1.200 | **codex** 0.141.0 | **agy** 1.0.16 | **grok** 0.2.82 | **deepcode** 0.1.33 | **gemini** 0.49.0 (legacy; recon-only) |
|---|---|---|---|---|---|---|
| Binary | `claude.cmd` (npm shim; emits harmless `'m' is not recognized` stderr noise) — witnessed(version) | `codex.ps1` (npm) — witnessed(version) | `agy.exe` (Go, `AppData\Local\agy\bin`) — witnessed(version) | `grok.exe` (`~\.grok\bin`) — witnessed(version) | `deepcode.ps1` (npm) — witnessed(version); **`deepseek` binary NOT FOUND** (DeepSeek-TUI absent) | `gemini.ps1` (npm) — witnessed(version) |
| Working headless call | `claude -p "<q>" --output-format json --tools ""` — witnessed(CL-1), exit 0 | `codex exec --sandbox read-only --skip-git-repo-check --json "<q>"` **with stdin closed** — witnessed(CX-1); open stdin hangs on "Reading additional input from stdin..." (witnessed, pre-CX-1) | `agy -p "<q>"` — witnessed(AG-1), exit 0 | `grok -p "<q>" --output-format json` — witnessed(GR-1), exit 0 | **NONE** — `deepcode -p` exits 1: "deepcode requires an interactive terminal (TTY)" — witnessed(DC-1) | `gemini -p "<q>" -o json` → exit 1 `IneligibleTierError` (consumer tier shut down; told to migrate to Antigravity) — witnessed(GM-1) |
| Output format / answer field | JSON; answer in `.result`; rich usage + cost fields — witnessed(CL-1) | JSONL events; answer in `item.completed → item.type=="agent_message" → .text` — witnessed(CX-1) | Plain text only; **no JSON flag exists** (help surface) — witnessed(AG-1/help) | JSON; answer in `.text`, plus `sessionId`, `stopReason`; reasoning models add `.thought` — witnessed(GR-1/GR-2) | n/a | n/a (auth-dead) |
| Model pin | `--model` honored — witnessed(CL-2): `--model haiku` → `modelUsage` key `claude-haiku-4-5-20251001` | `-m` honored — witnessed(CX-2): `-m gpt-5.4` (non-default; config default is `gpt-5.5`) echoed in banner | `--model` flag EXISTS but honor **REFUTED** — witnessed(AG-2): log `Model ID gemini-3.1-pro-preview not in local config, defaulting to CCPA` + `Model resolved via default`, while another line claims `model="gemini-3.1-pro-preview"` — silent swap, contradictory logging | `-m` honored — witnessed(GR-2): `-m grok-build-0.1` (non-default) → session record `"model_id":"grok-build-0.1"`; default is `grok-4.20-0309-non-reasoning` (witnessed via `grok models` + GR-1 session record) — **not** grok-build-0.1 as press implied | n/a — SKIPPED(no-headless-mode) | n/a |
| **Actual model identity readout** | **In-band:** `.modelUsage` JSON key — witnessed(CL-1: default `claude-opus-4-8`; CL-2: haiku) | **Out-of-band:** plain-mode stderr banner (`model: gpt-5.4`, `provider`, `sandbox`) — witnessed(CX-2). `--json` events carry **NO identity** — witnessed(CX-1) | **NONE** — no channel in output; log file (via `--log-file`) shows only the contradictory resolver lines — witnessed(AG-2) | **Out-of-band:** stdout gives `sessionId`; `~/.grok/sessions/<cwd-enc>/<sessionId>/events.jsonl` records authoritative `"model_id"` — witnessed(GR-1/GR-2) | n/a | n/a |
| Sandbox / read-only | `--tools ""` accepted — witnessed(CL-1). **NOT context isolation** (see canary) | `--sandbox read-only` accepted + echoed in banner — witnessed(CX-1/CX-2) | `--sandbox` flag exists (help) — SKIPPED(call-cap, unexercised) | `--sandbox <PROFILE>` + permission modes exist (help) — SKIPPED(call-cap, unexercised) | n/a | n/a |
| Auth / cost lane | Subscription OAuth (settings.json empty-key guard + shim strip active; probes ran with key stripped) — witnessed(env+CL-1 success) | **ChatGPT subscription** — witnessed(`codex login status` → "Logged in using ChatGPT") | Inconsistent: serves prompts while logging `You are not logged into Antigravity` / token-source failures on the models poll — witnessed(AG-2 log). Quota state invisible | **`XAI_API_KEY` (API-billed)** — witnessed(`grok models` header "You are using XAI_API_KEY"); `~/.grok/auth.json` absent — witnessed(Test-Path). Subscription lane would require `grok login` + shielding the env key (same trap class as the documented claude gotcha) | Unconfigured — no `~/.deepcode/settings.json` — witnessed(Test-Path); would bill `DEEPSEEK_API_KEY` (no subscription lane exists) | OAuth individual tier refused; **did not fall through to the present `GEMINI_API_KEY`** — API-key survival path SKIPPED(call-cap, needs auth-mode config) |
| cwd context ingestion (canary) | **INGESTS** — witnessed(CL-3): reply "Hello! CANARY" despite `--tools ""` | No ingestion observed (reply "Hello") — **residual false-negative risk**, not witnessed(clean) — CX-3. Possibly conditional on git-repo presence | SKIPPED(call-cap — 2 LLM calls spent on AG-1/AG-2) | **INGESTS** — witnessed(GR-3): reply "**CANARY** Hello" | SKIPPED(no-headless-mode) | SKIPPED(auth-dead) |
| Misc witnessed deltas vs press | — | — | `agy models` subcommand **HUNG** (>4 min, killed) — SKIPPED(hang) | `--no-auto-update` flag **does not exist** in 0.2.82 (help surface) — press claim stale. `grok-build-0.1` also exists as an xAI **API** model (model list) | Press said "-p for non-interactive mode"; the binary itself refuses non-TTY — press claim REFUTED | Shutdown message names Antigravity as successor — confirms the migration story |

---

## 3. Liveness matrix (Step 2)

Vehicle: the council's own `run_health_checks()` (`src/ai_council/healthcheck.py:58`) invoked standalone; ping prompt "Reply with the word OK only."; secrets loaded `override=True`; every pinged model is the **verbatim `settings.yaml` pin** (amendment 4 — zero corrections needed).

| Provider (seat) | Key (name only) | Model pinged | Result |
|---|---|---|---|
| claude (debate) | ANTHROPIC_API_KEY: SET | `claude-opus-4-7` | **PASS** — **Anthropic credit state: healthy** (a `billing`-class failure would have surfaced; none did) |
| claude-sonnet (spare seat) | ANTHROPIC_API_KEY: SET | `claude-sonnet-4-6` | **PASS** |
| gemini (debate + default synthesizer) | GEMINI_API_KEY: SET | `gemini-3.1-pro-preview` | **PASS** |
| openai (debate) | OPENAI_API_KEY: SET | `gpt-5.4` | **PASS** |
| grok (debate) | XAI_API_KEY: SET | `grok-4.3` | **PASS** |
| deepseek (debate + research summarizer) | DEEPSEEK_API_KEY: SET | `deepseek-v4-pro` | **PASS** |
| perplexity (research) | PERPLEXITY_API_KEY: SET | `sonar-pro` | **PASS** |
| grok_research | XAI_API_KEY: SET | `grok-4.20-reasoning` | **PASS** (but see currency — unlisted alias) |
| openai_mini (research) | OPENAI_API_KEY: SET | `gpt-5.4-mini` | **PASS** |
| openai_deep (research `--deep`) | OPENAI_API_KEY: SET | `gpt-5.5` | PRESENCE-ONLY (never invoked, per plan; auth shared with passing openai key; model listed live) |
| gemini_research (deep research agent) | GEMINI_API_KEY: SET | `deep-research-preview-04-2026` | PRESENCE-ONLY (Interactions agent never invoked; **agent ID IS in the live Gemini model list**) |

**Operational finding (P1 input):** `cli.py:388` loads the secrets .env with `override=False`. A council run launched from inside a Claude Code session — where the harness injects `ANTHROPIC_API_KEY=""` — keeps the empty string and the claude seat fails auth. The council must be run from a normal shell, or the loader needs an explicit stance.

---

## 4. Model-currency table (Step 3, #17 evidence)

Sources: live `/v1/models`-class listings (openai 120, xai 9, deepseek 2, anthropic 10, gemini 54 models) — raw lists in `captures/model-lists.json`.

| Seat | Pin (verbatim) | Verdict | Notes / suggested current ID |
|---|---|---|---|
| gemini (debate/synth) | `gemini-3.1-pro-preview` | **current** | Still the top Pro — no 3.5-pro exists (only `gemini-3.5-flash`) |
| openai | `gpt-5.4` | **current** (superseded available) | `gpt-5.5` / `gpt-5.5-pro` exist; pin listed & serving |
| claude | `claude-opus-4-7` | **current** (superseded available) | `claude-opus-4-8`, `claude-fable-5` exist; pin listed & serving |
| claude-sonnet | `claude-sonnet-4-6` | **current** (superseded available) | `claude-sonnet-5` exists |
| grok | `grok-4.3` | **current** | Listed |
| deepseek | `deepseek-v4-pro` | **current** | List is exactly `{deepseek-v4-pro, deepseek-v4-flash}` — the browser claim confirmed; **the #17 "stale pins" suspicion is refuted for deepseek** |
| research.perplexity | `sonar-pro` | unverifiable (no list endpoint) | Ping PASS — alive |
| research.grok | `grok-4.20-reasoning` | **stale → `grok-4.20-0309-reasoning`** | Serves via unlisted alias (ping PASS, absent from `/v1/models`) — classic deprecation posture |
| research.openai_mini | `gpt-5.4-mini` | **current** | Listed |
| research.openai_deep | `gpt-5.5` | **current** | Listed |
| research.gemini | `deep-research-preview-04-2026` | **current** | Listed (`models/deep-research-preview-04-2026`); a `deep-research-max-preview-04-2026` sibling also exists |

**#17 net:** the pins are in far better shape than suspected — one genuinely stale ID, several "newer exists" upgrade candidates. What's missing is the **process** (P1), not a rescue.

---

## 5. Reconciliation verdicts (Step 4)

### 5.1 Fable-audit spot-checks (§"How to verify this audit") — all re-run live, all reproduce

1. Synthesizer trace: eviction before pick — `orchestrator.py:61-74` → `runner.py:44-56`, `:59-75` ✔
2. Lane asymmetry: `cli.py:565-567` raw `read_text`, no `parse_file` ✔ (vs inbox parse `cli.py:466-482`)
3. Research return gap: `grep return_dir src/ai_council/research/` → **0 hits** ✔
4. No CLI backends: `grep subprocess|Popen src/` → **0 files** ✔
5. Guide drift: `:47` + `:384` + `:442` hardcode `synthesizer: openai`; `:481` mis-describes the mechanism ✔
6. Epic C recon absence: repo-wide grep for `codex exec`/`--output-format`/`claude -p` → only the audit document itself ✔

### 5.2 Decisions D1–D14

| D | Verdict | Live evidence |
|---|---|---|
| D1 two-lane contract | **HOLDS** | `cli.py:565-567` still raw-reads `--file`; no `protocols/COUNCIL_INVOCATION_CONTRACT.md` exists |
| D2 lane-parity fixes | **HOLDS** | Same + research `return_dir` grep = 0 hits |
| D3 `council.return_dir` reader seam | **HOLDS** | `cli.py:401-407` reserved-seam comment intact |
| D4 synthesizer doc reconciliation | **HOLDS** | GUIDE `:47/:384/:442/:481` unchanged |
| D5 Epic B #1 scoring guard | **HOLDS** | Contamination vector live: guide examples still hardcode `synthesizer: openai` |
| D6 CliProvider engine + backend axis | **HOLDS — coverage matrix MOVED** | Engine decision unchanged; per-CLI coverage rewritten by probes (see §7 ADR-12 markup): claude/codex eligible; grok CLI EXISTS (draft said none); agy excluded; deepcode non-headless |
| D7 default flip evidence-gated | **HOLDS** | Parity gate unchanged; note the cost prize shrank — grok CLI is API-billed as configured, codex/claude are the real subscription wins |
| D8 CLI-as-ABI | **HOLDS** | `VISION.md:26` "Standalone tool — invoked by other repos, not embedded as a library" live |
| D9 #9 deferral upheld | **HOLDS** | `BACKLOG.md:80` — "#9 … DEFERRED — do NOT build before the canonical-baseline settles" |
| D10 crux-check shape | **HOLDS** | `BACKLOG.md:86` #18 baseline-gated, unbuilt |
| D11 framing defense | **HOLDS** | `BACKLOG.md:87` #19 baseline-gated, unbuilt |
| D12 ADR-01/02 amendments stay with Epic B | **HOLDS** | `docs/decisions/ADR-01/-02` present, unamended |
| D13 minority-detection hardening → Epic B | **HOLDS** | `output.py:381` `extract_dissent` / `:403` `save_minority_report` heuristic unchanged |
| D14 hub reconciliation flags | **HOLDS** | Hub `protocols/AI_COUNCIL_PROCESS.md:11` + `:402` still point at `ai-council/docs/council-question-guide.md` (file lives at `protocols/COUNCIL_QUESTION_GUIDE.md`; old path absent — glob verified) |

### 5.3 Corrections #1–#5

| # | Verdict | Live evidence |
|---|---|---|
| #1 synthesizer premise false / gemini IS default | **HOLDS** | Eviction-then-pick trace reproduced (`orchestrator.py:61-74`, `runner.py:44-75`) |
| #2 twofold real finding (4-debater default; openai-by-guide-example) | **HOLDS** | GUIDE example/rule sites unchanged |
| #3 June-16 audit never claimed the bug + own staleness | **HOLDS** | `2026-06-16-…md:62` scores "Non-participating synthesizer ✅" with both functions cited |
| #4 Epic C recon absent from repo | **HOLDS** | Grep reproduces (only the audit doc matches) — **and this session now supplies the witnessed recon that was missing** |
| #5 brief §2 staleness | **HOLDS** | `output.py:403` (minority shipped); ADR-09/10 present in `docs/decisions/`; still zero CLI backend (0 subprocess hits) |

### 5.4 Browser-research claims (architect, 2026-07-05) — probe verdicts

| Claim | Verdict |
|---|---|
| Gemini CLI stopped serving free/Pro/Ultra accounts (2026-06-18); `agy` is the successor | **CONFIRMED in effect** — GM-1 `IneligibleTierError: This client is no longer supported … migrate to the Antigravity suite`. (The exact date is not independently witnessed; the shutdown state is) |
| Legacy `gemini` survives on paid API keys | **UNTESTED** — the client refused at the OAuth tier and never fell through to the present `GEMINI_API_KEY`; testing needs an auth-mode config change (1-call cap reached) |
| `agy -p` headless works | **CONFIRMED** (AG-1) |
| agy: NO model flag in headless mode | **REFUTED** — `--model` exists (help). But **pin honor REFUTED** (AG-2 silent default-swap) — worse than the claim |
| agy default = Gemini 3.5 Flash; TUI spans ~8 multi-lab models (incl. Claude/GPT-OSS) → identity not guaranteed | Default identity **UNVERIFIABLE** (no identity channel; the resolver's default alias "CCPA" is opaque; `gemini-3.5-flash` does exist as an API model). Multi-lab TUI menu UNTESTED. **The conclusion — seat identity NOT guaranteed — is CONFIRMED** by AG-2's silent swap |
| agy quota shared/invisible, fast exhaustion | Invisible: **consistent with probes** (no quota surface anywhere). Shared/fast-exhaustion: UNVERIFIED |
| grok: `-p` + `--output-format plain\|json\|streaming-json`; `-m` pins; `grok-build-0.1` coding line | **CONFIRMED** (GR-1/GR-2, help, `grok models`) — with correction: **default model is `grok-4.20-0309-non-reasoning`**, not the build line |
| grok: `--no-auto-update` recommended | **REFUTED** — flag does not exist in 0.2.82 |
| grok auth: subscription OAuth (`~/.grok/auth.json`) or `XAI_API_KEY` | **PARTIAL** — API-key lane witnessed ACTIVE; `auth.json` absent; `grok login` exists but no subscription auth is configured on this machine |
| grok auto-reads `CLAUDE.md`/`.claude/`/`AGENTS.md` from cwd | **CONFIRMED** (GR-3: "**CANARY** Hello") |
| grok sandbox + ACP agent mode | Flags/commands exist in help — PLAUSIBLE, unexercised |
| DeepSeek: no first-party CLI; community = Deep Code (`deepcode`) / DeepSeek-TUI (`deepseek`) | **CONFIRMED** — `deepcode` present, `deepseek` binary absent |
| Deep Code usable headless | **REFUTED** — TTY required (DC-1) |
| DeepSeek API models `deepseek-v4-pro`/`-flash` (1M context) | Model IDs **CONFIRMED** (live list is exactly these two); context size unverified |
| DeepSeek CLI seat = capability-only, API-billed (no subscription lane) | **MOOT/CONFIRMED-IN-PART** — no headless capability at all, so no seat regardless of billing |
| `claude -p … --output-format json --tools ""` works; `.result` field | **CONFIRMED** (CL-1) — plus the critical caveat: `--tools ""` does NOT block cwd `CLAUDE.md` ingestion (CL-3) |
| `codex exec --sandbox read-only --skip-git-repo-check --json`; JSONL `agent_message` | **CONFIRMED** (CX-1) — plus caveat: stdin must be closed or it hangs |

### 5.5 Architect-proposed invariants — probe-tested

| Invariant | Verdict |
|---|---|
| Synthesizer seat is never a CLI seat | **SUPPORTED** — every CLI has an identity gap (agy: none; codex `--json`: none; grok stdout: none) and two of four ingest cwd context. Highest-epistemic-load seat stays on pinned API identity |
| Identity logged or no seat | **SUPPORTED and now implementable** — witnessed channels: claude `.modelUsage` (in-band), codex plain-banner stderr (out-of-band), grok session `events.jsonl` via `sessionId` (out-of-band). agy has NO channel → excluded by this invariant, mechanically |
| Quota gradient grok > codex > claude > agy | **REVISED** — as configured: **codex (sub, idle) > claude (sub, competes with operator CC work) > grok (API-billed — zero cost advantage until `grok login` + env-key shielding) > agy (excluded)**. The original gradient becomes reachable only after grok subscription auth is set up |
| CLI probes/seats run with cwd = scratch, never a repo | **CONFIRMED as mandatory** — CL-3 + GR-3 ingestion witnessed |

### 5.6 ADR-12 draft premises hit by fleet reality

- "xai/deepseek stay API-only (**no CLI exists**)" — **INVALIDATED as stated**: both CLIs exist. Conclusion survives for deepseek on new grounds (no headless mode — DC-1); revised for grok (CLI seat feasible; blocker is billing lane, not existence).
- Gemini flag set `gemini -p -o json` — syntax is real, but the seat is **auth-dead on consumer tier** (GM-1); as recon'd, unusable.
- Quota assumptions — see revised gradient above.

---

## 6. Process designs (Step 5 — functional level, no code, no backlog items)

### P1 — Project liveness ("doctor")

- **Actors:** operator (on-demand); a scheduled/session-start trigger (cadence); any future `council doctor` command is one candidate *vehicle*, not the process.
- **Trigger/cadence:** (a) on demand before any `important`-profile or delegated (Lane A) run; (b) periodic — weekly is proportionate to observed drift rate (one stale pin in ~7 weeks); (c) after any provider/key/CLI change.
- **Inputs:** `config/settings.yaml` (single source of pins), secrets .env (names only), the CLI fleet list, provider list endpoints.
- **Steps:** 1) key presence by NAME; 2) provider pings via `run_health_checks()` with verbatim pins, errors classified by `classify_error` (billing/auth/network/…); 3) model-currency sweep against list endpoints (perplexity: ping-only, no list endpoint); 4) CLI fleet health — version + auth-state checks only (`codex login status`, grok auth-lane header, etc.), **zero LLM spend by default**; 5) verdict roll-up.
- **Outputs/records:** a dated machine-readable result (JSON) + human table; recorded under a local operational dir (e.g. `output/health/`) — NOT `docs/audits/` (operational telemetry, not review artifacts). Latest result referenced at session start.
- **RED criteria:** any missing key; any debate-seat or synthesizer-seat ping FAIL; research successes below the ADR-08 `min_successful_providers` analog. **YELLOW:** stale pin; CLI auth-lane drift (e.g. grok silently API-billed); superseded-model advisories.
- **Checks/failure handling:** the runner must load secrets with an explicit override stance — witnessed hazard: `cli.py:388` `override=False` + CC-injected `ANTHROPIC_API_KEY=""` = false auth failure when run from a CC session. Doctor must either force-override or refuse to run in a shell with an empty-but-set key.
- **Open forks:** (a) vehicle — `council doctor` subcommand vs standalone script vs extending the existing pre-run health gate. Deciding evidence: whether foreign repos need to invoke it (Lane A pre-flight) — if yes, subcommand wins. (b) CLI LLM smoke-test — zero-spend default vs optional 1-token smoke flag. Deciding evidence: whether auth-state checks alone catch real CLI breakage over a month of use. (c) record location & retention.
- **What the backlog item needs:** this spec; the session's `liveness.py` prototype (scratch) as the seed; the RED/YELLOW definitions above; the override-stance decision.

### P2 — Lane routing (CLI vs API vs research)

- **Actors:** orchestrator (mechanical routing); operator (profile choice — hub doctrine "cost gate lives in operator judgment" unchanged).
- **Trigger:** per run, at panel-build time.
- **Inputs:** profile (`standard`/`important`/research; flag > frontmatter > default), per-seat `backend:` axis (ADR-12), doctor's latest verdict (a RED seat routes to fallback or is dropped per existing health-gate semantics).
- **Probed seat-eligibility table (replaces the press-based one):**
  - **codex → openai seat:** eligible. Requirements witnessed: stdin closed; `--sandbox read-only`; pin via `-m` (else it runs its config default `gpt-5.5`, NOT the seat's pin); identity parsed from the plain-mode stderr banner (the `--json` stream carries none).
  - **claude → claude seat:** eligible. Pin via `--model`; identity in-band from `.modelUsage`; `--tools ""` for tool-lockdown but **cwd=scratch is the actual isolation** (CL-3); subscription lane guaranteed only under the existing key-strip guard. Contends with the operator's own CC quota.
  - **grok → grok seat:** mechanically eligible (pin honored; identity from session `events.jsonl` keyed by stdout `sessionId`), **no cost benefit as configured** — API-billed via env `XAI_API_KEY`. Enters the cost lane only after `grok login` (subscription OAuth) + shielding the env key for grok invocations (mirror the claude key-strip pattern).
  - **agy → gemini seat: EXCLUDED** — identity-logged-or-no-seat invariant fails (no identity channel; witnessed silent pin-swap). Re-admission evidence: a future agy version whose headless output (or log) reports the served model AND honors `--model`.
  - **deepcode → deepseek seat: EXCLUDED** — no headless mode exists.
- **Routing rules:** `standard` → CLI where eligible (codex, claude; grok post-OAuth), API elsewhere; `important` → all-API; research → always API. Fallback: CLI failure/quota error → same-seat API retry (extend `classify_error`).
- **Records (per run, per seat):** requested backend → actual backend → requested model → **actual model identity** (from the per-CLI channel above) → fallback events. Vehicle: extend the existing `_metrics.json` sidecar. A seat whose identity could not be read is recorded as a degradation event.
- **Safety floor (all witnessed):** cwd = scratch dir always; stdin closed; read-only/tools-off flags on every call; personas injected via prompt; outputs enter the normal ADR-03 anonymization path.
- **Open forks:** (a) grok seat now-vs-later — ship API-billed (capability parity, no savings) or wait for OAuth setup. Deciding evidence: operator runs `grok login`; re-probe shows subscription lane. (b) codex model policy — pin the seat's `settings.yaml` model via `-m` vs accept codex's own default. Deciding evidence: the ADR-12 §5 parity run, scored per rubric. (c) identity-parse robustness — banner/session-file parsing is version-fragile; decide a "doctor re-probes identity channels after CLI updates" rule vs per-run hard-fail.
- **What the backlog item needs:** §2's witnessed flag matrix (this is the build-start recon ADR-12 §Context demanded); the eligibility table; the metrics-extension field list.

### P3 — Open interface / delegation for foreign repos (ADR-11 lifecycle)

- **Actors:** caller repo's CC session (commissions); operator (approves spend/profile); ai-council CLI (executes); hub (owns WHEN/WHY per ADR-67 — referenced, not restated).
- **Trigger:** caller repo reaches an ADR-67 convene threshold.
- **Caller must provide:** a GUIDE-conformant brief (frontmatter: mode/models/synthesizer/rounds/target-project) authored under ADR-95 lane discipline; a `--return-dir` inside the caller repo; a profile choice (P2); the standing safety context (council runs from any cwd — `config_loader` resolves against repo root by design).
- **Steps (Lane A, the agent path):** author brief → optional P1 doctor pre-flight → `council --file <brief> --return-dir <dir> [--format json] [overrides]` → read exit code → consume artifacts at the return dir → CC drafts the ADR in the caller repo (ADR-67 step 5) → close-out. Lane B (inbox) remains the operator-mediated batch path, unchanged.
- **Caller gets back:** exit code 0/1/2/3 (ADR-08 semantics; 3 = degraded-but-complete → caller must surface the alarm in its ADR), `council-out-*.md`, `council-minority-*` when dissent, `*_metrics.json` (with P2's backend/identity fields once built), optional JSON on stdout.
- **Error/degradation semantics for the caller:** exit 3 → verdict usable, record the degradation in the caller ADR; hard failure → no artifacts at return dir, caller retries or falls back to Lane B; RoutingError stays fail-loud.
- **Preconditions (ordered):** D2 parity fixes first — until `--file` parses frontmatter, a guide-conformant brief silently leaks YAML into the question text on exactly the lane agents use; until research threads `return_dir`, research commissions cannot return deterministically.
- **Checks:** the caller never writes into ai-council; ai-council never writes into the caller outside the declared return dir; hub is never a default destination (ADR-10 held).
- **Open forks:** (a) default foreign-repo lane — Lane A synchronous vs Lane B drop. Deciding evidence: first 3 real cross-repo commissions — measure operator-mediation overhead. (b) verdict→ADR transform — free-form CC drafting vs a template keyed to synthesis sections. Deciding evidence: quality drift across the first few caller-side ADRs. (c) contract versioning — plain doc version vs ADR per breaking change. Deciding evidence: the first breaking flag change.
- **What the backlog item needs:** ADR-11 ratification + the contract doc outline (draft §4 list is complete); the D2 parity-fix pair as its own item; this lifecycle as the contract doc's "caller walkthrough" section.

### P4 — Debate lifecycle end-to-end with quality gates

- **Stages → existing mechanism → gap:**
  1. **Authoring** — GUIDE (protocols/COUNCIL_QUESTION_GUIDE.md), including the research-vs-decision recognition discipline. **Live-verified: the research-mode section EXISTS** (`:342-447`, incl. recognition test `:352-360`, format `:376+`, quick template `:435+`) — the prompt's expectation of absence is REFUTED. Gap: examples still teach `synthesizer: openai` (D4).
  2. **Gate** — absent by explicit deferral (#9, `BACKLOG.md:80`); hub tolerates manual pre-flight (`AI_COUNCIL_PROCESS.md:101` six-question pre-flight). Gap: none to build now; the gate targets the ADR-11 contract when Epic B settles.
  3. **Run** — health gate (`cli.py:430-441`), eviction mechanism (`runner.py:44-56`), blind critique (ADR-03), rounds ≤2. Gaps (both baseline-gated): crux grounding (#18), framing alarm (#19).
  4. **Synthesis-identity integrity** — mechanism real (gemini default synthesizes, correction #1) but identity is **habit-vulnerable**: guide-example frontmatter silently swaps the verdict author (D4/D5). Needed: verdict-author identity recorded first-class in output metadata + the D5 segmentation guard for any historical scoring.
  5. **Verdict/minority** — `extract_dissent`/`save_minority_report` (`output.py:381/:403`), heading-heuristic; hardening = D13, Epic B (synthesis-contract coupled).
  6. **Record** — output routing (ADR-10/43), metrics sidecar, transcripts. Gap: P2's backend/identity fields.
- **Checks & failure handling:** existing — health gate, exit codes, degradation alarm (ADR-08). Missing check with no owner: nothing verifies the *synthesizer identity used* matches the *intent* (default vs frontmatter override) — cheap to record, closes the D4/D5 loop permanently.
- **Open forks:** (a) research-mode recognition — keep prose discipline vs make `mode:` frontmatter mandatory for research briefs. Deciding evidence: mis-mode rate in inbox archive history. (b) verdict-author record — metadata-only vs a visible "Synthesized by X" line in the artifact. Deciding evidence: whether Epic B scoring needed the segmentation guard in practice (D5). (c) gate placement when built — caller-side skill (/council-question) vs council-side validator vs both. Deciding evidence: where bad questions actually originate (inbox archive audit).
- **What the backlog item needs:** the stage→mechanism→gap table above; the "record synthesizer identity" micro-item as the only new NOW-class candidate this spec surfaces.

---

## 7. ADR-draft markups (Step 6 — drafts stay drafts; nothing written to docs/decisions/)

### ADR-12 markup (against the 2026-07-04 draft)

- **§Context** — REPLACE the "operator-supplied flags … MUST be re-verified" clause: the re-verification now EXISTS (this report §2). Witnessed working invocations: `claude -p "<q>" --output-format json --tools "" --model <id>` (stdin closed; identity from `.modelUsage`); `codex exec --sandbox read-only --skip-git-repo-check [--json] -m <id> "<q>"` (stdin closed; identity from plain stderr banner; `--json` carries no identity); `grok -p "<q>" --output-format json -m <id>` (identity from session `events.jsonl` via stdout `sessionId`). STRIKE `gemini -p -o json` as a seat path (consumer tier dead, GM-1).
- **§2 adapters** — REVISE the coverage line "xai/deepseek remain API-only (no CLI exists)" to: *deepseek remains API-only (its CLI has no headless mode — witnessed DC-1); grok's CLI is seat-capable (pin honored, identity recoverable) but joins the cost lane only when subscription OAuth is configured; until then a grok CLI seat is API-billed and pointless. The gemini seat has NO usable CLI: legacy gemini is auth-dead (GM-1) and **agy is excluded until a probe witnesses both `--model` honor and a served-model identity channel** (AG-2 witnessed the opposite).* Adapter set v1 therefore: **claude + codex only.**
- **§3 safety invariants** — ADD (all witnessed): cwd = scratch dir is the PRIMARY isolation (tools-off does NOT prevent cwd `CLAUDE.md` ingestion — CL-3; grok ingests — GR-3); stdin must be closed (CX-1 hang); **identity-logged-or-no-seat** becomes a hard invariant with the three per-CLI channels named; **the synthesizer seat is never a CLI seat**.
- **§4 routing** — REVISE the quota gradient to the probed order: codex > claude > grok(post-OAuth only) > agy(excluded). ADD: seat model must be pinned explicitly per call (codex otherwise runs its own config default `gpt-5.5`, not the seat pin).
- **§5 default-flip clause** — unchanged (parity-evidence gate stands).

### ADR-11 — **CONFIRMED untouched by fleet reality.** All of its evidence is repo-internal (lanes, frontmatter, return-dir) and re-verified live (§5.2 D1–D3). No markup needed.

### ADR-13 — **CONFIRMED untouched** (baseline-gated shape). One annotation: rank v2 CLI-resolver candidates by witnessed sandbox posture — codex first (`--sandbox read-only` witnessed), claude second (tools-off + scratch-cwd), grok third (sandbox flag present, unexercised).

---

## 8. Consolidated fork list for the next architect

| # | Fork | Bounded options | Deciding evidence |
|---|---|---|---|
| F1 | Doctor vehicle | `council doctor` subcommand · standalone script · pre-run gate extension | Do foreign repos need pre-flight? (P3 usage) |
| F2 | Doctor CLI smoke spend | zero-spend auth checks · optional 1-token smoke | A month of auth-only checks: did they miss real breakage? |
| F3 | grok seat timing | ship API-billed now · wait for `grok login` OAuth + key shield | Re-probe after operator runs `grok login` (5 min) |
| F4 | codex seat model policy | pin seat model via `-m` · accept codex default | ADR-12 §5 parity run, rubric-scored |
| F5 | Identity-channel fragility | doctor re-probes channels after CLI updates · per-run hard-fail on unreadable identity | First CLI auto-update that breaks a parse |
| F6 | Foreign-repo default lane | Lane A sync · Lane B inbox | Operator-mediation overhead across first 3 real commissions |
| F7 | Verdict→ADR transform | free-form CC draft · section-keyed template | Quality drift across first caller-side ADRs |
| F8 | Contract versioning | doc version · ADR per breaking change | First breaking flag change |
| F9 | Research-mode recognition | prose discipline (status quo) · mandatory `mode:` frontmatter | Mis-mode rate in inbox archive |
| F10 | Verdict-author visibility | metadata-only · visible "Synthesized by X" line | Whether D5 guard fires during Epic B scoring |
| F11 | Legacy-gemini API-key lane | test it (auth-config change, 1 call) · declare dead | Only worth testing if a gemini CLI seat is ever wanted; agy exclusion makes it near-moot |
| F12 | Stale research pin | update `grok-4.20-reasoning` → `grok-4.20-0309-reasoning` now · fold into P1's first doctor run | None needed — alias still serves; zero-risk config edit |

## 9. Probe-capture manifest

Scratch root: `%LOCALAPPDATA%\Temp\claude\C--Users-1028120-Documents-Dev-ai-council\9fd449a5-e360-42ea-88f2-029288e5a120\scratchpad\`
- `probes/captures/` — 34+ files: `CL-1..3`, `CX-1..3`, `AG-1..2 (+AG-2.log)`, `GR-1..3`, `DC-1`, `GM-1` (`.out`/`.err` pairs), `help-*.txt`, `models-grok.txt`, `model-lists.json`. **Secret-scanned: 0 token-like strings.**
- `probes/canary/` — the amended canary files (CLAUDE.md / AGENTS.md / GEMINI.md); `probes/clean/` — empty probe cwd.
- `liveness.py` — the Step-2/3 runner (P1 seed).
- Skipped probes: DC-2/3 `SKIPPED(no-headless-mode)`; AG canary `SKIPPED(call-cap)`; `agy models` `SKIPPED(hang)`; agy/grok sandbox flags `SKIPPED(call-cap, present-in-help)`; legacy-gemini API-key lane `SKIPPED(call-cap + auth-config required)`.

---

**Done-contract check:** fleet cells all `witnessed`/`SKIPPED` ✔ · D1–D14 + #1–#5 verdicts with live evidence ✔ · four functional process specs with forks + backlog-needs lines ✔ · repo mutations: this file only, uncommitted ✔ · secret scan clean ✔ · self-sufficient for backlog construction ✔
