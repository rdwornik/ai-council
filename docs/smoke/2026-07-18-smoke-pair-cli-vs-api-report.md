# STREAM SMOKE — CLI vs API paired run (side-by-side, UNBLINDED)

**Date:** 2026-07-18
**Worktree/branch:** `smoke-pair` / `worktree-smoke-pair` (COMMIT-AND-STOP; never merged to main)
**Config:** `config/settings.smoke.yaml` (scoped witness; production `settings.yaml` untouched)
**Question (BURNED):** observability stack choice — see
`2026-07-18-BURNED-question-observability.md` for the verbatim prompt.

## Experiment design

One fresh, non-trivial `pick` decision run **twice** with an **identical** panel, rounds,
question, and synthesizer. The **only** independent variable is the seat backend axis.

- **Panel (both arms):** claude `claude-opus-4-8` + openai `gpt-5.6-terra` (the ruled pins)
- **Synthesizer (both arms):** gemini `gemini-3.1-pro-preview` — trial-scoped, non-participant
  (durable `settings.yaml` synthesizer `openai` untouched; ADR-01 unaffected)
- **Rounds:** 2 (initial + critique), mode `pick`
- **ARM 1 (CLI):** both seats via subscription CLI — claude 2.1.214, codex-cli 0.144.5 → $0 seats
- **ARM 2 (API):** same two seats via billed API — backend flipped to `api` in memory; model
  strings are the identical pins, so panel/rounds/models are byte-identical across arms
- **Isolation:** `AICOUNCIL_OUTPUT_DIR=./smoke-output/` (inside worktree); `PYTHONPATH=./src`;
  nothing landed in the primary's canonical `output/`.

No rubric — the operator judges verdict quality by eye. This report presents both arms
UNBLINDED and labeled.

## Seat-backend proof (from `_metrics.json` sidecars)

```
                     ARM 1 — CLI                      ARM 2 — API
claude seat   requested=cli  actual=cli        requested=api  actual=api
              model=claude-opus-4-8 (served)   model=claude-opus-4-8 (served)
              identity=modelUsage              identity=api-echo
              fallback_events=[]               fallback_events=[]
openai seat   requested=cli  actual=cli        requested=api  actual=api
              model=gpt-5.6-terra (served)     model=gpt-5.6-terra (served)
              identity=stderr-banner (codex)   identity=api-echo
              fallback_events=[]               fallback_events=[]
```

Both arms served the exact requested pins with **zero fallback**. CLI arm admitted both seats
on the subscription lane; API arm ran both on the billed lane. Clean pair.

## Cost / tokens / wall time (verbatim from sidecars)

```
metric                    ARM 1 — CLI            ARM 2 — API           note
------------------------  --------------------   -------------------   ------------------------
seat cost (4 panel calls) $0.000000              $0.281677             CLI subscription = $0
  R1 claude               $0.000000 (cli)        $0.079990 (api)
  R1 openai               $0.000000 (cli)        $0.028930 (api)
  R2 claude               $0.000000 (cli)        $0.112945 (api)
  R2 openai               $0.000000 (cli)        $0.059812 (api)
synthesizer (gemini,api)  $0.029696              $0.036416             billed both arms (API-only)
TOTAL cost                $0.029696              $0.318094             API ~10.7x CLI total
total tokens              37,877                 34,555                see token caveat below
  input / output          20,956 / 16,921        21,775 / 12,780
debate duration (sidecar) 165.08s                134.53s              CLI ~30s slower
driver wall-clock         170.4s                 134.6s               subprocess spawn overhead
```

**Headline:** the four panel seat-calls that cost **$0.28 on the API lane were served $0 on
the CLI subscription lane** — same pins, same rounds, no fallback. The only unavoidable spend
is the gemini synthesizer (always API per ADR-12), ~$0.03 either way.

**Token caveat (cross-arm counts are NOT apples-to-apples):** the CLI adapters measure
differently from the API SDK. codex reports a single combined total booked as `output_tokens`
(input shows 0 — see the R1/R2 openai calls in the CLI sidecar); the claude CLI books most of
the prompt to `cache_creation_input_tokens` (rolled into input). The API arm's per-call
input/output split is the clean measure. Seat **cost** on the CLI lane is $0 regardless of count.

**Cost caveat:** `gpt-5.6-terra` inherited `gpt-5.4`'s per-1M rates in the witness config
($2.50 in / $15.00 out), so the API-arm dollar figure is an *estimate from those rates*, not a
billed-exact invoice. The CLI arm's $0 seat cost is exact (no billing occurs).

---

# FULL SYNTHESIS VERDICTS (unblinded, side by side)

Both arms independently converged on the **same decision**: adopt Option A (managed SaaS) +
OpenTelemetry, keep cloud-native (C) as a break-glass fallback, reject self-hosting (B).
Full gemini synthesis text for each arm follows verbatim.

## ARM 1 — CLI verdict (`smoke-output/cli/council-out-20260718_210408-pick-smoke-cli.md`)

**Recommended Decision:** Adopt Option A (Managed SaaS) now, instrumented with OpenTelemetry
SDKs. Keep Option C (Cloud-native tooling) as a thin, out-of-band fallback.

**Consensus:** Both participants reached a strong, independently reasoned consensus — Option A
(Managed SaaS) + OpenTelemetry — driven by tight engineering economics, not groupthink. For a
3-person team engineer-hours are the scarcest resource; running 4–5 stateful telemetry
databases would consume an unacceptable share of capacity. Cloud-native (C) is insufficient as
a primary pane of glass due to the cognitive load of correlating signals across disconnected
UIs at 2am. Keep cloud-native metrics as a zero-effort fallback if the SaaS vendor goes down.

**Unresolved disagreements (tactical):**
- *OTel Collector on day one* — Claude initially wanted a self-hosted collector for
  sampling/cardinality control; OpenAI argued this smuggles back the self-hosted burden and a
  single point of failure. **Stronger: OpenAI.** Claude conceded in Round 2.
- *Threshold to revisit self-hosting* — Claude proposed a rigid dollar trigger ($3–5k/mo);
  OpenAI argued the trigger should be when the SaaS bill exceeds the fully-loaded cost of the
  engineering capacity to run OSS. **Stronger: OpenAI.**

**Argument quality:** Strongest = Claude's *blast-radius independence* (self-hosting telemetry
on the same infra means it dies during the outage it's meant to illuminate; SaaS gives
out-of-band debugging). Weakest = Claude's Round 1 claim that OTel "neutralizes lock-in" making
migration "a config change" — OpenAI dismantled it (dashboards, alerts, SLOs, query semantics,
RBAC stay vendor-locked); Claude corrected in Round 2.

**Blind spots flagged:** SaaS pricing-model differences (per-host+indexed vs per-event); cloud
egress fees on telemetry; OTel auto-instrumentation maturity varying by runtime.

**Action items:** (1) select a unified SaaS vendor; (2) instrument via OTel SDKs, no
self-hosted collector yet; (3) structured JSON logs + trace-ID injection, forbid PII/bodies;
(4) symptom-based RED alerts, no CPU/mem paging; (5) game-day: verify metric→trace→log pivot.

## ARM 2 — API verdict (`smoke-output/api/council-out-20260718_210624-pick-smoke-api.md`)

**Recommended Decision:** Adopt Option A (Managed SaaS) heavily instrumented via OpenTelemetry,
and explicitly reject self-hosting (Option B). Keep Option C strictly as a tested break-glass
fallback.

**Consensus:** Both independently arrived at the same conclusion with aligned reasoning (shared
operational logic, not groupthink): adopt A; a 3-person team cannot afford self-hosting (B);
cloud-native (C) lacks unified cross-signal correlation for 2am response. OpenTelemetry is
mandatory as the anti-lock-in escape hatch. Cost controls must be proactive (sampling, drop
rules, short retention). Instrumentation consistency (shared tagging libraries) is a
prerequisite. Cloud-native kept as an independent health-check fallback.

**Unresolved disagreement (the crux — cost enforcement):** OpenAI assumed an 8-service SaaS
bill lands in $500–$2,000/mo; Claude sharply disagreed — Datadog/Honeycomb bills blow past
that without ruthless automated enforcement. **Stronger: Claude** (hoping a bill stays low
because traffic is "moderate" is a known trap; automated collector-side drop rules + hard caps
are necessary, not reactive billing alerts). Strong counter from OpenAI: run a 7–14 day
real-data PoC to validate costs before signing.

**Argument quality:** Strongest = Claude's failure-domain point — a self-hosted stack "fails
correlated with the outages it's meant to illuminate… negative leverage." Weakest = OpenAI's
initial assumption that cost could be handled by "explicit cost controls on day one" without an
automated ingest-stop; Claude noted human discipline degrades under deadline pressure and
reactive alerts only tell you the money is already spent.

**Blind spots flagged:** cloud egress costs on third-party telemetry; PII redaction engineering
burden at the collector; CI/CD deployment markers as a first-class timeline citizen.

**Action items:** (1) assign a DRI for observability/cost/alerts; (2) build a shared telemetry
standard library (OTLP attributes); (3) real-data PoC on 1–2 vendors for 7–14 days; (4)
configure the OTel Collector to strip PII, drop debug logs, sample ~10% success traces, enforce
hard throughput caps; (5) 5–8 symptom-based alerts to pager; (6) test an independent
CloudWatch/Cloud-Monitoring break-glass alert.

---

## By-eye read (for the operator; no rubric applied)

- **Same decision, both arms:** adopt A (managed SaaS) + OTel, keep C as break-glass, reject B.
- **Same strongest argument surfaced both times:** the failure-domain / blast-radius point
  (self-hosted telemetry dies with the outage it should illuminate). Independent runs, same
  standout insight — a good consistency signal.
- **Overlapping blind spots:** both arms independently flagged cloud **egress costs** as the
  shared miss; both landed on OTel-for-lock-in + proactive cost caps.
- **Where they differ (emphasis, not direction):** the CLI synthesis foregrounded the
  *revisit-threshold* and *lock-in-overclaim* disagreements; the API synthesis foregrounded the
  *cost-enforcement crux* and added *PII redaction* + *deployment markers* to blind spots. The
  API action list is a touch more operational (DRI, PoC-before-contract); the CLI list slightly
  more incident-drill oriented (game-day). Neither is obviously weaker.
- **Cost verdict:** CLI seats delivered substantively equivalent decision quality at **$0**
  seat cost vs **$0.28** on the API lane for the identical panel/rounds. Only the gemini
  synthesizer is unavoidably billed (~$0.03).

## Artifacts (paths)

- Config: `config/settings.smoke.yaml`
- Burn note: `docs/smoke/2026-07-18-BURNED-question-observability.md`
- This report: `docs/smoke/2026-07-18-smoke-pair-cli-vs-api-report.md`
- Raw arm output (gitignored evidence, not committed):
  - `smoke-output/cli/` — transcript + `_metrics.json` + verdict package (+ minority report)
  - `smoke-output/api/` — transcript + `_metrics.json` + verdict package (+ minority report)
  - `smoke-output/smoke_summary.json`
