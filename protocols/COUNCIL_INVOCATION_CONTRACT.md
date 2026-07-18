# Council Invocation Contract

**Date:** 2026-07-05 · **Authority:** ADR-11 (Accepted) · **Compatibility:** the surface named here is a versioned commitment — breaking changes require an ADR (ADR-11 §5).

The machine-facing contract for commissioning the Council. Audience: a delegating agent
(typically a Claude Code session in another repo under `Dev/`) and any automation harness.
Question *authoring* is governed by `COUNCIL_QUESTION_GUIDE.md`; this document governs
*invocation and returns*.

**Ownership boundary:** the methodology hub owns WHEN/WHY a debate is convened (thresholds,
the six-step loop, caps, cost judgment — hub ADR-67, referenced, not restated). This repo's
`protocols/` owns HOW: question formats, invocation mechanics, output contract, quality rubric.

---

## 1. Lanes

**Lane A — delegated, synchronous (the agent lane):**

```
council --file <brief.md> --return-dir <dir> [--format json] [overrides]
```

Runnable from any cwd — config resolves output paths against the ai-council repo root by
design; global secrets load from `~/Documents/.secrets/.env`. The caller blocks on the run,
reads the exit code, and consumes artifacts from `<dir>`.

**Lane B — inbox, batch (the operator lane):** drop a `.md` brief into `council_inbox/`
(or `~/Downloads` — files carrying council frontmatter keys are detected); the operator runs
`council --inbox`. Fire-and-forget; results land per the same output rules; processed files
are archived. The interaction model stays batch — no context-pull or interactive concepts.

One brief format serves both lanes (ADR-11 §2).

## 2. Flag set (the committed surface)

| Flag | Meaning |
|---|---|
| `QUESTION` (positional) | Inline question text (alternative to `--file` / `--inbox`) |
| `--file <path>` | Read the question/brief from a `.md` file |
| `--inbox` / `--inbox-dir <path>` | Lane B batch processing (+ optional inbox override) |
| `--return-dir <dir>` | ADR-10 deterministic return: verdict + minority report copied to `<dir>` in addition to the canonical `./output/` write. Unset → canonical only; the hub is never a default |
| `--format text\|json` | `json` prints the structured `DebateResult` to stdout |
| `--models a,b,c` | Panel override (comma-separated provider names) |
| `--lite` | 3-model panel (claude, gemini, openai) |
| `--full` | No-op (full panel is the default; kept for backward compatibility) |
| `--synthesizer <name>` | Verdict author override; automatically evicted from the debate panel |
| `--rounds <n>` | Debate rounds (≤2 per hub caps) |
| `--mode <m>` / `-M` | Force mode (pick/ideas/judge/research or alias); skips auto-detection |
| `--modes` | Print modes + aliases, exit |
| `--output <dir>` | Canonical output dir override (default `./output/`) |
| `--target-project <name>` (repeatable) | ADR-43 transcript mirroring to allow-listed projects |
| `--deep` | Research mode: include slow deep-research providers |
| `--no-cache` | Research mode: bypass the research cache |
| `--skip-health-check` | Skip the startup connectivity gate |
| `--verbose` | DEBUG logging |

## 3. Frontmatter keys + precedence

Recognized brief frontmatter: `mode`, `rounds`, `models`, `synthesizer`, `full`,
`target-project`. Precedence, highest first: **CLI flag > frontmatter > config default**.
Files in `~/Downloads` are recognized as council briefs by carrying ≥1 of these keys.

## 4. Exit codes (ADR-08 convention)

| Code | Meaning | Caller obligation |
|---|---|---|
| 0 | Success | Consume artifacts |
| 1 | Hard error (run failed; e.g. no providers passed health check, research RuntimeError) | No artifacts guaranteed; retry or fall back to Lane B |
| 2 | CLI usage error (Click) | Fix the invocation |
| 3 | **Degraded but complete** (research: successful providers < `min_successful_providers`; inbox batch: ≥1 degraded run) | Verdict is usable; the caller MUST record the degradation (alarm banner content) in whatever ADR/decision it derives |

## 5. Artifacts + JSON payload

Deterministic names in every destination (canonical `./output/` always written first;
`--return-dir` and `--target-project` receive copies):

- `council-out-<ts>-<mode>-<slug>.md` — verdict/transcript (with a human-readable verdict mirror at the top)
- `council-verdict-<ts>-<mode>-<slug>.json` — transcript-free **verdict package** (DRAFT-INT-1, #26):
  decision, rationale, options, dissent pointer, panel/`seats[]`, verdict author, degradation — the
  machine-authoritative deliverable a caller consumes without reading the transcript.
  **Debate-path only** — research mode does not yet emit it (open parity gap, BACKLOG #34).
- `council-out-<ts>-<mode>-<slug>_metrics.json` — per-provider cost/latency sidecar
  (+ `seats[]` CLI-backend block and `synthesis` namespaced block, ADR-12/#16)
- `council-minority-<ts>-<mode>-<slug>.md` — emitted only when the verdict is non-unanimous
  (substantive dissent detected in the synthesis)

`--format json` prints the `DebateResult` structure to stdout (question, mode, per-round
responses with provider/model metadata, synthesis text, cost totals) for programmatic
consumption; artifacts are still written.

## 6. Failure semantics

- **Degradation** — a provider dropout (missing key, failed call) shrinks the panel; the run
  proceeds if ≥ the minimum panel remains; research degradation below threshold → exit 3 with
  an alarm banner (never a silent pass).
- **RoutingError — fail-loud:** an unknown `--target-project` name aborts with the allow-list
  shown; it is never silently skipped.
- **Health gate:** providers are pinged at startup; failures are reported per provider with
  a classified cause (auth / billing / timeout / …). `--skip-health-check` bypasses.

## 7. Known deviations (until the D2 parity fixes land)

Committed contract, **currently deviating** — an ADR-11 decision whose code has not
shipped yet (the fix is a separate, pause-gated session):

## 8. Caller walkthrough (Lane A lifecycle)

1. **Commission** — the caller repo hits an ADR-67 convene threshold (hub-owned judgment).
2. **Author** — write the brief per `COUNCIL_QUESTION_GUIDE.md` (ADR-95 lane discipline:
   architect frames substance; CC expands mechanically). Frontmatter or CLI flags both
   carry overrides (flag > frontmatter > config default).
3. **Pre-flight (optional)** — health/liveness check before an important run.
4. **Invoke** — `council --file <brief.md> --return-dir <caller>/docs/decisions/inbox
   [--format json] [flags]` from the caller's own cwd.
5. **Handle exit code** — per §4; on 3, carry the degradation note forward.
6. **Consume** — read verdict + minority report + metrics from the return dir.
7. **Record** — the caller's CC session drafts the ADR in the caller repo (ADR-67 step 5);
   the verdict artifact is referenced, not copied into the hub (hub is never a default).
8. **Close out** — caller-side commit; any transcript mirroring only via explicit
   `--target-project`.

Boundaries: the caller never writes into ai-council; ai-council writes into the caller only
at the declared `--return-dir` (and explicit `--target-project` mirrors).
