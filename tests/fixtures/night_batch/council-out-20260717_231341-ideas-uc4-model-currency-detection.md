# AI Council Debate: # Ideas: How could ai-council detect when its configured panel models fall behin

**Date:** 2026-07-17 23:13:41
**Panel:** claude-haiku-4-5-20251001, gpt-5.6-sol, gemini-3.1-pro-preview
**Synthesizer:** openai (non-participant)
**Rounds:** 1
**Duration:** 45.4s
**Panel Mode:** custom
**Debate Mode:** ideas
**Source:** C:\Users\1028120\Documents\Dev\ai-council\council_inbox\uc4-model-currency-detection.md
**Cost:** ~$0.0287 (6,377 tokens)

## Verdict Summary

**Decision:** Prototype a **doctor freshness check** with three layers:
**Dissent:** unanimous
**Panel seated:** 3/3
**Verdict author:** openai (non-participant)

_Machine-readable fields are authoritative in the council-verdict-*.json sibling._

---

## Question

# Ideas: How could ai-council detect when its configured panel models fall behind the latest releases? (#17)

## Context
ai-council pins specific model strings per provider in config/settings.yaml (e.g. gemini-3.1-pro-preview, gpt-5.4, claude-opus-4-7, grok-4.3, deepseek-v4-pro). Models are superseded frequently. Today nothing surfaces a stale pin — the Council can silently debate on an outdated model. config/settings.yaml is the single source of the model strings.

## The ask
Generate approaches for a check that reports any configured model no longer the latest available per provider, plus a documented refresh process. Optimize for breadth: cheap heuristics AND richer approaches, automated AND manual, plus wild cards.

## Constraints to respect
- Providers have no uniform "list latest models" API; each approach must state HOW it learns "latest" per provider (API listing endpoint, web check, changelog scrape, manual watch, etc.).
- Must run without spending on a full debate; a doctor-style advisory (never blocking) is the target surface.
- Windows-first, Python, config-as-single-source; no new hard dependency without justification.
- False "you are stale" alarms must be cheap to dismiss (heed-ability).

## Prior decisions in scope
config/settings.yaml single source, the doctor advisory surface (DRAFT-DOC-2 pin-upgrade lifecycle), ADR-08 advisory exit semantics.

## Round 1: Initial Responses

### Claude (claude-haiku-4-5-20251001)

## Ideas

- **Version-Pattern Matcher**: Parse model strings for date/version hints (opus-4-7 vs -4-8, gemini-3.1 vs -4.0) and flag if a newer pattern exists in public docs or GitHub issues. **Why it might work**: cheap regex + web grep, no API calls needed. **Feasibility**: easy

- **Provider API Poller**: Hit each provider's public `/models` or equivalent endpoint (OpenAI, Anthropic, Google, etc.—each needs custom logic) and diff against config pins; run once per week, cache results. **Why it might work**: authoritative source, one-liner per provider, optional persistence layer. **Feasibility**: medium

- **Staleness-by-Release-Date**: Hardcode model announcement dates (Anthropic drops a new model → add to lookup table), flag any pin older than 6 months or X major versions behind. **Why it might work**: trivial to maintain, cold-start solution while APIs come online. **Feasibility**: easy

- **Community Model Registry**: Maintain a curated `latest-models.json` versioned in the repo, updated via (infrequent) PR when providers announce; advisory reads from it. **Why it might work**: human review filters noise, explicit contract, zero API coupling. **Feasibility**: easy

- **Multi-Source Consensus Check**: Run API check + RSS feed scrape + regex-on-product-page in parallel, surface advisories only if 2+ agree on staleness. **Why it might work**: filters false alarms cheaply, survives one data source breaking. **Feasibility**: medium

- **Changelog Diffing**: Watch provider GitHub repos or public changelogs (RSS), extract model release announcements, maintain a moving window of "known latest per provider." **Why it might work**: source-of-truth agnostic, catches rapid minor bumps. **Feasibility**: medium

- **Performance Regression Canary**: Test a debate run with latest-available model pins in staging, compare output quality/latency/cost vs. current prod pins; flag if delta exceeds threshold. **Why it might work**: operationally grounded (detects real degradation risk), not just version drift. **Feasibility**: hard

- **Weak-Signal LLM Check**: Include a meta-prompt in advisories: "Are these model strings current as of your knowledge cutoff?" Treat response as heuristic (not binding). **Why it might work**: requires no new infrastructure, uses existing model capacity. **Feasibility**: easy

## Wild Card

**Vendor Webhook Registry**: Ask major providers (Anthropic, OpenAI) if they'll send structured webhooks on model releases; maintain a simple `/webhook/model-announced` endpoint that auto-updates a cache without polling. **Why it might work**: real-time, no polling overhead, shifts burden to vendors; high-impact if adoption rises. **Feasibility**: hard (requires vendor buy-in)

## Connections

- **Version-Pattern Matcher + Community Model Registry**: Registry holds the ground-truth patterns; matcher enforces them locally, catches skew before it lands.
- **Provider API Poller + Multi-Source Consensus**: API is primary source; if it fails, consensus falls back to RSS/changelog, only surfaces high-confidence staleness.
- **Staleness-by-Release-Date + Performance Regression Canary**: Date check is fast advisory; canary is the deep-dive validation, triggered only on flagged pins.

*Latency: 34.36s | Tokens: 1539*

### Deepseek (gpt-5.6-sol)

## Ideas

- **Provider Capability Probe**
  - **What**: Doctor calls each provider’s model-listing endpoint where available, compares normalized release families against `settings.yaml`, and caches results for 24 hours.
  - **Why it might work**: It is authoritative, costs no inference tokens, and should add only roughly 0.2–2 seconds per provider when checks run concurrently.
  - **Feasibility**: medium

- **Changelog Watchers**
  - **What**: Small provider-specific adapters fetch official release notes, model documentation, RSS feeds, or announcement pages and extract model identifiers using regexes.
  - **Why it might work**: It covers providers without discovery APIs while keeping every “latest” claim traceable to a URL and observation date.
  - **Feasibility**: medium

- **Curated Latest Registry**
  - **What**: Maintain a versioned, signed JSON file mapping provider and model class to the newest known pin, refreshed manually from official sources.
  - **Why it might work**: Runtime checks become deterministic and sub-millisecond, while stale registry data yields “unknown” rather than a false alarm.
  - **Feasibility**: easy

- **Hybrid Confidence Ladder**
  - **What**: Combine API listings, official-page detection, and the curated registry into `confirmed`, `probable`, or `unknown` findings, with only confirmed mismatches saying “stale.”
  - **Why it might work**: Confidence-ranked evidence makes alarms cheap to dismiss and limits noisy provider-page changes to informational notices.
  - **Feasibility**: medium

- **Alias Resolution Check**
  - **What**: Send metadata-only or minimal zero-output requests to provider aliases such as `latest`, then inspect resolved-model headers or response metadata without running a debate.
  - **Why it might work**: It asks the provider what “latest” currently resolves to and should cost zero or fractions of a cent where model resolution is exposed.
  - **Feasibility**: hard

- **Release Age Budget**
  - **What**: Record each configured pin’s first-seen release date from official changelogs and warn for manual review after a provider-specific age threshold.
  - **Why it might work**: It catches likely drift even when “latest” cannot be established, while wording the result as an age signal rather than a stale verdict.
  - **Feasibility**: easy

- **Scheduled Registry PR**
  - **What**: A weekly CI job runs provider adapters, proposes registry and documentation updates, and includes evidence links plus the exact `settings.yaml` pin changes for human approval.
  - **Why it might work**: Humans handle ambiguous naming and preview-versus-stable choices, while automation reduces refresh work to reviewing a small diff.
  - **Feasibility**: medium

- **Doctor Evidence Cards**
  - **What**: Doctor prints configured pin, candidate latest pin, source, confidence, checked-at time, and copyable refresh/ignore commands while always preserving ADR-08 advisory exit behavior.
  - **Why it might work**: Each alert can be verified or dismissed in seconds, improving heed-ability without blocking normal Council operation.
  - **Feasibility**: easy

## Wild Card

- **Community Freshness Quorum**: Consume several public model catalogs or SDK registries and report a candidate only when two independent sources agree; disagreement becomes “watching” rather than “stale,” creating broad coverage without trusting one unofficial feed.

## Connections

- **Provider Capability Probe + Changelog Watchers + Hybrid Confidence Ladder** combines authoritative discovery with fallback coverage and calibrated alarms.
- **Curated Latest Registry + Scheduled Registry PR + Doctor Evidence Cards** creates a low-latency runtime check backed by an auditable human refresh process.

*Latency: 28.62s | Tokens: 6097*

### Gemini (gemini-3.1-pro-preview)

## Ideas
- **Name**: Native API Polling
- **What**: Query each provider's `/v1/models` (or equivalent) endpoint using existing API keys and apply regex heuristics to identify newer version strings.
- **Why it might work**: Leverages existing infrastructure and credentials without requiring external dependencies, providing authoritative data directly from the source.
- **Feasibility**: medium

- **Name**: Community Version Manifest
- **What**: Maintain a lightweight, public `latest_models.json` on a known GitHub repository that `ai-council` fetches periodically to compare against local pins.
- **Why it might work**: Centralizes the complex logic of tracking provider updates into one easily updated file, simplifying the local application code.
- **Feasibility**: easy

- **Name**: Temporal Decay Heuristic
- **What**: Append a `last_updated` timestamp to each model pin in `settings.yaml` and trigger a doctor advisory if a model hasn't been updated in over 90 days.
- **Why it might work**: Provides a zero-external-call, ultra-cheap reminder that AI models move fast and require periodic manual verification.
- **Feasibility**: easy

- **Name**: Dependency Registry Sync
- **What**: Periodically poll the GitHub API for updates to a widely maintained LLM routing library (like LiteLLM or LangChain) and extract their latest supported model lists.
- **Why it might work**: Outsourcers the "latest model" discovery problem to larger open-source communities that already track these changes meticulously.
- **Feasibility**: medium

- **Name**: Automated Changelog Scraping
- **What**: Use a lightweight Python scraping library (like `requests` + `BeautifulSoup`) to periodically parse known provider release note URLs for new model string patterns.
- **Why it might work**: Catches announcements for providers that don't immediately update their API model endpoints or lack structured model discovery APIs.
- **Feasibility**: hard

- **Name**: Provider RSS Aggregator
- **What**: Subscribe a lightweight background task to provider engineering blog RSS/Atom feeds, parsing titles and summaries for known model family names and version bumps.
- **Why it might work**: A standardized, low-bandwidth way to monitor for release announcements without the brittleness of HTML scraping.
- **Feasibility**: medium

## Wild Card
- **Name**: LLM-Assisted Self-Audit
- **What**: Periodically run a cheap, fast model (like gpt-4o-mini) providing it the current `settings.yaml` and asking it to use web search (if available via the API) or its current knowledge cutoff to determine if newer versions exist for those specific model families.

## Connections
- **Native API Polling** + **Temporal Decay Heuristic**: Use the API polling where supported (OpenAI, Anthropic), but rely on the simple time-based decay for providers with opaque or non-existent model discovery endpoints.
- **Community Version Manifest** + **Provider RSS Aggregator**: Use the RSS aggregator in a separate, automated pipeline to keep the central GitHub `latest_models.json` manifest up-to-date, reducing the manual burden on maintainers.

*Latency: 17.59s | Tokens: 2152*

## Synthesis (by openai, non-participant)

## Idea Inventory
Total unique ideas generated (14).

## Top Tier (Implement Soon)
- **Provider-specific API/model listing adapters**
  - Query each provider’s official model discovery endpoint where available, normalize results, and compare against `config/settings.yaml` to report likely newer pins without running a debate.
  - **Who endorsed it:** claude, deepseek, gemini

- **Official changelog / release-note watchers**
  - For providers without reliable listing APIs, fetch official changelogs, docs pages, RSS feeds, or announcement pages and extract newer model identifiers with provider-specific rules.
  - **Who endorsed it:** claude, deepseek, gemini

- **Curated latest-model registry**
  - Maintain a repo-versioned JSON mapping of provider + model family to the newest known pin so doctor checks are fast, deterministic, and easy to audit.
  - **Who endorsed it:** claude, deepseek, gemini

- **Confidence-ranked advisory output**
  - Emit findings as `confirmed`, `probable`, or `unknown` with source URL, checked-at time, configured pin, and candidate replacement so false alarms are cheap to dismiss.
  - **Who endorsed it:** deepseek, claude

- **Time-based age/staleness heuristic**
  - Add a lightweight fallback warning when a pin has not been reviewed or updated for a fixed period, even if “latest” cannot be established automatically.
  - **Who endorsed it:** claude, deepseek, gemini

## Interesting But Unvalidated
- **Alias resolution check**
  - Probe provider aliases like `latest` or equivalent and inspect returned metadata/headers to learn what model the provider currently resolves to.
- **Scheduled registry PR bot**
  - Run a weekly job that gathers evidence, updates the registry, and proposes `settings.yaml` refreshes as a human-reviewed PR.
- **Community freshness quorum**
  - Compare several public catalogs or SDK registries and only surface a probable stale warning when multiple independent sources agree.
- **Dependency registry sync**
  - Mine model lists from actively maintained routing/tooling projects such as LiteLLM or LangChain as a secondary signal.
- **LLM-assisted self-audit**
  - Use a cheap model as a heuristic reviewer of current pins versus known newer releases, explicitly non-authoritative.

## Powerful Combinations
- **Adapter + watcher + confidence ladder**
  - Use official API listing as primary evidence, official changelog/doc scraping as fallback, and confidence tiers to keep advisories useful without being noisy.

- **Curated registry + scheduled PR + doctor evidence cards**
  - Put a human-reviewed registry in the repo, refresh it via automation-assisted PRs, and have doctor print evidence-rich advisories with exact suggested pin changes.

- **Age heuristic + authoritative checks**
  - Use age-based reminders as universal coverage for opaque providers, while letting API/changelog evidence upgrade warnings from “review soon” to “likely stale.”

## Discarded (and Why)
- **Performance regression canary**
  - Too heavy for the stated goal because it requires staged runs and evaluation infrastructure rather than a cheap doctor-style freshness advisory.

- **Vendor webhook registry**
  - Not practical near-term because it depends on provider buy-in and no current uniform webhook mechanism exists.

- **Pure regex/version-pattern matcher as sole mechanism**
  - Useful only as a weak heuristic, but too error-prone alone across inconsistent provider naming schemes to be trusted for stale verdicts.

## Suggested Next Step
Prototype a **doctor freshness check** with three layers:  
1. provider-specific official model-list adapters where possible,  
2. fallback official changelog/RSS/doc watchers, and  
3. a repo-versioned curated registry plus confidence-ranked evidence cards.

In parallel, document a **pin refresh process**: weekly or on-demand check, advisory output with source links, human review of candidate upgrades, update `config/settings.yaml`, and optional automation that opens a registry-refresh PR. If the team wants to narrow between “runtime live checks” and “registry-first workflow,” run a short pick debate on that implementation choice.
