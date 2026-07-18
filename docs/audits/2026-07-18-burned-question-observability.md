# BURNED question — STREAM SMOKE (worktree smoke-pair)

**Status:** BURNED — exclude from the blind parity corpus.
**Date:** 2026-07-18
**Stream:** SMOKE (operator quality test, independent of the 12-brief corpus).
**Reason:** This question was run live through both the CLI and API arms as the smoke
pair (see `2026-07-18-smoke-pair-cli-vs-api-report.md`). It has been consumed / seen by
the panel models, so it must NOT be reused as a blind-corpus brief.

> **Integration note for the primary:** at corpus-assembly / integration, exclude the
> topic below. The parity corpus was frozen with 12 topics (none observability); this
> burn is belt-and-suspenders so the smoke topic is never drafted into a future corpus.

---

## Burned question (verbatim, as run)

> We are a 3-engineer team running ~8 backend services in production. We need to choose an
> observability stack covering metrics, logs, distributed traces, and alerting. The options:
> (A) a managed SaaS platform (e.g. Datadog/Honeycomb) — one vendor, per-host + per-GB-ingest
> pricing, minimal setup; (B) self-hosted open-source (Prometheus + Loki + Tempo + Grafana +
> Alertmanager) running on our own infrastructure; or (C) rely solely on the cloud provider's
> native tooling (CloudWatch / Cloud Monitoring). Optimize for signal quality during a live
> incident at 2am, the ongoing operational burden on a 3-person team, and total cost at our
> scale (~8 services, moderate traffic). Which do we adopt now, what do we explicitly NOT do,
> and what do we defer until we have more scale or headcount?

**Topic domain:** observability / monitoring stack selection (managed SaaS vs self-hosted OSS
vs cloud-native). Confirmed zero topical overlap with the frozen 12 corpus topics.

---

## Also discarded (drafted, never used as the final pair)

An earlier draft — **webhook-ingestion topology** (managed queue vs serverless-per-vendor vs
monolithic in-process workers) — was drafted and briefly run, then **discarded before use**
for topical adjacency to the frozen corpus (backend-architecture domain shared with
"REST vs gRPC", "monorepo vs polyrepo", "PostgreSQL vs document store"). Its outputs were
cleared and it was replaced by the observability question above. Recorded here for honesty;
it is NOT the burned smoke pair. If ever considered for the corpus, note it was seen by the
panel during the discarded run and should also be treated as burned.
