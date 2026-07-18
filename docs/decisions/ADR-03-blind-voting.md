# ADR-03: Blind Voting in Round 2

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — `_anonymize_responses()` (`debate.py:20`). No open remainder. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-02-25
**Status:** Accepted
**Decision:** Anonymize Round 1 responses before Round 2 critique to eliminate anchoring bias.

**Context:**
Without anonymization, models can identify their own prior responses and those of known peers,
leading to tribal reinforcement rather than honest critique.

**Implementation:** `_anonymize_responses()` in `src/debate.py` — shuffles + labels as "Proposal A/B/C".
Provider names hidden; each model sees only anonymous proposals in Round 2 critique prompt.

**Trade-off:** Loses attribution in critique; accepted — impartiality > traceability in Round 2.
