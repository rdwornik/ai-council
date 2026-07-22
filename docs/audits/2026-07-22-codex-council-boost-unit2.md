# Codex Review — council-boost-unit2

**Date:** 2026-07-22
**Branch:** `feat/council-boost`
**HEAD:** `850f37f`
**Diff range:** `main..feat/council-boost`
**Codex version:** codex-cli 0.145.0
**Mode:** diff-review

---

## Focus

- Unit 2 P1: new `council boost` input stage (src/ai_council/boost.py, CLI registration, BoostConfig, settings.yaml boost block, tests/test_boost.py).
- Verify contracts on the tree as it will land, not per-file: boost's emitted briefs must be consumed by the EXISTING entry path unchanged (inbox.parse_file, resolve_mode, the six council frontmatter keys) — check nothing downstream changed.
- FR-B5 confabulation guard is the keystone: brief bodies must only ever contain caller text + fixed module constants; the decompose verbatim-token gate must be airtight (can any LLM text bypass _is_verbatim?).
- Exit-code contract per ADR-08 (0/1/3) in the boost CLI command.
- detect_mode, debate stage, verdict/output layer must be untouched.

---

## Findings
## CRITICAL

## [CRITICAL] src/ai_council/boost.py:229 — Rejected classifier output is emitted into briefs

**What:** An arbitrary invalid classifier response is interpolated into `SOURCE_FALLBACK_UNEXPECTED` and then written in the advisory block.  
**Why:** LLM-produced text can reach a brief without passing any guard, violating FR-B5’s caller-text-plus-fixed-scaffold contract.  
**Fix direction:** Log rejected content only; emit a fixed fallback reason in the brief.

## [CRITICAL] src/ai_council/boost.py:262 — “Verbatim” gate permits injected and reordered LLM prose

**What:** `_is_verbatim()` ignores all function words and checks token-set inclusion, so `RESEARCH: you should do migration` passes for a raw question containing only `migration`.  
**Why:** New LLM wording and altered meaning can enter hybrid briefs, defeating the FR-B5 keystone guard.  
**Fix direction:** Accept only exact caller spans (or an equivalently order-preserving, all-token validation), then emit the verified source text rather than the model response.

## HIGH

## [HIGH] src/ai_council/boost.py:537 — Caller frontmatter can override forced research routing

**What:** `**caller_meta` is merged after `"mode": "research"`, so a source file with `mode: pick` produces a research-classified brief that runs through debate.  
**Why:** This bypasses the required emitted research sub-commission path and reintroduces downstream mode detection/selection behavior.  
**Fix direction:** Apply the forced research mode after caller metadata and handle a conflicting caller mode as an advisory.

## [HIGH] src/ai_council/boost.py:543 — Boosted decision briefs pin global rounds, bypassing mode defaults

**What:** Every decision brief gets `rounds: config.defaults.rounds` (also hybrid decision briefs at line 528).  
**Why:** The existing entry path treats frontmatter rounds as higher precedence than mode defaults; e.g. an `ideas` brief runs two rounds instead of its configured one.  
**Fix direction:** Omit `rounds` unless the caller explicitly supplied it, allowing the unchanged entry path to select the resolved mode’s default.

## [HIGH] src/ai_council/boost.py:500 — Brief names can overwrite each other

**What:** Filenames use only second-resolution timestamps plus a six-word slug, followed by unconditional writes.  
**Why:** Concurrent or rapid identical/similarly-prefixed boosts can silently overwrite a prior brief.  
**Fix direction:** Use collision-resistant names or exclusive creation with a retry/error path.

## MEDIUM

(none)

## LOW

(none)
