#!/usr/bin/env python
"""verify_output_writes.py — re-runnable witness for the Lane A1 write-contract arc.

Codifies the fail-loud write semantics shipped for #35 / #62 / #63 / #60 as a
deterministic, $0, offline checker: it EXERCISES the shipped writer functions (never reads
diffs) and prints a PASS/FAIL/GAP table over ten legs, each mapped to the BACKLOG id it
proves.

Follows the scripts/ verify_*/validate_* sibling convention (verify_night_consolidation.py,
validate_backlog.py, validate_audit_casing.py): read-only w.r.t. tracked files,
self-contained, exit 0 only when no leg FAILs. Layer-2 invariant: it drives the library
in-process; it never spawns a debate, never calls a provider, never writes a tracked file.

Legs (claim -> shipped code):
  L1  #35  transcript required --return-dir miss raises, canonical survives  (output.save_to_file)
  L2  #35  minority report carries the same R4 guarantee            (output.save_minority_report)
  L3  #62  research report carries it too                   (research.output.save_research_to_file)
  L4  #26  verdict package still raises (regression guard)      (output.save_verdict_package)
  L5  #35  a common-mode fault aggregates: ONE raise naming all three, every canonical
           artifact still on disk                                  (output.raise_for_routing_failures)
  L6  #63  sidecar failure DEGRADES: transcript returns, verdict package still emitted,
           and the failure is machine-readable on the #26 degradation two-signal
  L7  #63  the verdict manifest never advertises a path that is not on disk
  L8  #60  a prose-only options heading falls back; a fallback that finds nothing does not
           clobber the synthesis heading                            (output._extracted_options)
  L9  ---  happy path: all four deliverables route to a good --return-dir, no false raise
  L10 #35  ORCHESTRATOR wiring: the aggregate really is raised by CouncilRunner.run after
           every canonical write, not just by the helper in isolation

Failure induction is HARNESS-SIDE only — no provider or shipped code is mutated:
  - return-dir faults: point --return-dir at an existing FILE, so its mkdir raises.
  - sidecar fault: intercept Path.write_text for *_metrics.json only. The plan called for
    pre-creating that path as a directory; the sidecar name is derived from a wall-clock
    _ts() stem and so is not predictable ahead of the call, hence the stdlib-boundary
    interception instead. Same effect, no shipped code touched.

Anything un-exercisable prints GAP, never a fake PASS. A GAP does not fail the run; a FAIL
does.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

# The legs deliberately induce failures, so the library logs warnings/exceptions by design.
# Silence them: the PASS/FAIL table is this script's only output contract.
logging.getLogger("ai_council").setLevel(logging.CRITICAL)

# Test THIS repo's code: prepend the co-located src + repo root so the checker never picks up a
# sibling worktree's editable install (the shared-.venv quirk). Order matters — front of path.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))


class LegResult:
    __slots__ = ("leg", "id_", "verdict", "evidence")

    def __init__(self, leg: str, id_: str, verdict: str, evidence: str) -> None:
        self.leg, self.id_, self.verdict, self.evidence = leg, id_, verdict, evidence


def _ok(leg: str, id_: str, evidence: str) -> LegResult:
    return LegResult(leg, id_, "PASS", evidence)


def _fail(leg: str, id_: str, evidence: str) -> LegResult:
    return LegResult(leg, id_, "FAIL", evidence)


def _gap(leg: str, id_: str, evidence: str) -> LegResult:
    """Un-exercisable in this harness. Reported, never counted as a pass."""
    return LegResult(leg, id_, "GAP", evidence)


# --------------------------------------------------------------------------------------------
# Shared fixtures — hand-built so the checker never depends on settings.yaml contents
# --------------------------------------------------------------------------------------------
_DISSENT_SYNTHESIS = (
    "## Consensus\nBoth options are viable.\n\n"
    "## Unresolved Disagreements\n"
    "The crux: Claude argued Postgres is sufficient at this scale, while Grok held that "
    "Cassandra is required for the projected write volume.\n\n"
    "## Recommended Decision\nStart with Postgres.\n"
)


def _debate_result(synthesis: str = _DISSENT_SYNTHESIS, **overrides):
    """A minimal but REAL DebateResult — the shipped writers are driven with this."""
    from ai_council.models import DebateResult, ModelResponse, Question, Round

    def _resp(provider: str, content: str) -> ModelResponse:
        return ModelResponse(
            provider=provider, model=f"{provider}-model", round_number=1,
            content=content, latency_sec=0.1, token_count=10,
        )

    responses = [
        _resp("claude", "Postgres is enough."),
        _resp("grok", "Cassandra is required."),
    ]
    base = dict(
        question=Question(text="Which datastore?\n\n## Options\n- (a) Postgres\n- (b) Cassandra\n",
                          source="cli"),
        rounds=[Round(number=1, responses=responses)],
        synthesis=synthesis,
        synthesizer="openai",
        total_duration_sec=1.0,
        mode="pick",
    )
    base.update(overrides)
    return DebateResult(**base)


def _blocker(td: Path) -> Path:
    """A FILE where a directory is expected, so mkdir on it raises. Harness-side induction."""
    p = td / "blocker"
    p.write_text("not a dir", encoding="utf-8")
    return p


def _break_sidecar_write():
    """Context manager: make only the *_metrics.json write fail."""
    original = Path.write_text

    def _selective(self: Path, *args, **kwargs):
        if self.name.endswith("_metrics.json"):
            raise PermissionError("sidecar blocked")
        return original(self, *args, **kwargs)

    return patch.object(Path, "write_text", _selective)


# --------------------------------------------------------------------------------------------
# L1 — #35 transcript required return-dir miss raises, canonical survives
# --------------------------------------------------------------------------------------------
def leg_l1() -> LegResult:
    from ai_council.output import OutputRoutingError, save_to_file

    with tempfile.TemporaryDirectory(prefix="vow-l1-") as td:
        root = Path(td)
        out = root / "output"
        raised = None
        try:
            save_to_file(_debate_result(), out, return_dir=_blocker(root))
        except OutputRoutingError as exc:
            raised = exc

        if raised is None:
            return _fail("L1", "#35", "no OutputRoutingError: a required return-dir miss was swallowed")
        artifacts = [f.artifact for f in raised.failures]
        canonical = list(out.glob("council-out-*.md"))
        ev = f"raised for {artifacts}; canonical transcripts on disk={len(canonical)}"
        return _ok("L1", "#35", ev) if artifacts == ["transcript"] and len(canonical) == 1 else _fail("L1", "#35", ev)


# --------------------------------------------------------------------------------------------
# L2 — #35 minority report carries the same guarantee (its docstring long claimed it did)
# --------------------------------------------------------------------------------------------
def leg_l2() -> LegResult:
    from ai_council.output import OutputRoutingError, save_minority_report

    with tempfile.TemporaryDirectory(prefix="vow-l2-") as td:
        root = Path(td)
        out = root / "output"
        raised = None
        try:
            save_minority_report(_debate_result(), out, return_dir=_blocker(root))
        except OutputRoutingError as exc:
            raised = exc

        if raised is None:
            return _fail("L2", "#35", "no OutputRoutingError: minority return-dir miss was swallowed")
        artifacts = [f.artifact for f in raised.failures]
        canonical = list(out.glob("council-minority-*.md"))
        ev = f"raised for {artifacts}; canonical minority reports on disk={len(canonical)}"
        good = artifacts == ["minority report"] and len(canonical) == 1
        return _ok("L2", "#35", ev) if good else _fail("L2", "#35", ev)


# --------------------------------------------------------------------------------------------
# L3 — #62 research path (before this arc it had no required-destination check at all)
# --------------------------------------------------------------------------------------------
def leg_l3() -> LegResult:
    from ai_council.output import OutputRoutingError
    from ai_council.research.models import MergedResearchReport
    from ai_council.research.output import save_research_to_file

    report = MergedResearchReport(
        query="datastore benchmarks",
        results=[],
        merged_report="# Findings\nNothing conclusive.",
        summary_2500="Short summary.",
        total_cost_usd=0.0,
        total_duration_sec=1.0,
        total_sources=0,
        cache_key="k",
    )
    with tempfile.TemporaryDirectory(prefix="vow-l3-") as td:
        root = Path(td)
        out = root / "output"
        raised = None
        try:
            save_research_to_file(report, out, return_dir=_blocker(root))
        except OutputRoutingError as exc:
            raised = exc

        if raised is None:
            return _fail("L3", "#62", "no OutputRoutingError: research return-dir miss was swallowed")
        artifacts = [f.artifact for f in raised.failures]
        canonical = list(out.glob("council-out-*research*.md"))
        ev = f"raised for {artifacts}; canonical research reports on disk={len(canonical)}"
        good = artifacts == ["research report"] and len(canonical) == 1
        return _ok("L3", "#62", ev) if good else _fail("L3", "#62", ev)


# --------------------------------------------------------------------------------------------
# L4 — #26 regression guard: the verdict package's original guarantee still holds
# --------------------------------------------------------------------------------------------
def leg_l4() -> LegResult:
    from ai_council.output import OutputRoutingError, save_to_file, save_verdict_package

    with tempfile.TemporaryDirectory(prefix="vow-l4-") as td:
        root = Path(td)
        out = root / "output"
        result = _debate_result()
        transcript = save_to_file(result, out)[0]
        raised = None
        try:
            save_verdict_package(result, out, transcript, return_dir=_blocker(root))
        except OutputRoutingError as exc:
            raised = exc

        if raised is None:
            return _fail("L4", "#26", "no OutputRoutingError: the pre-existing R4 guarantee regressed")
        artifacts = [f.artifact for f in raised.failures]
        canonical = list(out.glob("council-verdict-*.json"))
        ev = f"raised for {artifacts}; canonical verdict packages on disk={len(canonical)}"
        good = artifacts == ["verdict package"] and len(canonical) == 1
        return _ok("L4", "#26", ev) if good else _fail("L4", "#26", ev)


# --------------------------------------------------------------------------------------------
# L5 — #35 aggregate: ONE raise naming every missed deliverable, all canonical intact
# --------------------------------------------------------------------------------------------
def leg_l5() -> LegResult:
    from ai_council.output import (
        OutputRoutingError,
        RoutingFailure,
        raise_for_routing_failures,
        save_minority_report,
        save_to_file,
        save_verdict_package,
    )

    with tempfile.TemporaryDirectory(prefix="vow-l5-") as td:
        root = Path(td)
        out = root / "output"
        blocked = _blocker(root)
        result = _debate_result()

        # the orchestrator's sequence, with its shared accumulator
        failures: list[RoutingFailure] = []
        transcript = save_to_file(result, out, return_dir=blocked, routing_failures=failures)
        minority = save_minority_report(
            result, out, return_dir=blocked, routing_failures=failures,
            stem_base=transcript[0].stem[len("council-out-"):],
        )
        save_verdict_package(
            result, out, transcript[0], written={"minority": minority},
            return_dir=blocked, routing_failures=failures,
        )

        # every canonical artifact must have survived the common-mode fault
        present = (
            len(list(out.glob("council-out-*.md"))) == 1
            and len(list(out.glob("council-minority-*.md"))) == 1
            and len(list(out.glob("council-verdict-*.json"))) == 1
        )
        raised = None
        try:
            raise_for_routing_failures(failures)
        except OutputRoutingError as exc:
            raised = exc

        if raised is None:
            return _fail("L5", "#35", "aggregate did not raise despite recorded failures")
        artifacts = [f.artifact for f in raised.failures]
        expected = ["transcript", "minority report", "verdict package"]
        ev = f"one raise naming {artifacts}; all three canonical artifacts present={present}"
        return _ok("L5", "#35", ev) if artifacts == expected and present else _fail("L5", "#35", ev)


# --------------------------------------------------------------------------------------------
# L6 — #63 sidecar failure degrades AND is machine-readable, not log-only
# --------------------------------------------------------------------------------------------
def leg_l6() -> LegResult:
    from ai_council.models import DebateMetrics
    from ai_council.output import save_to_file, save_verdict_package

    with tempfile.TemporaryDirectory(prefix="vow-l6-") as td:
        out = Path(td) / "output"
        result = _debate_result(metrics=DebateMetrics())

        with _break_sidecar_write():
            transcript = save_to_file(result, out)[0]  # must NOT raise
            verdict = save_verdict_package(
                result, out, transcript, written={"transcript": [transcript]}
            )

        data = json.loads(verdict[0].read_text(encoding="utf-8"))
        degraded = data["degradation"]["degraded"] is True
        explained = "metrics sidecar not written" in (data["degradation"]["summary"] or "")
        exit_zero = data["exit_semantics"] == 0
        no_sidecar = not list(out.glob("*_metrics.json"))
        ev = (
            f"verdict emitted={verdict[0].exists()}; sidecar absent={no_sidecar}; "
            f"degraded={degraded}; summary names it={explained}; exit_semantics={data['exit_semantics']}"
        )
        good = verdict[0].exists() and no_sidecar and degraded and explained and exit_zero
        return _ok("L6", "#63", ev) if good else _fail("L6", "#63", ev)


# --------------------------------------------------------------------------------------------
# L7 — #63 the manifest never advertises a path that is not on disk
# --------------------------------------------------------------------------------------------
def leg_l7() -> LegResult:
    from ai_council.output import save_to_file, save_verdict_package

    with tempfile.TemporaryDirectory(prefix="vow-l7-") as td:
        out = Path(td) / "output"
        result = _debate_result()
        transcript = save_to_file(result, out)[0]
        phantom = out / "council-out-phantom_metrics.json"  # never written

        verdict = save_verdict_package(
            result, out, transcript,
            written={"transcript": [transcript], "metrics": [phantom]},
        )
        data = json.loads(verdict[0].read_text(encoding="utf-8"))
        kinds = {a["kind"] for a in data["artifacts"]}
        # the verdict's own entry is a prediction built before its write — excluded by design
        claimed = [
            p for a in data["artifacts"] if a["kind"] != "verdict" for p in a["paths"]
        ]
        all_real = all(Path(p).exists() for p in claimed)
        ev = f"kinds={sorted(kinds)}; phantom dropped={'metrics' not in kinds}; every claimed path exists={all_real}"
        return _ok("L7", "#63", ev) if "metrics" not in kinds and all_real else _fail("L7", "#63", ev)


# --------------------------------------------------------------------------------------------
# L8 — #60 prose-only options heading falls back, without clobbering the heading
# --------------------------------------------------------------------------------------------
def leg_l8() -> LegResult:
    from ai_council.output import _extracted_options, _split_sections

    prose_only = (
        "## Recommendation\nAdopt YAML.\n\n"
        "## Alternatives Considered\nThe panel weighed them at length and converged.\n"
    )
    with_options = "Which format?\n\n## Options\n- (a) Keep YAML\n- (b) Move to TOML\n"
    without_options = "Which format? No options section here.\n"

    fell_back = _extracted_options(
        _split_sections(prose_only), question_sections=_split_sections(with_options)
    )
    guarded = _extracted_options(
        _split_sections(prose_only), question_sections=_split_sections(without_options)
    )

    fallback_ok = fell_back["items"] == ["(a) Keep YAML", "(b) Move to TOML"]
    # the clobber guard: a fallback that yields nothing must not null out a real heading
    guard_ok = guarded["items"] == [] and guarded["heading"] == "Alternatives Considered"
    ev = f"fallback items={fell_back['items']}; guarded heading={guarded['heading']!r}"
    return _ok("L8", "#60", ev) if fallback_ok and guard_ok else _fail("L8", "#60", ev)


# --------------------------------------------------------------------------------------------
# L9 — happy path: a good --return-dir receives everything, and nothing raises
# --------------------------------------------------------------------------------------------
def leg_l9() -> LegResult:
    from ai_council.output import save_minority_report, save_to_file, save_verdict_package

    with tempfile.TemporaryDirectory(prefix="vow-l9-") as td:
        root = Path(td)
        out, ret = root / "output", root / "return"
        result = _debate_result()

        transcript = save_to_file(result, out, return_dir=ret)
        minority = save_minority_report(result, out, return_dir=ret)
        verdict = save_verdict_package(result, out, transcript[0], return_dir=ret)

        routed = [p for group in (transcript, minority, verdict) for p in group if p.parent == ret]
        ev = f"routed to return-dir={len(routed)}/3 deliverables, all present={all(p.exists() for p in routed)}"
        good = len(routed) == 3 and all(p.exists() for p in routed)
        return _ok("L9", "---", ev) if good else _fail("L9", "---", ev)


# --------------------------------------------------------------------------------------------
# L10 — #35 the ORCHESTRATOR really raises the aggregate after every canonical write
# --------------------------------------------------------------------------------------------
def leg_l10() -> LegResult:
    """Drives the real CouncilRunner.run with MockProvider seats.

    run_debate and synthesize are stubbed (they would call providers); every writer below
    them is the shipped one. This is the leg that proves the record-and-aggregate wiring,
    which the repo's own orchestrator tests do not cover — they patch save_to_file out.
    """
    from ai_council.models import DebateOutcome, RunRequest
    from ai_council.orchestrator import CouncilRunner
    from ai_council.output import OutputRoutingError
    from ai_council.policy import RunPolicy
    from config.config_loader import AppConfig, DefaultsConfig, ModelConfig, PromptsConfig
    from tests.conftest import MockProvider

    def _model(name: str) -> ModelConfig:
        return ModelConfig(name=name, sdk="test", model=f"{name}-model",
                           api_key_env="TEST_KEY", timeout_sec=60, max_tokens=1000)

    with tempfile.TemporaryDirectory(prefix="vow-l10-") as td:
        root = Path(td)
        out, blocked = root / "output", _blocker(root)
        result = _debate_result()

        config = AppConfig(
            defaults=DefaultsConfig(
                rounds=1, max_rounds=3, output_dir=out, synthesizer="openai",
                default_panel=["claude", "grok"], full_panel=["claude", "grok", "openai"],
            ),
            models={n: _model(n) for n in ("claude", "grok", "openai")},
            prompts=PromptsConfig(
                initial="{persona}\nAnswer: {question}",
                critique="{persona}\nRound {round}. {question}\n{previous_responses_anonymized}",
                synthesis="Question: {question}\n\n{full_transcript}\n\nSynthesize:",
                personas={"claude": "Be Claude.", "grok": "Be Grok.", "openai": "Be GPT."},
            ),
            available_providers={"claude", "grok", "openai"},
        )
        providers = {n: MockProvider(n) for n in ("claude", "grok", "openai")}
        request = RunRequest(
            question=result.question,
            panel_names=["claude", "grok"],
            synthesizer_name="openai",
            rounds=1,
            policy=RunPolicy.default(),
            panel_mode="custom",
            return_dir=blocked,
        )

        import asyncio

        raised = None
        # run() builds its own Console locally, so there is no module attribute to patch.
        # Redirect stdout instead — Rich resolves sys.stdout lazily. StringIO, never
        # /dev/null: this repo is Windows-first (repo anti-pattern).
        with (
            patch("ai_council.orchestrator.run_debate",
                  new=AsyncMock(return_value=DebateOutcome(rounds=result.rounds))),
            patch("ai_council.orchestrator.synthesize", new=AsyncMock(return_value=result)),
            patch("ai_council.orchestrator.print_round_summary"),
            patch("ai_council.orchestrator.print_synthesis"),
            patch("ai_council.orchestrator.print_cost_summary"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                asyncio.run(CouncilRunner(providers, config).run(request, output_dir=out))
            except OutputRoutingError as exc:
                raised = exc

        if raised is None:
            return _fail("L10", "#35", "CouncilRunner.run did not raise on a required return-dir miss")
        artifacts = [f.artifact for f in raised.failures]
        present = (
            len(list(out.glob("council-out-*.md"))) == 1
            and len(list(out.glob("council-minority-*.md"))) == 1
            and len(list(out.glob("council-verdict-*.json"))) == 1
        )
        ev = f"orchestrator raised once for {artifacts}; all canonical artifacts present={present}"
        good = artifacts == ["transcript", "minority report", "verdict package"] and present
        return _ok("L10", "#35", ev) if good else _fail("L10", "#35", ev)


LEGS = [leg_l1, leg_l2, leg_l3, leg_l4, leg_l5, leg_l6, leg_l7, leg_l8, leg_l9, leg_l10]


def main() -> int:
    results: list[LegResult] = []
    for fn in LEGS:
        name = fn.__name__.split("_")[-1].upper()
        try:
            results.append(fn())
        except Exception as exc:  # a leg that raises is a FAIL, never a silent skip
            results.append(_fail(name, "?", f"EXCEPTION: {type(exc).__name__}: {exc}"))

    print("=" * 78)
    print("OUTPUT-WRITE CONTRACT VERIFICATION  -  Lane A1 (#35 #62 #63 #60)")
    print("=" * 78)
    print(f"{'LEG':<6}{'ID':<8}{'VERDICT':<9}EVIDENCE")
    print("-" * 78)
    for r in results:
        print(f"{r.leg:<6}{r.id_:<8}{r.verdict:<9}{r.evidence}")
    print("-" * 78)
    passed = sum(r.verdict == "PASS" for r in results)
    failed = sum(r.verdict == "FAIL" for r in results)
    gaps = sum(r.verdict == "GAP" for r in results)
    print(f"RESULT: {passed}/{len(results)} PASS, {failed} FAIL, {gaps} GAP")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
