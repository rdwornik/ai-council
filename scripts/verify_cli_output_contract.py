#!/usr/bin/env python
"""verify_cli_output_contract.py — re-runnable witness for the CLI output contract (Lane A2).

Codifies the Lane A2 arc (#65 doctor honours the output controls · #71 --no-persist cleans up
· CLI-boundary fail-loud · #66 live witness) as a deterministic, $0 checker: it EXERCISES the
shipped code paths in-process (never reads diffs) and prints a PASS/FAIL/GAP table over ten
legs, each mapped to the BACKLOG id it proves.

Follows the scripts/ verify_*/validate_* sibling convention (verify_night_consolidation.py,
validate_backlog.py, validate_audit_casing.py): read-only, self-contained, exit 0 on all
PASS / exit 1 on any FAIL. Idempotent — every leg cleans up after itself, so repeat runs are
identical and the tree is unchanged (CLAUDE.md §5.9 "No leftovers").

Legs (claim -> shipped code):
  L1   arc   ONE output resolver — no second copy of the precedence chain in src/  (cli.py)
  L2   #65   doctor honours --output                                    (cli.doctor -> doctor.run_doctor)
  L3   #65   doctor honours AICOUNCIL_OUTPUT_DIR                        (cli._resolve_output_dir)
  L4   #65   doctor honours --no-persist; canonical output/ untouched   (cli.doctor)
  L5   #65   doctor record-write containment covers a NON-OSError       (doctor.run_doctor)
  L6   #71   --no-persist scratch removed on SUCCESS (temp-dir count)   (cli._remove_scratch_dir)
  L7   #71   --no-persist scratch removed on ABORT (temp-dir count)     (ctx.call_on_close)
  L8   #71   blocked cleanup: exit code unchanged, path named, root cause not masked
  L9   crit3 all four boundary sites exit non-zero, clean, no traceback (cli._report_boundary_failure)
  L10  crit3 inbox batch does NOT abort; failure dominates degradation  (cli.run --inbox)
  L11  #66   LIVE witness: --no-persist / AICOUNCIL_OUTPUT_DIR on a real run

GAP is a real runtime verdict here (verify_night_consolidation.py carries it only as docstring
prose). A GAP never counts as a discharge and never reads as a PASS — it is printed with the
reason it could not be exercised.

L11 / #66 and the $0-vs-live tension. #66's done-when requires a REAL run writing outside
canonical ./output/, but this checker must stay $0. Those are reconcilable ONLY on
CLI-subscription seats -- and the committed config arms none (no `backend: cli` in
settings.yaml; seat_router.py:134 defaults to "api"). The ADR-12 §5 backend=cli flip is gated
on #27, outside this lane. So L11 prices the run from live config and refuses to spend by
default, emitting GAP with the reason. It names #66 in BOTH states, so a GAP can never be
misread as the discharge. See the run report for #66's actual disposition.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Test THIS repo's code: prepend the co-located src + repo root so the checker never picks up a
# sibling worktree's editable install (the shared-.venv quirk). Order matters — front of path.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

PASS, FAIL, GAP = "PASS", "FAIL", "GAP"


class LegResult:
    __slots__ = ("leg", "id_", "verdict", "evidence")

    def __init__(self, leg: str, id_: str, verdict: str, evidence: str) -> None:
        self.leg, self.id_, self.verdict, self.evidence = leg, id_, verdict, evidence


def _ok(leg: str, id_: str, evidence: str) -> LegResult:
    return LegResult(leg, id_, PASS, evidence)


def _fail(leg: str, id_: str, evidence: str) -> LegResult:
    return LegResult(leg, id_, FAIL, evidence)


def _gap(leg: str, id_: str, evidence: str) -> LegResult:
    """Un-exercisable in this environment. NEVER a fake PASS, never a discharge."""
    return LegResult(leg, id_, GAP, evidence)


# --------------------------------------------------------------------------------------------
# Shared fixtures — a minimal in-memory config, and the scratch-dir census used by #71 legs
# --------------------------------------------------------------------------------------------

def _scratch_census() -> set[str]:
    """Names of aicouncil scratch dirs in the system temp dir (the #71 before/after count)."""
    return {p.name for p in Path(tempfile.gettempdir()).glob("aicouncil-scratch-*")}


def _make_config(out_dir: Path):
    from config.config_loader import (
        AppConfig,
        DefaultsConfig,
        ModelConfig,
        PromptsConfig,
    )

    return AppConfig(
        defaults=DefaultsConfig(
            rounds=1, max_rounds=2, output_dir=out_dir, synthesizer="claude",
            default_panel=["claude"], full_panel=["claude"],
        ),
        models={"claude": ModelConfig(
            name="claude", sdk="anthropic", model="claude-test",
            api_key_env="VCOC_TEST_KEY", timeout_sec=60, max_tokens=1024,
        )},
        prompts=PromptsConfig(
            initial="{persona}\n{question}",
            critique="{persona}\nRound {round}. {question}\n{previous_responses_anonymized}",
            synthesis="Q: {question}\n{full_transcript}",
            personas={"claude": "Be an architect."},
        ),
        available_providers={"claude"},
    )


def _invoke_doctor(config, args: list[str]):
    """Run `council doctor` with the PROBE surface patched (#32 is not this lane's)."""
    from click.testing import CliRunner

    from ai_council import doctor as doc
    from ai_council.cli import main
    from tests.conftest import MockProvider

    with patch("ai_council.cli.load_config", return_value=config), \
         patch("ai_council.cli.load_dotenv"), \
         patch.dict(os.environ, {"VCOC_TEST_KEY": "sk-real"}, clear=False), \
         patch.object(doc, "build_all_providers", return_value={"claude": MockProvider("claude")}), \
         patch.object(doc, "run_health_checks_sync", return_value={"claude": (True, "")}):
        return CliRunner().invoke(main, ["doctor", *args])


def _invoke_run(config, args: list[str], *, run_impl=None):
    """Run `council <args>` with CouncilRunner.run replaced by `run_impl`."""
    from click.testing import CliRunner

    from ai_council.cli import main
    from ai_council.models import DebateResult, Round
    from tests.conftest import MockProvider

    async def _default(request, output_dir=None, output_format="text"):
        return DebateResult(
            question=request.question, rounds=[Round(number=1, responses=[])],
            synthesis="ok", synthesizer="claude", total_duration_sec=1.0, panel_mode="custom",
        )

    with patch("ai_council.cli.load_config", return_value=config), \
         patch("ai_council.cli.build_all_providers", return_value={"claude": MockProvider("claude")}), \
         patch("ai_council.cli.CouncilRunner") as MockRunner:
        MockRunner.return_value.run = run_impl or _default
        return CliRunner().invoke(main, args)


# --------------------------------------------------------------------------------------------
# L1 — one resolver, no surviving second copy of the precedence chain
# --------------------------------------------------------------------------------------------

def leg_l1() -> LegResult:
    src = _REPO / "src"
    env_reads = [p for p in src.rglob("*.py") if "AICOUNCIL_OUTPUT_DIR" in p.read_text(encoding="utf-8")]
    mkdtemps = [
        p for p in src.rglob("*.py")
        if "mkdtemp(prefix=\"aicouncil-scratch-\")" in p.read_text(encoding="utf-8")
    ]
    # doctor.py mentions the env var in prose only; the READ must be unique to cli.py.
    real_reads = [
        p for p in env_reads
        if "os.environ.get(\"AICOUNCIL_OUTPUT_DIR\")" in p.read_text(encoding="utf-8")
    ]
    ev = (f"env-read sites={[p.name for p in real_reads]} scratch-construction="
          f"{[p.name for p in mkdtemps]}")
    ok = len(real_reads) == 1 and len(mkdtemps) == 1 and real_reads[0].name == "cli.py"
    return _ok("L1", "arc", ev) if ok else _fail("L1", "arc", f"expected exactly one of each in cli.py; {ev}")


# --------------------------------------------------------------------------------------------
# L2-L5 — #65 doctor honours the output controls
# --------------------------------------------------------------------------------------------

def leg_l2() -> LegResult:
    with tempfile.TemporaryDirectory(prefix="vcoc-l2-") as td:
        root = Path(td)
        config = _make_config(root / "canonical")
        explicit = root / "explicit"
        os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
        result = _invoke_doctor(config, ["--output", str(explicit)])
        landed = (explicit / "health" / "doctor-latest.json").exists()
        canonical = (config.defaults.output_dir / "health").exists()
        ev = f"exit={result.exit_code} record_in_--output={landed} canonical_written={canonical}"
        ok = result.exit_code == 0 and landed and not canonical
        return _ok("L2", "#65", ev) if ok else _fail("L2", "#65", ev)


def leg_l3() -> LegResult:
    with tempfile.TemporaryDirectory(prefix="vcoc-l3-") as td:
        root = Path(td)
        config = _make_config(root / "canonical")
        env_dir = root / "env_out"
        os.environ["AICOUNCIL_OUTPUT_DIR"] = str(env_dir)
        try:
            result = _invoke_doctor(config, [])
        finally:
            os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
        landed = (env_dir / "health" / "doctor-latest.json").exists()
        canonical = (config.defaults.output_dir / "health").exists()
        ev = f"exit={result.exit_code} record_in_env_dir={landed} canonical_written={canonical}"
        ok = result.exit_code == 0 and landed and not canonical
        return _ok("L3", "#65", ev) if ok else _fail("L3", "#65", ev)


def leg_l4() -> LegResult:
    with tempfile.TemporaryDirectory(prefix="vcoc-l4-") as td:
        config = _make_config(Path(td) / "canonical")
        os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
        before = _scratch_census()
        result = _invoke_doctor(config, ["--no-persist"])
        after = _scratch_census()
        canonical = (config.defaults.output_dir / "health").exists()
        ev = (f"exit={result.exit_code} canonical_written={canonical} "
              f"leaked={sorted(after - before) or 'none'}")
        ok = result.exit_code == 0 and not canonical and after == before
        return _ok("L4", "#65", ev) if ok else _fail("L4", "#65", ev)


def leg_l5() -> LegResult:
    """Containment must cover a NON-OSError (a json.dumps TypeError), not just filesystem errors."""
    from ai_council import doctor as doc

    with tempfile.TemporaryDirectory(prefix="vcoc-l5-") as td:
        config = _make_config(Path(td) / "canonical")
        os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
        with patch.object(doc, "write_record", side_effect=TypeError("not JSON serializable")):
            result = _invoke_doctor(config, [])
        ev = (f"exit={result.exit_code} warned={'WARNING' in result.output} "
              f"named_type={'TypeError' in result.output} "
              f"verdict_kept={'GREEN' in result.output or 'YELLOW' in result.output}")
        ok = result.exit_code == 0 and "WARNING" in result.output and "TypeError" in result.output
        return _ok("L5", "#65", ev) if ok else _fail("L5", "#65", ev)


# --------------------------------------------------------------------------------------------
# L6-L8 — #71 scratch dir lifecycle, proven by before/after temp-dir count
# --------------------------------------------------------------------------------------------

def leg_l6() -> LegResult:
    with tempfile.TemporaryDirectory(prefix="vcoc-l6-") as td:
        config = _make_config(Path(td) / "canonical")
        os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
        before = _scratch_census()
        result = _invoke_run(config, ["--skip-health-check", "--mode", "pick", "--no-persist", "q"])
        after = _scratch_census()
        ev = (f"exit={result.exit_code} before={len(before)} after={len(after)} "
              f"leaked={sorted(after - before) or 'none'}")
        return _ok("L6", "#71", ev) if after == before else _fail("L6", "#71", ev)


def leg_l7() -> LegResult:
    async def _boom(request, output_dir=None, output_format="text"):
        raise RuntimeError("injected mid-run failure")

    with tempfile.TemporaryDirectory(prefix="vcoc-l7-") as td:
        config = _make_config(Path(td) / "canonical")
        os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
        before = _scratch_census()
        result = _invoke_run(
            config, ["--skip-health-check", "--mode", "pick", "--no-persist", "q"], run_impl=_boom
        )
        after = _scratch_census()
        ev = (f"exit={result.exit_code} before={len(before)} after={len(after)} "
              f"leaked={sorted(after - before) or 'none'}")
        ok = after == before and result.exit_code != 0
        return _ok("L7", "#71", ev) if ok else _fail("L7", "#71", ev)


def leg_l8() -> LegResult:
    """A cleanup blocked by an open handle: non-fatal, names the path, does not mask the cause."""
    from ai_council import cli as cli_module
    from ai_council.models import DebateResult, Round

    fails: list[str] = []

    # (a) blocked cleanup on an otherwise SUCCESSFUL run -> exit unchanged, path named.
    printed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vcoc-l8a-") as td:
        config = _make_config(Path(td) / "canonical")
        os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
        captured: dict = {}

        async def _capture(request, output_dir=None, output_format="text"):
            captured["dir"] = output_dir
            return DebateResult(
                question=request.question, rounds=[Round(number=1, responses=[])],
                synthesis="ok", synthesizer="claude", total_duration_sec=1.0, panel_mode="custom",
            )

        with patch("ai_council.cli.shutil.rmtree", side_effect=PermissionError(13, "handle open")), \
             patch.object(cli_module.console, "print",
                          side_effect=lambda *a, **k: printed.append(str(a[0]))):
            result_a = _invoke_run(
                config, ["--skip-health-check", "--mode", "pick", "--no-persist", "q"],
                run_impl=_capture,
            )
        warning = "\n".join(printed)
        if result_a.exit_code != 0:
            fails.append(f"blocked cleanup changed exit code to {result_a.exit_code}")
        if "could not remove scratch dir" not in warning:
            fails.append("no warning emitted for the blocked cleanup")
        if captured.get("dir") and str(captured["dir"]) not in warning:
            fails.append("warning did not name the surviving path")
        if captured.get("dir"):
            shutil.rmtree(captured["dir"], ignore_errors=True)  # this leg cleans up after itself

    # (b) blocked cleanup while an exception is ALREADY in flight -> the run's error survives.
    captured_b: dict = {}

    async def _boom(request, output_dir=None, output_format="text"):
        captured_b["dir"] = output_dir
        raise RuntimeError("the real root cause")

    with tempfile.TemporaryDirectory(prefix="vcoc-l8b-") as td:
        config = _make_config(Path(td) / "canonical")
        with patch("ai_council.cli.shutil.rmtree", side_effect=PermissionError(13, "handle open")):
            result_b = _invoke_run(
                config, ["--skip-health-check", "--mode", "pick", "--no-persist", "q"],
                run_impl=_boom,
            )
        if isinstance(result_b.exception, PermissionError):
            fails.append("cleanup PermissionError MASKED the in-flight exception")
        if result_b.exit_code == 0:
            fails.append("aborted run exited 0")
        # This leg deliberately BLOCKS cleanup, so the scratch dir survives by construction.
        # Remove it here or the checker itself leaks (§5.9) and stops being idempotent.
        if captured_b.get("dir"):
            shutil.rmtree(captured_b["dir"], ignore_errors=True)

    ev = "; ".join(fails) if fails else (
        "blocked cleanup: exit unchanged, path named, root cause preserved"
    )
    return _fail("L8", "#71", ev) if fails else _ok("L8", "#71", ev)


# --------------------------------------------------------------------------------------------
# L9-L10 — criterion 3: required-write failures reach the exit code, cleanly
# --------------------------------------------------------------------------------------------

def leg_l9() -> LegResult:
    from ai_council.output import OutputRoutingError

    fails: list[str] = []
    routing = OutputRoutingError("verdict package failed to reach required return-dir: /nope")

    def _check(label: str, args: list[str], exc: Exception, *, research: bool, expect: str):
        with tempfile.TemporaryDirectory(prefix="vcoc-l9-") as td:
            config = _make_config(Path(td) / "canonical")
            os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
            if research:
                from config.config_loader import ModeConfig, ResearchConfig
                config.modes = {
                    "pick": ModeConfig(description="p", emoji="*", aliases=["p"], default=True,
                                       max_rounds=2, token_budget=1000),
                    "research": ModeConfig(description="r", emoji="*", aliases=["r"], default=False,
                                           max_rounds=1, token_budget=1000),
                }
                config.research = ResearchConfig(
                    default_providers=["perplexity"], deep_providers=["perplexity"],
                    cache_dir=Path(td) / "cache", cache_ttl_days=7,
                    summary_max_tokens=2500, summary_model="claude",
                )
                with patch("ai_council.cli._run_research_dispatch", side_effect=exc):
                    result = _invoke_run(config, args)
            else:
                async def _raise(request, output_dir=None, output_format="text"):
                    raise exc
                result = _invoke_run(config, args, run_impl=_raise)

        if result.exit_code == 0:
            fails.append(f"{label}: exited 0")
        if expect not in result.output:
            fails.append(f"{label}: missing '{expect}'")
        # "No raw traceback" means the failure was HANDLED at the boundary -- the surviving
        # exception is the deliberate SystemExit, not the original error escaping through
        # Click. (A logged traceback is intentional and required: the root cause of an
        # internal defect must not be discarded. It is context BELOW the clean message,
        # not the failure report itself.)
        if not isinstance(result.exception, SystemExit):
            fails.append(
                f"{label}: unhandled {type(result.exception).__name__} escaped the boundary"
            )

    debate_args = ["--skip-health-check", "--mode", "pick", "q"]
    research_args = ["--skip-health-check", "--mode", "research", "q"]
    _check("debate/routing", debate_args, routing, research=False, expect="Required write failed")
    _check("debate/internal", debate_args, TypeError("bad synthesis"), research=False,
           expect="Unexpected error")
    # OutputRoutingError subclasses RuntimeError -- must not be mislabelled "Research error".
    _check("research/routing", research_args, routing, research=True, expect="Required write failed")
    _check("research/oserror", research_args, OSError("disk full"), research=True,
           expect="Unexpected error")

    ev = "; ".join(fails) if fails else "all 4 sites: non-zero exit, clean message, no traceback"
    return _fail("L9", "crit3", ev) if fails else _ok("L9", "crit3", ev)


def leg_l10() -> LegResult:
    """The batch must NOT abort: every file attempted, bookkeeping intact, exit still non-zero."""
    from ai_council.models import DebateResult, Round
    from ai_council.output import OutputRoutingError
    from config.config_loader import InboxConfig

    with tempfile.TemporaryDirectory(prefix="vcoc-l10-") as td:
        root = Path(td)
        config = _make_config(root / "canonical")
        inbox_dir, archive_dir = root / "inbox", root / "archive"
        config.inbox = InboxConfig(dir=inbox_dir, archive_dir=archive_dir, scan_downloads=False)
        inbox_dir.mkdir(parents=True)
        for name in ("a", "b", "c"):
            (inbox_dir / f"{name}.md").write_text(f"question {name}\n", encoding="utf-8")

        seen: list[str] = []

        async def _run(request, output_dir=None, output_format="text"):
            stem = Path(request.question.source).stem
            seen.append(stem)
            if stem == "b":
                raise OutputRoutingError("failed to reach required return-dir: /nope")
            return DebateResult(
                question=request.question, rounds=[Round(number=1, responses=[])],
                synthesis="ok", synthesizer="claude", total_duration_sec=1.0, panel_mode="custom",
            )

        result = _invoke_run(config, ["--skip-health-check", "--inbox"], run_impl=_run)
        archived = len(list(archive_dir.rglob("*.md")))
        remaining = len(list(inbox_dir.glob("*.md")))

        fails = []
        if seen != ["a", "b", "c"]:
            fails.append(f"batch ABORTED early (processed {seen})")
        if result.exit_code == 0:
            fails.append("failed batch exited 0")
        if archived != 3:
            fails.append(f"archive bookkeeping changed ({archived}/3 archived)")
        if remaining:
            fails.append(f"{remaining} file(s) left in the inbox")

        ev = "; ".join(fails) if fails else (
            f"processed={seen} exit={result.exit_code} archived={archived}/3 (no abort)"
        )
        return _fail("L10", "crit3", ev) if fails else _ok("L10", "crit3", ev)


# --------------------------------------------------------------------------------------------
# L11 — #66 LIVE witness (gated). Names #66 whether it runs or GAPs.
# --------------------------------------------------------------------------------------------

def leg_l11() -> LegResult:
    """#66: witness --no-persist / AICOUNCIL_OUTPUT_DIR on a REAL run.

    COST PRECONDITION (verified, not assumed). #66 can only be discharged at $0 if the panel
    runs on CLI-subscription seats. The COMMITTED config arms none: settings.yaml declares no
    `backend: cli` seat, and seat_router.build_seat_router defaults requested_backend="api"
    (seat_router.py:134). The backend=cli flip is ADR-12 §5, gated on #27 -- out of this
    lane's scope. So a live run here bills the API and needs explicit operator authorization.

    This leg therefore probes the config first and refuses to spend by default. It emits GAP
    (never a fake PASS, never a discharge) unless BOTH are true:
      * AICOUNCIL_LIVE_WITNESS=1  -- the operator authorized a live run, and
      * either a cli-backend seat exists ($0), or AICOUNCIL_LIVE_WITNESS_BILLED=1 confirms
        the operator accepted a billed debate.
    """
    from config.config_loader import load_config

    try:
        cfg = load_config()
        cli_seats = sorted(n for n, m in cfg.models.items() if getattr(m, "backend", "api") == "cli")
    except Exception as exc:
        return _gap("L11", "#66", f"NOT DISCHARGED - could not read config to price the run: {exc}")

    if os.environ.get("AICOUNCIL_LIVE_WITNESS") != "1":
        cost = f"$0 via cli seats {cli_seats}" if cli_seats else "BILLED (no cli-backend seat armed)"
        return _gap(
            "L11", "#66",
            f"NOT DISCHARGED - live run not attempted; would be {cost}. "
            "Set AICOUNCIL_LIVE_WITNESS=1 to authorize.",
        )

    if not cli_seats and os.environ.get("AICOUNCIL_LIVE_WITNESS_BILLED") != "1":
        return _gap(
            "L11", "#66",
            "NOT DISCHARGED - refusing to spend: no backend=cli seat in the committed config "
            "(ADR-12 §5 flip is gated on #27), so a live run bills the API. Re-run with "
            "AICOUNCIL_LIVE_WITNESS_BILLED=1 to accept the cost.",
        )

    panel = ",".join(cli_seats) if cli_seats else "claude"

    canonical = _REPO / "output"
    before_canonical = {p.name for p in canonical.rglob("*")} if canonical.exists() else set()
    before_scratch = _scratch_census()
    fails: list[str] = []

    # (a) --no-persist: nothing new in canonical output/, no scratch survivor.
    proc_a = subprocess.run(
        [sys.executable, "-m", "ai_council.cli", "--no-persist", "--lite",
         "--models", panel, "--mode", "pick", "Name one tradeoff of feature flags."],
        cwd=_REPO, capture_output=True, text=True, timeout=900,
    )
    after_canonical = {p.name for p in canonical.rglob("*")} if canonical.exists() else set()
    after_scratch = _scratch_census()
    if proc_a.returncode != 0:
        fails.append(f"--no-persist run exited {proc_a.returncode}")
    if after_canonical - before_canonical:
        fails.append(f"--no-persist WROTE to canonical output/: {sorted(after_canonical - before_canonical)}")
    if after_scratch - before_scratch:
        fails.append(f"scratch leaked: {sorted(after_scratch - before_scratch)}")

    # (b) AICOUNCIL_OUTPUT_DIR: artifacts land in the env dir, canonical untouched.
    with tempfile.TemporaryDirectory(prefix="vcoc-l11-") as td:
        env_dir = Path(td) / "env_out"
        env = {**os.environ, "AICOUNCIL_OUTPUT_DIR": str(env_dir)}
        proc_b = subprocess.run(
            [sys.executable, "-m", "ai_council.cli", "--lite",
             "--models", panel, "--mode", "pick", "Name one tradeoff of feature flags."],
            cwd=_REPO, capture_output=True, text=True, timeout=900, env=env,
        )
        landed = sorted(p.name for p in env_dir.rglob("*.md")) if env_dir.exists() else []
        final_canonical = {p.name for p in canonical.rglob("*")} if canonical.exists() else set()
        if proc_b.returncode != 0:
            fails.append(f"env-override run exited {proc_b.returncode}")
        if not landed:
            fails.append("AICOUNCIL_OUTPUT_DIR received no artifacts")
        if final_canonical - after_canonical:
            fails.append(f"env run WROTE to canonical: {sorted(final_canonical - after_canonical)}")

    ev = "; ".join(fails) if fails else (
        f"DISCHARGED - live $0 run: --no-persist left canonical output/ unchanged and no "
        f"scratch survivor; AICOUNCIL_OUTPUT_DIR received {landed}"
    )
    return _fail("L11", "#66", ev) if fails else _ok("L11", "#66", ev)


LEGS = [leg_l1, leg_l2, leg_l3, leg_l4, leg_l5, leg_l6, leg_l7, leg_l8, leg_l9, leg_l10, leg_l11]


def main() -> int:
    results: list[LegResult] = []
    for fn in LEGS:
        name = fn.__name__.split("_")[-1].upper()
        try:
            results.append(fn())
        except Exception as exc:  # a leg that raises is a FAIL, never a silent skip
            results.append(_fail(name, "?", f"EXCEPTION: {type(exc).__name__}: {exc}"))

    print("=" * 78)
    print("CLI OUTPUT-CONTRACT VERIFICATION  -  Lane A2 (#65 / #71 / #66 / boundary)")
    print("=" * 78)
    print(f"{'LEG':<6}{'ID':<8}{'VERDICT':<9}EVIDENCE")
    print("-" * 78)
    for r in results:
        print(f"{r.leg:<6}{r.id_:<8}{r.verdict:<9}{r.evidence}")
    print("-" * 78)
    passed = sum(r.verdict == PASS for r in results)
    failed = sum(r.verdict == FAIL for r in results)
    gapped = sum(r.verdict == GAP for r in results)
    summary = f"RESULT: {passed}/{len(results)} PASS"
    if gapped:
        summary += f", {gapped} GAP"
    if failed:
        summary += f", {failed} FAIL"
    print(summary)

    # #66's discharge is attributed by name, in both states -- a GAP is never a discharge.
    l11 = next((r for r in results if r.leg == "L11"), None)
    if l11 is not None:
        state = "DISCHARGED" if l11.verdict == PASS else f"NOT DISCHARGED ({l11.verdict})"
        print(f"#66 discharge leg: L11 -- {state}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
