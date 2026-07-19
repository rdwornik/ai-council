"""End-to-end output-contract witness — the COMPOSED guarantee, across Lanes A1 and A2.

Sibling of verify_output_writes.py (A1, writer layer) and verify_cli_output_contract.py
(A2, CLI boundary). Those two verify each half in isolation. This one verifies the thing
neither can see: that a real filesystem failure travels the WHOLE path.

Why a third checker exists
--------------------------
At integration, A1 was green and A2 was green and merged main was RED. Two per-lane
checkers are structurally blind to that class of defect: each half is sound, the
composition is broken. If a later change narrows the boundary catch or alters the raise,
both sibling checkers still exit 0 while the user-facing guarantee dies silently.

The claim under test, for each of the four boundary paths: a REQUIRED --return-dir write
to an unwritable destination must
  1. raise at the WRITER layer, from a real filesystem error (not an injected exception)
  2. propagate through the BOUNDARY
  3. exit NON-ZERO, naming the destination, with the deliberate SystemExit surviving
  4. report EVERY lost deliverable, not just the first
  5. still leave the CANONICAL artifacts on disk

Legs
----
  L1  interactive debate      crit3   all five criteria
  L2  interactive research    crit3   all five criteria
  L3  inbox debate            crit3   all five criteria
  L4  inbox research          crit3   all five criteria

Discrimination established 2026-07-19 (recorded here because this checker cannot reach a
prior commit to re-derive it). The same five criteria were run against original main
27a45d1 and merged main 74e8359, via a temporary detached worktree:

  path                  27a45d1 (pre)                74e8359 (post)
  interactive debate    exit 1, cause NOT surfaced   PASS
  interactive research  exit 0  <- silent            PASS
  inbox debate          exit 0  <- silent            PASS
  inbox research        exit 0  <- silent            PASS
  RESULT                1/4 criteria sets met        4/4 PASS

Two criteria carry the discrimination: exit_nonzero (three paths flipped 0 -> 1) and
real_fs_cause (0/4 -> 4/4). Interactive debate exited 1 on BOTH — original main had a
single pre-existing verdict-package raise — so on that path the discriminator is the
deliverable COUNT: pre reports 1 lost deliverable, post reports 2 (transcript AND verdict).
A no-traceback criterion was deliberately DROPPED: CliRunner captures SystemExit rather
than printing a traceback, so it passed against pre-fix behaviour and discriminates nothing.

Cost: $0. Providers are mocked (tests/conftest.py MockProvider); everything below the
provider call is real — real CouncilRunner, real orchestrator, real writer layer, real
filesystem. Idempotent: every leg works in its own TemporaryDirectory and leaves no state.
Anything un-exercisable prints GAP, never a fake PASS. A GAP does not fail the run; a FAIL
does. A leg that raises is a FAIL, never a silent skip.

Run:  py scripts/verify_output_contract_e2e.py
"""

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PANEL = ("claude", "gemini", "openai")
RESEARCH_PANEL = ("perplexity", "gemini_research", "grok_research")


@dataclass
class LegResult:
    leg: str
    id_: str
    verdict: str
    evidence: str


def _ok(leg: str, id_: str, evidence: str) -> LegResult:
    return LegResult(leg, id_, "PASS", evidence)


def _fail(leg: str, id_: str, evidence: str) -> LegResult:
    return LegResult(leg, id_, "FAIL", evidence)


def _gap(leg: str, id_: str, evidence: str) -> LegResult:
    return LegResult(leg, id_, "GAP", evidence)


# ------------------------------------------------------------------------------------
# Harness — mocked providers, real everything else
# ------------------------------------------------------------------------------------

def _mock_research_provider(name: str):
    """A ResearchProvider double. Built lazily so an import error is a FAIL, not a skip."""
    from ai_council.research.models import ResearchResult
    from ai_council.research.provider import ResearchProvider

    class _Mock(ResearchProvider):
        def name(self) -> str:
            return name

        def model_string(self) -> str:
            return f"{name}-mock"

        async def research(self, query: str):
            return ResearchResult(
                provider=name, query=query,
                content=f"## Findings\n\n- {name} finding for {query}\n",
                sources=[], token_count=10, cost_usd=0.0, duration_sec=0.1,
                timestamp="2026-07-19T00:00:00Z",
            )

    return _Mock()


async def _stub_summarize(report, research_cfg, models_cfg):
    """Skip the summarizer's model call; it is not on the path under test."""
    report.summary_2500 = "stub summary"
    return report


def _make_config(out_dir: Path, *, research: bool = False, inbox: tuple | None = None):
    from config.config_loader import (
        AppConfig,
        DefaultsConfig,
        InboxConfig,
        ModeConfig,
        ModelConfig,
        PromptsConfig,
        ResearchConfig,
        ResearchProviderConfig,
    )

    cfg = AppConfig(
        defaults=DefaultsConfig(
            # openai synthesizes and is evicted from the panel (ADR-01), leaving 2 debaters,
            # which is the orchestrator's minimum.
            rounds=1, max_rounds=2, output_dir=out_dir, synthesizer="openai",
            default_panel=["claude", "gemini"], full_panel=["claude", "gemini"],
        ),
        models={
            n: ModelConfig(name=n, sdk="anthropic", model=f"{n}-test",
                           api_key_env="VOCE_TEST_KEY", timeout_sec=60, max_tokens=1024)
            for n in PANEL
        },
        prompts=PromptsConfig(
            initial="{persona}\n{question}",
            critique="{persona}\nRound {round}. {question}\n{previous_responses_anonymized}",
            synthesis="Q: {question}\n{full_transcript}",
            personas={n: "Be an architect." for n in PANEL},
        ),
        available_providers=set(PANEL),
    )
    if research:
        cfg.modes = {
            "pick": ModeConfig(description="p", emoji="*", aliases=["p"], default=True,
                               max_rounds=2, token_budget=1000),
            "research": ModeConfig(description="r", emoji="*", aliases=["r"], default=False,
                                   max_rounds=1, token_budget=1000),
        }
        cfg.research = ResearchConfig(
            default_providers=list(RESEARCH_PANEL), deep_providers=list(RESEARCH_PANEL),
            cache_dir=out_dir.parent / "cache", cache_ttl_days=7,
            summary_max_tokens=2500, summary_model="claude",
            providers={
                n: ResearchProviderConfig(name=n, model=f"{n}-mock",
                                          api_key_env="VOCE_TEST_KEY", timeout_sec=60)
                for n in RESEARCH_PANEL
            },
        )
    if inbox:
        cfg.inbox = InboxConfig(dir=inbox[0], archive_dir=inbox[1], scan_downloads=False)
    return cfg


def _blocker(root: Path) -> Path:
    """A FILE where --return-dir expects a directory, so the real mkdir raises.

    Harness-side induction only: the failure is a genuine OS error, not a patched raise.
    """
    p = root / "blocker"
    p.write_text("not a dir", encoding="utf-8")
    return p


def _invoke(cfg, args: list[str], *, research: bool):
    """Drive the real CLI with only the PROVIDERS mocked."""
    from click.testing import CliRunner

    from ai_council.cli import main as cli_main
    from tests.conftest import MockProvider

    os.environ.pop("AICOUNCIL_OUTPUT_DIR", None)
    stack = [
        patch("ai_council.cli.load_config", return_value=cfg),
        patch("ai_council.cli.build_all_providers",
              return_value={n: MockProvider(n) for n in PANEL}),
    ]
    if research:
        stack += [
            patch("ai_council.research.runner.build_research_providers",
                  return_value=[_mock_research_provider(n) for n in RESEARCH_PANEL]),
            patch("ai_council.research.runner.summarize_report", _stub_summarize),
        ]
    if len(stack) == 2:
        with stack[0], stack[1]:
            return CliRunner().invoke(cli_main, args)
    with stack[0], stack[1], stack[2], stack[3]:
        return CliRunner().invoke(cli_main, args)


def _assess(leg: str, result, blocker: Path, canonical: Path, *, expect_lost: int) -> LegResult:
    """Score one path against the five criteria. Every failure is named, not just the first."""
    squashed = "".join(result.output.split())
    flat = " ".join(result.output.split())
    artifacts = (
        sorted(p.name for p in canonical.rglob("*") if p.is_file())
        if canonical.exists() else []
    )
    noun = "deliverable" if expect_lost == 1 else "deliverables"

    bad: list[str] = []
    if result.exit_code == 0:
        bad.append("exit 0: a required-write failure was swallowed")
    # A2's formulation: the surviving exception must be the DELIBERATE SystemExit, not the
    # original error escaping through Click. Replaces a no-traceback check, which cannot
    # discriminate -- CliRunner captures SystemExit instead of printing a traceback.
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        bad.append(f"unhandled {type(result.exception).__name__} escaped the boundary")
    if str(blocker) not in squashed:
        bad.append("destination not named in the operator message")
    # The real OS error must survive to the operator. Without this the message can be
    # technically non-zero yet say nothing about why.
    if "FileExistsError" not in squashed and "NotADirectoryError" not in squashed:
        bad.append("real filesystem cause not surfaced")
    # Completeness. On the debate path the exit code alone does not discriminate (original
    # main also exited 1, from the single pre-existing verdict raise) -- the count does.
    if f"{expect_lost} {noun} not delivered" not in flat:
        bad.append(f"deliverable count != {expect_lost} (lost deliverables under-reported)")
    if not artifacts:
        bad.append("canonical artifacts absent: a routing failure destroyed the run's output")

    ev = "; ".join(bad) if bad else (
        f"exit={result.exit_code}; dest named; {expect_lost} {noun} reported; "
        f"real FS cause; canonical={len(artifacts)} file(s)"
    )
    return _fail(leg, "crit3", ev) if bad else _ok(leg, "crit3", ev)


# ------------------------------------------------------------------------------------
# L1-L4 — the four boundary paths
# ------------------------------------------------------------------------------------

def leg_l1() -> LegResult:
    """Interactive debate. Transcript AND verdict package both lose the return-dir."""
    with tempfile.TemporaryDirectory(prefix="voce-l1-") as td:
        root = Path(td)
        canonical = root / "canonical"
        blocker = _blocker(root)
        cfg = _make_config(canonical)
        r = _invoke(cfg, ["--skip-health-check", "--mode", "pick",
                          "--return-dir", str(blocker), "which config format?"],
                    research=False)
        return _assess("L1", r, blocker, canonical, expect_lost=2)


def leg_l2() -> LegResult:
    """Interactive research. The research report is the required deliverable (#62)."""
    with tempfile.TemporaryDirectory(prefix="voce-l2-") as td:
        root = Path(td)
        canonical = root / "canonical"
        blocker = _blocker(root)
        cfg = _make_config(canonical, research=True)
        r = _invoke(cfg, ["--skip-health-check", "--mode", "research",
                          "--return-dir", str(blocker), "which config format?"],
                    research=True)
        return _assess("L2", r, blocker, canonical, expect_lost=1)


def leg_l3() -> LegResult:
    """Inbox debate. Same guarantee on the batch path, which has its own boundary site."""
    with tempfile.TemporaryDirectory(prefix="voce-l3-") as td:
        root = Path(td)
        canonical = root / "canonical"
        blocker = _blocker(root)
        inbox_dir, archive_dir = root / "inbox", root / "archive"
        inbox_dir.mkdir(parents=True)
        (inbox_dir / "q.md").write_text("which config format?\n", encoding="utf-8")
        cfg = _make_config(canonical, inbox=(inbox_dir, archive_dir))
        r = _invoke(cfg, ["--skip-health-check", "--inbox", "--return-dir", str(blocker)],
                    research=False)
        return _assess("L3", r, blocker, canonical, expect_lost=2)


def leg_l4() -> LegResult:
    """Inbox research — frontmatter-routed, the fourth and last boundary site."""
    with tempfile.TemporaryDirectory(prefix="voce-l4-") as td:
        root = Path(td)
        canonical = root / "canonical"
        blocker = _blocker(root)
        inbox_dir, archive_dir = root / "inbox", root / "archive"
        inbox_dir.mkdir(parents=True)
        (inbox_dir / "q.md").write_text(
            "---\nmode: research\n---\n\nwhich config format?\n", encoding="utf-8"
        )
        cfg = _make_config(canonical, research=True, inbox=(inbox_dir, archive_dir))
        r = _invoke(cfg, ["--skip-health-check", "--inbox", "--return-dir", str(blocker)],
                    research=True)
        return _assess("L4", r, blocker, canonical, expect_lost=1)


LEGS = [leg_l1, leg_l2, leg_l3, leg_l4]

LABELS = {
    "L1": "interactive debate",
    "L2": "interactive research",
    "L3": "inbox debate",
    "L4": "inbox research",
}


def main() -> int:
    results: list[LegResult] = []
    for fn in LEGS:
        name = fn.__name__.split("_")[-1].upper()
        try:
            results.append(fn())
        except Exception as exc:  # a leg that raises is a FAIL, never a silent skip
            results.append(_fail(name, "crit3", f"EXCEPTION: {type(exc).__name__}: {exc}"))

    print("=" * 78)
    print("END-TO-END OUTPUT-CONTRACT WITNESS  -  composed, Lanes A1 x A2")
    print("=" * 78)
    print(f"{'LEG':<6}{'PATH':<24}{'VERDICT':<9}EVIDENCE")
    print("-" * 78)
    for r in results:
        print(f"{r.leg:<6}{LABELS.get(r.leg, '?'):<24}{r.verdict:<9}{r.evidence}")
    print("-" * 78)
    passed = sum(r.verdict == "PASS" for r in results)
    failed = sum(r.verdict == "FAIL" for r in results)
    gaps = sum(r.verdict == "GAP" for r in results)
    print(f"RESULT: {passed}/{len(results)} PASS, {failed} FAIL, {gaps} GAP")
    if failed == 0:
        print("The composed guarantee holds on all four boundary paths.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
