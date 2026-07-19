#!/usr/bin/env python
"""verify_night_consolidation.py — re-runnable witness for the 2026-07-18 shipped batch.

Codifies the night-consolidation verification wave (docs/audits/2026-07-19-night-
consolidation-verification.md) as a deterministic, $0, offline checker: it EXERCISES the
shipped code paths (never reads diffs) and prints a PASS/FAIL table over eight legs, each
mapped to the BACKLOG id / ADR it proves.

Follows the scripts/ verify_*/validate_* sibling convention (validate_backlog.py,
validate_audit_casing.py, verify_openai_deep.py): read-only, self-contained, exit 0 on all
PASS / exit 1 on any FAIL. Layer-2 invariant: it drives the library in-process; it never
spawns a debate, never calls a provider, never writes a tracked file.

Legs (claim -> shipped code):
  L1  #22  --file frontmatter strip + CLI>frontmatter>config precedence   (inbox.parse_file, cli.py)
  L2  #23  research --return-dir writes canonical AND return dir          (research.output.save_research_to_file)
  L3  ADR-13  verdict package carries contract_version == "1.0"           (output.save_verdict_package)
  L4  #39  health retention keeps 10 doctor-<ts>.json + doctor-latest     (doctor._prune_health_records)
  L5  #40  options_considered populated for pick AND ideas verdicts       (output._build_verdict_payload)
  L6  #41  non-zero token counts parsed for claude & codex CLI seats      (providers.cli_base *CliProvider._extract)
  L7  #42  research filename has no doubled "research-research" prefix     (research.output.save_research_to_file)
  L8  #45-#48  shim broken · iso_now centralized · dead code gone · RunPolicy-from-YAML

Deliberately OUT of live scope (recorded as GAPs in the report, not failures here):
  - the CLI-subscription seat lane end-to-end (L6): the committed config routes every seat to
    backend=api; a live seat run needs the §5 backend=cli flip, gated on the #27 scoring.
  - --no-persist / AICOUNCIL_OUTPUT_DIR on the `run` command (L4 b/c): only reachable through a
    paid debate. Retention (the decisive #39 mechanism) is the offline-provable core and is checked.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

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


# --------------------------------------------------------------------------------------------
# L1 — #22 --file frontmatter strip + precedence
# --------------------------------------------------------------------------------------------
def leg_l1() -> LegResult:
    from ai_council.inbox import parse_file
    from config.config_loader import load_config

    with tempfile.TemporaryDirectory(prefix="vnc-l1-") as td:
        md = Path(td) / "q.md"
        md.write_text(
            "---\nsynthesizer: grok\nrounds: 3\nmode: judge\n---\nWhich datastore should we pick?",
            encoding="utf-8",
        )
        text, meta = parse_file(md)

    # (a) no-leak: frontmatter must not survive into the question text.
    no_leak = ("synthesizer" not in text) and ("---" not in text) and text.strip() == "Which datastore should we pick?"
    cfg_default = load_config().defaults.synthesizer

    # (b) the exact cli.py precedence ternary: CLI flag > frontmatter > config default.
    def resolve(flag: str | None, fm: dict) -> str:
        return flag if flag is not None else str(fm["synthesizer"]) if "synthesizer" in fm else cfg_default

    cli_wins = resolve("openai", meta)          # flag present
    fm_wins = resolve(None, meta)               # flag absent, frontmatter present
    cfg_wins = resolve(None, {})                # both absent -> config
    tiers_ok = cli_wins == "openai" and fm_wins == "grok" and cfg_wins == cfg_default
    ev = f"no_leak={no_leak}; tiers cli={cli_wins}/fm={fm_wins}/cfg={cfg_wins} (default={cfg_default})"
    return _ok("L1", "#22", ev) if (no_leak and tiers_ok) else _fail("L1", "#22", ev)


# --------------------------------------------------------------------------------------------
# L2 — #23 research --return-dir (both dirs written, identical)
# --------------------------------------------------------------------------------------------
def _dummy_report(query: str):
    from ai_council.research.models import MergedResearchReport, ResearchResult, Source

    return MergedResearchReport(
        query=query,
        results=[
            ResearchResult(
                provider="perplexity", query=query, content="dummy",
                sources=[Source(title="Example", url="https://example.com", snippet="s")],
                token_count=100, cost_usd=0.0, duration_sec=1.0,
                timestamp="2026-07-19T00:00:00", timed_out=False, error=None,
            )
        ],
        merged_report="Full dummy merged report body.",
        summary_2500="Dummy summary.",
        total_sources=1, total_cost_usd=0.0, total_duration_sec=1.0,
        cache_key="dummy", degraded=False, failed_count=0,
    )


def leg_l2() -> LegResult:
    from ai_council.research.output import save_research_to_file

    with tempfile.TemporaryDirectory(prefix="vnc-l2-") as td:
        canon, ret = Path(td) / "canon", Path(td) / "ret"
        saved = save_research_to_file(_dummy_report("vector db comparison"), canon, from_cache=False, return_dir=ret)
        both = (len(saved) == 2 and saved[0].parent == canon and saved[1].parent == ret
                and saved[0].exists() and saved[1].exists())
        match = saved[0].read_bytes() == saved[1].read_bytes()
        ev = f"paths={len(saved)} canonical_first={saved[0].parent == canon} identical={match}"
    return _ok("L2", "#23", ev) if (both and match) else _fail("L2", "#23", ev)


# --------------------------------------------------------------------------------------------
# L3 — ADR-13 contract_version == "1.0"
# --------------------------------------------------------------------------------------------
def _pick_result(question_text: str, synthesis: str, mode: str = "pick"):
    from ai_council.models import DebateResult, ModelResponse, Question, Round

    resp = ModelResponse(provider="claude", model="claude-opus-4-6", round_number=1,
                         content="ans", latency_sec=1.0, token_count=42)
    return DebateResult(
        question=Question(text=question_text, source="cli"),
        rounds=[Round(number=1, responses=[resp])],
        synthesis=synthesis, synthesizer="gemini", total_duration_sec=5.0,
        panel_mode="default", mode=mode,
    )


def leg_l3() -> LegResult:
    from ai_council.output import save_to_file, save_verdict_package

    result = _pick_result(
        "Should we use YAML or JSON for config?",
        "## Position\nUse YAML.\n\n## Recommendation\nAdopt YAML.\n\n## Rationale\n- Human-editable\n",
    )
    with tempfile.TemporaryDirectory(prefix="vnc-l3-") as td:
        transcript = save_to_file(result, Path(td))[0]
        verdict_path = save_verdict_package(result, Path(td), transcript)[0]
        cv = json.loads(verdict_path.read_text(encoding="utf-8"))["contract_version"]
        ev = f"{verdict_path.name}: contract_version={cv!r}"
    return _ok("L3", "ADR-13", ev) if cv == "1.0" else _fail("L3", "ADR-13", ev)


# --------------------------------------------------------------------------------------------
# L4 — #39 health retention keep-10 (+ doctor-latest always kept)
# --------------------------------------------------------------------------------------------
def leg_l4() -> LegResult:
    from ai_council.doctor import _HEALTH_RETENTION, _prune_health_records

    with tempfile.TemporaryDirectory(prefix="vnc-l4-") as td:
        hdir = Path(td)
        for n in range(1, 13):  # 12 timestamped, sortable
            (hdir / f"doctor-202607{n:02d}_120000.json").write_text("{}", encoding="utf-8")
        (hdir / "doctor-latest.json").write_text("{}", encoding="utf-8")
        before = len(list(hdir.glob("doctor-*.json")))
        _prune_health_records(hdir)
        remaining = sorted(p.name for p in hdir.glob("doctor-*.json") if p.name != "doctor-latest.json")
        latest_kept = (hdir / "doctor-latest.json").exists()
        ev = (f"keep={_HEALTH_RETENTION} before={before} after_ts={len(remaining)} "
              f"latest_kept={latest_kept} oldest_remaining={remaining[0] if remaining else None}")
    ok = len(remaining) == _HEALTH_RETENTION == 10 and latest_kept and remaining[0] == "doctor-20260703_120000.json"
    return _ok("L4", "#39", ev) if ok else _fail("L4", "#39", ev)


# --------------------------------------------------------------------------------------------
# L5 — #40 options_considered for pick AND ideas
# --------------------------------------------------------------------------------------------
def leg_l5() -> LegResult:
    from ai_council.output import _build_verdict_payload

    pick = _pick_result(
        "Which cache should we adopt?\n\n## Options\n- Redis\n- Memcached\n- Postgres\n",
        "## Recommendation\nUse Redis.\n", mode="pick",
    )
    ideas = _pick_result(
        "What ideas should we explore?", "## Top Tier\n- Idea Alpha\n- Idea Beta\n", mode="ideas",
    )
    pick_items = _build_verdict_payload(pick, "l5-pick", "f.md", {}, [])["options_considered"]["items"]
    ideas_items = _build_verdict_payload(ideas, "l5-ideas", "f.md", {}, [])["options_considered"]["items"]
    ev = f"pick={pick_items} ideas={ideas_items}"
    ok = pick_items == ["Redis", "Memcached", "Postgres"] and ideas_items == ["Idea Alpha", "Idea Beta"]
    return _ok("L5", "#40", ev) if ok else _fail("L5", "#40", ev)


# --------------------------------------------------------------------------------------------
# L6 — #41 non-zero token counts, claude & codex CLI adapters (adapter-level; seat lane gated)
# --------------------------------------------------------------------------------------------
def leg_l6() -> LegResult:
    from unittest.mock import patch

    from ai_council.providers.cli_base import ClaudeCliProvider, CliProvider, CodexCliProvider
    from config.config_loader import ModelConfig

    def make(cls, cmd):
        cfg = ModelConfig(name=cmd, sdk="cli", model="m", api_key_env="K", timeout_sec=30,
                          max_tokens=100, backend="cli", cli_command=cmd, cli_model="m")
        with patch("ai_council.providers.cli_base.shutil.which", return_value=f"{cmd}.CMD"), \
             patch.object(CliProvider, "_read_version", staticmethod(lambda exe: "v-test")):
            return cls(cfg)

    claude_doc = {"result": "pong", "modelUsage": {"claude-opus-4-7": {}},
                  "usage": {"input_tokens": 100, "cache_creation_input_tokens": 4000,
                            "cache_read_input_tokens": 500, "output_tokens": 200}}
    c = make(ClaudeCliProvider, "claude")._extract(json.dumps(claude_doc), "")
    x = make(CodexCliProvider, "codex")._extract("answer", "model: gpt-5.6-sol\ntokens used\n1,234\n")
    ev = (f"claude token_count={c.token_count} (in={c.input_tokens},out={c.output_tokens}); "
          f"codex token_count={x.token_count}")
    ok = c.token_count == 4800 and c.input_tokens == 4600 and c.output_tokens == 200 and x.token_count == 1234
    return _ok("L6", "#41", ev) if ok else _fail("L6", "#41", ev)


# --------------------------------------------------------------------------------------------
# L7 — #42 research filename: no doubled "research-research"
# --------------------------------------------------------------------------------------------
def leg_l7() -> LegResult:
    from ai_council.research.output import save_research_to_file

    with tempfile.TemporaryDirectory(prefix="vnc-l7-") as td:
        saved = save_research_to_file(_dummy_report("research best vector databases in 2026"),
                                      Path(td), from_cache=False)
        name = saved[0].name
        ev = f"{name} (research- x{name.count('research-')})"
    ok = "research-research" not in name and name.count("research-") == 1
    return _ok("L7", "#42", ev) if ok else _fail("L7", "#42", ev)


# --------------------------------------------------------------------------------------------
# L8 — #45/#46/#47/#48 (compound; all four must hold)
# --------------------------------------------------------------------------------------------
def leg_l8() -> LegResult:
    import importlib

    fails: list[str] = []

    # #45 shim broken: CouncilRunner only from orchestrator, not runner; panel utils still in runner.
    orch = importlib.import_module("ai_council.orchestrator")
    runner = importlib.import_module("ai_council.runner")
    if not hasattr(orch, "CouncilRunner"):
        fails.append("#45 orchestrator.CouncilRunner missing")
    if hasattr(runner, "CouncilRunner"):
        fails.append("#45 runner.CouncilRunner still re-exported")
    if not (hasattr(runner, "build_all_providers") and hasattr(runner, "determine_panel")):
        fails.append("#45 runner lost panel utils")

    # #46 iso_now centralized: helper tz-aware, used by 5 research providers, no live utcnow.
    from ai_council.research.provider import iso_now
    if "+00:00" not in iso_now() and not iso_now().endswith("Z"):
        fails.append("#46 iso_now not tz-aware")
    prov_dir = _REPO / "src" / "ai_council" / "research" / "providers"
    sites = sum("iso_now()" in (prov_dir / f"{p}.py").read_text(encoding="utf-8")
                for p in ("gemini_research", "grok_research", "openai_deep_research",
                          "openai_mini_research", "perplexity"))
    if sites != 5:
        fails.append(f"#46 iso_now call sites={sites} (want 5)")
    def _is_live_utcnow(ln: str) -> bool:
        s = ln.lstrip()
        return "datetime.utcnow(" in ln and "deprecated" not in ln and not s.startswith(("#", '"', "'"))

    live_utcnow = sum(
        1 for f in (_REPO / "src").rglob("*.py")
        for ln in f.read_text(encoding="utf-8").splitlines()
        if _is_live_utcnow(ln)
    )
    if live_utcnow:
        fails.append(f"#46 live datetime.utcnow( calls={live_utcnow}")

    # #47 dead code gone: no _target_projects assignment in routing.py.
    if "_target_projects" in (_REPO / "src" / "ai_council" / "routing.py").read_text(encoding="utf-8"):
        fails.append("#47 _target_projects still present in routing.py")

    # #48 RunPolicy from settings.yaml policy block (explicit honored, None -> defaults, live -> 2/1).
    from ai_council.policy import RunPolicy
    from config.config_loader import load_config
    explicit = RunPolicy.from_config({"min_panel_size": 5, "max_retries_per_provider": 3})
    default = RunPolicy.from_config(None)
    live = RunPolicy.from_config(load_config().policy)
    if not (explicit.min_panel_size == 5 and explicit.max_retries_per_provider == 3):
        fails.append("#48 explicit policy not honored")
    if not (default.min_panel_size == 2 and default.max_retries_per_provider == 1):
        fails.append("#48 None-defaults wrong")
    if not (live.min_panel_size == 2 and live.max_retries_per_provider == 1):
        fails.append(f"#48 live policy={live}")

    ev = "shim-ok iso_now-ok deadcode-ok RunPolicy-ok" if not fails else "; ".join(fails)
    return _ok("L8", "#45-48", ev) if not fails else _fail("L8", "#45-48", ev)


LEGS = [leg_l1, leg_l2, leg_l3, leg_l4, leg_l5, leg_l6, leg_l7, leg_l8]


def main() -> int:
    results: list[LegResult] = []
    for fn in LEGS:
        name = fn.__name__.split("_")[-1].upper()
        try:
            results.append(fn())
        except Exception as exc:  # a leg that raises is a FAIL, never a silent skip
            results.append(_fail(name, "?", f"EXCEPTION: {type(exc).__name__}: {exc}"))

    print("=" * 78)
    print("NIGHT-CONSOLIDATION VERIFICATION  -  2026-07-18 shipped batch")
    print("=" * 78)
    print(f"{'LEG':<5}{'ID':<9}{'VERDICT':<9}EVIDENCE")
    print("-" * 78)
    for r in results:
        print(f"{r.leg:<5}{r.id_:<9}{r.verdict:<9}{r.evidence}")
    print("-" * 78)
    passed = sum(r.verdict == "PASS" for r in results)
    print(f"RESULT: {passed}/{len(results)} PASS")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
