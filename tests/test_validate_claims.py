"""Tests for scripts/validate_claims.py -- #97 claim-vs-reality checker (Unit 1).

Two kinds of test live here:
  * HARNESS / MODEL tests (this file's first half) -- assert structural properties of the
    Finding model, the exit-code contract, the registry, and the read-only guarantee. These
    are CC-built: they test the skeleton, not drift detection.
  * RULE-LEG tests (second half) -- RED-first drift tests for rules 2/3/4/8. Their fixtures are
    luna-sourced (real night-batch drift); CC transcribes them verbatim. Authorship separation
    applies to these drift fixtures only.

Loader + tmp_path git-repo pattern copied from tests/test_validate_docs_registry.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "validate_claims.py"


def _load():
    spec = importlib.util.spec_from_file_location("validate_claims", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_claims"] = module
    spec.loader.exec_module(module)
    return module


vc = _load()


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert out.returncode == 0, f"git {args} failed: {out.stderr}"
    return out.stdout


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _run_evidence(finding, repo: Path) -> subprocess.CompletedProcess:
    """Parse the PRINTED evidence back to argv (proving human-re-runnability), then execute it,
    mapping the bare `python` runner to THIS interpreter at exec time only."""
    import shlex
    argv = shlex.split(finding.printed())
    assert tuple(argv) == finding.evidence, "printed evidence did not round-trip to argv"
    if argv[0] == "python":
        argv = [sys.executable, *argv[1:]]
    elif argv[0] == "pytest":
        argv = [sys.executable, "-m", "pytest", *argv[1:]]
    elif argv[0] == "git":
        argv = ["git", "-C", str(repo), *argv[1:]]
    return subprocess.run(argv, cwd=str(repo), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


# =============================================================================
# HARNESS / MODEL tests (CC-built)
# =============================================================================

# --- Finding: re-runnable-by-a-human evidence (rule 12) ----------------------

def test_finding_evidence_roundtrips_through_shlex():
    f = vc.Finding(2, "claim", "DOC.md:1", "reality",
                   ("python", "-c", "print(1)"), reproduces="exit-0")
    import shlex
    assert tuple(shlex.split(f.printed())) == f.evidence


def test_finding_rejects_prose_evidence():
    # Prose in the evidence slot: arg0 is not a runner AND the token carries spaces.
    with pytest.raises(ValueError):
        vc.Finding(2, "c", "D:1", "r", ("this is prose, not a command",))


def test_finding_rejects_absolute_interpreter_path():
    # sys.executable (absolute, backslashes/space on Windows) must NOT be the printed arg0.
    with pytest.raises(ValueError):
        vc.Finding(2, "c", "D:1", "r", (sys.executable, "-c", "print(1)"))


def test_finding_rejects_empty_evidence():
    with pytest.raises(ValueError):
        vc.Finding(2, "c", "D:1", "r", ())


def test_finding_rejects_bare_runner_evidence():
    # A runner with no argument is not a runnable command (H2): ("python",) / ("git",).
    with pytest.raises(ValueError):
        vc.Finding(2, "c", "D:1", "r", ("python",))
    with pytest.raises(ValueError):
        vc.Finding(2, "c", "D:1", "r", ("git",))


def test_finding_evidence_executes_and_reproduces(tmp_path):
    # The teeth of rule 12: a Finding's printed command is parsed back and RUN, and it
    # reproduces the reality (here, prints a token) -- verified, not promised.
    repo = _init_repo(tmp_path)
    f = vc.Finding(2, "demo", "DOC.md:1", "the-token",
                   ("python", "-c", "print('the-token')"), reproduces="stdout-contains")
    res = _run_evidence(f, repo)
    assert res.returncode == 0
    assert "the-token" in res.stdout


# --- RuleResult status derivation --------------------------------------------

def test_ruleresult_with_findings_is_fail():
    f = vc.Finding(2, "c", "D:1", "r", ("git", "status"), reproduces="exit-0")
    r = vc.RuleResult(2, "path-existence", findings=(f,))
    assert r.status == "fail"


def test_ruleresult_fail_without_findings_raises():
    with pytest.raises(ValueError):
        vc.RuleResult(2, "path-existence", status="fail")


def test_ruleresult_anchor_missing_is_not_fail():
    r = vc.RuleResult(2, "path-existence", status="anchor-missing", detail="doc reworded")
    assert r.status == "anchor-missing"
    assert not r.findings


# --- exit-code contract (0 / 1 / >=2) ----------------------------------------

def test_exit_code_clean_is_zero():
    results = [vc.RuleResult(2, "x", status="pass"), vc.RuleResult(3, "y", status="skipped")]
    assert vc.exit_code(results, []) == 0


def test_exit_code_findings_is_one():
    f = vc.Finding(2, "c", "D:1", "r", ("git", "status"), reproduces="exit-0")
    results = [vc.RuleResult(2, "x", findings=(f,))]
    assert vc.exit_code(results, []) == 1


def test_exit_code_error_is_two_and_dominates_findings():
    f = vc.Finding(2, "c", "D:1", "r", ("git", "status"), reproduces="exit-0")
    results = [vc.RuleResult(2, "x", findings=(f,))]
    # A checker error must dominate: a crash is never reported as a pass (or even as findings).
    assert vc.exit_code(results, [("rule_9", "RuntimeError: boom")]) == 2


def test_report_header_names_known_limitations():
    # The checker must not overclaim about itself: the report header states R2's precision
    # tradeoff, the unparsed-negation class (named, not allowlisted), and the Unit-2 SKIP caveat.
    out = vc.format_report([vc.RuleResult(2, "path-existence", status="pass")], [])
    assert "KNOWN LIMITATIONS" in out
    assert "precision-over-recall" in out
    assert ".claude/skills/" in out
    # The coverage block replaced the old "Rules 5/6/... are Unit-2 stubs" line: it must
    # still carry the SKIP caveat, and now also a denominator against the 14-rule spec.
    assert "A clean run is NOT a clean repo." in out
    assert "of 14 accounted for" in out
    # ...and it must disclose that the spec set is hand-maintained, so the report does not
    # overclaim: a change to the SPEC itself is not detected by anything here.
    assert "maintained BY HAND against BACKLOG #97" in out


def test_run_all_isolates_a_crashing_leg():
    def boom(ctx):
        raise RuntimeError("leg exploded")
    good = lambda ctx: vc.RuleResult(2, "x", status="pass")  # noqa: E731
    results, errors = vc.run_all(None, legs=[good, boom])
    assert len(results) == 1 and results[0].status == "pass"
    assert len(errors) == 1 and "leg exploded" in errors[0][1]
    assert vc.exit_code(results, errors) == 2


# --- registry auto-test (a new leg is auto-covered) --------------------------

@pytest.mark.parametrize("leg", vc.RULES, ids=lambda leg: getattr(leg, "__name__", "?"))
def test_every_registered_leg_returns_a_ruleresult(leg, tmp_path):
    ctx = vc.RepoContext(_init_repo(tmp_path))
    r = leg(ctx)
    assert isinstance(r, vc.RuleResult)
    assert r.status in vc._STATUSES


def test_registry_ids_are_unique_and_sorted_on_output():
    ctx_results = [leg.__name__ for leg in vc.RULES]
    assert len(ctx_results) == len(set(ctx_results))


# --- registry completeness vs the #97 spec (EXTERNAL denominator) ------------

# Written out literally and deliberately NOT derived from vc.*: this is the external
# denominator #97 gate (ii) requires. A test that iterates vc.RULES covers whatever is
# present and structurally cannot detect an omission -- which is how rules 1 and 7 sat
# absent from the registry while 41 registry-parametrized tests passed green.
# Independently derived from BACKLOG #97's inline spec by sol (gpt-5.6-sol, 2026-07-26)
# with no sight of this file or scripts/validate_claims.py: {1..14}, contiguous, no gaps.
SPEC_RULE_IDS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14})

# Rule 12 is the ONE sanctioned exemption from registration: realized structurally in the
# Finding model (evidence argv + shlex round-trip), so every finding carries a re-runnable
# command by construction and there is no leg to register. #97: "rule 12 is structural
# rather than a leg" / "the one sanctioned exemption".
STRUCTURAL_EXEMPTIONS = frozenset({12})


def test_registry_covers_the_full_spec_id_set(tmp_path):
    assert len(SPEC_RULE_IDS) == 14, "the #97 spec defines fourteen rules"
    ctx = vc.RepoContext(_init_repo(tmp_path))
    registered = {leg(ctx).rule_id for leg in vc.RULES}
    expected = SPEC_RULE_IDS - STRUCTURAL_EXEMPTIONS
    assert registered == expected, (
        "RULES drifted from the #97 spec: "
        f"missing {sorted(expected - registered)}, "
        f"unexpected {sorted(registered - expected)}"
    )


def test_spec_constants_match_the_checkers_own():
    # Two literals exist on purpose (the test's is the external denominator), so they need a
    # binding or they can drift apart silently -- one updated, the other not, both green.
    assert vc._SPEC_RULE_IDS == SPEC_RULE_IDS
    assert vc._STRUCTURAL_EXEMPTIONS == STRUCTURAL_EXEMPTIONS


def test_report_discloses_the_structural_exemption():
    # Rule 12's exemption must be visible in the report with its reason, not just asserted in
    # a constant -- otherwise the exemption set could be widened and no run would say so.
    # Asserting STRUCTURAL_EXEMPTIONS == {12} against itself would be tautological (#94).
    out = vc.format_report([vc.RuleResult(2, "path-existence", status="pass")], [])
    assert "structural  (1): 12" in out
    assert "shlex round-trip" in out


def test_report_absent_count_is_live_not_hardcoded():
    # The whole point of computing the denominator: drop a leg and the report SAYS so.
    # Here rule 7 is simply not among the results, and `absent` must name it.
    partial = [vc.RuleResult(rid, f"r{rid}", status="skipped", detail="Unit 2")
               for rid in sorted(SPEC_RULE_IDS - STRUCTURAL_EXEMPTIONS - {7})]
    out = vc.format_report(partial, [])
    assert "absent      (1): 7" in out
    assert "TOTAL 14 of 14 accounted for" in out


# --- read-only proof: the checker mutates nothing under the repo root ---------

def _manifest(root: Path) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            out[str(p.relative_to(root))] = (len(data), hashlib.sha256(data).hexdigest())
    return out


def test_checker_mutates_nothing(tmp_path):
    # Build a small git repo with real-shaped canonical docs, run EVERY leg, and assert the
    # working tree is byte-identical afterwards. Any accidental mutating call breaks this.
    repo = _init_repo(tmp_path)
    (repo / "CLAUDE.md").write_text("# CLAUDE\n\nSee `scripts/` and `tests/`.\n", encoding="utf-8")
    (repo / "ARCHITECTURE.md").write_text("# ARCH\n\n## Validators\n- `check.ps1`\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    (repo / "tests").mkdir()
    (repo / ".pre-commit-config.yaml").write_text(
        "repos:\n  - repo: local\n    hooks:\n      - id: ruff\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "decisions").mkdir()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")

    before = _manifest(repo)
    ctx = vc.RepoContext(repo)
    vc.run_all(ctx)
    after = _manifest(repo)
    assert before == after, "checker mutated the working tree"


# =============================================================================
# RULE-LEG tests (luna-sourced drift fixtures; CC transcribes verbatim)
# =============================================================================

# --- R2: path existence ------------------------------------------------------

_R2_DOC_DRIFTED = """\
# DOC.md

## 4. Conventions
- Config lives in `src/ai_council/config/` -- the single source of truth is `config/settings.yaml`.
- CLI entry point at `src/ai_council/cli.py`.
- Hub protocols: `.dev-knowledge/protocols/ESSENTIALS.md`.

## 12. Section history
- v1.0 -- see old path `src/ai_council/legacy/guide.py` for the pre-migration GUIDE location.
"""

_R2_DOC_CLEAN = """\
# DOC.md

## 4. Conventions
- Config lives in `config/` -- the single source of truth is `config/settings.yaml`.
- CLI entry point at `src/ai_council/cli.py`.
- Hub protocols: `.dev-knowledge/protocols/ESSENTIALS.md`.

## 12. Section history
- v1.0 -- see old path `src/ai_council/legacy/guide.py` for the pre-migration GUIDE location.
"""


def _r2_tree(tmp_path: Path, doc_text: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "ai_council").mkdir(parents=True)
    (repo / "src" / "ai_council" / "cli.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "config").mkdir()
    (repo / "config" / "settings.yaml").write_text("k: v\n", encoding="utf-8")
    (repo / "protocols").mkdir()
    (repo / "protocols" / "COUNCIL_QUESTION_GUIDE.md").write_text("# g\n", encoding="utf-8")
    (repo / "DOC.md").write_text(doc_text, encoding="utf-8")
    return repo


def test_r2_flags_missing_path(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
    ctx = vc.RepoContext(_r2_tree(tmp_path, _R2_DOC_DRIFTED))
    r = vc.rule_2(ctx)
    assert r.status == "fail"
    locs = {f.location for f in r.findings}
    claims = " ".join(f.claim for f in r.findings)
    # The one real miss: src/ai_council/config/ (config is at repo-root config/).
    assert "src/ai_council/config/" in claims
    assert "DOC.md:4" in locs


def test_r2_does_not_flag_resolving_or_allowlisted_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
    ctx = vc.RepoContext(_r2_tree(tmp_path, _R2_DOC_DRIFTED))
    r = vc.rule_2(ctx)
    flagged = " ".join(f"{f.location} {f.claim}" for f in r.findings)
    # resolving paths
    assert "config/settings.yaml" not in flagged
    assert "src/ai_council/cli.py" not in flagged
    # hub-qualified allowlist
    assert ".dev-knowledge/protocols/ESSENTIALS.md" not in flagged
    # historical-section allowlist (## 12. Section history)
    assert "src/ai_council/legacy/guide.py" not in flagged


def test_r2_clean_twin_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
    ctx = vc.RepoContext(_r2_tree(tmp_path, _R2_DOC_CLEAN))
    r = vc.rule_2(ctx)
    assert r.status == "pass"
    assert not r.findings


def test_r2_evidence_reproduces(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
    repo = _r2_tree(tmp_path, _R2_DOC_DRIFTED)
    ctx = vc.RepoContext(repo)
    r = vc.rule_2(ctx)
    for f in r.findings:
        res = _run_evidence(f, repo)
        assert res.returncode == 1, f"R2 evidence did not reproduce the miss: {f.printed()}"


# --- R3: hook-roster parity --------------------------------------------------

_R3_ARCH_DRIFTED = """\
# ARCH.md

## Validators
- normalize-headers
- floor-hash-verify
- canonical_freshness
"""

_R3_ARCH_CLEAN = """\
# ARCH.md

## Validators
- normalize-headers
- floor-hash-verify
- canonical_freshness
- block-ff-push
- ruff
"""

_R3_PRECOMMIT = """\
repos:
  - repo: local
    hooks:
      - id: normalize-headers
      - id: floor-hash-verify
      - id: canonical_freshness
      - id: block-ff-push
      - id: ruff
"""


def _r3_tree(tmp_path: Path, arch_text: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ARCH.md").write_text(arch_text, encoding="utf-8")
    (repo / ".pre-commit-config.yaml").write_text(_R3_PRECOMMIT, encoding="utf-8")
    return repo


def test_r3_flags_roster_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_hook_roster_docs", lambda ctx: ["ARCH.md"])
    ctx = vc.RepoContext(_r3_tree(tmp_path, _R3_ARCH_DRIFTED))
    r = vc.rule_3(ctx)
    assert r.status == "fail"
    f = r.findings[0]
    # config declares block-ff-push and ruff, which the doc roster omits.
    assert "block-ff-push" in f.reality and "ruff" in f.reality
    assert f.location.startswith("ARCH.md:3")   # the ## Validators section


def test_r3_clean_twin_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_hook_roster_docs", lambda ctx: ["ARCH.md"])
    ctx = vc.RepoContext(_r3_tree(tmp_path, _R3_ARCH_CLEAN))
    r = vc.rule_3(ctx)
    assert r.status == "pass"


def test_r3_evidence_reproduces(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_hook_roster_docs", lambda ctx: ["ARCH.md"])
    repo = _r3_tree(tmp_path, _R3_ARCH_DRIFTED)
    r = vc.rule_3(vc.RepoContext(repo))
    for f in r.findings:
        res = _run_evidence(f, repo)
        assert res.returncode == 0
        assert "block-ff-push" in res.stdout and "ruff" in res.stdout


# --- R4: ADR-roster parity ---------------------------------------------------

_R4_DOC_DRIFTED = """\
# DOC.md

## 11. Recent ADRs binding here
- ADR-01: Synthesizer Selection
- ADR-02: Default Panel Composition
- ADR-12: Provider Backend Engine
- ADR-14: ADR lifecycle states
"""

_R4_DOC_CLEAN = """\
# DOC.md

## 11. Recent ADRs binding here
- ADR-01: Synthesizer Selection
- ADR-02: Default Panel Composition
- ADR-12: Provider Backend Engine
- ADR-13: Invocation-contract versioning
- ADR-14: ADR lifecycle states
"""

_R4_ADR_FILES = [
    "ADR-01-synthesizer-selection.md",
    "ADR-02-default-panel.md",
    "ADR-12-provider-backend-engine-and-cost-lanes.md",
    "ADR-13-invocation-contract-versioning.md",
    "ADR-14-adr-lifecycle-states.md",
]


def _r4_tree(tmp_path: Path, doc_text: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs" / "decisions").mkdir(parents=True)
    for name in _R4_ADR_FILES:
        (repo / "docs" / "decisions" / name).write_text(f"# {name}\n", encoding="utf-8")
    (repo / "DOC.md").write_text(doc_text, encoding="utf-8")
    return repo


def test_r4_flags_unrostered_adr(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_adr_roster_docs", lambda ctx: ["DOC.md"])
    ctx = vc.RepoContext(_r4_tree(tmp_path, _R4_DOC_DRIFTED))
    r = vc.rule_4(ctx)
    assert r.status == "fail"
    f = r.findings[0]
    assert "ADR-13" in f.reality
    assert f.location.startswith("DOC.md:3")


def test_r4_clean_twin_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_adr_roster_docs", lambda ctx: ["DOC.md"])
    ctx = vc.RepoContext(_r4_tree(tmp_path, _R4_DOC_CLEAN))
    r = vc.rule_4(ctx)
    assert r.status == "pass"


def test_r4_evidence_reproduces(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_adr_roster_docs", lambda ctx: ["DOC.md"])
    repo = _r4_tree(tmp_path, _R4_DOC_DRIFTED)
    r = vc.rule_4(vc.RepoContext(repo))
    for f in r.findings:
        res = _run_evidence(f, repo)
        assert res.returncode == 0
        assert "ADR-13" in res.stdout


# --- R8: SHA reachability ----------------------------------------------------

def _r8_repo(tmp_path: Path, drifted: bool):
    """Build a throwaway repo. SHA_A is HEAD (reachable). When drifted, SHA_B is a commit made
    on a temp branch that is then DELETED -- a loose object, unreachable from any ref (git
    rev-parse would still resolve it: the exact existence-vs-reachability bug R8 targets)."""
    repo = _init_repo(tmp_path)
    _git(repo, "commit", "--allow-empty", "-q", "-m", "init commit")
    branch = _git(repo, "symbolic-ref", "--short", "HEAD").strip()
    sha_a = _git(repo, "rev-parse", "--short=7", "HEAD").strip()
    sha_b = None
    if drifted:
        _git(repo, "checkout", "-q", "-b", "spike")
        _git(repo, "commit", "--allow-empty", "-q", "-m", "spike work")
        sha_b = _git(repo, "rev-parse", "--short=7", "HEAD").strip()
        _git(repo, "checkout", "-q", branch)
        _git(repo, "branch", "-D", "spike")
    if drifted:
        doc = (
            "# JOURNAL.md (fixture)\n\n"
            "### 2026-07-24 -- night batch\n"
            f"**Did:** Landed initial scaffolding at `{sha_a}` (reachable, verified via merge-base).\n"
            f"Spike work at `{sha_b}` was explored on a throwaway branch and later deleted -- do not cite.\n"
            "Incorporated edit-distance feedback from the reviewer; ADR-11 verdict pending.\n"
        )
    else:
        doc = (
            "# JOURNAL.md (fixture)\n\n"
            "### 2026-07-24 -- night batch\n"
            f"**Did:** Landed initial scaffolding at `{sha_a}` (reachable, verified via merge-base).\n"
            f"Follow-up polish landed in the same commit `{sha_a}` before the branch closed.\n"
            "Incorporated edit-distance feedback from the reviewer; ADR-11 verdict pending.\n"
        )
    (repo / "JOURNAL.md").write_text(doc, encoding="utf-8")
    return repo, sha_a, sha_b


def test_r8_distinguishes_existence_from_reachability(tmp_path):
    # The premise: SHA_B EXISTS as an object (cat-file -e succeeds) but is UNREACHABLE.
    # rev-parse/cat-file resolving it is exactly why an existence check would wrongly pass it.
    repo, sha_a, sha_b = _r8_repo(tmp_path, drifted=True)
    ctx = vc.RepoContext(repo)
    assert ctx.git("cat-file", "-e", sha_b).returncode == 0            # object EXISTS
    assert ctx.git("merge-base", "--is-ancestor", sha_b, "HEAD").returncode == 1  # UNREACHABLE


def test_r8_flags_unreachable_sha_not_reachable_nor_feedbac(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_sha_citation_docs", lambda ctx: ["JOURNAL.md"])
    repo, sha_a, sha_b = _r8_repo(tmp_path, drifted=True)
    r = vc.rule_8(vc.RepoContext(repo))
    assert r.status == "fail"
    blob = " ".join(f.claim + " " + f.reality + " " + f.location for f in r.findings)
    assert sha_b in blob                 # the dangling spike SHA is reported
    assert sha_a not in blob             # the reachable SHA is not
    assert "feedbac" not in blob         # 'feedback' prose is not a SHA (word-boundary + hex-only)


def test_r8_clean_twin_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_sha_citation_docs", lambda ctx: ["JOURNAL.md"])
    repo, _, _ = _r8_repo(tmp_path, drifted=False)
    r = vc.rule_8(vc.RepoContext(repo))
    assert r.status == "pass"


def test_r8_evidence_reproduces(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_sha_citation_docs", lambda ctx: ["JOURNAL.md"])
    repo, _, _ = _r8_repo(tmp_path, drifted=True)
    r = vc.rule_8(vc.RepoContext(repo))
    for f in r.findings:
        res = _run_evidence(f, repo)
        # `git describe --all --contains <dangling>` fails (no ref reaches it) -> any-ref
        # reachability, faithful to the finding's claim (H5). Non-zero exit, not specifically 1.
        assert res.returncode != 0, f"R8 evidence should fail for a dangling SHA: {f.printed()}"


def test_r8_git_failure_is_a_checker_error_not_a_pass(tmp_path, monkeypatch):
    # H1: a broken git query must surface as a checker ERROR (exit >=2), never a silent pass.
    monkeypatch.setattr(vc, "_sha_citation_docs", lambda ctx: ["JOURNAL.md"])
    repo, _, _ = _r8_repo(tmp_path, drifted=True)
    ctx = vc.RepoContext(repo)
    orig_git = ctx.git

    def _broken(*args, **kwargs):
        if args[:1] == ("rev-list",):
            return subprocess.CompletedProcess(args, 128, "", "fatal: broken")
        return orig_git(*args, **kwargs)

    ctx.git = _broken
    results, errors = vc.run_all(ctx, legs=[vc.rule_8])
    assert errors, "a failed git query must be recorded as a checker error"
    assert vc.exit_code(results, errors) == 2
