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

# --- #106: evidence must paste into PowerShell as well as git-bash -----------

_POSIX_SPLICE = "'\"'\"'"   # what shlex.join emits for an inner single quote


def _r2_shaped_finding(tok="logs/TOKEN-LOG.md"):
    return vc.Finding(
        rule_id=2, claim="c", location="DOC.md:1", reality="r",
        evidence=("python", "-c",
                  f"import pathlib,sys; sys.exit(0 if pathlib.Path({tok!r}).exists() else 1)"),
        reproduces="exit-nonzero")


def test_r2_evidence_carries_no_posix_only_splice():
    # #106's named acceptance test. The '"'"' idiom is valid POSIX and unparseable in
    # PowerShell, which is this repo's default shell -- so its presence made rule 12's
    # "re-runnable by a human" claim false for the most common finding shape.
    printed = _r2_shaped_finding().printed()
    assert _POSIX_SPLICE not in printed
    assert printed == (
        'python -c "import pathlib,sys; '
        "sys.exit(0 if pathlib.Path('logs/TOKEN-LOG.md').exists() else 1)\"")


def test_shell_quote_prefers_double_quotes_only_when_both_shells_agree():
    # Single quotes inside, nothing interpolable -> double-quote: identical in both shells.
    assert vc._shell_quote("say 'hi'") == "\"say 'hi'\""
    # A safe bare token is left alone rather than gratuitously quoted.
    assert vc._shell_quote("git") == "git"
    assert vc._shell_quote("src/ai_council/cli.py") == "src/ai_council/cli.py"


def test_shell_quote_falls_back_when_double_quoting_would_be_wrong():
    # $ interpolates in BOTH shells; a backtick escapes in PowerShell; \ escapes in POSIX; a
    # double quote would terminate the string. Emitting a confidently-wrong double-quoted form
    # is worse than falling back to POSIX-correct quoting, so these degrade rather than guess.
    for hostile in ("cost is $5", "tick ` here", "back\\slash", 'has "quotes"'):
        assert vc._shell_quote(hostile) == __import__("shlex").quote(hostile)


def test_printed_still_round_trips_after_the_quoting_change():
    # The __post_init__ invariant #106 must not break: the printed string parses back to argv.
    import shlex as _s
    f = _r2_shaped_finding()
    assert tuple(_s.split(f.printed())) == f.evidence


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
    # tradeoff, what context adjudication does and does not do (#108), and the coverage
    # denominator. The old "negation is not parsed" bullet was retired when #108 landed --
    # leaving it would have been the checker overclaiming a limitation it no longer has.
    out = vc.format_report([vc.RuleResult(2, "path-existence", status="pass")], [])
    assert "KNOWN LIMITATIONS" in out
    assert "precision-over-recall" in out
    for cls in ("negated", "externally-attributed", "hypothetical"):
        assert cls in out
    # ...and it must say the suppression is not counted, so a silent suppression stays disclosed.
    assert "Suppressions are not currently counted" in out
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


# --- harness determinism: RepoContext reads the COMMIT TREE, not the disk ----
# Frozen acceptance for the 2026-07-26 ruling. These are written against the HARNESS, not any
# one rule: the point is that every leg -- the four implemented, the nine stubs, and every
# future one -- becomes deterministic by construction rather than one rule at a time.


def _seeded_repo(tmp_path: Path) -> Path:
    """A committed repo shaped like the real one, for harness-level determinism tests."""
    repo = _init_repo(tmp_path)
    (repo / "src" / "ai_council").mkdir(parents=True)
    (repo / "src" / "ai_council" / "cli.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "docs" / "decisions").mkdir(parents=True)
    (repo / "docs" / "decisions" / "ADR-01-first.md").write_text("# ADR-01\n", encoding="utf-8")
    (repo / ".pre-commit-config.yaml").write_text(
        "repos:\n  - repo: local\n    hooks:\n      - id: alpha\n      - id: beta\n",
        encoding="utf-8")
    (repo / "CLAUDE.md").write_text(
        "# CLAUDE\n\n## 9. Hooks active\n- `alpha`\n- `beta`\n\n"
        "## 11. Recent ADRs\n- ADR-01: first\n", encoding="utf-8")
    (repo / "ARCHITECTURE.md").write_text("# ARCH\n\n## Validators\n- `alpha`\n- `beta`\n",
                                          encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
    return repo


def test_ctx_read_returns_committed_content_not_the_working_copy(tmp_path):
    repo = _seeded_repo(tmp_path)
    ctx = vc.RepoContext(repo)
    (repo / "CLAUDE.md").write_text("# CLAUDE\n\nDIRTIED, uncommitted\n", encoding="utf-8")
    assert "DIRTIED" not in vc.RepoContext(repo).read("CLAUDE.md"), (
        "ctx.read must return HEAD's content; reading the working copy is rule 2's defect "
        "one layer up")
    assert "Hooks active" in ctx.read("CLAUDE.md")


def test_ctx_exists_and_glob_ignore_untracked_files(tmp_path):
    repo = _seeded_repo(tmp_path)
    (repo / "docs" / "decisions" / "ADR-99-scratch.md").write_text("# scratch\n", encoding="utf-8")
    ctx = vc.RepoContext(repo)
    assert not ctx.exists("docs/decisions/ADR-99-scratch.md"), "untracked file must not exist to ctx"
    names = [p.name for p in ctx.glob("docs/decisions/ADR-*.md")]
    assert names == ["ADR-01-first.md"], f"glob must not see untracked files, got {names}"


def test_disk_access_is_opt_in_and_no_rule_uses_it(tmp_path):
    # The opt-in exists (a harness may still need the disk) but must be unused by rule legs --
    # otherwise determinism is a convention rather than a property.
    repo = _seeded_repo(tmp_path)
    ctx = vc.RepoContext(repo)
    (repo / "docs" / "decisions" / "ADR-99-scratch.md").write_text("# s\n", encoding="utf-8")
    assert ctx.disk_exists("docs/decisions/ADR-99-scratch.md"), "opt-in disk access must work"
    assert not ctx.exists("docs/decisions/ADR-99-scratch.md")

    src = _P.read_text(encoding="utf-8")
    body = src[src.index("# --- rule legs"):]
    for name in ("disk_exists", "disk_read", "disk_glob"):
        assert name not in body, f"a rule leg calls ctx.{name}() -- determinism must not be optional"


# --- FROZEN ACCEPTANCE (2026-07-26 ruling) -----------------------------------

def test_acceptance_rule3_passes_with_a_dirty_uncommitted_config(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_hook_roster_docs", lambda ctx: ["CLAUDE.md", "ARCHITECTURE.md"])
    repo = _seeded_repo(tmp_path)
    assert vc.rule_3(vc.RepoContext(repo)).status == "pass"
    (repo / ".pre-commit-config.yaml").write_text(
        "repos:\n  - repo: local\n    hooks:\n      - id: alpha\n      - id: GAMMA-DIRTY\n",
        encoding="utf-8")
    assert vc.rule_3(vc.RepoContext(repo)).status == "pass", (
        "a dirty uncommitted config must not change rule 3's verdict")


def test_acceptance_rule4_passes_with_an_untracked_adr_present(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_adr_roster_docs", lambda ctx: ["CLAUDE.md"])
    repo = _seeded_repo(tmp_path)
    assert vc.rule_4(vc.RepoContext(repo)).status == "pass"
    (repo / "docs" / "decisions" / "ADR-99-scratch.md").write_text("# s\n", encoding="utf-8")
    assert vc.rule_4(vc.RepoContext(repo)).status == "pass", (
        "an untracked scratch ADR must not change rule 4's verdict")


def test_acceptance_dirty_primary_agrees_with_a_clean_clone(tmp_path, monkeypatch):
    # The clause that matters: the earlier determinism run compared two CLEAN checkouts, which is
    # the same blind spot as the ls-files acceptance. This compares a DIRTY tree to a clean clone.
    monkeypatch.setattr(vc, "_hook_roster_docs", lambda ctx: ["CLAUDE.md", "ARCHITECTURE.md"])
    monkeypatch.setattr(vc, "_adr_roster_docs", lambda ctx: ["CLAUDE.md"])
    repo = _seeded_repo(tmp_path)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True,
                   capture_output=True, text=True)

    # dirty the primary in every way the harness could notice
    (repo / "CLAUDE.md").write_text("# CLAUDE\n\nDIRTIED\n", encoding="utf-8")
    (repo / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
    (repo / "docs" / "decisions" / "ADR-99-scratch.md").write_text("# s\n", encoding="utf-8")

    def verdicts(root):
        c = vc.RepoContext(root)
        return {r.rule_id: r.status for r in (vc.rule_3(c), vc.rule_4(c))}

    assert verdicts(repo) == verdicts(clone), (
        "a DIRTY primary and a clean clone at the same commit must agree")


# --- #116 R2 resolution model (tracked tree, declared bases) -----------------

_REPO = Path(__file__).resolve().parent.parent


def _r2_tracked(tmp_path: Path, doc_text: str, tracked: tuple[str, ...]) -> Path:
    """A git repo whose listed files are actually COMMITTED, so ls-files sees them."""
    repo = _init_repo(tmp_path)
    for rel in tracked:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n", encoding="utf-8")
    (repo / "DOC.md").write_text(doc_text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
    return repo


def test_declared_bases_exist_in_the_tracked_tree():
    # The base list validates ITSELF: a base that does not exist is a base that silently
    # suppresses every token written relative to it -- the failure #116 is repairing, one
    # level up. Run against the real repo, because it is a claim about THIS repo.
    ctx = vc.RepoContext(_REPO)
    files, dirs = ctx.committed_paths()
    for base in vc._R2_BASES:
        if base == "":
            continue                      # the repo root needs no proof
        assert base in dirs, f"declared R2 base {base!r} is not in the tracked tree"


def test_declared_runtime_paths_really_are_gitignored_and_untracked():
    # Same self-validation for the runtime list: each entry must be genuinely gitignored AND
    # genuinely untracked, or it is an excuse rather than a declaration.
    ctx = vc.RepoContext(_REPO)
    files, dirs = ctx.committed_paths()
    # Validated against the TRACKED .gitignore, not `git check-ignore` (terra 2026-07-26):
    # check-ignore also consults .git/info/exclude and the user's global excludesfile, both
    # untracked, so a green result could come from local config -- the same checkout-dependence
    # the checker itself refuses. .gitignore is tracked, so reading it is commit-deterministic.
    ignore_lines = {ln.strip() for ln in (_REPO / ".gitignore").read_text(
        encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")}
    for p in vc._R2_RUNTIME_PATHS:
        assert p.rstrip("/") + "/" not in dirs, f"{p} is committed -- it does not belong here"
        assert p in ignore_lines or p.rstrip("/") in ignore_lines, (
            f"{p} is not an entry in the tracked .gitignore -- the declaration is false")


def test_r2_verdict_is_identical_with_and_without_untracked_debris(tmp_path):
    # THE #116 PROPERTY. logs/ is gitignored; on the primary checkout it existed as untracked
    # debris and on a fresh clone it did not, so the old disk-based guard reported the SAME
    # COMMIT two different ways. Resolving against the tracked tree makes that impossible.
    doc = "# DOC.md\n\n- The log is `logs/TOKEN-LOG.md`, append-only.\n"
    repo = _r2_tracked(tmp_path, doc, ("src/ai_council/cli.py",))
    (repo / ".gitignore").write_text("logs/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore logs")

    def verdict():
        import importlib
        m = importlib.import_module("validate_claims")
        m._canonical_docs = lambda ctx: ["DOC.md"]
        return m.rule_2(m.RepoContext(repo))

    without = verdict()
    (repo / "logs").mkdir()                                  # untracked, gitignored debris
    (repo / "logs" / "TOKEN-LOG.md").write_text("x\n", encoding="utf-8")
    with_debris = verdict()

    assert without.status == "fail" and with_debris.status == "fail"
    assert [f.location for f in without.findings] == [f.location for f in with_debris.findings]
    assert [f.claim for f in without.findings] == [f.claim for f in with_debris.findings]


def test_r2_ignores_a_staged_but_uncommitted_path(tmp_path, monkeypatch):
    # terra 2026-07-26: `git ls-files` reports the INDEX, so a staged-but-uncommitted path would
    # satisfy a claim and the verdict would still depend on working state. The two agree on a
    # clean tree -- which is exactly why the first fresh-clone acceptance run passed while this
    # was still wrong -- so this pins the difference directly.
    monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
    repo = _r2_tracked(tmp_path, "# DOC.md\n\n- `docs/later.md`\n", ("src/ai_council/cli.py",))
    assert vc.rule_2(vc.RepoContext(repo)).status == "fail"

    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "later.md").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "docs/later.md")          # staged, deliberately NOT committed
    assert vc.rule_2(vc.RepoContext(repo)).status == "fail", (
        "a staged-but-uncommitted path must not satisfy a claim -- that is index state, "
        "not commit state")

    _git(repo, "commit", "-qm", "commit it")
    assert vc.rule_2(vc.RepoContext(repo)).status == "pass"


def test_r2_resolves_base_relative_and_excludes_external(tmp_path, monkeypatch):
    # Frozen acceptance, in shape: base-relative paths resolve; ecosystem paths and org/repo
    # slugs are excluded by DECLARATION; an unresolvable path still fires.
    monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
    doc = (
        "# DOC.md\n\n"
        "- package-relative: `research/merger.py` and `providers/`\n"
        "- docs-relative: `decisions/` and `audits/`\n"
        "- hub-routed, base-relative: `handoffs/` and `transcripts/`\n"
        "- ecosystem: `Dev/` and `ai-council/output/`\n"
        "- slug: `astral-sh/ruff-pre-commit`\n"
        "- dot-prefixed: `./src/ai_council/cli.py`\n"
        "- genuinely missing: `logs/TOKEN-LOG.md`\n"
    )
    repo = _r2_tracked(tmp_path, doc, (
        "src/ai_council/cli.py", "src/ai_council/research/merger.py",
        "src/ai_council/providers/openai.py",
        "docs/decisions/ADR-01-x.md", "docs/audits/a.md",
    ))
    r = vc.rule_2(vc.RepoContext(repo))
    assert r.status == "fail"
    claims = [f.claim for f in r.findings]
    assert len(claims) == 1, f"expected only the missing path to fire, got {claims}"
    assert "logs/TOKEN-LOG.md" in claims[0]


def test_r2_evidence_probes_the_commit_tree_not_the_disk(tmp_path, monkeypatch):
    # The evidence must test what the RULE tests. A Path.exists() probe would contradict the
    # rule and reproduce the #116 nondeterminism inside the evidence itself; `ls-files
    # --error-unmatch` would consult the INDEX and carry a staged-path dependence instead.
    monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
    repo = _r2_tracked(tmp_path, "# DOC.md\n\n- `logs/TOKEN-LOG.md`\n", ("src/ai_council/cli.py",))
    f = vc.rule_2(vc.RepoContext(repo)).findings[0]
    assert f.evidence[:3] == ("git", "cat-file", "-e")
    assert f.evidence[3].startswith("HEAD:"), "the probe must read the COMMIT, not the index"
    assert _run_evidence(f, repo).returncode != 0


# --- #108 shared context predicate -------------------------------------------
# Fixtures are the REAL lines the two false positives came from, transcribed verbatim, so the
# tests fail if the predicate stops handling the instances that motivated it.

_CTX_NEGATED_LINE = (
    "(`session-summary`/`codex-review` are **commands**, not skills — see §7. This repo has "
    "**no** repo-level `.claude/skills/` directory; a repo-specific gotchas skill, if added, "
    "would go under `.claude/skills/gotchas/`.)"
)
_CTX_EXTERNAL_LINE = "> Note: the hub's `protocols/AI_COUNCIL_PROCESS.md` is a **different** artifact —"
_CTX_STANDING_LINE = (
    "1. **`LESSONS.md` and `logs/TOKEN-LOG.md` are append-only** — never edit old entries; "
    "only append (ADR-29, ADR-39)"
)


def _span_of(line: str, token: str) -> tuple[int, int]:
    i = line.index(token)
    return (i, i + len(token))


def test_ctx_negation_withdraws_the_claim():
    # "This repo has **no** repo-level `.claude/skills/` directory"
    assert vc.context_withdraws_claim(
        _CTX_NEGATED_LINE, _span_of(_CTX_NEGATED_LINE, ".claude/skills/`")) == "negated"


def test_ctx_hypothetical_withdraws_the_claim():
    # Same LINE, different SENTENCE: "if added, would go under `.claude/skills/gotchas/`".
    # This is the case line-scoped adjudication would have gotten right by accident and
    # sentence-scoping has to get right on purpose.
    assert vc.context_withdraws_claim(
        _CTX_NEGATED_LINE, _span_of(_CTX_NEGATED_LINE, ".claude/skills/gotchas/")) == "hypothetical"


def test_ctx_external_attribution_withdraws_the_claim():
    assert vc.context_withdraws_claim(
        _CTX_EXTERNAL_LINE,
        _span_of(_CTX_EXTERNAL_LINE, "protocols/AI_COUNCIL_PROCESS.md")) == "externally-attributed"


def test_ctx_leaves_a_standing_claim_alone():
    # The #111 line: a genuinely stale address with no withdrawal marker. The predicate must NOT
    # swallow it -- that would convert a real finding into a silent false negative, which is the
    # exact trade the narrow classes exist to refuse.
    assert vc.context_withdraws_claim(
        _CTX_STANDING_LINE, _span_of(_CTX_STANDING_LINE, "logs/TOKEN-LOG.md")) is None


def test_ctx_is_sentence_scoped_not_line_scoped():
    # One clause withdraws, the next asserts. Line-scoping would let the first silence the second.
    line = "This repo has no `docs/absent/` directory. The CLI entry point is `src/x/cli.py`."
    assert vc.context_withdraws_claim(line, _span_of(line, "docs/absent/")) == "negated"
    assert vc.context_withdraws_claim(line, _span_of(line, "src/x/cli.py")) is None


def test_ctx_is_pure():
    # Same inputs, same verdict, no I/O -- purity is why two rules can share it without coupling.
    line = _CTX_NEGATED_LINE
    span = _span_of(line, ".claude/skills/`")
    assert vc.context_withdraws_claim(line, span) == vc.context_withdraws_claim(line, span)


def test_ctx_suppression_added_no_allowlist_entries():
    # #108's done-when: both instances go silent with NO path added to the allowlist. If a future
    # change "fixes" a context case by widening the allowlist, this fails.
    assert vc._R2_ALLOWLIST == (
        ".dev-knowledge/", "docs/handoffs/", "docs/decisions/transcripts/")


def test_r2_suppresses_negated_and_hypothetical_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_canonical_docs", lambda ctx: ["DOC.md"])
    doc = "# DOC.md\n\n## 8. Skills\n" + _CTX_NEGATED_LINE + "\n"
    ctx = vc.RepoContext(_r2_tree(tmp_path, doc))
    assert vc.rule_2(ctx).status == "pass"


def test_r8_classifies_only_on_external_attribution(tmp_path, monkeypatch):
    # R8 must act on "externally-attributed" ALONE. A negated sentence is not a coherent way to
    # cite a SHA, and letting it silence R8 would widen the predicate past its witnessed need.
    hub = "the hub's commit deadbee is upstream"          # externally-attributed -> classified
    neg = "this repo has no commit deadbee to speak of"   # negated -> must NOT silence R8
    assert vc.context_withdraws_claim(hub, _span_of(hub, "deadbee")) == "externally-attributed"
    assert vc.context_withdraws_claim(neg, _span_of(neg, "deadbee")) == "negated"


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
    # Files must be COMMITTED, not merely written: since #116 rule 2 resolves against git's
    # tracked tree rather than the disk, an uncommitted fixture would resolve nothing.
    repo = _init_repo(tmp_path)
    (repo / "src" / "ai_council").mkdir(parents=True)
    (repo / "src" / "ai_council" / "cli.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "config").mkdir()
    (repo / "config" / "settings.yaml").write_text("k: v\n", encoding="utf-8")
    (repo / "protocols").mkdir()
    (repo / "protocols" / "COUNCIL_QUESTION_GUIDE.md").write_text("# g\n", encoding="utf-8")
    (repo / "DOC.md").write_text(doc_text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
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
        # reproduces="exit-nonzero" is the declared contract -- pinning ==1 over-specified an
        # incidental code (git cat-file exits 128), which is not what the Finding promises.
        assert res.returncode != 0, f"R2 evidence did not reproduce the miss: {f.printed()}"


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
    repo = _init_repo(tmp_path)
    (repo / "ARCH.md").write_text(arch_text, encoding="utf-8")
    (repo / ".pre-commit-config.yaml").write_text(_R3_PRECOMMIT, encoding="utf-8")
    # Committed, not merely written: RepoContext reads HEAD's tree, so an uncommitted
    # fixture is invisible to every leg (2026-07-26 harness determinism ruling).
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
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
    repo = _init_repo(tmp_path)
    (repo / "docs" / "decisions").mkdir(parents=True)
    for name in _R4_ADR_FILES:
        (repo / "docs" / "decisions" / name).write_text(f"# {name}\n", encoding="utf-8")
    (repo / "DOC.md").write_text(doc_text, encoding="utf-8")
    # Committed, not merely written: RepoContext reads HEAD's tree, so an uncommitted
    # fixture is invisible to every leg (2026-07-26 harness determinism ruling).
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
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
    # Committed: RepoContext reads HEAD's tree, so an uncommitted fixture doc is invisible
    # to the leg (2026-07-26 harness determinism ruling). The DANGLING sha stays dangling --
    # committing the doc that CITES it does not make the cited commit reachable.
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture doc")
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
