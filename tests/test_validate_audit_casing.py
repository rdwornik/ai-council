"""Tests for scripts/validate_audit_casing.py (fleet ruling d1; ADR-101 R4 casing gate).

Firing tests, not presence: the pure classifier (the R4 casing branch of hub Rule B) is
exercised directly, and the prospective-only/grandfather behavior is proven end-to-end
against a REAL temp git repo (an added UPPERCASE audit name BLOCKS; a modified grandfathered
one PASSES). This carry is CASING-ONLY -- the hub 11-class enum + date-shape grammar are NOT
carried, so free-form leading tokens (e.g. `code-quality`) are allowed; the tests assert that.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_P = Path(__file__).resolve().parent.parent / "scripts" / "validate_audit_casing.py"


def _load():
    spec = importlib.util.spec_from_file_location("validate_audit_casing", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_audit_casing"] = module
    spec.loader.exec_module(module)
    return module


vac = _load()


# --- BLOCKS: R4 casing violations --------------------------------------------

def test_blocks_uppercase_underscore():
    # The motivating fixture: the corp-monorepo UPPERCASE _AUDIT_/_BRIEF_ divergence (R4).
    r = vac.audit_casing_violation("docs/audits/2026-07-12-TECHNICAL_AUDIT_BRIEF.md")
    assert r is not None and "casing" in r


def test_blocks_underscore_lowercase():
    r = vac.audit_casing_violation("docs/audits/2026-07-12-technical_foo.md")
    assert r is not None and "casing" in r


def test_blocks_camelcase():
    r = vac.audit_casing_violation("docs/audits/2026-07-12-TechnicalFoo.md")
    assert r is not None and "casing" in r


def test_blocks_uppercase_md_extension():
    # An uppercase .MD extension must NOT dodge the check (apply is extension-case-insensitive).
    r = vac.audit_casing_violation("docs/audits/2026-07-12-technical-good.MD")
    assert r is not None and "casing" in r


# --- PASSES: lowercase-kebab, incl. free-form tokens the enum would reject ----

def test_allows_free_form_leading_token():
    # CASING-ONLY: `code-quality` is NOT in the hub 11-class enum, but casing is clean -> PASS.
    # This proves the enum is deliberately NOT enforced by this carry.
    assert vac.audit_casing_violation("docs/audits/2026-07-06-code-quality-audit.md") is None
    assert vac.audit_casing_violation("docs/audits/2026-07-04-fable-architecture-audit.md") is None


def test_allows_existing_repo_names():
    assert vac.audit_casing_violation(
        "docs/audits/2026-07-11-technical-root-parity-disposition.md") is None
    assert vac.audit_casing_violation("docs/audits/2026-07-09-qa-lived-exercise.md") is None


def test_allows_dot_carveout():
    # `.` allowed inside the name for repo/version tokens (S3-1 carve-out).
    assert vac.audit_casing_violation("docs/audits/2026-07-12-census-.dev-knowledge-sweep.md") is None
    assert vac.audit_casing_violation("docs/audits/2026-07-12-technical-v3.4-abort.md") is None


def test_allows_readme_index():
    assert vac.audit_casing_violation("docs/audits/README.md") is None


def test_shape_agnostic_no_date_required():
    # CASING-ONLY: no date-shape grammar is carried, so a name with no leading date but clean
    # casing PASSES (the hub gate would reject it; this carry deliberately does not).
    assert vac.audit_casing_violation("docs/audits/plain-lowercase-note.md") is None


# --- SILENT: out of Rule B's 3-part docs/audits/*.md scope --------------------

def test_silent_on_legacy_quarantine():
    # docs/audits/archive/legacy/* is 5 path parts, not 3 -> structurally out of scope.
    # Proves the quarantined UPPERCASE_UNDERSCORE legacy set is never policed.
    assert vac.audit_casing_violation(
        "docs/audits/archive/legacy/2026-03-15_CODE_REVIEW_REPORT.md") is None
    assert vac.audit_casing_violation(
        "docs/audits/archive/legacy/2026-03-26_CODE_REVIEW_REPORT.md") is None


def test_silent_on_non_audit_paths():
    assert vac.audit_casing_violation("src/ai_council/foo.py") is None
    assert vac.audit_casing_violation("README.md") is None
    assert vac.audit_casing_violation("docs/decisions/ADR-102-x.md") is None


# --- check() aggregation -----------------------------------------------------

def test_check_aggregates_and_labels():
    reasons = vac.check([
        "src/ai_council/ok.py",                                  # silent
        "docs/audits/2026-07-12-code-quality-audit.md",          # clean (free-form token)
        "docs/audits/2026-07-12-BAD_CASE.md",                    # casing violation
    ])
    assert len(reasons) == 1
    assert reasons[0].startswith("docs/audits/2026-07-12-BAD_CASE.md:")
    assert "casing" in reasons[0]


def test_check_empty_is_clean():
    assert vac.check([]) == []
    assert vac.check(["src/ai_council/a.py", "docs/audits/2026-07-12-technical-a.md"]) == []


# --- prospective-only / grandfather: REAL temp git repo ----------------------

def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, f"git {args} failed: {out.stderr}"
    return out.stdout


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "docs" / "audits").mkdir(parents=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _run_hook(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_P)], cwd=str(repo),
                          capture_output=True, text=True, encoding="utf-8")


def test_prospective_grandfathers_modified_existing(tmp_path):
    # An UPPERCASE audit name that ALREADY EXISTS (committed) is grandfathered: modifying and
    # staging it is status M, not A, so the gate is silent.
    repo = _init_repo(tmp_path)
    bad = repo / "docs" / "audits" / "BADNAME_UPPER.md"
    bad.write_text("# grandfathered\n", encoding="utf-8")
    _git(repo, "add", "-f", str(bad))
    _git(repo, "commit", "-q", "-m", "grandfather the bad name")
    bad.write_text("# grandfathered, edited\n", encoding="utf-8")
    _git(repo, "add", "-f", str(bad))            # staged as MODIFIED
    res = _run_hook(repo)
    assert res.returncode == 0, res.stderr


def test_blocks_newly_added_uppercase_audit(tmp_path):
    repo = _init_repo(tmp_path)
    new = repo / "docs" / "audits" / "2026-07-12-TECHNICAL_AUDIT.md"
    new.write_text("# new bad\n", encoding="utf-8")
    _git(repo, "add", str(new))
    res = _run_hook(repo)
    assert res.returncode == 1
    assert "casing" in res.stderr


def test_allows_newly_added_lowercase_freeform(tmp_path):
    # A free-form leading token (not in the hub enum) with clean casing is ALLOWED.
    repo = _init_repo(tmp_path)
    good = repo / "docs" / "audits" / "2026-07-12-code-quality-audit.md"
    good.write_text("# ok\n", encoding="utf-8")
    _git(repo, "add", str(good))
    res = _run_hook(repo)
    assert res.returncode == 0, res.stderr


def test_rename_to_bad_audit_name_is_blocked(tmp_path):
    # A rename INTRODUCES a new pathname; --no-renames surfaces it as an ADD so the off-casing
    # destination is policed (not silently skipped as status R).
    repo = _init_repo(tmp_path)
    good = repo / "docs" / "audits" / "2026-07-12-technical-ok.md"
    good.write_text("# ok\n", encoding="utf-8")
    _git(repo, "add", str(good))
    _git(repo, "commit", "-q", "-m", "add a good audit")
    _git(repo, "mv", str(good), str(repo / "docs" / "audits" / "2026-07-12-BAD_RENAME.md"))
    res = _run_hook(repo)
    assert res.returncode == 1
    assert "casing" in res.stderr


def test_main_fail_open_on_git_error(monkeypatch, capsys):
    def _boom():
        raise RuntimeError("simulated git failure")

    monkeypatch.setattr(vac, "staged_added_paths", _boom)
    assert vac.main() == 0                          # fail OPEN
    assert "skipped" in capsys.readouterr().err     # but LOUD
