"""Tests for scripts/canonical_freshness_gate.py (#126 output contract + gate behavior).

The #126 lane: a gating validator's success must be a POSITIVE assertion — name, verdict,
the predicate it evaluated, and how many items it checked — never exit-0 silence. A run
that examined zero items must be distinguishable from a clean run. The A2 FAIL path
(exit 1) is pinned unchanged.
"""

import importlib.util
import subprocess
import sys
from datetime import date
from pathlib import Path

_P = Path(__file__).resolve().parent.parent / "scripts" / "canonical_freshness_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("canonical_freshness_gate", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules["canonical_freshness_gate"] = module
    spec.loader.exec_module(module)
    return module


cfg = _load()


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
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _run_hook(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_P)], cwd=str(repo),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def _canonical_doc(repo: Path, reviewed: str) -> None:
    (repo / "CLAUDE.md").write_text(
        f"---\nlast_reviewed: {reviewed}\n---\n# doc\n", encoding="utf-8")
    _git(repo, "add", "CLAUDE.md")
    _git(repo, "commit", "-q", "-m", "add canonical doc")


# --- #126 output contract: success is a positive assertion --------------------

def test_success_prints_positive_assertion(tmp_path):
    repo = _init_repo(tmp_path)
    _canonical_doc(repo, date.today().isoformat())
    res = _run_hook(repo)
    assert res.returncode == 0, res.stderr
    assert "canonical_freshness: OK" in res.stdout
    assert "1 canonical doc(s) checked" in res.stdout
    assert "last_reviewed" in res.stdout  # the predicate clause names what was evaluated


def test_zero_item_run_is_distinguishable(tmp_path):
    # No canonical doc exists at all: the gate must SAY it checked nothing, not stay silent.
    repo = _init_repo(tmp_path)
    res = _run_hook(repo)
    assert res.returncode == 0, res.stderr
    assert "canonical_freshness: OK" in res.stdout
    assert "0 canonical doc(s) checked" in res.stdout


def test_warn_only_run_still_reports_ok_with_warn_count(tmp_path):
    # A stale-but-uncommitted-history doc (A2 unavailable) past cadence -> WARN, exit 0.
    repo = _init_repo(tmp_path)
    (repo / "VISION.md").write_text(
        "---\nlast_reviewed: 2020-01-01\n---\n# doc\n", encoding="utf-8")
    # deliberately NOT committed: git date is None, so only the A1 calendar warn fires
    res = _run_hook(repo)
    assert res.returncode == 0, res.stderr
    assert "WARN" in res.stdout
    assert "canonical_freshness: OK" in res.stdout
    assert "1 warning(s)" in res.stdout


# --- existing behavior pinned unchanged ---------------------------------------

def test_a2_stale_doc_still_fails_exit_1(tmp_path):
    # last_reviewed predates the doc's last commit (authored today) -> A2 FAIL, exit 1,
    # and no OK line is printed on the failure path.
    repo = _init_repo(tmp_path)
    _canonical_doc(repo, "2020-01-01")
    res = _run_hook(repo)
    assert res.returncode == 1
    assert "FAIL" in res.stdout
    assert "canonical_freshness: OK" not in res.stdout
