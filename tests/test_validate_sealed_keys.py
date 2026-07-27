"""Tests for scripts/validate_sealed_keys.py (#67 sealed-key staging guard).

Firing tests, not presence. Every bypass found during the 2026-07-19 proof round has a named
regression test here, because the guard code changed three times AFTER its violation-proof
transcripts were written and `check.ps1` gives `scripts/` no other durable coverage.

The transcripts in `docs/audits/2026-07-19-guards-violation-proof.md` are point-in-time; these
are the thing that stops a regression.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_P = Path(__file__).resolve().parent.parent / "scripts" / "validate_sealed_keys.py"


def _load():
    spec = importlib.util.spec_from_file_location("validate_sealed_keys", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_sealed_keys"] = module
    spec.loader.exec_module(module)
    return module


vsk = _load()


# --- MATCHES: both real key name shapes --------------------------------------

def test_matches_cli4_parity_shape():
    # docs/audits/2026-07-18-cli4-parity/SEALED-KEY.json
    assert vsk.is_sealed_key("docs/audits/2026-07-18-cli4-parity/SEALED-KEY.json")


def test_matches_epi1_reversed_shape():
    # THE reason the pattern is wider than #67's literal `SEALED-KEY*.json`: the second real key
    # reverses the suffix order, so the literal pattern would have missed it entirely.
    assert vsk.is_sealed_key("docs/audits/2026-07-17-epi1-archaeology-KEY-SEALED.json")


def test_match_is_case_insensitive_and_order_free():
    assert vsk.is_sealed_key("docs/audits/sealed-key.json")
    assert vsk.is_sealed_key("x/KEY_SEALED.JSON")
    assert vsk.is_sealed_key("a/b/c/my-Sealed-Key-v2.json")


def test_position_in_tree_is_irrelevant():
    # A key is a key wherever it is staged -- the guard keys off the FILENAME, not the location.
    assert vsk.is_sealed_key("SEALED-KEY.json")
    assert vsk.is_sealed_key("src/ai_council/SEALED-KEY.json")


# --- IGNORES: must not fire on the real tracked tree -------------------------

def test_ignores_tracked_json_files():
    # Proven zero-match repo-wide before arming; a guard that fires on a tracked file would
    # block every future commit.
    assert not vsk.is_sealed_key(".claude/settings.json")
    assert not vsk.is_sealed_key(".vscode/settings.json")


def test_ignores_non_json_and_near_misses():
    assert not vsk.is_sealed_key("docs/audits/2026-07-18-cli4-parity-corpus.md")
    assert not vsk.is_sealed_key("config/settings.yaml")
    assert not vsk.is_sealed_key("docs/audits/2026-07-17-epi1-archaeology-SECOND-OPINION-judge.md")
    assert not vsk.is_sealed_key("keys.json")      # 'key' but no 'sealed'
    assert not vsk.is_sealed_key("sealed.json")    # 'sealed' but no 'key'


# --- override: exact-path only, never blanket --------------------------------

def test_override_requires_the_exact_path():
    p = "docs/audits/x/SEALED-KEY.json"
    blocked, overridden = vsk.check([p], {p})
    assert blocked == [] and overridden == [p]


def test_override_naming_a_different_path_still_blocks():
    # The constraint that makes the override safe: it cannot blanket-disarm.
    p = "docs/audits/x/SEALED-KEY.json"
    blocked, overridden = vsk.check([p], {"some/other/key-SEALED.json"})
    assert blocked == [p] and overridden == []


def test_bare_truthy_env_authorizes_nothing(monkeypatch):
    monkeypatch.setenv(vsk._ALLOW_ENV, "1")
    # "1" is a path-shaped token that matches no real path -> nothing is authorized.
    assert vsk.allowed_paths() == {"1"}
    blocked, overridden = vsk.check(["docs/a/SEALED-KEY.json"], vsk.allowed_paths())
    assert blocked == ["docs/a/SEALED-KEY.json"] and overridden == []


def test_override_parses_semicolon_separated_paths(monkeypatch):
    monkeypatch.setenv(vsk._ALLOW_ENV, "a/SEALED-KEY.json; b/KEY-SEALED.json ")
    assert vsk.allowed_paths() == {"a/SEALED-KEY.json", "b/KEY-SEALED.json"}


# --- fail CLOSED (opposite of the peer casing gate) --------------------------

def test_main_fails_closed_on_git_error(monkeypatch, capsys):
    def _boom():
        raise RuntimeError("simulated git failure")

    monkeypatch.setattr(vsk, "staged_added_paths", _boom)
    assert vsk.main() == 1                                  # fail CLOSED, not open
    err = capsys.readouterr().err
    assert "GUARD MALFUNCTION" in err
    assert "NOT a policy violation" in err                  # distinguishable at a glance


# --- end-to-end against a REAL temp git repo ---------------------------------

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
    (repo / "docs").mkdir()
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _run_hook(repo: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    # errors="replace": on Windows the child writes stderr in the console codepage (cp1252), so
    # a non-ASCII path in a rejection message is not valid UTF-8. That is a harness decoding
    # concern only -- the guard's exit code, which is what gates the commit, is unaffected.
    return subprocess.run([sys.executable, str(_P)], cwd=str(repo), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


def test_e2e_blocks_staged_sealed_key(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "SEALED-KEY.json").write_text('{"k":1}\n', encoding="utf-8")
    _git(repo, "add", "SEALED-KEY.json")
    res = _run_hook(repo)
    assert res.returncode == 1
    assert "SEALED-KEY.json" in res.stderr


def test_e2e_clean_tree_passes(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "docs" / "note.md").write_text("# ok\n", encoding="utf-8")
    _git(repo, "add", "docs/note.md")
    assert _run_hook(repo).returncode == 0


def test_e2e_unicode_path_does_not_bypass(tmp_path):
    """REGRESSION -- this was a real bypass: a sealed key was ADMITTED (exit 0).

    `core.quotePath` is on by default, so `git diff --name-only` (no `-z`) C-quotes a non-ASCII
    path as `"docs/\\303\\251vasion/SEALED-KEY.json"`. The TRAILING double quote defeats the
    `\\.json$` anchor and the key sailed through. Fixed by reading NUL-delimited (`-z`).
    """
    repo = _init_repo(tmp_path)
    d = repo / "docs" / "évasion"
    d.mkdir(parents=True)
    (d / "SEALED-KEY.json").write_text('{"k":1}\n', encoding="utf-8")
    _git(repo, "add", "-A")
    res = _run_hook(repo)
    assert res.returncode == 1, f"unicode sealed key BYPASSED the guard: {res.stdout}{res.stderr}"


def test_e2e_git_mv_into_tracked_path_is_caught(tmp_path):
    # --no-renames forces a rename to surface as delete+ADD, so a `git mv` that drags a key into
    # a tracked location is policed. This is the shape of the 2026-07-18 near-miss.
    repo = _init_repo(tmp_path)
    src = repo / "docs" / "SEALED-KEY.json"
    src.write_text('{"k":1}\n', encoding="utf-8")
    _git(repo, "add", "-f", "docs/SEALED-KEY.json")
    _git(repo, "commit", "-q", "-m", "seed a key")
    _git(repo, "mv", "docs/SEALED-KEY.json", "docs/moved-KEY-SEALED.json")
    res = _run_hook(repo)
    assert res.returncode == 1
    assert "moved-KEY-SEALED.json" in res.stderr


def test_e2e_override_emits_audit_banner(tmp_path):
    """The override's ONLY audit trail is this banner -- an env var is invisible in git log."""
    import os
    repo = _init_repo(tmp_path)
    (repo / "SEALED-KEY.json").write_text('{"k":1}\n', encoding="utf-8")
    _git(repo, "add", "SEALED-KEY.json")
    env = dict(os.environ, AICOUNCIL_SEALED_KEY_ALLOW="SEALED-KEY.json")
    res = _run_hook(repo, env=env)
    assert res.returncode == 0
    assert "DELIBERATELY BYPASSED" in res.stderr


# --- #126 output contract: success is a positive assertion --------------------

def test_e2e_success_prints_positive_assertion(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "docs" / "note.md").write_text("# ok\n", encoding="utf-8")
    _git(repo, "add", "docs/note.md")
    res = _run_hook(repo)
    assert res.returncode == 0, res.stderr
    assert "validate_sealed_keys: OK" in res.stdout
    assert "1 staged add(s) checked" in res.stdout
    assert "sealed-key" in res.stdout            # the predicate clause names what was scanned for


def test_e2e_zero_item_run_is_distinguishable(tmp_path):
    # Nothing staged: the gate must SAY it checked zero items, not stay silent.
    repo = _init_repo(tmp_path)
    res = _run_hook(repo)
    assert res.returncode == 0, res.stderr
    assert "validate_sealed_keys: OK" in res.stdout
    assert "0 staged add(s) checked" in res.stdout


def test_e2e_override_run_reports_authorized_count(tmp_path):
    # An authorized key exits 0: the OK line must carry the override count so the success
    # assertion and the stderr banner tell the same story.
    import os
    repo = _init_repo(tmp_path)
    (repo / "SEALED-KEY.json").write_text('{"k":1}\n', encoding="utf-8")
    _git(repo, "add", "SEALED-KEY.json")
    env = dict(os.environ, AICOUNCIL_SEALED_KEY_ALLOW="SEALED-KEY.json")
    res = _run_hook(repo, env=env)
    assert res.returncode == 0
    assert "validate_sealed_keys: OK" in res.stdout
    assert "1 sealed key(s) explicitly authorized" in res.stdout
    assert "NO record in git log" in res.stderr
