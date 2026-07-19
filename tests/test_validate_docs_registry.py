"""Tests for scripts/validate_docs_registry.py (#68 docs/ directory registry check).

Firing tests, not presence. Every bypass and lifecycle bug found during the 2026-07-19 proof
round has a named regression test, because the guard code changed three times AFTER its
violation-proof transcripts were written and `check.ps1` gives `scripts/` no other coverage.

Provenance of the named regressions:
  * sol (adversarial): unicode C-quoting, the `docs/` taxonomy token, index-vs-working-tree,
    gitlink/symlink with no child component, arbitrary-depth `archive/` laundering.
  * terra (pre-merge): a registered corpus gaining internal structure, and a structurally
    valid EMPTY registry table read as a malfunction.
  * self-found: the guard blocking the corpus-exit move that #27 itself must perform.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_P = Path(__file__).resolve().parent.parent / "scripts" / "validate_docs_registry.py"


def _load():
    spec = importlib.util.spec_from_file_location("validate_docs_registry", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_docs_registry"] = module
    spec.loader.exec_module(module)
    return module


vdr = _load()


# A minimal registry with the same SHAPE as the real docs/audits/README.md.
REGISTRY_MD = """# Audits

## Directory invariant

| # | Class | Rule |
|---|---|---|
| a | **Date-slug markdown records** | `YYYY-MM-DD-<topic>.md` |
| b | **`archive/`** | governed by its own `archive/README.md` |
| c | **A registered live corpus** | permitted only while live |

## Live corpora

| Path | What it is | Ruling | Essence markdown | Exit condition |
|---|---|---|---|---|
| `2026-07-17-epi1-archaeology/` | pack | ruling | `essence.md` | settled |
| `2026-07-18-cli4-parity/` | trial | ruling | The report written at unseal | unseal |

## Convention
"""


def _reg():
    return vdr.parse_registry(REGISTRY_MD)


# --- registry parsing --------------------------------------------------------

def test_parses_registered_corpora_and_taxonomy():
    registered, taxonomy = _reg()
    assert registered == {"2026-07-17-epi1-archaeology", "2026-07-18-cli4-parity"}
    assert taxonomy == {"archive"}


def test_prose_essence_cell_does_not_break_parsing():
    # The real cli4-parity row's essence column is PROSE, not a path -- a parser that assumed a
    # filename there would fail on the live registry.
    registered, _ = _reg()
    assert "2026-07-18-cli4-parity" in registered


# --- admissible vs blocked ---------------------------------------------------

def _v(d):
    registered, taxonomy = _reg()
    return vdr.violation(d, registered, taxonomy)


def test_blocks_unregistered_top_level_dir():
    # The real docs/smoke/ incident.
    assert _v("docs/smoke") is not None


def test_blocks_unregistered_corpus_in_audits():
    r = _v("docs/audits/2026-07-20-unregistered")
    assert r is not None and "no row in the 'Live corpora' table" in r


def test_blocks_unregistered_dir_under_another_section():
    assert _v("docs/decisions/sneaky") is not None


def test_allows_registered_live_corpora():
    assert _v("docs/audits/2026-07-17-epi1-archaeology") is None
    assert _v("docs/audits/2026-07-18-cli4-parity") is None


def test_allows_archive_taxonomy_children():
    assert _v("docs/audits/archive") is None
    assert _v("docs/decisions/archive") is None


def test_not_a_blanket_ban():
    # The whole point: a blanket new-directory ban would reject archive/ and both live corpora.
    for d in ("docs/audits/archive", "docs/audits/2026-07-17-epi1-archaeology",
              "docs/audits/2026-07-18-cli4-parity"):
        assert _v(d) is None


# --- named regressions -------------------------------------------------------

def test_regression_does_not_block_the_corpus_exit_move():
    """SELF-FOUND: the guard rejected the very operation #27 must perform at unseal.

    Moving a corpus to `docs/audits/archive/<corpus>/` was blocked because its parent is
    `docs/audits/archive` and its own name is not `archive`.
    """
    assert _v("docs/audits/archive/2026-07-18-cli4-parity") is None


def test_regression_registered_corpus_owns_its_internal_structure():
    """TERRA: a registered corpus gaining a new subdirectory was blocked.

    cli4-parity already carries a `blinded/` child, which passes today only by grandfathering.
    """
    assert _v("docs/audits/2026-07-18-cli4-parity/new-section") is None
    assert _v("docs/audits/2026-07-18-cli4-parity/blinded/deeper") is None


def test_regression_empty_but_valid_registry_is_not_a_malfunction():
    """TERRA, the sharpest one: zero live corpora is a NORMAL lifecycle state.

    #27's unseal is precisely the event that empties this table. Treating a structurally valid
    empty table as a malfunction would have bricked EVERY commit at that moment.
    """
    empty = REGISTRY_MD.replace(
        "| `2026-07-17-epi1-archaeology/` | pack | ruling | `essence.md` | settled |\n", ""
    ).replace(
        "| `2026-07-18-cli4-parity/` | trial | ruling | The report written at unseal | unseal |\n", ""
    )
    registered, taxonomy = vdr.parse_registry(empty)     # must NOT raise
    assert registered == set()
    assert taxonomy == {"archive"}
    # ...and with an empty registry the guard still blocks an unregistered corpus.
    assert vdr.violation("docs/audits/2026-07-20-new", registered, taxonomy) is not None


def test_regression_docs_token_cannot_become_a_taxonomy_name():
    """SOL: a backticked `docs/` token anywhere in the invariant table became a global allow.

    It was admitted as a taxonomy name and then matched as an ancestor segment at ANY depth,
    so `docs/anything/` passed.
    """
    poisoned = REGISTRY_MD.replace(
        "| c | **A registered live corpus** | permitted only while live |",
        "| d | note | `docs/` |\n| c | **A registered live corpus** | permitted only while live |",
    )
    registered, taxonomy = vdr.parse_registry(poisoned)
    assert "docs" not in taxonomy
    assert vdr.violation("docs/completely-unregistered", registered, taxonomy) is not None


def test_regression_multi_segment_token_is_not_a_taxonomy_name():
    # Hardening behind the same finding: taxonomy tokens must be single-segment.
    poisoned = REGISTRY_MD.replace(
        "| b | **`archive/`** | governed by its own `archive/README.md` |",
        "| b | **`archive/`** | governed by its own `archive/README.md` |\n"
        "| d | note | `docs/audits/` |",
    )
    _, taxonomy = vdr.parse_registry(poisoned)
    assert taxonomy == {"archive"}


def test_regression_archive_cannot_launder_at_arbitrary_depth():
    """SOL: `archive` matched at ANY depth, so docs/a/b/archive/rogue/ laundered anything."""
    assert _v("docs/a/b/archive") is not None
    assert _v("docs/a/b/archive/rogue") is not None


def test_regression_gitlink_and_symlink_are_treated_as_directories():
    """SOL: `git submodule add <url> docs/rogue` stages ONE path with no child component.

    Treated as a file it yielded no directory prefix, so a fully populated directory entered
    unregistered.
    """
    tracked = {"docs", "docs/audits"}
    assert vdr.new_dirs([(vdr._GITLINK_MODE, "docs/rogue")], tracked) == ["docs/rogue"]
    assert vdr.new_dirs([(vdr._SYMLINK_MODE, "docs/rogue")], tracked) == ["docs/rogue"]
    # A regular file at the same path introduces no directory.
    assert vdr.new_dirs([("100644", "docs/rogue")], tracked) == []


def test_new_dirs_grandfathers_tracked_dirs():
    tracked = {"docs", "docs/audits", "docs/audits/archive"}
    assert vdr.new_dirs([("100644", "docs/audits/archive/legacy/x.md")], tracked) == [
        "docs/audits/archive/legacy"]
    assert vdr.new_dirs([("100644", "docs/audits/note.md")], tracked) == []


# --- fail CLOSED, distinguishably --------------------------------------------

def test_malfunction_on_missing_live_corpora_section():
    bad = REGISTRY_MD.replace("## Live corpora", "## Corpora currently live")
    try:
        vdr.parse_registry(bad)
        raise AssertionError("expected RegistryError")
    except vdr.RegistryError as exc:
        assert "Live corpora" in str(exc)


def test_malfunction_on_missing_invariant_section():
    bad = REGISTRY_MD.replace("## Directory invariant", "## Rules")
    try:
        vdr.parse_registry(bad)
        raise AssertionError("expected RegistryError")
    except vdr.RegistryError as exc:
        assert "Directory invariant" in str(exc)


def test_malfunction_on_reformatted_path_column():
    bad = REGISTRY_MD.replace("| `2026-07-17-epi1-archaeology/` |", "| 2026-07-17-epi1-archaeology/ |") \
                     .replace("| `2026-07-18-cli4-parity/` |", "| 2026-07-18-cli4-parity/ |")
    try:
        vdr.parse_registry(bad)
        raise AssertionError("expected RegistryError")
    except vdr.RegistryError as exc:
        assert "parseable path" in str(exc)


def test_malfunction_on_renamed_path_header():
    bad = REGISTRY_MD.replace("| Path | What it is |", "| Folder | What it is |")
    try:
        vdr.parse_registry(bad)
        raise AssertionError("expected RegistryError")
    except vdr.RegistryError as exc:
        assert "Path" in str(exc)


def test_main_reports_malfunction_distinguishably(monkeypatch, capsys):
    def _boom():
        raise vdr.RegistryError("simulated registry failure")

    monkeypatch.setattr(vdr, "load_registry", _boom)
    assert vdr.main() == 1                                  # fail CLOSED
    err = capsys.readouterr().err
    assert "GUARD MALFUNCTION" in err
    assert "NOT a policy violation" in err
    assert "Nothing is wrong with what you staged" in err


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
    audits = repo / "docs" / "audits"
    audits.mkdir(parents=True)
    (audits / "README.md").write_text(REGISTRY_MD, encoding="utf-8")
    (audits / "archive").mkdir()
    (audits / "archive" / ".keep").write_text("", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed registry")
    return repo


def _run_hook(repo: Path) -> subprocess.CompletedProcess:
    # errors="replace": see the note in test_validate_sealed_keys.py -- Windows console codepage.
    return subprocess.run([sys.executable, str(_P)], cwd=str(repo), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def test_e2e_clean_tree_passes(tmp_path):
    repo = _init_repo(tmp_path)
    assert _run_hook(repo).returncode == 0


def test_e2e_blocks_unregistered_dir(tmp_path):
    repo = _init_repo(tmp_path)
    d = repo / "docs" / "smoke"
    d.mkdir()
    (d / "report.md").write_text("# x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    res = _run_hook(repo)
    assert res.returncode == 1
    assert "docs/smoke" in res.stderr


def test_e2e_allows_registered_corpus(tmp_path):
    repo = _init_repo(tmp_path)
    d = repo / "docs" / "audits" / "2026-07-18-cli4-parity"
    d.mkdir()
    (d / "note.md").write_text("# x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _run_hook(repo).returncode == 0


def test_e2e_regression_unicode_dir_does_not_bypass(tmp_path):
    """SOL REGRESSION -- this was a full bypass: the guard exited 0.

    `core.quotePath` C-quotes a non-ASCII path, and the leading `"` made `parts[0] != "docs"`,
    so every target was skipped. Fixed by reading `--raw -z`.
    """
    repo = _init_repo(tmp_path)
    d = repo / "docs" / "audits" / "évasion"
    d.mkdir()
    (d / "f.md").write_text("# x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    res = _run_hook(repo)
    assert res.returncode == 1, f"unicode dir BYPASSED the guard: {res.stdout}{res.stderr}"


def test_e2e_regression_registry_read_from_index_not_worktree(tmp_path):
    """SOL REGRESSION: `git rm --cached` staged the registry's deletion while leaving an
    untracked copy on disk. The guard parsed that copy and passed, so a commit could delete
    the registry and add an unregistered corpus at the same time."""
    repo = _init_repo(tmp_path)
    _git(repo, "rm", "--cached", "-q", "docs/audits/README.md")
    assert (repo / "docs" / "audits" / "README.md").exists()   # still on disk
    res = _run_hook(repo)
    assert res.returncode == 1
    assert "GUARD MALFUNCTION" in res.stderr


def test_e2e_allows_corpus_exit_to_archive(tmp_path):
    """The operation #27 must perform at unseal must not be blocked."""
    repo = _init_repo(tmp_path)
    src = repo / "docs" / "audits" / "2026-07-18-cli4-parity"
    src.mkdir()
    (src / "report.md").write_text("# x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add live corpus")
    dst = repo / "docs" / "audits" / "archive" / "2026-07-18-cli4-parity"
    _git(repo, "mv", str(src), str(dst))
    res = _run_hook(repo)
    assert res.returncode == 0, f"blocked the sanctioned exit move: {res.stderr}"
