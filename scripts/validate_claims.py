#!/usr/bin/env python
"""validate_claims.py -- #97 claim-vs-reality checker (Unit 1: skeleton + rules 2/3/4/8).

Read-only. Reads canonical docs, compares each self-contained, deterministically-checkable
CLAIM to repo ground truth (filesystem / YAML / git), and reports PASS/FAIL per rule with a
re-runnable evidence command per finding (rule 12). Surfaced as a NON-BLOCKING check.ps1
section; NOT a pre-commit gate at first (#97/#73).

Design (Unit 1 freezes these; Unit 2 legs + any Codex leg plug in without rework):
  * Finding      -- one violation. `evidence` is an argv tuple whose printed form a HUMAN can
                    paste into git-bash OR PowerShell; the harness parses it back and RUNS it,
                    so "re-runnable" is verified, not promised (rule 12).
  * RuleResult   -- one rule's verdict: pass | fail | anchor-missing | skipped. anchor-missing
                    (a reworded doc the leg cannot locate) is a WARN, never a synthesized
                    finding -- the precision lever borrowed from the hub #89 checker.
  * check(ctx)   -- the FROZEN rule-leg signature: (RepoContext) -> RuleResult.
  * exit codes   -- 0 clean; 1 findings (swallowed at the call site); >=2 checker error, which
                    the call site surfaces LOUD (Red) but still does not fail the gate. A crash
                    must never masquerade as a pass -- the defect class this checker ends.

Layer-2 read-only contract: reads docs + .pre-commit-config.yaml + read-only git queries
(rev-parse / cat-file -e / merge-base --is-ancestor / ls-tree); writes NOTHING; never gates.

Authority: BACKLOG #97 / [S18]; docs/audits/2026-07-22-pre-handoff-cleanup.md Section 3 (rules
1-12 + the 2026-07-23 rule-5 amendment); #97 (rules 13-14). Fleet sibling (not imported):
../.dev-knowledge/scripts/validate_doc_claims.py (#89).
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent

# arg0 stays a BARE runner (never sys.executable's absolute path): the printed command must
# paste cleanly into both git-bash and PowerShell without shell-specific quoting. See the
# report header caveat -- on Windows a bare `python` may resolve to a Store stub or a non-venv
# interpreter; that is documented, not engineered around.
_ALLOWED_RUNNERS = frozenset({"python", "git", "pytest"})

# Cross-shell contract (A3): evidence is emitted with shlex/POSIX quoting -- natively valid in
# git-bash. arg0 is a BARE runner (never sys.executable's absolute path), so it resolves on PATH
# in PowerShell too; the common `python -c '<script>'` / `git <args>` shapes carry POSIX
# single-quoting that PowerShell also reads as a literal string. We do NOT constrain argument
# tokens to be quote-free -- that would reject legitimate `-c` scripts. The enforced guarantees
# are: arg0 is a bare runner, and the printed string round-trips through shlex back to the argv.

_REPRODUCES = frozenset({"exit-0", "exit-nonzero", "stdout-contains"})
_STATUSES = frozenset({"pass", "fail", "anchor-missing", "skipped"})

# The #97 fourteen-rule spec, enumerated literally rather than as range(1, 15) -- in the one
# file whose job is catching comment-vs-code mismatch, the ids should be readable as ids.
# `format_report` uses this as the DENOMINATOR so no run can understate the checker's own
# coverage: `absent` is computed against it, not hardcoded, and goes non-zero the moment a
# leg is dropped. Maintained BY HAND against BACKLOG #97 -- see the KNOWN LIMITATIONS note
# on that blind spot, and #114 for the doc-side check that would close it.
_SPEC_RULE_IDS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14})

# Rule 12 is the ONE sanctioned exemption from registration: realized structurally in the
# Finding model (evidence argv + shlex round-trip), so every finding carries a re-runnable
# command by construction and there is no leg to register. BACKLOG #97 gate (i).
_STRUCTURAL_EXEMPTIONS = frozenset({12})


@dataclass(frozen=True)
class Finding:
    """One rule violation with a re-runnable (by a human) evidence command."""

    rule_id: int
    claim: str                    # the false claim, human-readable
    location: str                 # where the claim lives, e.g. "CLAUDE.md:159"
    reality: str                  # what the checker found instead
    evidence: tuple[str, ...]     # canonical argv; arg0 a BARE runner in _ALLOWED_RUNNERS
    reproduces: str = "stdout-contains"   # how evidence reproduces reality (see _REPRODUCES)

    def __post_init__(self) -> None:
        if len(self.evidence) < 2:
            raise ValueError(
                "Finding.evidence must be a runner PLUS at least one argument (rule 12): a bare "
                f"runner like {self.evidence!r} is not a runnable command")
        arg0 = self.evidence[0]
        if arg0 not in _ALLOWED_RUNNERS:
            raise ValueError(
                f"Finding.evidence[0]={arg0!r} must be a bare runner in "
                f"{sorted(_ALLOWED_RUNNERS)} -- not an absolute interpreter path, so the "
                f"printed command stays shell-portable")
        if self.reproduces not in _REPRODUCES:
            raise ValueError(f"Finding.reproduces={self.reproduces!r} not in {sorted(_REPRODUCES)}")
        # Round-trip invariant: the printed string parses back to the exact argv, i.e. it is
        # genuinely what a human would type. Guaranteed by the portability constraint above,
        # asserted here so a future relaxation of that constraint fails loudly.
        if tuple(shlex.split(self.printed())) != self.evidence:
            raise ValueError("Finding.evidence does not round-trip through shlex")

    def printed(self) -> str:
        """The human-copyable command string (POSIX/shlex quoting)."""
        return shlex.join(self.evidence)


@dataclass(frozen=True)
class RuleResult:
    """One rule's verdict. `findings` non-empty => status 'fail'."""

    rule_id: int
    rule_name: str
    findings: tuple[Finding, ...] = ()
    status: str = "pass"          # pass | fail | anchor-missing | skipped
    detail: str = ""              # reason for anchor-missing / skipped

    def __post_init__(self) -> None:
        if self.findings and self.status == "pass":
            object.__setattr__(self, "status", "fail")
        if self.status not in _STATUSES:
            raise ValueError(f"RuleResult.status={self.status!r} not in {sorted(_STATUSES)}")
        if self.findings and self.status != "fail":
            raise ValueError("RuleResult has findings but status is not 'fail'")
        if self.status == "fail" and not self.findings:
            raise ValueError("RuleResult status 'fail' but carries no findings")


RuleLeg = Callable[["RepoContext"], RuleResult]


class RepoContext:
    """Read-only access to the repo under `root`. Treat as immutable. Shared by every leg so
    no leg re-implements file/glob/yaml/git access."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._text: dict[str, str] = {}
        self._yaml: dict[str, Any] = {}

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def read(self, rel: str) -> str:
        if rel not in self._text:
            self._text[rel] = (self.root / rel).read_text(encoding="utf-8")
        return self._text[rel]

    def glob(self, pat: str) -> list[Path]:
        return sorted(self.root.glob(pat))

    def load_yaml(self, rel: str) -> Any:
        if rel not in self._yaml:
            self._yaml[rel] = yaml.safe_load(self.read(rel))
        return self._yaml[rel]

    def git(self, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        """A read-only git query, `git -C root <args>`. `input_text` is fed on stdin (used to
        batch object lookups through a single `cat-file --batch-check`)."""
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            input=input_text, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )


# --- rule legs (Unit 1: 2/3/4/8 -- stubs until each is implemented RED-first) ---------------

# Canonical docs R2 scans for backtick-quoted repo paths. Append-only logs (JOURNAL/BACKLOG/
# LESSONS) are deliberately excluded -- they record historical paths that no longer resolve.
_CANONICAL_DOCS = ("CLAUDE.md", "ARCHITECTURE.md", "VISION.md", "CONTRIBUTING.md")

_BACKTICK = re.compile(r"`([^`]+)`")
_PATHISH = re.compile(r"^[\w./@-]+/?$")          # path-shaped, single token, no spaces
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_HISTORICAL_HEADING = re.compile(r"section history", re.IGNORECASE)
# Template/placeholder segments -- an illustration of a naming convention, not a real path.
_PLACEHOLDER = re.compile(r"NN|YYYY|MMDD|HHMMSS|<[^>]*>")
# R2 allowlist: subtrees documented to live OUTSIDE this repo tree. Kept minimal; each carries
# its reason inline. A path under one of these is a routing reference, not a local-existence claim.
_R2_ALLOWLIST = (
    ".dev-knowledge/",              # hub-qualified sibling repo
    "docs/handoffs/",               # ADR-42: handoffs centralized in the hub, none local
    "docs/decisions/transcripts/",  # ADR-43: Council transcripts routed to the hub
)


def _canonical_docs(ctx: RepoContext) -> list[str]:
    docs = [d for d in _CANONICAL_DOCS if ctx.exists(d)]
    docs += [str(p.relative_to(ctx.root)).replace("\\", "/") for p in ctx.glob("protocols/*.md")]
    return docs


def rule_2(ctx: RepoContext) -> RuleResult:
    """Every backtick-quoted repo path in a canonical doc resolves on disk.

    Allowlist (kept minimal, each with a reason):
      * hub-qualified `.dev-knowledge/...` -- lives in a sibling repo, never on this tree;
      * a `## ... Section history` section -- records superseded paths by design.
    """
    findings: list[Finding] = []
    for rel in _canonical_docs(ctx):
        in_historical = False
        for i, line in enumerate(ctx.read(rel).splitlines(), start=1):
            mh = _HEADING.match(line)
            if mh:
                in_historical = bool(_HISTORICAL_HEADING.search(mh.group(1)))
                continue
            if in_historical:
                continue
            for m in _BACKTICK.finditer(line):
                tok = m.group(1)
                if "/" not in tok or not _PATHISH.match(tok):
                    continue
                if tok.startswith("/"):              # reason: slash-command / absolute, not a repo path
                    continue
                if _PLACEHOLDER.search(tok):         # reason: naming-convention template, not a real path
                    continue
                if any(tok.startswith(p) for p in _R2_ALLOWLIST):
                    continue
                # reason: only a REPO-ROOTED path is a claim about this tree. A token whose first
                # segment is not an existing top-level dir (`Dev/`, `ai-council/output/`) is an
                # ecosystem/illustrative path, not a local-existence claim -- out of scope.
                if not (ctx.root / tok.split("/", 1)[0]).is_dir():
                    continue
                if ctx.exists(tok):
                    continue
                findings.append(Finding(
                    rule_id=2,
                    claim=f"path `{tok}` does not resolve on disk",
                    location=f"{rel}:{i}",
                    reality=f"{tok} is absent from the repo tree",
                    evidence=("python", "-c",
                              f"import pathlib,sys; sys.exit(0 if pathlib.Path({tok!r}).exists() "
                              f"else 1)"),
                    reproduces="exit-nonzero",
                ))
    if findings:
        return RuleResult(2, "path-existence", findings=tuple(findings))
    return RuleResult(2, "path-existence", status="pass")


_PRECOMMIT_CFG = ".pre-commit-config.yaml"
_VALIDATOR_SECTION = re.compile(r"(validators|hooks active)", re.IGNORECASE)


def _hook_roster_docs(ctx: RepoContext) -> list[str]:
    return [d for d in ("CLAUDE.md", "ARCHITECTURE.md") if ctx.exists(d)]


def _extract_config_hook_ids(cfg_text: str) -> list[str]:
    data = yaml.safe_load(cfg_text) or {}
    return [h["id"] for repo in data.get("repos", []) for h in repo.get("hooks", []) if "id" in h]


def _section_body(text: str, section_re: re.Pattern[str]) -> tuple[list[str] | None, int | None]:
    """The body lines of the section whose heading matches `section_re`, and its heading line,
    or (None, None) when no such section is anchored (-> anchor-missing, never a false roster)."""
    lines = text.splitlines()
    in_section = False
    heading_line: int | None = None
    body: list[str] = []
    for i, line in enumerate(lines, start=1):
        mh = _HEADING.match(line)
        if mh:
            if section_re.search(mh.group(1)):
                in_section, heading_line = True, i
            elif in_section:
                break
            continue
        if in_section:
            body.append(line)
    if heading_line is None:
        return None, None
    return body, heading_line


def _id_mentioned(hook_id: str, text: str) -> bool:
    """Whether a hook id appears as a whole token (hyphens are part of the token, so `ruff`
    does not match inside `truffle` and `block-ff-push` matches as a unit)."""
    return re.search(r"(?<![\w-])" + re.escape(hook_id) + r"(?![\w-])", text) is not None


def rule_3(ctx: RepoContext) -> RuleResult:
    """Every hook id in .pre-commit-config.yaml is DOCUMENTED in a doc that carries a
    Validators/Hooks-active roster.

    Precision-over-recall: a hook is "documented" if its id appears as a whole token anywhere in
    the doc -- so a multi-id bullet (`toc-freshness` / `toc-generate`) or a hook named in prose
    is never falsely reported missing (structural leading-bullet parsing false-positived `ruff`
    and `toc-generate` on the live CLAUDE.md). Detection and the emitted evidence use the SAME
    whole-doc word-boundary test, so the evidence always reproduces the finding (H4). The section
    anchor is used only to require that a roster exists (else anchor-missing) and to locate it.
    """
    if not ctx.exists(_PRECOMMIT_CFG):
        return RuleResult(3, "hook-roster-parity", status="anchor-missing",
                          detail=f"no {_PRECOMMIT_CFG}")
    config_ids = set(_extract_config_hook_ids(ctx.read(_PRECOMMIT_CFG)))
    findings: list[Finding] = []
    anchored = False
    for rel in _hook_roster_docs(ctx):
        _body, hl = _section_body(ctx.read(rel), _VALIDATOR_SECTION)
        if _body is None:
            continue
        anchored = True
        doc_text = ctx.read(rel)                 # whole doc -- matches the evidence scope (H4)
        config_only = sorted(i for i in config_ids if not _id_mentioned(i, doc_text))
        if config_only:
            findings.append(Finding(
                rule_id=3,
                claim=f"{rel} does not document every hook in {_PRECOMMIT_CFG}",
                location=f"{rel}:{hl}",
                reality=f"declared in config but not documented: {config_only}",
                evidence=("python", "-c",
                          f"import re,pathlib; "
                          f"ids=set(re.findall(r'id:\\s*([\\w.-]+)', "
                          f"pathlib.Path({_PRECOMMIT_CFG!r}).read_text(encoding='utf-8'))); "
                          f"doc=pathlib.Path({rel!r}).read_text(encoding='utf-8'); "
                          f"print('config-not-documented:', sorted(i for i in ids if not "
                          f"re.search(r'(?<![\\w-])'+re.escape(i)+r'(?![\\w-])', doc)))"),
                reproduces="stdout-contains",
            ))
    if not anchored:
        return RuleResult(3, "hook-roster-parity", status="anchor-missing",
                          detail="no Validators/Hooks-active roster located")
    if findings:
        return RuleResult(3, "hook-roster-parity", findings=tuple(findings))
    return RuleResult(3, "hook-roster-parity", status="pass")


_ADR_SECTION = re.compile(r"recent adrs|adrs? binding", re.IGNORECASE)
_ADR_ID = re.compile(r"(ADR-\d+)")


def _adr_roster_docs(ctx: RepoContext) -> list[str]:
    return [d for d in ("CLAUDE.md",) if ctx.exists(d)]


def rule_4(ctx: RepoContext) -> RuleResult:
    """Every local ADR file (docs/decisions/ADR-*.md) is NAMED in the doc's ADR roster.

    One direction only (disk -> roster), by whole-token mention: an ADR file that exists but is
    unrostered is real drift (ADR-13 was). The reverse (rostered id absent on disk) is NOT
    checked -- the roster legitimately also names ecosystem ADRs that live in the sibling hub,
    so flagging them would be a false positive.
    """
    disk_ids = sorted({m.group(1) for p in ctx.glob("docs/decisions/ADR-*.md")
                       if (m := _ADR_ID.match(p.name))})
    if not disk_ids:
        return RuleResult(4, "adr-roster-parity", status="anchor-missing",
                          detail="no docs/decisions/ADR-*.md files")
    findings: list[Finding] = []
    anchored = False
    for rel in _adr_roster_docs(ctx):
        _body, hl = _section_body(ctx.read(rel), _ADR_SECTION)
        if _body is None:
            continue
        anchored = True
        doc_text = ctx.read(rel)                 # whole doc -- matches the evidence scope (H4)
        missing = [a for a in disk_ids if not _id_mentioned(a, doc_text)]
        if missing:
            findings.append(Finding(
                rule_id=4,
                claim=f"ADR roster in {rel} omits an ADR that exists on disk",
                location=f"{rel}:{hl}",
                reality=f"on disk but not documented: {missing}",
                evidence=("python", "-c",
                          f"import re,pathlib; "
                          f"disk=sorted(re.match(r'(ADR-\\d+)', p.name).group(1) "
                          f"for p in pathlib.Path('docs/decisions').glob('ADR-*.md')); "
                          f"doc=pathlib.Path({rel!r}).read_text(encoding='utf-8'); "
                          f"print('on-disk-not-documented:', [a for a in disk if not "
                          f"re.search(r'(?<![\\w-])'+re.escape(a)+r'(?![\\w-])', doc)])"),
                reproduces="stdout-contains",
            ))
    if not anchored:
        return RuleResult(4, "adr-roster-parity", status="anchor-missing",
                          detail="no ADR roster section located")
    if findings:
        return RuleResult(4, "adr-roster-parity", findings=tuple(findings))
    return RuleResult(4, "adr-roster-parity", status="pass")


# Hex-only, 7-40 chars, bounded by non-alphanumerics. `feedbac` inside `feedback` is not a
# match: the trailing `k` is alphanumeric, so the required right boundary fails (the exact
# false-positive class rule 8 must avoid).
_HEXTOKEN = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")


def _sha_citation_docs(ctx: RepoContext) -> list[str]:
    return [d for d in ("JOURNAL.md", "BACKLOG.md") if ctx.exists(d)]


def _reachable_full_shas(ctx: RepoContext) -> set[str]:
    """Full SHAs of every commit reachable from ANY ref (branches + tags + HEAD), in one call.
    A commit not in this set is unreachable and gc-pruned on a fresh clone (A4). Computed once,
    not per token -- `for-each-ref --contains` per SHA was ~90s on the live JOURNAL.

    Raises on a git failure rather than returning empty: an empty set would make every cited SHA
    look unreachable (or, with cat-file also failing, produce a false clean pass) -- a broken
    query is a CHECKER ERROR (exit >=2), never silent findings or a silent pass (H1)."""
    out = ctx.git("rev-list", "--all")
    if out.returncode != 0:
        raise RuntimeError(f"git rev-list --all failed: {out.stderr.strip() or out.returncode}")
    return set(out.stdout.split())


def rule_8(ctx: RepoContext) -> RuleResult:
    """Every 7+ hex SHA token cited in an append-only log is REACHABLE in git.

    Reachability, not existence: `git rev-parse` resolves unreachable loose objects until gc, so
    an existence check passes a citation that vanishes on a fresh clone. A token is flagged only
    when it (a) resolves to a commit object AND (b) is unreachable from every ref. A hex-shaped
    token that is not a commit object (a coincidental hex word) is skipped -- precision.
    """
    docs = _sha_citation_docs(ctx)
    if not docs:
        return RuleResult(8, "sha-reachability", status="anchor-missing",
                          detail="no JOURNAL/BACKLOG log to scan")
    reachable = _reachable_full_shas(ctx)

    # Every hex token cited anywhere in the logs, resolved to a full commit SHA (or None if not
    # a commit) in ONE cat-file --batch-check call -- one subprocess for all tokens, not one
    # rev-parse each (which was ~45s on the live JOURNAL: Windows process spawn dominates).
    tokens = sorted({m.group(0) for rel in docs
                     for m in _HEXTOKEN.finditer(ctx.read(rel))})
    resolved: dict[str, str | None] = {}
    if tokens:
        query = "".join(f"{t}^{{commit}}\n" for t in tokens)
        out = ctx.git("cat-file", "--batch-check=%(objectname) %(objecttype)", input_text=query)
        if out.returncode != 0:                   # a broken query is a checker error (H1), not "no findings"
            raise RuntimeError(
                f"git cat-file --batch-check failed: {out.stderr.strip() or out.returncode}")
        for tok, line in zip(tokens, out.stdout.splitlines()):
            parts = line.split()
            resolved[tok] = parts[0] if len(parts) >= 2 and parts[1] == "commit" else None

    findings: list[Finding] = []
    for rel in docs:
        for i, line in enumerate(ctx.read(rel).splitlines(), start=1):
            for m in _HEXTOKEN.finditer(line):
                sha = m.group(0)
                full = resolved.get(sha)
                if full is None:            # not a commit object -> a coincidental hex word
                    continue
                if full in reachable:       # reachable from some ref
                    continue
                findings.append(Finding(
                    rule_id=8,
                    claim=f"SHA `{sha}` cited in {rel} is unreachable from any ref (dangling)",
                    location=f"{rel}:{i}",
                    reality=(f"{sha} resolves as a commit object but no branch or tag contains "
                             f"it -- it is gc-pruned on a fresh clone"),
                    # describe --all --contains fails (exit != 0) iff no ref reaches the commit --
                    # faithful to the any-ref claim (merge-base ..HEAD tests only HEAD's line, H5).
                    evidence=("git", "describe", "--all", "--contains", sha),
                    reproduces="exit-nonzero",
                ))
    if findings:
        return RuleResult(8, "sha-reachability", findings=tuple(findings))
    return RuleResult(8, "sha-reachability", status="pass")


def _unit2_stub(rule_id: int, name: str) -> RuleLeg:
    def leg(ctx: RepoContext) -> RuleResult:
        return RuleResult(rule_id, name, status="skipped", detail="Unit 2")
    leg.__name__ = f"rule_{rule_id}"
    return leg


# Registry -- a new leg is one appended row (the parametrized registry test auto-covers it).
RULES: list[RuleLeg] = [
    rule_2, rule_3, rule_4, rule_8,
    _unit2_stub(1, "module-table-completeness"),
    _unit2_stub(5, "config-parity"),
    _unit2_stub(6, "cli-surface-parity"),
    _unit2_stub(7, "stamp-honesty"),
    _unit2_stub(9, "durations-regression"),
    _unit2_stub(10, "dep-parity"),
    _unit2_stub(11, "invariant-spot-checks"),
    _unit2_stub(13, "ticket-reference"),
    _unit2_stub(14, "layer-edge-conformance"),
]


# --- orchestration (pure; reads only) -------------------------------------------------------

def run_all(ctx: RepoContext,
            legs: list[RuleLeg] | None = None) -> tuple[list[RuleResult], list[tuple[str, str]]]:
    """Run every leg. A leg raising is a CHECKER ERROR (not a finding) -- collected separately
    so a crash forces exit >=2 and cannot masquerade as a pass."""
    results: list[RuleResult] = []
    errors: list[tuple[str, str]] = []
    for leg in (RULES if legs is None else legs):
        name = getattr(leg, "__name__", repr(leg))
        try:
            results.append(leg(ctx))
        except Exception as exc:  # noqa: BLE001 -- a leg must never take down the checker
            errors.append((name, f"{type(exc).__name__}: {exc}"))
    results.sort(key=lambda r: r.rule_id)
    return results, errors


def exit_code(results: list[RuleResult], errors: list[tuple[str, str]]) -> int:
    """0 clean; 1 findings; >=2 checker error. Error dominates (a crash is not a pass)."""
    if errors:
        return 2
    if any(r.findings for r in results):
        return 1
    return 0


def _sorted_findings(results: list[RuleResult]) -> list[Finding]:
    """All findings in deterministic (rule_id, location) order for clean cross-session diffs."""
    fs = [f for r in results for f in r.findings]
    return sorted(fs, key=lambda f: (f.rule_id, f.location))


def _ids(ids: list[int]) -> str:
    """Render an id list for the coverage block; 'none' beats an empty string for a zero row."""
    return ", ".join(str(i) for i in ids) if ids else "none"


def format_report(results: list[RuleResult], errors: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    # KNOWN LIMITATIONS -- the checker must not overclaim about itself.
    lines.append("KNOWN LIMITATIONS (v1):")
    lines.append("  - R2 is precision-over-recall: the repo-rooted guard suppresses some real")
    lines.append("    missing-path claims to kill the ecosystem-path false-positive class")
    lines.append("    (terra H3; accepted for v1, revisit at gating promotion).")
    lines.append("  - Negation is not parsed: a doc that asserts a path is ABSENT is still")
    lines.append("    reported as a missing path. Known instance: .claude/skills/ and")
    lines.append("    .claude/skills/gotchas/ (CLAUDE.md section 8) -- named, not allowlisted:")
    lines.append("    an allowlist would hide the whole negation class.")
    # Coverage denominator -- COUNTED FROM THIS RUN, never hardcoded. A hardcoded roster
    # would go stale the first time a stub is implemented, which is the exact drift class
    # this checker exists to catch. Computed, `absent` is a live number.
    implemented = sorted(r.rule_id for r in results if r.status != "skipped")
    stubbed = sorted(r.rule_id for r in results if r.status == "skipped")
    structural = sorted(_STRUCTURAL_EXEMPTIONS)
    absent = sorted(_SPEC_RULE_IDS - {r.rule_id for r in results} - _STRUCTURAL_EXEMPTIONS)
    spec_n = len(_SPEC_RULE_IDS)
    total = len(implemented) + len(stubbed) + len(structural) + len(absent)
    lines.append(f"  - Rule coverage of the #97 {spec_n}-rule spec (counted from this run):")
    lines.append(f"      implemented ({len(implemented)}): {_ids(implemented)}")
    lines.append(f"      stubbed     ({len(stubbed)}): {_ids(stubbed)} -- registered, report SKIP,")
    lines.append("        check nothing")
    lines.append(f"      structural  ({len(structural)}): {_ids(structural)} -- realized in the Finding")
    lines.append("        model (evidence argv + shlex round-trip), so every finding")
    lines.append("        carries a re-runnable command by construction; no leg to register")
    lines.append(f"      absent      ({len(absent)}): {_ids(absent)}")
    lines.append(f"      TOTAL {total} of {spec_n} accounted for. A clean run is NOT a clean repo.")
    lines.append("  - The spec id set is maintained BY HAND against BACKLOG #97: a change to the")
    lines.append("    SPEC itself (say a fifteenth rule) leaves this literal and the test's")
    lines.append("    literal both at fourteen, and every test green. The doc-side check that")
    lines.append("    would close it -- parse #97's rule list, compare to _SPEC_RULE_IDS -- is")
    lines.append("    NOT built; tracked as BACKLOG #114.")
    lines.append("")
    for r in results:
        tag = {"pass": "PASS", "fail": "FAIL", "anchor-missing": "WARN", "skipped": "SKIP"}[r.status]
        note = f"  ({r.detail})" if r.detail and r.status in ("anchor-missing", "skipped") else ""
        lines.append(f"  [{tag}] rule {r.rule_id:>2} {r.rule_name}{note}")
    for f in _sorted_findings(results):
        lines.append(f"    - rule {f.rule_id} @ {f.location}: {f.claim}")
        lines.append(f"      reality:  {f.reality}")
        lines.append(f"      evidence: {f.printed()}")
    for name, msg in errors:
        lines.append(f"  [ERROR] {name}: {msg}")
    n_find = sum(len(r.findings) for r in results)
    n_failr = sum(1 for r in results if r.status == "fail")
    n_pass = sum(1 for r in results if r.status == "pass")
    n_anchor = sum(1 for r in results if r.status == "anchor-missing")
    n_skip = sum(1 for r in results if r.status == "skipped")
    lines.append(
        f"SUMMARY: pass {n_pass} | FINDINGS {n_find} across {n_failr} rules | "
        f"anchor-missing {n_anchor} | skipped(Unit2) {n_skip} | errors {len(errors)}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ctx = RepoContext(_REPO_ROOT)
    results, errors = run_all(ctx)
    print("validate_claims (#97) -- claim-vs-reality, read-only, non-blocking")
    print(format_report(results, errors))
    return exit_code(results, errors)


if __name__ == "__main__":
    sys.exit(main())
