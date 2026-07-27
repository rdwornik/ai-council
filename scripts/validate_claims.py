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

import fnmatch
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
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

# Cross-shell contract (A3, corrected by #106): evidence must paste into BOTH git-bash and
# PowerShell. The earlier claim -- "POSIX single-quoting that PowerShell also reads as a literal
# string" -- was false for the shape R2 actually emits: `shlex.join` on an argument that itself
# contains single quotes produces the '"'"' splice, which PowerShell does not parse at all. Since
# this repo's default shell is PowerShell (CLAUDE section 4), that made rule 12's "re-runnable by
# a human" claim weaker than it read -- and rule 12's exemption from registration rests on that
# claim, so the gap undercut the exemption, not just the ergonomics.
#
# The fix is quoting STYLE, not payload restriction: an argument holding single quotes but no
# character either shell would interpolate is emitted DOUBLE-quoted, which both shells read
# identically. Arguments containing " $ ` or \ fall back to shlex.quote (POSIX-correct, possibly
# PowerShell-hostile) rather than silently emitting something wrong in both.
# The enforced guarantees remain: arg0 is a bare runner, and the printed string round-trips
# through shlex back to the exact argv.
_SHELL_SAFE_BARE = re.compile(r"^[\w@%+=:,./-]+$")
# Unsafe INSIDE double quotes in at least one target shell: `"` ends the string; `$` interpolates
# in both; a backtick escapes in PowerShell and substitutes in POSIX; `\` escapes in POSIX.
_DOUBLE_UNSAFE = re.compile(r'["$`\\]')


def _shell_quote(arg: str) -> str:
    """Quote one argv element so the printed command pastes into PowerShell AND git-bash."""
    if not arg:
        return "''"
    if _SHELL_SAFE_BARE.match(arg):
        return arg
    if "'" in arg and not _DOUBLE_UNSAFE.search(arg):
        return f'"{arg}"'
    return shlex.quote(arg)

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
        """The human-copyable command string, quoted to paste into PowerShell AND git-bash."""
        return " ".join(_shell_quote(a) for a in self.evidence)


@dataclass(frozen=True)
class RuleResult:
    """One rule's verdict. `findings` non-empty => status 'fail'."""

    rule_id: int
    rule_name: str
    findings: tuple[Finding, ...] = ()
    status: str = "pass"          # pass | fail | anchor-missing | skipped
    detail: str = ""              # reason for anchor-missing / skipped
    # The surfaces this rule actually read, filled in BY THE HARNESS (run_all) from the
    # observed read-set -- never by the leg itself. A verdict published without the surface
    # set it rests on is a value published without its predicate (LESSONS 2026-07-26); rule 4
    # spent its whole life reporting `pass` while reading one of the four docs it claimed.
    surfaces: tuple[str, ...] = ()

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
    """Read-only access to the repo under `root`, AS OF HEAD'S COMMIT. Treat as immutable.
    Shared by every leg so no leg re-implements file/glob/yaml/git access.

    DETERMINISM BY CONSTRUCTION (2026-07-26 ruling). `exists`/`read`/`glob` answer questions
    about **HEAD's tree**, never the working directory. This is #116's fix moved one layer up:
    rule 2 was repaired individually, but `ctx.read` still returned the working copy and
    `ctx.glob` still hit the disk, so the same defect was live in rules 3 and 4 and would have
    been inherited by all nine stubs. Fixing the harness makes every leg -- implemented,
    stubbed, and future -- deterministic without each one having to remember.

    Both failure modes were witnessed, not hypothesised: a dirty uncommitted
    `.pre-commit-config.yaml` flipped rule 3 PASS->FAIL, and one *untracked*
    `docs/decisions/ADR-99-scratch.md` flipped rule 4 PASS->FAIL, while a fresh clone at the
    identical commit passed both.

    Disk access survives as `disk_exists` / `disk_read` / `disk_glob`: explicit, documented,
    and used by NO rule leg (asserted by test). A harness may legitimately need the working
    tree; a rule never does, and making that opt-in rather than the default is what stops
    determinism from being a convention someone has to remember.

    A repo with no commits has a genuinely empty tree -- not an error. Any other git failure
    raises, so a broken query can never masquerade as "nothing found" (H1 discipline).
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._text: dict[str, str] = {}
        self._yaml: dict[str, Any] = {}
        self._tree: set[str] | None = None
        self._dirs: set[str] | None = None
        # Observed read-set (#125 leg 2). Every content read is appended here, so the harness can
        # report, per rule, the surface set that rule ACTUALLY read -- see `run_all`. Observed,
        # never declared: a hand-written per-rule surface string would go stale exactly the way
        # rule 4's four-surface spec did, which is the defect this disclosure answers.
        self._reads: list[str] = []

    # --- observed read-set (harness-generic disclosure) ---------------------------------

    def _record(self, surface: str) -> None:
        """Note that `surface`'s CONTENT was consulted. Recorded on every call, including a
        cache hit: recording only on a miss would make disclosure depend on which leg happened
        to read a doc first, so a later leg would silently under-report its own basis."""
        self._reads.append(surface)

    def reads_mark(self) -> int:
        """An opaque cursor into the read log, for measuring one leg's reads (see run_all)."""
        return len(self._reads)

    def surfaces_read(self) -> tuple[str, ...]:
        """Every surface read so far, deduplicated, in first-read order."""
        return tuple(dict.fromkeys(self._reads))

    def surfaces_since(self, mark: int) -> tuple[str, ...]:
        """The surfaces read since `mark` -- one leg's own basis when bracketed around it."""
        return tuple(dict.fromkeys(self._reads[mark:]))

    # --- committed-tree view (the default; what every rule leg uses) --------------------

    def _load_tree(self) -> None:
        if self._tree is not None:
            return
        if self.git("rev-parse", "--verify", "-q", "HEAD").returncode != 0:
            self._tree, self._dirs = set(), set()
            return
        out = self.git("ls-tree", "-r", "--name-only", "HEAD")
        if out.returncode != 0:
            raise RuntimeError(f"git ls-tree HEAD failed: {out.stderr.strip() or out.returncode}")
        files = {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
        dirs: set[str] = set()
        for f in files:
            parts = f.split("/")
            for k in range(1, len(parts)):
                dirs.add("/".join(parts[:k]))
        self._tree, self._dirs = files, dirs

    def exists(self, rel: str) -> bool:
        """Is `rel` a file or directory in HEAD's tree? (Not: is it on disk.)"""
        self._load_tree()
        key = rel.rstrip("/")
        assert self._tree is not None and self._dirs is not None
        return key in self._tree or key in self._dirs

    def read(self, rel: str) -> str:
        """The COMMITTED content of `rel`. A dirty working copy cannot change a verdict."""
        self._record(rel)
        if rel not in self._text:
            out = self.git("show", f"HEAD:{rel}")
            if out.returncode != 0:
                raise FileNotFoundError(f"{rel} is not in HEAD's tree")
            self._text[rel] = out.stdout
        return self._text[rel]

    def glob(self, pat: str) -> list[Path]:
        """Committed paths matching `pat`. Returns Paths for caller convenience (`.name`),
        built from the tree -- nothing is stat'd, so an untracked file cannot appear."""
        self._record(pat)
        self._load_tree()
        assert self._tree is not None
        return [self.root / r for r in sorted(fnmatch.filter(sorted(self._tree), pat))]

    def committed_paths(self) -> tuple[set[str], set[str]]:
        """HEAD's tree as (files, dirs-with-trailing-slash). One source of truth: rule 2's
        resolution model and every other leg read the same set, so the two cannot drift."""
        self._record("HEAD:<tracked tree>")
        self._load_tree()
        assert self._tree is not None and self._dirs is not None
        return set(self._tree), {d + "/" for d in self._dirs}

    def load_yaml(self, rel: str) -> Any:
        if rel not in self._yaml:
            self._yaml[rel] = yaml.safe_load(self.read(rel))
        return self._yaml[rel]

    # --- disk view: EXPLICIT opt-in, used by no rule leg --------------------------------

    def disk_exists(self, rel: str) -> bool:
        """Working-directory check. NOT for rule legs -- a verdict must not depend on it."""
        return (self.root / rel).exists()

    def disk_read(self, rel: str) -> str:
        """Working-directory read. NOT for rule legs."""
        return (self.root / rel).read_text(encoding="utf-8")

    def disk_glob(self, pat: str) -> list[Path]:
        """Working-directory glob. NOT for rule legs."""
        return sorted(self.root.glob(pat))

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

# R2 resolution model (#116), replacing the first-segment-is-a-top-level-dir heuristic.
#
# Declared bases, in order. Canonical docs routinely write a path relative to a base rather than
# to the repo root -- `research/merger.py` means src/ai_council/research/merger.py, `decisions/`
# means docs/decisions/. The old guard read those as "first segment is not a top-level dir" and
# skipped them, which silently suppressed 16 legitimate claims while excluding ecosystem paths
# only as a side effect. Bases are declared, and a test asserts each one exists in the tracked
# tree, so the base list validates itself instead of being folklore.
_R2_BASES = (
    "",                   # repo root
    "src/ai_council/",    # ARCHITECTURE writes module paths relative to the package
    "docs/",              # the docs taxonomy is written base-relative (ADR-60)
    "docs/decisions/",    # reaches docs/decisions/transcripts/, which is hub-routed (ADR-43)
)

# Declared external prefixes -- first segments naming something OUTSIDE this repo. An EXPLICIT
# exclusion, not an accident of a heuristic: these must be excluded because they are not claims
# about this tree, and saying so by name is what lets the resolution model be strict everywhere
# else.
_R2_EXTERNAL_PREFIXES = (
    "Dev/",           # the ecosystem root, one level ABOVE this repo
    "ai-council/",    # this repo named from outside itself (VISION's ecosystem view)
    "astral-sh/",     # a GitHub org/repo slug (ruff-pre-commit) -- not a filesystem path at all
)

# Declared RUNTIME paths: gitignored by design, so correctly absent from the tracked tree while
# the docs that name them are telling the truth. Surfaced by the resolution model itself -- these
# never reached the old guard, because they exist on disk and the old disk check passed them, so
# no drop-set measurement could have predicted them.
#
# Declared rather than derived: `git check-ignore` also consults .git/info/exclude and the user's
# global excludesfile, both untracked, which would reintroduce exactly the checkout-dependence
# #116 exists to remove. A test asserts each entry really is gitignored AND untracked, so this
# list self-validates the same way the base list does.
_R2_RUNTIME_PATHS = (
    "output/",                 # .gitignore:38 -- council run artifacts
    "council_inbox/archive/",  # .gitignore:41 -- processed-brief archive
)


def _resolves_under_a_base(tok: str, files: set[str], dirs: set[str]) -> bool:
    """Does `tok` name a tracked path under ANY declared base? A finding fires only if none do."""
    for base in _R2_BASES:
        cand = base + tok
        if cand in files or cand.rstrip("/") + "/" in dirs:
            return True
    return False


def _externally_routed(tok: str) -> bool:
    """Allowlisted subtree, checked against every base expansion as well as the raw token.

    ARCHITECTURE writes `handoffs/` and `transcripts/` base-relative; both name subtrees an ADR
    routes to the hub, so they are legitimately absent here. Matching the allowlist only against
    the root-relative form would report them as drift.
    """
    for base in ("", *_R2_BASES):
        cand = base + tok
        if any(cand.startswith(p) for p in _R2_ALLOWLIST):
            return True
    return False


# --- shared context predicate (#108) --------------------------------------------------------
# ONE pure function, two consumers. Rules 2 and 8 both had the same blind spot -- they tested a
# token and never read the sentence holding it -- so they get one shared answer rather than two
# drifting copies of the same judgement.
_MD_NOISE = re.compile(r"[*_`]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+")

# Each class is deliberately TIGHT. A context predicate trades a false POSITIVE for a false
# NEGATIVE, and here the false negative is the worse trade: an over-broad marker silently
# suppresses real drift, which is the failure this checker exists to prevent. Widen only on a
# witnessed instance, never speculatively -- and prefer a new narrow alternative to loosening
# an existing one.
_CTX_NEGATED = re.compile(
    r"\b(?:has|have|carries|contains|holds|is|are|was|were)\s+no\b"
    r"|\bno\s+(?:repo-level|local|such)\b"
    r"|\b(?:absent from|does not exist|do not exist|never had|never existed|not present)\b",
    re.IGNORECASE)
_CTX_EXTERNAL = re.compile(
    r"\bhub(?:'s)?\b|\.dev-knowledge\b|\bsibling repo\b|\bupstream repo\b", re.IGNORECASE)
_CTX_HYPOTHETICAL = re.compile(
    r"\bif added\b|\bif it existed\b|\bif ever added\b"
    r"|\bwould\s+(?:go|live|sit|be placed|be added)\b", re.IGNORECASE)

_CTX_CLASSES = (
    ("negated", _CTX_NEGATED),
    ("externally-attributed", _CTX_EXTERNAL),
    ("hypothetical", _CTX_HYPOTHETICAL),
)


def context_withdraws_claim(line: str, span: tuple[int, int]) -> str | None:
    """Does the prose AROUND a token withdraw the claim that it exists here, now?

    PURE -- no filesystem, no git, no module state. The same (line, span) always yields the same
    verdict. That purity is load-bearing twice: it is what makes the predicate testable in
    isolation, and it is why two rules can share it without becoming coupled to each other.

    Adjudication is scoped to the token's own SENTENCE, not its line: a line routinely holds one
    clause that withdraws a claim and another that makes one, and line-scoping would let the
    first silence the second.

    Two consumers, same answer used differently:
      * rule 2 -- SUPPRESSION: a path the sentence says is absent, attributes to another repo, or
        describes hypothetically is not a false existence claim about this tree.
      * rule 8 -- CLASSIFICATION: a SHA in a hub-attributed sentence cites another repo's
        history, so it is not a dangling LOCAL citation. R8 acts on that class alone -- a
        "negated" or "hypothetical" SHA is not a coherent notion, so it must not silence one.

    Returns the withdrawal reason, or None when the claim stands and the rule should check it.
    """
    start, _ = span
    bounds = [0] + [m.end() for m in _SENTENCE_SPLIT.finditer(line)] + [len(line)]
    sentence = line
    for lo, hi in zip(bounds, bounds[1:]):
        if lo <= start < hi:
            sentence = line[lo:hi]
            break
    # Emphasis and backticks are stripped before matching so `**no**` reads as `no`; the token
    # itself is located on the RAW line, so stripping cannot shift the span.
    probe = _MD_NOISE.sub("", sentence)
    for name, pattern in _CTX_CLASSES:
        if pattern.search(probe):
            return name
    return None


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
    files, dirs = ctx.committed_paths()
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
                # `./x` and `x` are the same claim; resolve on the normalized form but keep the
                # token as WRITTEN for the report, so file:line and text match the doc.
                norm = tok.removeprefix("./")
                if _externally_routed(norm):       # hub-routed subtree (ADR-42/43), any base
                    continue
                if any(norm.startswith(p) for p in _R2_EXTERNAL_PREFIXES):
                    continue                       # declared external: not a claim about this tree
                if any(norm.startswith(p) for p in _R2_RUNTIME_PATHS):
                    continue                       # declared runtime path: gitignored by design
                # #108: adjudicate the claim in its sentence before resolving it. Deliberately NOT
                # an allowlist entry -- an allowlist hides one path, this reads the prose and so
                # handles the whole class, including instances nobody has hit yet.
                if context_withdraws_claim(line, m.span(1)) is not None:
                    continue
                # #116: resolve against the TRACKED TREE under the declared bases. Not the disk --
                # a disk check made the verdict depend on untracked debris, so the same commit
                # reported differently on different checkouts.
                if _resolves_under_a_base(norm, files, dirs):
                    continue
                bases = ", ".join(b or "<repo root>" for b in _R2_BASES)
                findings.append(Finding(
                    rule_id=2,
                    claim=f"path `{tok}` does not resolve under any declared base",
                    location=f"{rel}:{i}",
                    reality=f"{tok} is not in HEAD's tree under any of: {bases}",
                    # A COMMIT-tree probe, matching what the rule now tests. `ls-files
                    # --error-unmatch` would consult the INDEX and so carry the same staged-path
                    # dependence the rule just removed; `cat-file -e HEAD:<p>` reads the commit
                    # and works for files and directories alike. Shown root-relative; `reality`
                    # carries the base list, since no single probe asserts "no base matched".
                    evidence=("git", "cat-file", "-e", f"HEAD:{norm}"),
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

# The FOUR surfaces #97 rule 4 specifies, quoted from the spec line: "ADR files on disk ==
# CLAUDE section 11 == ARCHITECTURE Governing-ADRs == `docs/decisions/README.md` index".
# The implementation read ONE of them (`("CLAUDE.md",)`) in ONE direction, so ADR-15 could sit
# absent from ARCHITECTURE's roster under a `pass 4 | FINDINGS 0` report -- witnessed
# 2026-07-26, filed as #125. Each surface carries its own heading anchor because the three
# rosters are named differently; a doc absent from the tree is simply not compared, but a doc
# that is PRESENT and cannot be anchored is a finding, not a silent omission.
_ADR_SURFACES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("CLAUDE.md", _ADR_SECTION, "adrs"),
    ("ARCHITECTURE.md", re.compile(r"governing adrs", re.IGNORECASE), "governing adrs"),
    ("docs/decisions/README.md", re.compile(r"^\s*index\s*$", re.IGNORECASE), "index"),
)

# Within a roster section, the LOCAL sub-block. All three surfaces separate local from
# hub-owned ADRs with an explicit marker; the README's `## Index` has no marker because its
# whole table is local (its cross-repo refs live in a sibling section). So: narrow to the
# Local block when the body declares one, else the body IS the local block.
#
# Deliberately case-insensitive SUBSTRING tokens rather than regexes, and deliberately a slice
# rather than a stateful scan. Terra pre-merge finding, accepted: the evidence command must
# reproduce detection exactly, and it has to do so as a shell-pasteable one-liner from a bare
# checkout. Mirroring a multi-branch regex + stateful loop in that payload is where fidelity
# silently rots -- so the PREDICATE was simplified to one both sides can express identically,
# instead of inflating the payload to chase it. Converging the two beats asserting they agree.
_ADR_LOCAL_TOKEN = "**local"
_ADR_ECOSYSTEM_TOKEN = "**ecosystem"
# The entry shapes that constitute a roster CLAIM (list item / table row), shared with evidence.
_ADR_ENTRY_PREFIXES = ("-", "*", "|")
# Entries are one-per-line, `·`-packed on one bullet (ARCHITECTURE), or `;`-packed (the
# ecosystem block's style). Splitting on all three keeps one entry == one declaration.
_ADR_ENTRY_SPLIT = re.compile(r"[\n·;]")


def _adr_roster_docs(ctx: RepoContext) -> list[str]:
    """The roster surfaces to compare -- every one the spec names that exists in HEAD's tree."""
    return [rel for rel, _sec, _w in _ADR_SURFACES if ctx.exists(rel)]


def _adr_surface_spec(rel: str) -> tuple[re.Pattern[str], str]:
    """(heading anchor, lowercase heading keyword) for `rel`; the CLAUDE-style roster is the
    default so a caller naming a doc outside the declared set still gets a sane anchor."""
    for name, sec, word in _ADR_SURFACES:
        if name == rel:
            return sec, word
    return _ADR_SECTION, "adrs"


def _adr_local_block(body: list[str]) -> list[str]:
    """The LOCAL sub-block of a roster section: from the Local marker (or the top, when the
    section declares none) up to the FIRST Ecosystem marker that follows it.

    "That follows it" is load-bearing: an Ecosystem marker sitting ABOVE the Local block must
    not collapse the slice to empty, because an empty local block reports every ADR on disk as
    missing -- turning a layout quirk into a wall of false findings.
    """
    lm = [k for k, ln in enumerate(body) if _ADR_LOCAL_TOKEN in ln.lower()]
    em = [k for k, ln in enumerate(body) if _ADR_ECOSYSTEM_TOKEN in ln.lower()]
    start = lm[0] if lm else 0
    after = [k for k in em if k > start]
    return body[start:(after[0] if after else len(body))]


def _adr_first_ids(lines: list[str]) -> set[str]:
    """The id each entry is ABOUT -- its FIRST id -- across `lines` split into entries.

    Measured, not stylistic: all three live surfaces name ADR-43 (hub-owned) inside ADR-07's
    own entry, so treating every mention as a declaration would report three false findings.
    Precision boundary, stated: two local ADRs packed into ONE entry yield only the first --
    every surface writes one entry per ADR, and the fix if that changes is to split the entry.
    """
    ids: set[str] = set()
    for entry in _ADR_ENTRY_SPLIT.split("\n".join(lines)):
        m = _ADR_ID.search(entry)
        if m:
            ids.add(m.group(1))
    return ids


def _declared_local_adrs(body: list[str]) -> set[str]:
    """PERMISSIVE set, for the disk -> roster direction: any entry in the local block naming an
    ADR documents it. Erring permissive here is the established precision-over-recall posture
    (rule 3's whole-token mention): an ADR named in prose must not be reported as undocumented.
    """
    return _adr_first_ids(_adr_local_block(body))


def _claimed_local_adrs(body: list[str]) -> set[str]:
    """STRICT set, for the roster -> disk direction: only ENTRY-SHAPED lines (list items and
    table rows) make a local claim.

    The two directions ask different questions and must err in opposite directions, so they get
    different sets. Terra pre-merge finding, verified: with one permissive set, a section that
    carries an Ecosystem marker but no Local marker treats its introductory prose as the local
    block, so a cross-reference in that prose is reported as "rostered but not on disk" -- a
    false finding invented out of a sentence. Requiring entry shape kills that without touching
    the disk -> roster direction, where an unparsed prose line would instead have invented a
    false "missing" finding. All three live surfaces are entry-shaped (`- `, `| `).
    """
    block = [ln for ln in _adr_local_block(body) if ln.lstrip()[:1] in _ADR_ENTRY_PREFIXES]
    return _adr_first_ids(block)


def _adr_evidence(rel: str, word: str, strict: bool, label: str) -> tuple[str, ...]:
    """A re-runnable command that recomputes BOTH sides of the set-equality the way the rule
    did, then prints the delta.

    Terra pre-merge finding, accepted: an earlier form embedded the already-computed id list
    and only re-derived the roster side, so it could print ids the on-disk set no longer
    supports -- evidence that agrees with the finding by construction rather than by
    recomputation. Both sides are now derived here, and the disk side comes from HEAD's TREE
    (`git ls-tree`), not a disk glob, because the rule resolves against the committed tree
    (#116/ADR-15) -- a `pathlib.glob` here would reintroduce the very tree-vs-disk divergence
    that determinism fix closed, and the evidence would disagree with the rule on a dirty tree.

    Deliberately backslash-free so it double-quotes cleanly into both PowerShell and git-bash
    (#106), and deliberately not an import of this checker: evidence must run from a bare
    checkout with no scripts/ on sys.path.
    """
    entry_filter = (
        f"blk=[l for l in blk if l.lstrip()[:1] in {_ADR_ENTRY_PREFIXES!r}]; " if strict else ""
    )
    delta = "sorted(decl-disk)" if strict else "sorted(disk-decl)"
    payload = (
        "import re,subprocess,pathlib; "
        f"ls=pathlib.Path({rel!r}).read_text(encoding='utf-8').split(chr(10)); "
        f"h=[k for k,l in enumerate(ls) if l.startswith('#') and {word!r} in l.lower()]; "
        "i=h[0] if h else -1; "
        "nx=[k for k in range(i+1,len(ls)) if ls[k].startswith('#')] if i>=0 else []; "
        "body=ls[i+1:(nx[0] if nx else len(ls))] if i>=0 else []; "
        f"lm=[k for k,l in enumerate(body) if {_ADR_LOCAL_TOKEN!r} in l.lower()]; "
        f"em=[k for k,l in enumerate(body) if {_ADR_ECOSYSTEM_TOKEN!r} in l.lower()]; "
        "st=lm[0] if lm else 0; af=[k for k in em if k>st]; "
        "blk=body[st:(af[0] if af else len(body))]; "
        f"{entry_filter}"
        "txt=chr(10).join(blk).replace(chr(183),chr(10)).replace(';',chr(10)); "
        "decl={m.group(1) for e in txt.split(chr(10)) if (m:=re.search('(ADR-[0-9]+)',e))}; "
        "tr=subprocess.run(['git','ls-tree','-r','--name-only','HEAD'],capture_output=True,"
        "text=True).stdout.split(chr(10)); "
        "disk={m.group(1) for p in tr if p.endswith('.md') "
        "if (m:=re.match('docs/decisions/(ADR-[0-9]+)',p))}; "
        f"print({label!r}, {delta})"
    )
    return ("python", "-c", payload)


def rule_4(ctx: RepoContext) -> RuleResult:
    """SET-EQUALITY between the ADR files on disk and every roster surface the spec names.

    Four surfaces, both directions (#125). `docs/decisions/ADR-*.md` in HEAD's tree is the
    ground truth; CLAUDE section 11, ARCHITECTURE's Governing-ADRs and the
    `docs/decisions/README.md` index must each declare exactly that set:

      * on disk, not declared  -> the roster is stale (ADR-13 was; ADR-15 was, invisibly);
      * declared, not on disk  -> the roster claims an ADR that does not exist.

    The reverse direction is scoped to each section's LOCAL block and to entry-head ids, which
    is what makes it safe: the rosters legitimately name hub-owned ecosystem ADRs, and
    comparing against every mention would flag those (measured: ADR-43, on all three surfaces).
    """
    disk_ids = sorted({m.group(1) for p in ctx.glob("docs/decisions/ADR-*.md")
                       if (m := _ADR_ID.match(p.name))})
    if not disk_ids:
        return RuleResult(4, "adr-roster-parity", status="anchor-missing",
                          detail="no docs/decisions/ADR-*.md files")
    findings: list[Finding] = []
    anchored = False
    for rel in _adr_roster_docs(ctx):
        section_re, word = _adr_surface_spec(rel)
        body, hl = _section_body(ctx.read(rel), section_re)
        if body is None:
            # Present but unanchorable: the surface drops out of the comparison and nothing
            # would say so -- the #104 criterion-(iv) hazard (a doc silently un-gating its own
            # check) one level down. Reported, never skipped.
            findings.append(Finding(
                rule_id=4,
                claim=f"{rel} is a declared ADR-roster surface but its roster section could "
                      f"not be located, so it silently drops out of the comparison",
                location=f"{rel}:1",
                reality=f"no heading matching /{section_re.pattern}/ found in {rel}",
                evidence=("python", "-c",
                          f"import pathlib; "
                          f"ls=pathlib.Path({rel!r}).read_text(encoding='utf-8').split(chr(10)); "
                          f"print('roster-headings:', [l for l in ls if l.startswith('#') "
                          f"and {word!r} in l.lower()])"),
                reproduces="stdout-contains",
            ))
            continue
        anchored = True
        missing = [a for a in disk_ids if a not in _declared_local_adrs(body)]
        extra = sorted(_claimed_local_adrs(body) - set(disk_ids), key=lambda a: (len(a), a))
        if missing:
            findings.append(Finding(
                rule_id=4,
                claim=f"ADR roster in {rel} omits an ADR that exists on disk",
                location=f"{rel}:{hl}",
                reality=f"on disk but not declared in {rel}: {missing}",
                evidence=_adr_evidence(rel, word, strict=False, label="missing-from-roster"),
                reproduces="stdout-contains",
            ))
        if extra:
            findings.append(Finding(
                rule_id=4,
                claim=f"ADR roster in {rel} declares a local ADR with no file on disk",
                location=f"{rel}:{hl}",
                reality=f"declared in {rel} but absent from docs/decisions/: {extra}",
                evidence=_adr_evidence(rel, word, strict=True, label="rostered-but-not-on-disk"),
                reproduces="stdout-contains",
            ))
    if not anchored and not findings:
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
                # #108, classification half: a SHA the sentence attributes to the hub cites
                # ANOTHER repo's history, so its unreachability here is expected, not drift.
                # Only this class counts -- a "negated" or "hypothetical" SHA is not a coherent
                # notion, and letting those classes silence R8 would widen the predicate past
                # what it was witnessed to need.
                if context_withdraws_claim(line, m.span()) == "externally-attributed":
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
    so a crash forces exit >=2 and cannot masquerade as a pass.

    Each leg is bracketed by a read-set mark so its RuleResult carries the surfaces it actually
    read (#125 leg 2). Harness-generic by construction: a new leg discloses its basis without
    writing a line of disclosure code, and no leg can claim a surface it never opened.
    """
    results: list[RuleResult] = []
    errors: list[tuple[str, str]] = []
    for leg in (RULES if legs is None else legs):
        name = getattr(leg, "__name__", repr(leg))
        mark = ctx.reads_mark()
        try:
            results.append(replace(leg(ctx), surfaces=ctx.surfaces_since(mark)))
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
    lines.append("  - Context IS adjudicated (#108), in the token's own sentence, in three tight")
    lines.append("    classes: negated ('this repo has no X'), externally-attributed ('the hub's")
    lines.append("    X'), and hypothetical ('if added, would go under X'). R2 uses this to")
    lines.append("    SUPPRESS, R8 to CLASSIFY hub-cited SHAs -- one shared pure predicate, so the")
    lines.append("    two rules cannot drift apart. No path was allowlisted to achieve this.")
    lines.append("    The classes are deliberately narrow: over-broad markers would silently")
    lines.append("    suppress real drift, so a suppression you did not expect is a BUG, not a")
    lines.append("    tuning opportunity. Suppressions are not currently counted in the report.")
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
    lines.append("  - Each rule line ends with `reads:` -- the surface set that rule ACTUALLY")
    lines.append("    read this run, observed by the harness, not declared by the rule. It is")
    lines.append("    the predicate behind that rule's verdict: `pass` over a surface set that")
    lines.append("    omits a doc the rule claims to compare is exactly how a stale ARCHITECTURE")
    lines.append("    roster sat under FINDINGS 0 (#125). Scope: doc/tree surfaces whose CONTENT")
    lines.append("    was consulted; git ref-graph queries (R8's reachability) are not listed.")
    lines.append("  - Evidence commands paste into BOTH PowerShell and git-bash (#106). One")
    lines.append("    residual: arg0 is a BARE runner, so on Windows `python` may resolve to a")
    lines.append("    Store stub or a non-venv interpreter. That is documented, not engineered")
    lines.append("    around -- an absolute interpreter path would not be portable either.")
    lines.append("  - The spec id set is maintained BY HAND against BACKLOG #97: a change to the")
    lines.append("    SPEC itself (say a fifteenth rule) leaves this literal and the test's")
    lines.append("    literal both at fourteen, and every test green. The doc-side check that")
    lines.append("    would close it -- parse #97's rule list, compare to _SPEC_RULE_IDS -- is")
    lines.append("    NOT built; tracked as BACKLOG #114.")
    lines.append("")
    for r in results:
        tag = {"pass": "PASS", "fail": "FAIL", "anchor-missing": "WARN", "skipped": "SKIP"}[r.status]
        note = f"  ({r.detail})" if r.detail and r.status in ("anchor-missing", "skipped") else ""
        # The predicate published beside the value: what this verdict actually rests on.
        reads = ", ".join(r.surfaces) if r.surfaces else "none"
        lines.append(f"  [{tag}] rule {r.rule_id:>2} {r.rule_name}{note}  reads: {reads}")
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
