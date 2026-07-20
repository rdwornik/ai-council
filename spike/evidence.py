"""SPIKE (throwaway): decision evidence for #80 / #81 / hang+perf.

Runs the SAME input through both implementations and prints PASS/FAIL per case.
Run:  py -m spike.evidence
"""

from __future__ import annotations

import time

from spike.worktree_path import activate

activate()

from ai_council.output import _top_level_bullets as scanner  # noqa: E402
from spike.md_it_options import token_tree  # noqa: E402
from spike.md_it_options import top_level_bullets as library  # noqa: E402


def _run(fn, body: str) -> tuple[list[str] | str, float]:
    started = time.perf_counter()
    try:
        got: list[str] | str = fn(body)
    except Exception as exc:  # noqa: BLE001
        got = f"RAISED {type(exc).__name__}: {exc}"
    return got, time.perf_counter() - started


def case(title: str, body: str, *, expect: list[str] | None = None, show_tree: bool = False) -> None:
    print("=" * 78)
    print(title)
    print("-" * 78)
    print("input:", repr(body[:180] + ("..." if len(body) > 180 else "")))
    for name, fn in (("scanner", scanner), ("library", library)):
        got, elapsed = _run(fn, body)
        verdict = ""
        if expect is not None:
            verdict = "  PASS" if got == expect else "  FAIL"
        print(f"  {name:<8} {elapsed*1000:8.1f}ms  {got!r}{verdict}")
    if expect is not None:
        print(f"  expected {expect!r}")
    if show_tree:
        print("-" * 78)
        print("markdown-it token tree:")
        print(token_tree(body))
    print()


# --- #81: fenced code block whose lines start with "- " ---------------------
FENCE = (
    "- Real option A\n"
    "\n"
    "```yaml\n"
    "- not_an_option: 1\n"
    "- also_not_an_option: 2\n"
    "```\n"
    "\n"
    "- Real option B\n"
)
case(
    "#81  fenced block with '- ' lines  (expect ONLY the two real options)",
    FENCE,
    expect=["Real option A", "Real option B"],
    show_tree=True,
)

# --- #81b: tilde fence + indented (4-space) code block ----------------------
case(
    "#81b tilde fence + indented code block",
    "- Real option\n\n~~~\n- fabricated tilde\n~~~\n\n    - fabricated indented\n",
    expect=["Real option"],
)

# --- #80: multi-line option payload with a nested annotation ----------------
MULTI = (
    "- Adopt the library\n"
    "  because it is spec-backed\n"
    "  - Who endorsed it: sol\n"
    "- Keep the scanner\n"
)
case(
    "#80  multi-line payload + nested annotation",
    MULTI,
    show_tree=True,
)

# --- hang + perf ------------------------------------------------------------
case(
    "hang  Windows path backslash",
    "- C:\\Users\\rob\n- trailing\\\n",
    expect=["C:\\Users\\rob", "trailing\\"],
)

print("=" * 78)
print("PERF SCALING  (' *a' repeated -- the terra pass-1 pathological input)")
print("-" * 78)
print(f"{'n':>8} {'scanner':>12} {'library':>12}   ratio(lib/scan)")
for n in (2_000, 4_000, 8_000, 16_000, 30_000):
    body = "- " + " *a" * n + "\n"
    _, t_scan = _run(scanner, body)
    _, t_lib = _run(library, body)
    print(f"{n:>8} {t_scan*1000:>10.1f}ms {t_lib*1000:>10.1f}ms   {t_lib/max(t_scan,1e-9):>8.1f}x")

print()
print("PERF SCALING  ('!_!*' repeated -- the terra pass-2 close-and-open input)")
print("-" * 78)
print(f"{'n':>8} {'scanner':>12} {'library':>12}")
for n in (2_000, 5_000, 10_000, 20_000):
    body = "- " + "!_!*" * n + "\n"
    _, t_scan = _run(scanner, body)
    _, t_lib = _run(library, body)
    print(f"{n:>8} {t_scan*1000:>10.1f}ms {t_lib*1000:>10.1f}ms")
