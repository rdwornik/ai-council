"""Repo-root conftest: the checkout guard.

Fails COLLECTION, loudly, if `ai_council` resolves outside the checkout pytest is running in.

WHY THIS EXISTS (2026-07-26). The editable install lives in the SYSTEM interpreter's
site-packages, not only in a venv, so `import ai_council` resolves to the PRIMARY checkout's
`src/` from any working directory — including a git worktree that has its own `src/`. A test
run in a worktree would then exercise the primary's code while reporting on the worktree's
branch: green tests about the wrong tree, with nothing in the output saying so.

The remedy is a per-worktree venv with its own editable install. This file is the GUARANTEE
that the remedy is actually in effect: a note in a doc does not survive three sessions, and the
failure is silent without a check. Root-level on purpose -- pytest loads the rootdir conftest
before `tests/conftest.py`, which imports `ai_council` at module level, so the guard has to run
first to be worth anything.

This is a checkout-identity check, NOT a venv check: a venv is one way to satisfy it, and the
guard deliberately does not care which way. See BACKLOG #121 (shared-primary writers) and
LESSONS 2026-07-26 (a value published without the predicate that produces it -- here, "the tests
passed" published without "against which tree").
"""

from __future__ import annotations

import pathlib

import ai_council

_CHECKOUT_ROOT = pathlib.Path(__file__).resolve().parent
_RESOLVED = pathlib.Path(ai_council.__file__).resolve()

if _CHECKOUT_ROOT not in _RESOLVED.parents:
    raise RuntimeError(
        "\n"
        "=================== WRONG-TREE IMPORT -- COLLECTION ABORTED ===================\n"
        f"  pytest is running in : {_CHECKOUT_ROOT}\n"
        f"  ai_council resolves to: {_RESOLVED}\n"
        "\n"
        "  The package resolves OUTSIDE this checkout, so the suite would test a different\n"
        "  tree than the one you are on and report the result against this branch.\n"
        "\n"
        "  Cause: the editable install lives in the SYSTEM interpreter's site-packages, so a\n"
        "  bare `python` / `py` resolves `ai_council` to the primary checkout from any cwd.\n"
        "\n"
        "  Fix, in a worktree:\n"
        "      py -m venv .venv\n"
        "      .venv\\Scripts\\python.exe -m pip install -e \".[dev]\"\n"
        "      .venv\\Scripts\\python.exe -m pytest        <-- call it explicitly; `python` leaks\n"
        "\n"
        "  See BACKLOG #121 and LESSONS 2026-07-26.\n"
        "==============================================================================="
    )
