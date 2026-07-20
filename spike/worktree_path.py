"""SPIKE (throwaway): force `ai_council` to resolve to THIS worktree's src.

The shared `.venv` editable-installs `ai_council` from the MAIN checkout, so an
unqualified import inside a worktree silently tests the main checkout's code.
Prepending the worktree `src` to `sys.path` shadows it — and `activate()`
ASSERTS the resolved path afterwards, so a run can never quietly test main.

Git Bash caveat (this cost a false start): `PYTHONPATH` does work, but MSYS
converts POSIX paths (`/c/...`) to Windows form only for values it recognises as
a single path. Joining two paths with `;` defeats that heuristic, Python then
gets an unresolvable `/c/...`, and the import silently falls back to MAIN src.
Pass Windows-form paths (`C:/...`) from Git Bash, or set one path only.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"


def activate() -> str:
    for mod in [m for m in sys.modules if m == "ai_council" or m.startswith("ai_council.")]:
        del sys.modules[mod]
    sys.path.insert(0, str(_SRC))
    import ai_council

    resolved = Path(ai_council.__file__).resolve()
    if _SRC not in resolved.parents:
        raise RuntimeError(f"WRONG SOURCE: importing {resolved}, expected under {_SRC}")
    return str(resolved)
