"""SPIKE (throwaway): force `ai_council` to resolve to THIS worktree's src.

The shared `.venv` carries an editable install whose `__editable___ai_council_..._finder`
is a MetaPathFinder pinned to the MAIN checkout's `src/`. A MetaPathFinder is consulted
before `sys.path`, so setting PYTHONPATH alone does NOT redirect the import — the worktree
silently tests the main checkout's code. Drop the finder, then prepend the worktree src.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"


def activate() -> str:
    sys.meta_path[:] = [
        f for f in sys.meta_path if "__editable__" not in type(f).__module__
    ]
    for mod in [m for m in sys.modules if m == "ai_council" or m.startswith("ai_council.")]:
        del sys.modules[mod]
    sys.path.insert(0, str(_SRC))
    import ai_council

    resolved = Path(ai_council.__file__).resolve()
    if _SRC not in resolved.parents:
        raise RuntimeError(f"WRONG SOURCE: importing {resolved}, expected under {_SRC}")
    return str(resolved)
