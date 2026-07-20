"""SPIKE (throwaway): baseline plugin — worktree src, scanner untouched.

Same import redirect as `plugin_swap`, WITHOUT the extractor swap, so the two
runs differ in exactly one variable.
"""

from __future__ import annotations

from spike.worktree_path import activate

print(f"\n[spike] ai_council from: {activate()}")
print("[spike] _top_level_bullets -> hand-written scanner (baseline)\n")
