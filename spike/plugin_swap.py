"""SPIKE (throwaway): pytest plugin that swaps the scanner for the library extractor.

Loaded with `-p spike.plugin_swap` so the EXISTING merged #77 contract suite runs
unmodified against markdown-it-py. `src/ai_council/output.py` is never touched.
"""

from __future__ import annotations

from spike.worktree_path import activate

_RESOLVED = activate()

import ai_council.output as output  # noqa: E402
from spike.md_it_options import top_level_bullets  # noqa: E402

_SCANNER = output._top_level_bullets
output._top_level_bullets = top_level_bullets  # type: ignore[assignment]

print(f"\n[spike] ai_council from: {_RESOLVED}")
print("[spike] _top_level_bullets -> markdown-it-py\n")
