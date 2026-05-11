"""Target project resolver for per-invocation transcript routing.

Names come from frontmatter (target-project:) or the --target-project CLI flag.
Paths come from config/settings.yaml target_projects map.
Unknown names fail loud with the full list of known targets shown.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TRANSCRIPTS_SUBPATH = Path("docs") / "decisions" / "transcripts"


class RoutingError(Exception):
    pass


class TargetResolver:
    """Resolves target project names to transcript directory paths."""

    def __init__(self, target_projects: dict[str, str]) -> None:
        self._target_projects = target_projects

    def resolve(
        self,
        target_project: str | list[str] | tuple[str, ...] | None,
    ) -> list[Path]:
        """Return list of transcripts dirs for the given target name(s).

        Returns empty list when target_project is None or empty.
        Raises RoutingError on any unknown name (fast-fail, checks all names).
        Accepts tuple to match Click's multiple=True output type.
        """
        if not target_project:
            return []

        if isinstance(target_project, str):
            names: list[str] = [target_project]
        elif isinstance(target_project, (list, tuple)):
            names = list(target_project)
            bad = [(i, v) for i, v in enumerate(names) if not isinstance(v, str)]
            if bad:
                idx, val = bad[0]
                raise RoutingError(
                    f"target-project items must be strings; got {type(val).__name__!r} at index {idx}: {val!r}"
                )
        else:
            raise RoutingError(
                f"target-project must be a string or list of strings; got {type(target_project).__name__!r}"
            )

        if not names:
            return []

        unknown = [n for n in names if n not in self._target_projects]
        if unknown:
            known_sorted = sorted(self._target_projects.keys())
            raise RoutingError(
                f"Unknown target-project {unknown[0]!r}. Known targets: {known_sorted}"
            )

        return [
            Path(self._target_projects[name]) / _TRANSCRIPTS_SUBPATH
            for name in names
        ]
