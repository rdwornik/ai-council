"""Tests for src/ai_council/routing.py."""

from pathlib import Path

import pytest

from ai_council.routing import RoutingError, TargetResolver

_DEV_ROOT = Path("C:/Dev/")
_PROJECTS = [".dev-knowledge", "foo"]

_TRANSCRIPTS = Path("docs") / "decisions" / "transcripts"


@pytest.fixture
def resolver() -> TargetResolver:
    return TargetResolver(_DEV_ROOT, _PROJECTS)


@pytest.fixture
def empty_resolver() -> TargetResolver:
    return TargetResolver(Path("."), [])


# ---------------------------------------------------------------------------
# None / empty → no targets
# ---------------------------------------------------------------------------


def test_resolve_none_returns_empty(resolver: TargetResolver) -> None:
    assert resolver.resolve(None) == []


def test_resolve_empty_list_returns_empty(resolver: TargetResolver) -> None:
    assert resolver.resolve([]) == []


def test_resolve_empty_tuple_returns_empty(resolver: TargetResolver) -> None:
    assert resolver.resolve(()) == []


# ---------------------------------------------------------------------------
# Single known name
# ---------------------------------------------------------------------------


def test_resolve_single_string(resolver: TargetResolver) -> None:
    result = resolver.resolve(".dev-knowledge")
    assert len(result) == 1
    assert result[0] == _DEV_ROOT / ".dev-knowledge" / _TRANSCRIPTS


def test_resolve_single_name_list(resolver: TargetResolver) -> None:
    result = resolver.resolve([".dev-knowledge"])
    assert len(result) == 1
    assert result[0] == _DEV_ROOT / ".dev-knowledge" / _TRANSCRIPTS


# ---------------------------------------------------------------------------
# Multiple known names
# ---------------------------------------------------------------------------


def test_resolve_list_of_known_names(resolver: TargetResolver) -> None:
    result = resolver.resolve([".dev-knowledge", "foo"])
    assert len(result) == 2
    assert result[0] == _DEV_ROOT / ".dev-knowledge" / _TRANSCRIPTS
    assert result[1] == _DEV_ROOT / "foo" / _TRANSCRIPTS


def test_resolve_list_preserves_order(resolver: TargetResolver) -> None:
    result = resolver.resolve(["foo", ".dev-knowledge"])
    assert result[0] == _DEV_ROOT / "foo" / _TRANSCRIPTS
    assert result[1] == _DEV_ROOT / ".dev-knowledge" / _TRANSCRIPTS


def test_resolve_tuple_of_known_names(resolver: TargetResolver) -> None:
    result = resolver.resolve((".dev-knowledge", "foo"))
    assert len(result) == 2
    assert result[0] == _DEV_ROOT / ".dev-knowledge" / _TRANSCRIPTS


def test_resolve_tuple_preserves_order(resolver: TargetResolver) -> None:
    result = resolver.resolve(("foo", ".dev-knowledge"))
    assert result[0] == _DEV_ROOT / "foo" / _TRANSCRIPTS


# ---------------------------------------------------------------------------
# Unknown names → RoutingError
# ---------------------------------------------------------------------------


def test_resolve_unknown_string_raises(resolver: TargetResolver) -> None:
    with pytest.raises(RoutingError, match="Unknown target-project 'unknown'"):
        resolver.resolve("unknown")


def test_resolve_unknown_error_lists_known_names(resolver: TargetResolver) -> None:
    with pytest.raises(RoutingError) as exc_info:
        resolver.resolve("unknown")
    msg = str(exc_info.value)
    assert ".dev-knowledge" in msg
    assert "foo" in msg


def test_resolve_list_with_unknown_raises(resolver: TargetResolver) -> None:
    with pytest.raises(RoutingError, match="Unknown target-project"):
        resolver.resolve([".dev-knowledge", "not-real"])


# ---------------------------------------------------------------------------
# Empty config + non-None → RoutingError
# ---------------------------------------------------------------------------


def test_resolve_nonempty_with_empty_config_raises(empty_resolver: TargetResolver) -> None:
    with pytest.raises(RoutingError):
        empty_resolver.resolve(".dev-knowledge")


def test_empty_resolver_empty_input_returns_empty(empty_resolver: TargetResolver) -> None:
    assert empty_resolver.resolve(None) == []
    assert empty_resolver.resolve([]) == []


# ---------------------------------------------------------------------------
# Malformed types → RoutingError (not TypeError)
# ---------------------------------------------------------------------------


def test_resolve_integer_raises_routing_error(resolver: TargetResolver) -> None:
    with pytest.raises(RoutingError, match="must be a string or list"):
        resolver.resolve(123)  # type: ignore[arg-type]


def test_resolve_list_with_integer_item_raises_routing_error(resolver: TargetResolver) -> None:
    with pytest.raises(RoutingError, match="must be strings"):
        resolver.resolve([".dev-knowledge", 42])  # type: ignore[list-item]


def test_resolve_list_with_none_item_raises_routing_error(resolver: TargetResolver) -> None:
    with pytest.raises(RoutingError, match="must be strings"):
        resolver.resolve([None])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Path computation — paths resolve under dev_root
# ---------------------------------------------------------------------------


def test_resolve_path_computed_under_dev_root(tmp_path: Path) -> None:
    resolver = TargetResolver(tmp_path, ["myproject"])
    result = resolver.resolve("myproject")
    assert result[0] == tmp_path / "myproject" / _TRANSCRIPTS
