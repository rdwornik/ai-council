"""SPIKE (throwaway): options extractor over markdown-it-py's token stream.

Parallel implementation of ``ai_council.output._top_level_bullets`` built on a
CommonMark parser instead of the hand-written line scanner. Nothing here is
imported by the package; this exists only to produce decision evidence for the
buy-vs-build call on #80 / #81.

Signature is deliberately identical to the scanner's so the merged #77 contract
suite can be run against it unmodified via monkeypatch.
"""

from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

# `commonmark` preset, not the default `gfm-like`: the scanner's contract is
# CommonMark, and the default preset enables linkify/tables, which would change
# what counts as a list item.
_MD = MarkdownIt("commonmark")

# Inline token types whose `.content` IS payload. Everything else is markup.
_CONTENT_TYPES = {"text", "code_inline", "html_inline"}
# Delimiter tokens the parser has already resolved into a span — emit nothing.
_MARKUP_TYPES = {
    "em_open", "em_close", "strong_open", "strong_close",
    "link_open", "link_close", "s_open", "s_close",
}


def _inline_text(inline: Token) -> str:
    """Flatten a resolved inline token to plain text.

    The parser has already decided what is emphasis, what is a code span, and
    what is a literal backslash-escaped character. This function only reads that
    decision off the tree — there is no flanking logic, no backtick pairing and
    no escape handling here, because none of it is this layer's business.
    """
    out: list[str] = []
    for child in inline.children or ():
        if child.type in _CONTENT_TYPES:
            out.append(child.content)
        elif child.type in _MARKUP_TYPES:
            continue
        elif child.type == "softbreak":
            # A lazy-continuation line inside the same paragraph. CommonMark
            # says it is the SAME inline run; rendering it as a space is the
            # HTML renderer's convention.
            out.append(" ")
        elif child.type == "hardbreak":
            out.append(" ")
        elif child.type == "image":
            out.append(child.content)
    return "".join(out).strip()


def top_level_bullets(body: str | None) -> list[str]:
    """Top-level list items in a section body, as plain text.

    Only depth-1 lists are read, so a nested sub-bullet (an ideas entry's
    annotation) is dropped rather than scooped as its own option — same rule the
    scanner enforces by testing for leading whitespace.
    """
    if not body:
        return []
    items: list[str] = []
    list_depth = 0
    item_depth = 0
    want_inline = False
    for tok in _MD.parse(body):
        if tok.type in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif tok.type in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
        elif tok.type == "list_item_open":
            item_depth += 1
            # Capture the first paragraph of a TOP-LEVEL item only.
            want_inline = list_depth == 1 and item_depth == 1
        elif tok.type == "list_item_close":
            item_depth -= 1
            want_inline = False
        elif tok.type == "inline" and want_inline:
            text = _inline_text(tok)
            if text:
                items.append(text)
            want_inline = False  # first paragraph only; later blocks are continuation
    return items


def token_tree(body: str, indent: int = 0) -> str:
    """Readable dump of the token stream — used for the #80 continuation evidence."""
    lines: list[str] = []

    def walk(tokens: list[Token], depth: int) -> None:
        for tok in tokens:
            label = f"{'  ' * depth}{tok.type}"
            if tok.content and tok.type not in ("inline",):
                label += f"  content={tok.content!r}"
            elif tok.type == "inline":
                label += f"  content={tok.content!r}"
            lines.append(label)
            if tok.children:
                walk(list(tok.children), depth + 1)

    walk(_MD.parse(body), indent)
    return "\n".join(lines)
