# ruff: noqa: E501  # CLASSIFICATION_PROMPT is an LLM prompt string; line-wrapping changes model behavior
"""Cheap LLM call to classify a question into a debate mode."""

import asyncio
import logging

from ai_council.providers.base import AIProvider

logger = logging.getLogger(__name__)

# Cheapest-first preference for classification call
CHEAPEST_PREFERENCE = ["deepseek", "gemini", "openai", "grok", "claude"]

CLASSIFICATION_PROMPT = """\
Classify this question into exactly one debate mode:
- pick: CHOOSE between options. Signals: "should I", "which", "vs", "or", "best way to", comparing two or more concrete options.
- ideas: GENERATE possibilities. Signals: "brainstorm", "ideas for", "what could", "what features", "how might", "possibilities".
- judge: EVALUATE something existing. Signals: "how good", "review", "assess", "audit", "is this", "rate", "score", evaluating a specific thing.

Question: {question}

Respond with ONLY one word: pick, ideas, or judge"""


async def detect_mode(
    question: str,
    providers: dict[str, AIProvider],
    valid_modes: set[str],
    timeout_sec: float = 10.0,
) -> tuple[str, str]:
    """Classify question into a mode using the cheapest available provider.

    Returns:
        (mode, source_label) where source_label describes how the mode was chosen.
        Falls back to ("pick", "fallback — <reason>") on any error.
    """
    provider = _pick_cheapest(providers)
    if provider is None:
        return "pick", "fallback — no providers available"

    prompt = CLASSIFICATION_PROMPT.format(question=question)
    try:
        response = await asyncio.wait_for(
            provider.generate(prompt, round_number=1),
            timeout=timeout_sec,
        )
        mode = response.content.strip().lower()
        if mode in valid_modes:
            return mode, f"auto-detected via {provider.name()}"
        logger.warning("Mode detector returned unknown mode '%s', falling back to pick", mode)
        return "pick", f"fallback — unexpected response '{mode}' from {provider.name()}"
    except asyncio.TimeoutError:
        logger.warning("Mode detection timed out after %.0fs, falling back to pick", timeout_sec)
        return "pick", f"fallback — timeout after {timeout_sec:.0f}s"
    except Exception as exc:
        logger.warning("Mode detection failed (%s), falling back to pick", exc)
        return "pick", f"fallback — {type(exc).__name__}"


def _pick_cheapest(providers: dict[str, AIProvider]) -> AIProvider | None:
    """Return cheapest available provider according to CHEAPEST_PREFERENCE."""
    for name in CHEAPEST_PREFERENCE:
        if name in providers:
            return providers[name]
    # Any provider will do
    return next(iter(providers.values()), None)
