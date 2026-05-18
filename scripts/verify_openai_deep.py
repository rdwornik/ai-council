"""Minimal one-shot live verification of the openai_deep research provider.

Sub-dollar verification: short query, asserts non-empty content AND sources.
Uses the migrated gpt-5.5 path; NOT the deprecated o3-deep-research.
"""

import asyncio
import logging
import os
import sys

from ai_council.research.provider import ResearchProviderError
from ai_council.research.providers.openai_deep_research import OpenAIDeepResearchProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

QUERY = "What is the current latest stable Python release as of 2026? One sentence."


async def main() -> int:
    api_key = os.environ["OPENAI_API_KEY"]
    # Use medium effort for the verification probe (still uses migrated reasoning param shape).
    provider = OpenAIDeepResearchProvider(
        api_key=api_key, timeout_sec=600, reasoning_effort="medium"
    )
    print(f"=== verify_openai_deep model={provider.model_string()} effort=medium ===")
    try:
        result = await provider.research(QUERY)
    except ResearchProviderError as exc:
        print(f"FAILED: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED {type(exc).__name__}: {exc}")
        return 1

    content_len = len(result.content or "")
    src_count = len(result.sources or [])
    print(
        f"content_len={content_len} sources={src_count} "
        f"cost=${result.cost_usd:.4f} dur={result.duration_sec:.1f}s"
    )
    snippet = (result.content or "")[:300].replace("\n", " ")
    print(f"content_snippet: {snippet!r}")
    for s in (result.sources or [])[:5]:
        print(f"  source: {s.title} -> {s.url}")
    if content_len == 0 or src_count == 0:
        print("VERIFY FAIL: empty content or zero sources (silent parse mismatch)")
        return 1
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
