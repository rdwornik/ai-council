"""One-shot live verification of the openai_mini research provider.

Two modes:
- pre: run against the currently-configured (deprecated) model — expect failure, capture exact error.
- post: run against the post-migration model — expect non-empty content AND non-empty sources.

Usage:
    py scripts/verify_openai_mini.py pre
    py scripts/verify_openai_mini.py post

Operator-approved live call. Cost ~$0.20-2.50.
"""

import asyncio
import logging
import os
import sys

from ai_council.research.providers.openai_mini_research import OpenAIMiniResearchProvider
from ai_council.research.provider import ResearchProviderError

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(message)s")

QUERY = "What is the current latest stable Python release as of 2026? One sentence."


async def main(mode: str) -> int:
    api_key = os.environ["OPENAI_API_KEY"]
    if mode == "pre":
        provider = OpenAIMiniResearchProvider(api_key=api_key, timeout_sec=600)
    elif mode == "post":
        # Caller is expected to have already edited the provider defaults; we just instantiate.
        provider = OpenAIMiniResearchProvider(api_key=api_key, timeout_sec=600)
    else:
        print(f"Unknown mode {mode!r}", file=sys.stderr)
        return 2

    print(f"=== verify_openai_mini mode={mode} model={provider.model_string()} ===")
    try:
        result = await provider.research(QUERY)
    except ResearchProviderError as exc:
        print(f"FAILED: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - surface anything else
        print(f"UNEXPECTED {type(exc).__name__}: {exc}")
        return 1

    content_len = len(result.content or "")
    src_count = len(result.sources or [])
    print(f"content_len={content_len} sources={src_count} cost=${result.cost_usd:.4f} dur={result.duration_sec:.1f}s")
    snippet = (result.content or "")[:300].replace("\n", " ")
    print(f"content_snippet: {snippet!r}")
    for s in (result.sources or [])[:5]:
        print(f"  source: {s.title} -> {s.url}")
    if mode == "post":
        if content_len == 0 or src_count == 0:
            print("VERIFY FAIL: empty content or zero sources (silent parse mismatch)")
            return 1
        print("VERIFY OK")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre"
    sys.exit(asyncio.run(main(mode)))
