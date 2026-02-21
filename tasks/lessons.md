# AI Council — Lessons Learned

## Session: Phase 1 Foundation (2026-02-21)

### Rules to Follow

1. **google-genai async**: Use `client.aio.models.generate_content()` — NOT `asyncio.to_thread`. Package is `google-genai`, not deprecated `google-generativeai`.

2. **Click + asyncio**: Use `asyncio.run()` inside the sync Click handler — don't add `asyncclick` dependency.

3. **pytest-asyncio 0.24+**: Requires `asyncio_mode = auto` in `pytest.ini`. Without this, async tests silently skip or fail.

4. **`config/__init__.py`**: Required for `from config.config_loader import ...` to work as a package import.

5. **`asyncio.wait_for`**: Takes the coroutine object directly — `asyncio.wait_for(coro, timeout=n)` — not a lambda.

6. **Provider isolation**: Providers must NOT import each other per CLAUDE.md spec. XAI and OpenAI have near-identical code — that's intentional, no shared base class.

7. **`output_dir.mkdir`**: Called lazily in `output.py` when saving, not at startup.

8. **Synthesizer also debates**: Claude instance participates in debate rounds AND synthesizes. Same provider instance, by design.

9. **No bare except**: Always catch specific exceptions. Log with `logging`, never `print()`.

10. **Type hints**: Use `X | None` not `Optional[X]`. Use `Path` objects, not raw strings.
