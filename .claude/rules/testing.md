# Testing Standards

- pytest + pytest-asyncio for all tests
- asyncio_mode = auto
- Test files: tests/test_*.py
- Run: pytest tests/ -v
- 266 unit tests passing (+ integration tests requiring API keys)
- No silent failures — log warnings, raise on errors
