# Python Environment — ai-council

- Python >=3.12
- Virtual env: .venv\Scripts\Activate.ps1
- Install: pip install -e ".[dev]"
- Click CLI entry point: `ai-council`
- Async-first architecture (asyncio.to_thread for blocking providers)
- 5 LLM providers: keep separate, do NOT merge provider implementations
- Config single source of truth: config/settings.yaml
