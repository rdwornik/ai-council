# AI Council — Task Tracking

## Phase 1 Implementation

### Wave 1 — Foundation
- [x] Create feat/phase1-foundation branch
- [x] requirements.txt
- [x] config/settings.yaml
- [x] src/models.py
- [x] tasks/todo.md + tasks/lessons.md

### Wave 2 — Config + Base
- [ ] config/__init__.py
- [ ] config/config_loader.py
- [ ] src/providers/base.py

### Wave 3 — Provider Implementations
- [ ] src/providers/gemini.py
- [ ] src/providers/openai_provider.py
- [ ] src/providers/anthropic.py
- [ ] src/providers/xai.py

### Wave 4 — Orchestration
- [ ] src/debate.py
- [ ] src/synthesis.py
- [ ] src/output.py

### Wave 5 — CLI + Tests
- [ ] src/cli.py
- [ ] tests/__init__.py
- [ ] tests/conftest.py
- [ ] tests/test_config.py
- [ ] tests/test_models.py
- [ ] tests/test_debate.py
- [ ] tests/test_synthesis.py
- [ ] tests/test_output.py
- [ ] tests/test_integration.py
- [ ] pytest.ini
- [ ] README.md

### Verification
- [ ] pip install -r requirements.txt
- [ ] pytest tests/ -m "not integration" -v
- [ ] python -m src.cli "test question" --rounds 1 --models claude,openai
- [ ] pytest tests/test_integration.py -v
