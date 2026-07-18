"""Tests for `council doctor` — liveness + config pre-flight (src/ai_council/doctor.py).

No real API calls: seat pings are patched at the run_health_checks_sync seam, provider
build is patched, and env is controlled with patch.dict. Console output is captured via
io.StringIO (Windows /dev/null anti-pattern).
"""

import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from rich.console import Console

from ai_council import doctor as doc
from ai_council.cli import main
from ai_council.doctor import (
    ADVISORY,
    FAIL,
    GREEN,
    PASS,
    RED,
    YELLOW,
    Check,
    build_record,
    check_keys,
    evaluate_verdict,
    run_doctor,
    validate_config,
)
from config.config_loader import (
    AppConfig,
    DefaultsConfig,
    ModelConfig,
    PromptsConfig,
    ResearchConfig,
    ResearchProviderConfig,
)
from tests.conftest import MockProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, *, with_research: bool = False) -> AppConfig:
    """Minimal AppConfig whose synthesizer + panels resolve; single seat 'claude'."""
    model = ModelConfig(
        name="claude", sdk="anthropic", model="claude-test",
        api_key_env="TEST_KEY", timeout_sec=60, max_tokens=1024,
    )
    defaults = DefaultsConfig(
        rounds=1, max_rounds=2,
        output_dir=tmp_path / "output",
        synthesizer="claude",
        default_panel=["claude"],
        full_panel=["claude"],
    )
    prompts = PromptsConfig(initial="{question}", critique="{question}", synthesis="{question}")
    research = None
    if with_research:
        research = ResearchConfig(
            default_providers=["perplexity"],
            deep_providers=["perplexity"],
            cache_dir=tmp_path / "cache",
            cache_ttl_days=7,
            summary_max_tokens=2500,
            summary_model="claude",  # resolves against top-level models, not research.providers
            providers={"perplexity": ResearchProviderConfig(
                name="perplexity", model="sonar", api_key_env="PPLX_KEY", timeout_sec=60,
            )},
            min_successful_providers=1,
        )
    return AppConfig(
        defaults=defaults,
        models={"claude": model},
        prompts=prompts,
        available_providers={"claude"},
        research=research,
    )


def _sio_console() -> tuple[Console, io.StringIO]:
    sio = io.StringIO()
    return Console(file=sio, force_terminal=False, width=200), sio


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------


def test_evaluate_verdict_precedence() -> None:
    assert evaluate_verdict([Check("key", "K", PASS, "")]) == GREEN
    assert evaluate_verdict([Check("key", "K", ADVISORY, "")]) == YELLOW
    assert evaluate_verdict([Check("key", "K", ADVISORY, ""), Check("seat", "s", FAIL, "")]) == RED


def test_validate_config_all_resolve(tmp_path: Path) -> None:
    checks = validate_config(_make_config(tmp_path))
    assert all(c.status == PASS for c in checks)


def test_validate_config_unresolved_synthesizer(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.defaults.synthesizer = "ghost"
    checks = validate_config(config)
    synth = next(c for c in checks if c.subject == "synthesizer")
    assert synth.status == FAIL and "ghost" in synth.detail


def test_validate_config_unresolved_research_ref(tmp_path: Path) -> None:
    config = _make_config(tmp_path, with_research=True)
    assert config.research is not None
    config.research.default_providers = ["nonexistent"]
    checks = validate_config(config)
    ref = next(c for c in checks if c.subject == "research.default_providers")
    assert ref.status == FAIL and "nonexistent" in ref.detail


def test_validate_config_summary_model_resolves_against_models(tmp_path: Path) -> None:
    """research.summary_model resolves against top-level models (merger.py/cli.py), not
    research.providers -- a valid model name is PASS."""
    config = _make_config(tmp_path, with_research=True)  # summary_model="claude" (a model)
    checks = validate_config(config)
    sm = next(c for c in checks if c.subject == "research.summary_model")
    assert sm.status == PASS


def test_validate_config_summary_model_unresolved_is_advisory(tmp_path: Path) -> None:
    """An unresolved summary_model is ADVISORY (merger degrades to truncation), not FAIL."""
    config = _make_config(tmp_path, with_research=True)
    assert config.research is not None
    config.research.summary_model = "ghost"
    checks = validate_config(config)
    sm = next(c for c in checks if c.subject == "research.summary_model")
    assert sm.status == ADVISORY and "ghost" in sm.detail


def test_validate_config_empty_roster_unsatisfiable_threshold(tmp_path: Path) -> None:
    """Empty default_providers with a positive min_successful is unsatisfiable -> ADVISORY,
    never a false PASS (regression guard for the short-circuit bug)."""
    config = _make_config(tmp_path, with_research=True)
    assert config.research is not None
    config.research.default_providers = []
    config.research.min_successful_providers = 3
    checks = validate_config(config)
    m = next(c for c in checks if c.subject == "research.min_successful (default)")
    assert m.status == ADVISORY and "unsatisfiable" in m.detail
    assert evaluate_verdict(checks) != GREEN


def test_validate_config_deep_threshold_checked(tmp_path: Path) -> None:
    """min_successful is validated against deep_providers too (--deep uses it standalone)."""
    config = _make_config(tmp_path, with_research=True)
    assert config.research is not None
    config.research.default_providers = ["perplexity", "perplexity", "perplexity"]
    config.research.deep_providers = ["perplexity"]  # smaller than threshold
    config.research.min_successful_providers = 3
    checks = validate_config(config)
    deep = next(c for c in checks if c.subject == "research.min_successful (deep)")
    assert deep.status == ADVISORY and "unsatisfiable" in deep.detail


def test_validate_config_empty_panel_is_fail(tmp_path: Path) -> None:
    """An empty debate panel is FAIL (a run would select zero seats), never a false PASS."""
    config = _make_config(tmp_path)
    config.defaults.default_panel = []
    checks = validate_config(config)
    panel = next(c for c in checks if c.subject == "default_panel")
    assert panel.status == FAIL and "empty" in panel.detail
    assert evaluate_verdict(checks) == RED


def test_check_keys_present_absent_and_shadow(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False):
        present = check_keys(config, {})
        assert present[0].status == PASS
        shadow = check_keys(config, {"TEST_KEY": ""})  # set-but-empty in shell
        assert shadow[0].status == ADVISORY
    with patch.dict(os.environ, {}, clear=True):
        absent = check_keys(config, {})
        assert absent[0].status == FAIL


def test_check_keys_research_only_absent_is_advisory(tmp_path: Path) -> None:
    """A missing research-only key is ADVISORY (research degrades), not FAIL; a missing
    debate/synth model key stays FAIL."""
    config = _make_config(tmp_path, with_research=True)  # claude->TEST_KEY, perplexity->PPLX_KEY
    with patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=True):  # PPLX_KEY absent
        checks = check_keys(config, {})
    by_env = {c.subject: c for c in checks}
    assert by_env["TEST_KEY"].status == PASS
    assert by_env["PPLX_KEY"].status == ADVISORY
    assert evaluate_verdict(checks) == YELLOW  # advisory, not RED


def test_check_keys_model_absent_is_fail_even_with_research(tmp_path: Path) -> None:
    config = _make_config(tmp_path, with_research=True)
    with patch.dict(os.environ, {"PPLX_KEY": "pk-real"}, clear=True):  # TEST_KEY (model) absent
        checks = check_keys(config, {})
    by_env = {c.subject: c for c in checks}
    assert by_env["TEST_KEY"].status == FAIL
    assert evaluate_verdict(checks) == RED


def test_check_keys_never_prints_values(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with patch.dict(os.environ, {"TEST_KEY": "super-secret-value"}, clear=False):
        checks = check_keys(config, {})
    serialized = " ".join(c.subject + c.detail for c in checks)
    assert "super-secret-value" not in serialized
    assert "TEST_KEY" in serialized  # name IS shown


def test_build_record_shape_and_no_values() -> None:
    checks = [
        Check("key", "TEST_KEY", PASS, "present (claude)"),
        Check("seat", "claude", PASS, "ping OK", role="synthesizer"),
    ]
    record = build_record(checks, GREEN, "2026-07-17T09:00:00")
    assert record["schema_version"] == doc.SCHEMA_VERSION
    assert record["verdict"] == GREEN
    assert record["seats"] == {"synthesizer": {"claude": PASS}}
    assert record["checks"][0]["class"] == "key"


# ---------------------------------------------------------------------------
# run_doctor end-to-end (hermetic — pings patched)
# ---------------------------------------------------------------------------


def _run(config: AppConfig, ping_results: dict, shell_snapshot: dict | None = None):
    console, sio = _sio_console()
    with patch.object(doc, "build_all_providers", return_value={"claude": MockProvider("claude")}), \
         patch.object(doc, "run_health_checks_sync", return_value=ping_results):
        code = run_doctor(
            config, {"claude": MockProvider},
            shell_snapshot=shell_snapshot or {}, console=console,
            output_dir=config.defaults.output_dir,
        )
    return code, sio.getvalue()


def test_run_doctor_green(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False):
        code, output = _run(config, {"claude": (True, "")})
    assert code == 0
    assert "GREEN" in output
    assert "synthesizer" in output


def test_run_doctor_red_on_seat_fail(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False):
        code, output = _run(config, {"claude": (False, "authentication failed (check API key)")})
    assert code == 1
    assert "RED" in output


def test_run_doctor_red_on_missing_key(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        code, output = _run(config, {"claude": (True, "")})
    assert code == 1  # key absent -> RED regardless of ping


def test_run_doctor_yellow_on_shadowing(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False):
        code, output = _run(config, {"claude": (True, "")}, shell_snapshot={"TEST_KEY": ""})
    assert code == 3
    assert "YELLOW" in output


def test_run_doctor_writes_record(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False):
        code, _ = _run(config, {"claude": (True, "")})
    health = config.defaults.output_dir / "health"
    latest = health / "doctor-latest.json"
    assert latest.exists()
    record = json.loads(latest.read_text(encoding="utf-8"))
    assert record["schema_version"] == doc.SCHEMA_VERSION
    assert record["verdict"] == GREEN
    # a timestamped sibling was also written
    assert any(p.name.startswith("doctor-") and p.name != "doctor-latest.json"
               for p in health.glob("doctor-*.json"))
    # no secret value leaked into the record
    assert "sk-real" not in latest.read_text(encoding="utf-8")


def test_run_doctor_redacts_secret_in_error_detail(tmp_path: Path) -> None:
    """A raw ping-error string carrying the key VALUE is redacted before screen + record."""
    config = _make_config(tmp_path)
    secret = "sk-live-abcdef0123456789"
    console, sio = _sio_console()
    with patch.dict(os.environ, {"TEST_KEY": secret}, clear=False), \
         patch.object(doc, "build_all_providers", return_value={"claude": MockProvider("claude")}), \
         patch.object(doc, "run_health_checks_sync",
                      return_value={"claude": (False, f"401 from https://x?key={secret}")}):
        run_doctor(config, {"claude": MockProvider}, console=console,
                   output_dir=config.defaults.output_dir)
    assert secret not in sio.getvalue()
    assert "[REDACTED]" in sio.getvalue()
    latest = (config.defaults.output_dir / "health" / "doctor-latest.json").read_text(encoding="utf-8")
    assert secret not in latest


def test_run_doctor_redacts_short_secret(tmp_path: Path) -> None:
    """Redaction is length-agnostic: even a short credential value is stripped."""
    config = _make_config(tmp_path)
    secret = "sk9z"  # short, under the old 8-char guard
    console, sio = _sio_console()
    with patch.dict(os.environ, {"TEST_KEY": secret}, clear=False), \
         patch.object(doc, "build_all_providers", return_value={"claude": MockProvider("claude")}), \
         patch.object(doc, "run_health_checks_sync",
                      return_value={"claude": (False, f"bad token {secret} rejected")}):
        run_doctor(config, {"claude": MockProvider}, console=console,
                   output_dir=config.defaults.output_dir)
    assert f"{secret} rejected" not in sio.getvalue()
    latest = (config.defaults.output_dir / "health" / "doctor-latest.json").read_text(encoding="utf-8")
    assert f"token {secret}" not in latest


def test_collect_secret_values_orders_longest_first(tmp_path: Path) -> None:
    config = _make_config(tmp_path)  # single model -> TEST_KEY
    with patch.dict(os.environ, {"TEST_KEY": "short", "OTHER": "muchlongervalue"}, clear=True):
        # only TEST_KEY is a configured env, so OTHER is ignored; verify sorting on a 2-key config
        config.models["extra"] = ModelConfig(
            name="extra", sdk="x", model="m", api_key_env="OTHER", timeout_sec=1, max_tokens=1,
        )
        values = doc._collect_secret_values(config)
    assert values == sorted(values, key=len, reverse=True)


def test_redact_overlapping_secret_no_suffix_leak() -> None:
    """A shorter secret that is a prefix of a longer one must not leave the suffix exposed."""
    short = "abc12345"
    longer = short + "def67890"
    ordered = sorted({short, longer}, key=len, reverse=True)  # the contract: longest-first
    out = doc._redact(f"error {longer} here", ordered)
    assert short not in out and longer not in out and "def67890" not in out


def test_run_doctor_seat_build_failure_contained(tmp_path: Path) -> None:
    """A provider-build blow-up becomes a FAIL row, not a doctor crash."""
    config = _make_config(tmp_path)
    console, sio = _sio_console()
    with patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False), \
         patch.object(doc, "build_all_providers", side_effect=RuntimeError("boom")):
        code = run_doctor(config, {"claude": MockProvider}, console=console,
                          output_dir=config.defaults.output_dir)
    assert code == 1  # contained FAIL -> RED
    assert "RED" in sio.getvalue()


# ---------------------------------------------------------------------------
# CLI wiring — `council doctor` subcommand
# ---------------------------------------------------------------------------


def test_run_doctor_record_write_failure_contained(tmp_path: Path) -> None:
    """An unwritable record dir warns and keeps the health verdict -- no crash, no RED flip."""
    config = _make_config(tmp_path)
    console, sio = _sio_console()
    with patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False), \
         patch.object(doc, "build_all_providers", return_value={"claude": MockProvider("claude")}), \
         patch.object(doc, "run_health_checks_sync", return_value={"claude": (True, "")}), \
         patch.object(doc, "write_record", side_effect=OSError("disk full")):
        code = run_doctor(config, {"claude": MockProvider}, console=console,
                          output_dir=config.defaults.output_dir)
    assert code == 0  # GREEN health verdict preserved despite write failure
    out = sio.getvalue()
    assert "WARNING" in out and "GREEN" in out
    assert "(not written)" in out


def test_doctor_subcommand_invokes(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    runner = CliRunner()
    with patch("ai_council.cli.load_config", return_value=config), \
         patch("ai_council.cli.load_dotenv"), \
         patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False), \
         patch.object(doc, "build_all_providers", return_value={"claude": MockProvider("claude")}), \
         patch.object(doc, "run_health_checks_sync", return_value={"claude": (True, "")}):
        result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "COUNCIL DOCTOR" in result.output


def test_doctor_subcommand_registered() -> None:
    assert "doctor" in main.commands
    assert "run" in main.commands


def test_doctor_subcommand_survives_unreadable_secrets_file(tmp_path: Path) -> None:
    """An unreadable/corrupt global secrets file warns but does not abort the doctor."""
    config = _make_config(tmp_path)
    runner = CliRunner()
    with patch("ai_council.cli.load_config", return_value=config), \
         patch("ai_council.cli.load_dotenv", side_effect=OSError("permission denied")), \
         patch.object(Path, "exists", return_value=True), \
         patch.dict(os.environ, {"TEST_KEY": "sk-real"}, clear=False), \
         patch.object(doc, "build_all_providers", return_value={"claude": MockProvider("claude")}), \
         patch.object(doc, "run_health_checks_sync", return_value={"claude": (True, "")}):
        result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "WARNING" in result.output and "COUNCIL DOCTOR" in result.output


# ---------------------------------------------------------------------------
# #39: bounded retention for output/health/ doctor records
# ---------------------------------------------------------------------------

def test_prune_health_records_keeps_recent_n(tmp_path):
    """#39: only the most recent N timestamped records survive; doctor-latest.json is kept."""
    health = tmp_path / "health"
    health.mkdir()
    for i in range(15):
        (health / f"doctor-20260718_{i:06d}.json").write_text("{}", encoding="utf-8")
    (health / "doctor-latest.json").write_text("{}", encoding="utf-8")

    doc._prune_health_records(health, keep=10)

    remaining = sorted(
        p.name for p in health.glob("doctor-*.json") if p.name != "doctor-latest.json"
    )
    assert len(remaining) == 10
    assert remaining[0] == "doctor-20260718_000005.json"  # oldest kept (records 05..14)
    assert (health / "doctor-latest.json").exists()  # never pruned


def test_write_record_prunes_to_retention(tmp_path):
    """#39: write_record applies the bounded retention on each write."""
    out = tmp_path / "out"
    for i in range(12):
        doc.write_record({"n": i}, out, f"20260718_{i:06d}")
    health = out / "health"
    records = [p for p in health.glob("doctor-*.json") if p.name != "doctor-latest.json"]
    assert len(records) == doc._HEALTH_RETENTION
    assert (health / "doctor-latest.json").exists()
