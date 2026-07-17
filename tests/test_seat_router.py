"""Tests for the seat router — CLI admission gate, same-seat API fallback, seats[] telemetry."""

from unittest.mock import AsyncMock, MagicMock, patch

from ai_council.models import ModelResponse
from ai_council.providers.base import ProviderError
from ai_council.seat_router import SeatRouter, SeatSpec, build_seat_router
from config.config_loader import ModelConfig
from tests.conftest import MockProvider


def _cli(served_model: str = "srv-model", channel: str = "modelUsage",
         version: str = "v1", raises: Exception | None = None) -> MagicMock:
    cli = MagicMock()
    cli.identity_channel = channel
    cli.version = version
    if raises is not None:
        cli.generate = AsyncMock(side_effect=raises)
    else:
        cli.generate = AsyncMock(
            return_value=ModelResponse("seat", served_model, 1, "cli answer", 0.1, 10)
        )
    return cli


def _router(cli: MagicMock | None, backend: str = "cli") -> SeatRouter:
    spec = SeatSpec(
        api_provider=MockProvider("seat", "api answer"),
        requested_backend=backend,
        requested_model="pin-model",
        cli_provider=cli,
        cli_command="claude" if cli else None,
    )
    return SeatRouter({"seat": spec})


# --- try_cli ---


async def test_try_cli_success_admits_and_records_identity() -> None:
    r = _router(_cli(served_model="claude-opus-4-8"))
    resp = await r.try_cli("seat", "prompt", 1)
    assert resp is not None and resp.content == "cli answer"
    sm = r.collect()[0]
    assert sm.actual_backend == "cli"
    assert sm.actual_model == "claude-opus-4-8"
    assert sm.identity_channel == "modelUsage"
    assert sm.identity_readable is True
    assert sm.cli == {"name": "claude", "version": "v1"}
    assert sm.fallback_events == []


async def test_try_cli_failure_records_fallback_and_returns_none() -> None:
    r = _router(_cli(raises=ProviderError("seat", "CLI timed out after 30s")))
    resp = await r.try_cli("seat", "prompt", 2)
    assert resp is None  # signals the caller to run the API leg
    sm = r.collect()[0]
    assert len(sm.fallback_events) == 1
    fe = sm.fallback_events[0]
    assert fe.round == 2 and fe.from_backend == "cli" and fe.to_backend == "api"
    assert fe.cause == "timeout"  # classified into the shared vocabulary


async def test_try_cli_identity_unreadable_flips_flag() -> None:
    r = _router(_cli(raises=ProviderError("seat", "identity-unreadable: no served model")))
    assert await r.try_cli("seat", "p", 1) is None
    sm = r.collect()[0]
    assert sm.identity_readable is False
    assert sm.fallback_events[0].cause == "identity-unreadable"


async def test_try_cli_api_backend_returns_none_immediately() -> None:
    r = _router(None, backend="api")
    assert await r.try_cli("seat", "p", 1) is None
    sm = r.collect()[0]
    assert sm.actual_backend == "api" and sm.cli is None


# --- record_api ---


async def test_record_api_labels_seat_api_echo() -> None:
    r = _router(None, backend="api")
    r.record_api("seat", ModelResponse("seat", "gpt-x", 1, "api answer", 0.1, 5))
    sm = r.collect()[0]
    assert sm.actual_backend == "api"
    assert sm.identity_channel == "api-echo"
    assert sm.actual_model == "gpt-x"
    assert sm.identity_readable is True


async def test_record_api_provider_error_leaves_model_null() -> None:
    r = _router(_cli(raises=ProviderError("seat", "process error")))
    await r.try_cli("seat", "p", 1)  # CLI fails -> fallback
    r.record_api("seat", ProviderError("seat", "api also failed"))  # both lanes down
    sm = r.collect()[0]
    assert sm.actual_backend == "api"
    assert sm.actual_model is None  # degradation record
    assert len(sm.fallback_events) == 1


async def test_full_cli_then_api_fallback_flow() -> None:
    r = _router(_cli(raises=ProviderError("seat", "quota exceeded")))
    cli_resp = await r.try_cli("seat", "p", 1)
    assert cli_resp is None
    r.record_api("seat", ModelResponse("seat", "api-model", 1, "recovered", 0.1, 5))
    sm = r.collect()[0]
    assert sm.actual_backend == "api" and sm.actual_model == "api-model"
    assert sm.fallback_events[0].cause == "quota"


# --- build_seat_router ---


def _model_cfg(name: str, backend: str = "api", cli_command: str | None = None) -> ModelConfig:
    return ModelConfig(
        name=name, sdk="s", model=f"{name}-api", api_key_env="K", timeout_sec=30, max_tokens=100,
        backend=backend, cli_command=cli_command, cli_model=(f"{name}-cli" if cli_command else None),
    )


def test_build_seat_router_cli_seat_builds_adapter() -> None:
    fake_cls = MagicMock(return_value=MagicMock())
    with patch.dict("ai_council.seat_router.CLI_PROVIDER_CLASSES", {"claude": fake_cls}, clear=True):
        router = build_seat_router(
            ["claude"],
            {"claude": MockProvider("claude")},
            {"claude": _model_cfg("claude", backend="cli", cli_command="claude")},
        )
    spec = router._specs["claude"]
    assert spec.requested_backend == "cli"
    assert spec.requested_model == "claude-cli"
    assert spec.cli_provider is not None
    fake_cls.assert_called_once()


def test_build_seat_router_api_seat_no_cli() -> None:
    router = build_seat_router(
        ["openai"], {"openai": MockProvider("openai")}, {"openai": _model_cfg("openai")}
    )
    spec = router._specs["openai"]
    assert spec.requested_backend == "api" and spec.cli_provider is None


def test_build_seat_router_unknown_cli_command_degrades_to_api() -> None:
    router = build_seat_router(
        ["x"], {"x": MockProvider("x")},
        {"x": _model_cfg("x", backend="cli", cli_command="nonexistent")},
    )
    spec = router._specs["x"]
    assert spec.requested_backend == "api" and spec.cli_provider is None


def test_build_seat_router_build_failure_degrades_to_api() -> None:
    boom = MagicMock(side_effect=ProviderError("claude", "CLI not found on PATH"))
    with patch.dict("ai_council.seat_router.CLI_PROVIDER_CLASSES", {"claude": boom}, clear=True):
        router = build_seat_router(
            ["claude"], {"claude": MockProvider("claude")},
            {"claude": _model_cfg("claude", backend="cli", cli_command="claude")},
        )
    assert router._specs["claude"].cli_provider is None  # degraded, not fatal


# --- end-to-end through run_debate ---


async def test_run_debate_threads_seats_through_router(sample_prompts_config, sample_question):
    """run_debate routes each seat via the router and returns per-seat SeatMetrics."""
    from ai_council.debate import run_debate

    cli = _cli(served_model="served-m")
    a = MockProvider("a", "api a")
    b = MockProvider("b", "api b")
    router = SeatRouter({
        "a": SeatSpec(a, "cli", "pin-a", cli, "claude"),
        "b": SeatSpec(b, "api", "b-model", None, None),
    })
    outcome = await run_debate(
        question=sample_question, providers=[a, b],
        prompts=sample_prompts_config, num_rounds=1, seat_router=router,
    )
    seats = {s.seat: s for s in outcome.seats}
    assert seats["a"].actual_backend == "cli" and seats["a"].actual_model == "served-m"
    assert seats["b"].actual_backend == "api" and seats["b"].identity_channel == "api-echo"
    cli.generate.assert_awaited()  # the CLI seat actually went through the CLI adapter
