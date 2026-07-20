"""Tests for src/metrics.py — cost computation and metric aggregation."""



from ai_council.metrics import build_call_metrics, build_debate_metrics, compute_call_cost
from ai_council.models import ModelResponse, ProviderCallMetrics, Round
from config.config_loader import ModelConfig


def _make_config(name: str, cost_in: float, cost_out: float) -> ModelConfig:
    return ModelConfig(
        name=name,
        sdk="test",
        model=f"{name}-model",
        api_key_env="TEST_KEY",
        timeout_sec=60,
        max_tokens=1000,
        cost_per_1m_input=cost_in,
        cost_per_1m_output=cost_out,
    )


def _make_response(
    provider: str,
    round_number: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency: float = 1.0,
) -> ModelResponse:
    token_count = None
    if input_tokens is not None and output_tokens is not None:
        token_count = input_tokens + output_tokens
    return ModelResponse(
        provider=provider,
        model=f"{provider}-model",
        round_number=round_number,
        content="response",
        latency_sec=latency,
        token_count=token_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# --- compute_call_cost ---


def test_cost_calculation_basic():
    cfg = _make_config("claude", cost_in=15.0, cost_out=75.0)
    # 1000 input tokens + 500 output tokens
    cost = compute_call_cost(1_000, 500, cfg)
    expected = (1_000 * 15.0 + 500 * 75.0) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_cost_calculation_zero_tokens():
    cfg = _make_config("deepseek", cost_in=0.55, cost_out=2.19)
    assert compute_call_cost(0, 0, cfg) == 0.0


def test_cost_calculation_cheapest_provider():
    cfg = _make_config("deepseek", cost_in=0.55, cost_out=2.19)
    cost = compute_call_cost(1_000_000, 1_000_000, cfg)
    assert abs(cost - (0.55 + 2.19)) < 1e-6


def test_cost_calculation_most_expensive_provider():
    cfg = _make_config("claude", cost_in=15.0, cost_out=75.0)
    cost = compute_call_cost(1_000_000, 1_000_000, cfg)
    assert abs(cost - (15.0 + 75.0)) < 1e-6


# --- build_call_metrics ---


def test_build_call_metrics_known_tokens():
    configs = {"claude": _make_config("claude", cost_in=15.0, cost_out=75.0)}
    response = _make_response("claude", round_number=1, input_tokens=1000, output_tokens=500)
    m = build_call_metrics(response, configs, round_number=1)

    assert m.provider == "claude"
    assert m.round_number == 1
    assert m.input_tokens == 1000
    assert m.output_tokens == 500
    expected_cost = (1000 * 15.0 + 500 * 75.0) / 1_000_000
    assert abs(m.estimated_cost_usd - expected_cost) < 1e-9
    assert m.was_retry is False


def test_build_call_metrics_missing_tokens_defaults_to_zero():
    configs = {"gemini": _make_config("gemini", cost_in=2.0, cost_out=10.0)}
    response = _make_response("gemini", round_number=1, input_tokens=None, output_tokens=None)
    m = build_call_metrics(response, configs, round_number=1)

    assert m.input_tokens == 0
    assert m.output_tokens == 0
    assert m.estimated_cost_usd == 0.0


def test_build_call_metrics_unknown_provider_cost_is_zero():
    configs = {}  # no config for this provider
    response = _make_response("unknown", round_number=1, input_tokens=500, output_tokens=200)
    m = build_call_metrics(response, configs, round_number=1)
    assert m.estimated_cost_usd == 0.0


def test_build_call_metrics_was_retry_flag():
    configs = {"openai": _make_config("openai", cost_in=1.75, cost_out=14.0)}
    response = _make_response("openai", round_number=2, input_tokens=100, output_tokens=50)
    m = build_call_metrics(response, configs, round_number=2, was_retry=True)
    assert m.was_retry is True


# --- build_debate_metrics ---


def test_build_debate_metrics_aggregates_correctly():
    configs = {
        "claude": _make_config("claude", cost_in=15.0, cost_out=75.0),
        "gemini": _make_config("gemini", cost_in=2.0, cost_out=10.0),
    }

    round1 = Round(
        number=1,
        responses=[
            _make_response("claude", 1, input_tokens=1000, output_tokens=500),
            _make_response("gemini", 1, input_tokens=800, output_tokens=400),
        ],
    )
    synthesis_call = ProviderCallMetrics(
        provider="claude",
        round_number=0,
        input_tokens=2000,
        output_tokens=1000,
        estimated_cost_usd=(2000 * 15.0 + 1000 * 75.0) / 1_000_000,
        latency_sec=1.5,
    )

    metrics = build_debate_metrics([round1], synthesis_call, configs, total_duration_sec=10.0)

    assert metrics.total_input_tokens == 1000 + 800 + 2000
    assert metrics.total_output_tokens == 500 + 400 + 1000
    assert len(metrics.calls) == 3  # 2 round1 + 1 synthesis
    assert metrics.total_duration_sec == 10.0

    expected_cost = (
        compute_call_cost(1000, 500, configs["claude"])
        + compute_call_cost(800, 400, configs["gemini"])
        + synthesis_call.estimated_cost_usd
    )
    assert abs(metrics.total_estimated_cost_usd - expected_cost) < 1e-9


def test_build_debate_metrics_synthesis_is_round_zero():
    configs = {"claude": _make_config("claude", cost_in=15.0, cost_out=75.0)}
    synth = ProviderCallMetrics(
        provider="claude", round_number=0, input_tokens=100, output_tokens=50,
        estimated_cost_usd=0.0, latency_sec=1.0,
    )
    metrics = build_debate_metrics([], synth, configs, total_duration_sec=5.0)
    assert len(metrics.calls) == 1
    assert metrics.calls[0].round_number == 0


def test_build_debate_metrics_empty_rounds():
    configs = {"claude": _make_config("claude", cost_in=15.0, cost_out=75.0)}
    synth = ProviderCallMetrics(
        provider="claude", round_number=0, input_tokens=500, output_tokens=200,
        estimated_cost_usd=compute_call_cost(500, 200, configs["claude"]),
        latency_sec=0.5,
    )
    metrics = build_debate_metrics([], synth, configs, total_duration_sec=1.0)
    assert metrics.total_input_tokens == 500
    assert metrics.total_output_tokens == 200


def test_cli_backend_call_is_zero_marginal_cost() -> None:
    """A CLI (subscription-lane) response is $0 marginal; the same call via API is priced."""
    configs = {"claude": _make_config("claude", cost_in=10.0, cost_out=30.0)}
    kw = dict(provider="claude", model="m", round_number=1, content="x", latency_sec=0.1,
              token_count=2000, input_tokens=1000, output_tokens=1000)
    cli = build_call_metrics(ModelResponse(backend="cli", **kw), configs, round_number=1)
    api = build_call_metrics(ModelResponse(backend="api", **kw), configs, round_number=1)
    assert cli.estimated_cost_usd == 0.0 and cli.backend == "cli"
    assert api.estimated_cost_usd > 0.0 and api.backend == "api"
    assert cli.input_tokens == 1000  # tokens still recorded for transparency


# --- extra_calls (#18 crux check) ---


def _crux_call(cost: float = 0.05, tokens_in: int = 200, tokens_out: int = 50):
    """A ProviderCallMetrics standing in for the one crux extraction call."""
    return ProviderCallMetrics(
        provider="openai",
        round_number=-1,  # sentinel: 0 is synthesis, 1..n are rounds
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        estimated_cost_usd=cost,
        latency_sec=1.0,
    )


def test_extra_calls_default_none_preserves_existing_totals():
    """The defaulted param must not perturb any existing caller."""
    configs = {"a": _make_config("a", 1.0, 2.0)}
    round1 = Round(
        number=1,
        responses=[ModelResponse("a", "a-model", 1, "x", 1.0, 30, input_tokens=10, output_tokens=20)],
    )
    synthesis_call = _crux_call(cost=0.01)

    without = build_debate_metrics([round1], synthesis_call, configs, total_duration_sec=5.0)
    with_none = build_debate_metrics(
        [round1], synthesis_call, configs, total_duration_sec=5.0, extra_calls=None
    )

    assert without.total_estimated_cost_usd == with_none.total_estimated_cost_usd
    assert without.total_input_tokens == with_none.total_input_tokens
    assert without.total_output_tokens == with_none.total_output_tokens
    assert len(without.calls) == len(with_none.calls)


def test_extra_calls_included_in_totals():
    configs = {"a": _make_config("a", 1.0, 2.0)}
    round1 = Round(
        number=1,
        responses=[ModelResponse("a", "a-model", 1, "x", 1.0, 30, input_tokens=10, output_tokens=20)],
    )
    synthesis_call = _crux_call(cost=0.01, tokens_in=5, tokens_out=5)
    crux = _crux_call(cost=0.05, tokens_in=200, tokens_out=50)

    baseline = build_debate_metrics([round1], synthesis_call, configs, total_duration_sec=5.0)
    with_crux = build_debate_metrics(
        [round1], synthesis_call, configs, total_duration_sec=5.0, extra_calls=[crux]
    )

    assert len(with_crux.calls) == len(baseline.calls) + 1
    assert with_crux.total_input_tokens == baseline.total_input_tokens + 200
    assert with_crux.total_output_tokens == baseline.total_output_tokens + 50
    assert with_crux.total_estimated_cost_usd == baseline.total_estimated_cost_usd + 0.05


def test_extra_calls_appear_in_the_calls_list():
    """The crux call must be individually visible, not just folded into the totals."""
    configs = {"a": _make_config("a", 1.0, 2.0)}
    round1 = Round(number=1, responses=[])
    crux = _crux_call()

    metrics = build_debate_metrics(
        [round1], _crux_call(cost=0.0), configs, total_duration_sec=1.0, extra_calls=[crux]
    )
    assert crux in metrics.calls
    assert any(c.round_number == -1 for c in metrics.calls)


def test_extra_calls_empty_list_is_a_noop():
    configs = {"a": _make_config("a", 1.0, 2.0)}
    round1 = Round(number=1, responses=[])
    synthesis_call = _crux_call(cost=0.01)

    baseline = build_debate_metrics([round1], synthesis_call, configs, total_duration_sec=1.0)
    empty = build_debate_metrics(
        [round1], synthesis_call, configs, total_duration_sec=1.0, extra_calls=[]
    )
    assert len(empty.calls) == len(baseline.calls)
    assert empty.total_estimated_cost_usd == baseline.total_estimated_cost_usd
