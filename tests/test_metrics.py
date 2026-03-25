"""Tests for src/metrics.py — cost computation and metric aggregation."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from config.config_loader import ModelConfig
from src.metrics import build_call_metrics, build_debate_metrics, compute_call_cost
from src.models import DebateMetrics, ModelResponse, ProviderCallMetrics, Round


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
