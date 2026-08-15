"""Backtesting and research-to-execution export contracts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from signalglass.backtesting import BacktestConfig, run_backtest
from signalglass.signal_export import build_execution_signal_frame, serialize_execution_signals


@pytest.fixture
def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=5),
            "actual_return": [0.02, -0.01, 0.03, -0.02, 0.01],
            "predicted_return": [0.01, -0.02, 0.02, -0.01, 0.03],
        }
    )


def test_backtest_is_deterministic_cost_aware_and_does_not_mutate_input(predictions) -> None:
    original = predictions.copy(deep=True)

    free = run_backtest(predictions, config=BacktestConfig(transaction_cost_bps=0))
    costed = run_backtest(predictions, config=BacktestConfig(transaction_cost_bps=25))

    pd.testing.assert_frame_equal(predictions, original)
    assert list(costed.timeline.columns) == [
        "date",
        "actual_return",
        "predicted_return",
        "signal",
        "turnover",
        "transaction_cost",
        "strategy_return",
        "strategy_equity",
        "benchmark_equity",
    ]
    assert costed.metrics.total_return < free.metrics.total_return
    assert costed.metrics.trade_count == 5
    assert costed.timeline["signal"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert costed.timeline["benchmark_equity"].iloc[-1] == pytest.approx(
        (1 + predictions["actual_return"]).prod()
    )


def test_backtest_reports_reproducible_risk_metrics(predictions) -> None:
    result = run_backtest(
        predictions,
        config=BacktestConfig(transaction_cost_bps=5, allow_short=True),
    )

    assert result.metrics.observation_count == len(predictions)
    assert result.metrics.max_drawdown <= 0
    assert 0 <= result.metrics.win_rate <= 1
    assert result.metrics.annualized_volatility >= 0
    assert result.metrics.total_turnover > 0
    assert result.metrics.benchmark_return == pytest.approx(result.timeline["benchmark_equity"].iloc[-1] - 1)


def test_execution_export_excludes_realized_returns_and_uses_stable_schema(predictions) -> None:
    frame = build_execution_signal_frame(predictions, ticker="aapl", model_name="ridge")
    payload = json.loads(serialize_execution_signals(frame))

    assert list(frame.columns) == ["date", "symbol", "model", "action", "score"]
    assert "actual_return" not in frame.columns
    assert frame["symbol"].unique().tolist() == ["AAPL"]
    assert frame["action"].tolist() == ["BUY", "SELL", "BUY", "SELL", "BUY"]
    assert payload["schema_version"] == "signalglass.execution.v1"
    assert len(payload["signals"]) == len(predictions)
    assert "actual_return" not in payload["signals"][0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"transaction_cost_bps": -1}, "transaction"),
        ({"signal_threshold": -0.1}, "threshold"),
    ],
)
def test_backtest_config_rejects_unsafe_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        BacktestConfig(**kwargs)
