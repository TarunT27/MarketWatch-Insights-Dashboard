"""Leakage-safe strategy backtesting with explicit trading frictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Immutable assumptions used to translate predictions into positions."""

    transaction_cost_bps: float = 5.0
    signal_threshold: float = 0.0
    allow_short: bool = False
    annualization: int = 252

    def __post_init__(self) -> None:
        if not np.isfinite(self.transaction_cost_bps) or self.transaction_cost_bps < 0:
            raise ValueError("transaction cost must be a finite, non-negative number of basis points")
        if not np.isfinite(self.signal_threshold) or self.signal_threshold < 0:
            raise ValueError("signal threshold must be finite and non-negative")
        if not isinstance(self.annualization, int) or isinstance(self.annualization, bool):
            raise ValueError("annualization must be a positive integer")
        if self.annualization < 1:
            raise ValueError("annualization must be a positive integer")


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Reproducible performance and risk statistics."""

    total_return: float
    benchmark_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_turnover: float
    trade_count: int
    observation_count: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Backtest timeline and its aggregate metrics."""

    timeline: pd.DataFrame
    metrics: BacktestMetrics
    config: BacktestConfig


_REQUIRED_COLUMNS = ("date", "actual_return", "predicted_return")


def _validated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    missing = sorted(set(_REQUIRED_COLUMNS).difference(predictions.columns))
    if missing:
        raise ValueError(f"prediction schema is missing required columns: {', '.join(missing)}")

    frame = predictions.loc[:, _REQUIRED_COLUMNS].copy(deep=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
    for column in ("actual_return", "predicted_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("predictions must contain at least one complete observation")
    values = frame[["actual_return", "predicted_return"]].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("predictions must contain only finite returns")
    if frame["actual_return"].le(-1).any():
        raise ValueError("actual returns cannot be less than or equal to -100%")
    return frame


def _annualized_return(total_return: float, observations: int, annualization: int) -> float:
    ending_value = max(1.0 + total_return, np.finfo(float).eps)
    return float(ending_value ** (annualization / observations) - 1.0)


def run_backtest(
    predictions: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest signals that were produced before each realized test return."""

    assumptions = config or BacktestConfig()
    frame = _validated_predictions(predictions)
    raw_signal = np.where(
        frame["predicted_return"] > assumptions.signal_threshold,
        1.0,
        np.where(frame["predicted_return"] < -assumptions.signal_threshold, -1.0, 0.0),
    )
    if not assumptions.allow_short:
        raw_signal = np.maximum(raw_signal, 0.0)
    frame["signal"] = raw_signal
    previous = frame["signal"].shift(1, fill_value=0.0)
    frame["turnover"] = (frame["signal"] - previous).abs()
    frame["transaction_cost"] = frame["turnover"] * assumptions.transaction_cost_bps / 10_000.0
    frame["strategy_return"] = (frame["signal"] * frame["actual_return"] - frame["transaction_cost"]).clip(
        lower=-0.999999
    )
    frame["strategy_equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["actual_return"]).cumprod()

    strategy_returns = frame["strategy_return"]
    total_return = float(frame["strategy_equity"].iloc[-1] - 1.0)
    volatility = float(strategy_returns.std(ddof=1) * np.sqrt(assumptions.annualization))
    daily_std = float(strategy_returns.std(ddof=1))
    sharpe = (
        float(strategy_returns.mean() / daily_std * np.sqrt(assumptions.annualization))
        if daily_std > 0 and np.isfinite(daily_std)
        else 0.0
    )
    drawdown = frame["strategy_equity"].div(frame["strategy_equity"].cummax()).sub(1.0)
    active = frame.loc[frame["signal"].ne(0), "strategy_return"]
    win_rate = float(active.gt(0).mean()) if not active.empty else 0.0
    metrics = BacktestMetrics(
        total_return=total_return,
        benchmark_return=float(frame["benchmark_equity"].iloc[-1] - 1.0),
        annualized_return=_annualized_return(total_return, len(frame), assumptions.annualization),
        annualized_volatility=volatility if np.isfinite(volatility) else 0.0,
        sharpe_ratio=sharpe,
        max_drawdown=float(drawdown.min()),
        win_rate=win_rate,
        total_turnover=float(frame["turnover"].sum()),
        trade_count=int(frame["turnover"].gt(0).sum()),
        observation_count=len(frame),
    )
    return BacktestResult(timeline=frame, metrics=metrics, config=assumptions)


__all__ = ["BacktestConfig", "BacktestMetrics", "BacktestResult", "run_backtest"]
