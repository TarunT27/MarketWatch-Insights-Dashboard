"""Pure multi-asset portfolio analytics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .validation import validate_ticker


@dataclass(frozen=True, slots=True)
class PortfolioAnalysis:
    weights: dict[str, float]
    timeline: pd.DataFrame
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    risk_contributions: dict[str, float]
    observation_count: int


def normalize_allocations(allocations: Mapping[object, object]) -> dict[str, float]:
    if not isinstance(allocations, Mapping) or not allocations:
        raise ValueError("allocations must contain at least one positive weight")
    normalized: dict[str, float] = {}
    for raw_symbol, raw_weight in allocations.items():
        symbol = validate_ticker(raw_symbol)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as error:
            raise ValueError(f"allocation for {symbol} must be numeric") from error
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"allocation for {symbol} must be finite and non-negative")
        normalized[symbol] = normalized.get(symbol, 0.0) + weight
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("allocations must contain at least one positive weight")
    return {symbol: weight / total for symbol, weight in normalized.items() if weight > 0}


def _return_series(frame: pd.DataFrame, symbol: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or not {"date", "Close"}.issubset(frame.columns):
        raise ValueError(f"price history for {symbol} must contain date and Close columns")
    values = frame.loc[:, ["date", "Close"]].copy(deep=True)
    values["date"] = pd.to_datetime(values["date"], errors="coerce", utc=True).dt.tz_convert(None)
    values["Close"] = pd.to_numeric(values["Close"], errors="coerce")
    values = values.dropna().sort_values("date").drop_duplicates("date", keep="last")
    if len(values) < 2 or values["Close"].le(0).any():
        raise ValueError(f"price history for {symbol} must contain at least two positive closes")
    series = values.set_index("date")["Close"].pct_change(fill_method=None).dropna()
    series.name = symbol
    return series


def analyze_portfolio(
    prices: Mapping[str, pd.DataFrame],
    allocations: Mapping[object, object],
    *,
    annualization: int = 252,
) -> PortfolioAnalysis:
    """Calculate return, drawdown, volatility, Sharpe, and risk contribution."""

    if not isinstance(prices, Mapping):
        raise TypeError("prices must map symbols to DataFrames")
    weights = normalize_allocations(allocations)
    available = {validate_ticker(symbol): frame for symbol, frame in prices.items()}
    missing = sorted(set(weights).difference(available))
    if missing:
        raise ValueError(f"allocations reference unavailable symbols: {', '.join(missing)}")
    if not isinstance(annualization, int) or annualization < 1:
        raise ValueError("annualization must be a positive integer")

    returns = pd.concat([_return_series(available[symbol], symbol) for symbol in weights], axis=1).dropna()
    if returns.empty:
        raise ValueError("portfolio assets do not share enough overlapping price history")
    weight_vector = np.array([weights[column] for column in returns.columns], dtype="float64")
    portfolio_returns = returns.to_numpy(dtype="float64") @ weight_vector
    timeline = pd.DataFrame({"date": returns.index, "portfolio_return": portfolio_returns})
    timeline["portfolio_equity"] = (1.0 + timeline["portfolio_return"]).cumprod()

    daily_std = float(timeline["portfolio_return"].std(ddof=1))
    total_return = float(timeline["portfolio_equity"].iloc[-1] - 1.0)
    annualized_return = float(
        max(1.0 + total_return, np.finfo(float).eps) ** (annualization / len(timeline)) - 1.0
    )
    volatility = daily_std * np.sqrt(annualization) if np.isfinite(daily_std) else 0.0
    sharpe = (
        float(timeline["portfolio_return"].mean() / daily_std * np.sqrt(annualization))
        if daily_std > 0 and np.isfinite(daily_std)
        else 0.0
    )
    drawdown = timeline["portfolio_equity"].div(timeline["portfolio_equity"].cummax()).sub(1.0)

    covariance = returns.cov().to_numpy(dtype="float64")
    marginal = covariance @ weight_vector
    variance = float(weight_vector @ marginal)
    if variance > 0 and np.isfinite(variance):
        contributions = weight_vector * marginal / variance
    else:
        contributions = weight_vector.copy()
    risk_contributions = {
        symbol: float(value) for symbol, value in zip(returns.columns, contributions, strict=True)
    }
    return PortfolioAnalysis(
        weights=weights,
        timeline=timeline.reset_index(drop=True),
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=float(volatility),
        sharpe_ratio=sharpe,
        max_drawdown=float(drawdown.min()),
        risk_contributions=risk_contributions,
        observation_count=len(timeline),
    )


__all__ = ["PortfolioAnalysis", "analyze_portfolio", "normalize_allocations"]
