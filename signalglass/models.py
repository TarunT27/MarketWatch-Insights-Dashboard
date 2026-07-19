"""Immutable result containers shared by SignalGlass core modules."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class MarketBundle:
    """A normalized market snapshot and its provenance.

    The dataclass is frozen so callers cannot replace its members. DataFrames are
    copied at provider boundaries because pandas objects are themselves mutable.
    """

    ticker: str
    prices: pd.DataFrame
    news: pd.DataFrame
    is_demo: bool
    source: str = "demo"
    notice: str | None = None
    price_source: str = "SignalGlass deterministic demo"
    news_source: str = "SignalGlass deterministic demo"


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    """Out-of-sample results from a chronological walk-forward evaluation."""

    predictions: pd.DataFrame
    mean_absolute_error: float
    directional_accuracy: float
    sample_size: int
    latest_predicted_return: float | None = None

    @property
    def predicted_direction(self) -> str | None:
        """Return the direction of the most recent out-of-sample prediction."""

        if self.latest_predicted_return is None:
            return None
        return "Up" if self.latest_predicted_return >= 0 else "Down"


__all__ = ["MarketBundle", "SignalEvaluation"]
