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
    company: str = ""


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    """Out-of-sample results from a chronological walk-forward evaluation."""

    predictions: pd.DataFrame
    mean_absolute_error: float
    directional_accuracy: float
    sample_size: int
    latest_predicted_return: float | None = None
    model_name: str = "linear"
    accuracy_low: float = 0.0
    accuracy_high: float = 1.0
    majority_baseline: float = 0.5

    @property
    def predicted_direction(self) -> str | None:
        """Return the direction of the most recent out-of-sample prediction."""

        if self.latest_predicted_return is None:
            return None
        return "Up" if self.latest_predicted_return >= 0 else "Down"

    @property
    def beats_coin_flip(self) -> bool:
        """True when the 95% interval excludes chance, so the edge is real."""

        return self.accuracy_low > 0.5

    @property
    def beats_baseline(self) -> bool:
        """True when the model outperforms always predicting the common direction."""

        return self.directional_accuracy > self.majority_baseline

    @property
    def verdict(self) -> str:
        """Plain-language reading of whether this result means anything."""

        if not self.beats_baseline:
            return "Below baseline"
        if not self.beats_coin_flip:
            return "Not significant"
        return "Significant"


@dataclass(frozen=True, slots=True)
class ModelSuite:
    """Comparable walk-forward evaluations ranked by out-of-sample evidence."""

    evaluations: tuple[SignalEvaluation, ...]
    leaderboard: pd.DataFrame
    best_model: str
    selection_is_separable: bool = True

    @property
    def best_evaluation(self) -> SignalEvaluation:
        return next(item for item in self.evaluations if item.model_name == self.best_model)


__all__ = ["MarketBundle", "ModelSuite", "SignalEvaluation"]
