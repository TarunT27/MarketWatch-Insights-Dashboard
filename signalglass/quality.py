"""Measured data-quality scoring for a loaded market window.

The product previously displayed a constant. This module computes the number
from the data actually in hand so the figure can be defended: it is the mean of
four observable components, each reported alongside the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

_PRICE_FIELDS = ("Open", "High", "Low", "Close", "Volume")
# Sessions older than this many days make the window stale rather than current.
_FRESHNESS_TOLERANCE_DAYS = 5


@dataclass(frozen=True, slots=True)
class DataQuality:
    """A defensible completeness score and the components behind it."""

    score: float
    session_coverage: float
    field_completeness: float
    news_coverage: float
    freshness: float
    session_count: int
    expected_sessions: int

    @property
    def components(self) -> dict[str, float]:
        return {
            "Session coverage": self.session_coverage,
            "Field completeness": self.field_completeness,
            "News coverage": self.news_coverage,
            "Freshness": self.freshness,
        }

    @property
    def summary(self) -> str:
        """One line explaining what the headline percentage is made of."""

        # Symbols that trade every day (crypto, some indices) report more
        # sessions than the business-day calendar predicts, which is complete
        # coverage rather than a shortfall.
        if self.session_count >= self.expected_sessions > 0:
            coverage = f"{self.session_count} sessions, no gaps"
        else:
            coverage = f"{self.session_count} of {self.expected_sessions} expected sessions"
        return (
            f"{coverage} · fields {self.field_completeness:.0%} · "
            f"news {self.news_coverage:.0%} · freshness {self.freshness:.0%}"
        )


def _session_coverage(prices: pd.DataFrame) -> tuple[float, int, int]:
    dates = pd.to_datetime(prices["date"], errors="coerce").dropna()
    if dates.empty:
        return 0.0, 0, 0
    expected = len(pd.bdate_range(start=dates.min(), end=dates.max()))
    observed = int(dates.dt.normalize().nunique())
    if expected <= 0:
        return 0.0, observed, 0
    # Holidays mean observed can never quite reach the business-day count.
    return float(min(1.0, observed / expected)), observed, expected


def _field_completeness(prices: pd.DataFrame) -> float:
    present = [field for field in _PRICE_FIELDS if field in prices.columns]
    if not present or prices.empty:
        return 0.0
    values = prices.loc[:, present].apply(pd.to_numeric, errors="coerce")
    total = values.size
    return float(values.notna().to_numpy().sum() / total) if total else 0.0


def _news_coverage(prices: pd.DataFrame, news: pd.DataFrame) -> float:
    if news is None or not isinstance(news, pd.DataFrame) or news.empty:
        return 0.0
    if "publishedAt" not in news.columns:
        return 0.0
    sessions = pd.to_datetime(prices["date"], errors="coerce").dt.normalize().dropna().unique()
    if len(sessions) == 0:
        return 0.0
    published = pd.to_datetime(news["publishedAt"], errors="coerce").dropna()
    if published.empty:
        return 0.0
    covered = published.dt.normalize().isin(sessions).sum()
    # Coverage saturates once roughly a third of sessions carry a headline;
    # daily wire coverage of a single mid-cap name is not a realistic bar.
    density = float(published.dt.normalize().nunique() / max(1, len(sessions)))
    attach_rate = float(covered / len(published))
    return float(np.clip(0.5 * attach_rate + 0.5 * min(1.0, density / 0.33), 0.0, 1.0))


def _freshness(prices: pd.DataFrame, *, today: date | None = None) -> float:
    dates = pd.to_datetime(prices["date"], errors="coerce").dropna()
    if dates.empty:
        return 0.0
    reference = pd.Timestamp(today or date.today()).normalize()
    latest = dates.max().normalize()
    lag_sessions = len(pd.bdate_range(start=latest, end=reference)) - 1 if latest <= reference else 0
    if lag_sessions <= 1:
        return 1.0
    return float(max(0.0, 1.0 - (lag_sessions - 1) / _FRESHNESS_TOLERANCE_DAYS))


def assess_data_quality(
    prices: pd.DataFrame,
    news: pd.DataFrame | None = None,
    *,
    today: date | None = None,
) -> DataQuality:
    """Score the loaded window on coverage, completeness, news, and freshness."""

    if not isinstance(prices, pd.DataFrame) or prices.empty or "date" not in prices.columns:
        return DataQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    coverage, observed, expected = _session_coverage(prices)
    completeness = _field_completeness(prices)
    news_score = _news_coverage(prices, news if isinstance(news, pd.DataFrame) else pd.DataFrame())
    freshness = _freshness(prices, today=today)
    score = float(np.mean([coverage, completeness, news_score, freshness]) * 100)
    return DataQuality(
        score=score,
        session_coverage=coverage,
        field_completeness=completeness,
        news_coverage=news_score,
        freshness=freshness,
        session_count=observed,
        expected_sessions=expected,
    )


__all__ = ["DataQuality", "assess_data_quality"]
