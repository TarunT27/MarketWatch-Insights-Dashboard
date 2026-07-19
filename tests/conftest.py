"""Shared fixtures for SignalGlass contract tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd
import pytest


@pytest.fixture
def price_frame_factory() -> Callable[[int], pd.DataFrame]:
    """Build a small, deterministic business-day price history."""

    def build(periods: int = 12) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-02", periods=periods)
        closes = pd.Series(
            [100.0 + (index * 0.8) + ((-1) ** index) * 0.35 for index in range(periods)],
            dtype="float64",
        )
        return pd.DataFrame(
            {
                "date": dates,
                "Open": closes - 0.25,
                "High": closes + 1.0,
                "Low": closes - 1.0,
                "Close": closes,
                "Volume": [1_000_000 + index * 10_000 for index in range(periods)],
                "ticker": "AAPL",
            }
        )

    return build


@pytest.fixture
def combined_market_frame(price_frame_factory: Callable[[int], pd.DataFrame]) -> pd.DataFrame:
    """Market sessions with enough varying sentiment for signal evaluation."""

    frame = price_frame_factory(14)
    frame = frame.assign(
        avg_sentiment=[-0.25, 0.15, 0.3, -0.1, 0.05, 0.4, -0.2, 0.1, 0.35, -0.3, 0.2, 0.0, 0.45, -0.05],
        positive_count=[0, 2, 3, 0, 1, 4, 0, 2, 3, 0, 2, 1, 4, 0],
        negative_count=[2, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 1, 0, 1],
        neutral_count=[1, 1, 0, 1, 2, 1, 0, 1, 1, 0, 1, 2, 1, 1],
        headline_count=[3, 3, 3, 2, 3, 5, 2, 3, 4, 3, 3, 4, 5, 2],
    )
    return frame


@pytest.fixture
def valid_range() -> tuple[date, date]:
    return date(2025, 1, 2), date(2025, 1, 31)
