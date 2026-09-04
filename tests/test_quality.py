"""Data-quality scoring must be measured, never asserted."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from signalglass.demo_data import generate_demo_news, generate_demo_prices
from signalglass.quality import assess_data_quality


def _prices(sessions: int, end: date = date(2026, 7, 17)) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=sessions)
    return pd.DataFrame(
        {
            "date": dates,
            "Open": np.linspace(100, 110, sessions),
            "High": np.linspace(101, 111, sessions),
            "Low": np.linspace(99, 109, sessions),
            "Close": np.linspace(100, 110, sessions),
            "Volume": np.full(sessions, 1_000_000, dtype="int64"),
            "ticker": "TEST",
        }
    )


def test_empty_prices_score_zero_rather_than_a_flattering_constant() -> None:
    quality = assess_data_quality(pd.DataFrame())

    assert quality.score == 0.0
    assert quality.session_count == 0


def test_complete_recent_window_with_news_scores_highly() -> None:
    end = date(2026, 7, 17)
    quality = assess_data_quality(
        generate_demo_prices("AAPL", 90, end_date=end),
        generate_demo_news("AAPL", 90, end_date=end),
        today=end,
    )

    assert quality.score > 80
    assert quality.session_coverage == 1.0
    assert quality.freshness == 1.0


def test_missing_values_reduce_field_completeness() -> None:
    prices = _prices(40)
    prices.loc[0:9, "Close"] = np.nan

    quality = assess_data_quality(prices, today=date(2026, 7, 17))

    assert quality.field_completeness < 1.0


def test_a_stale_window_loses_freshness() -> None:
    quality = assess_data_quality(_prices(40, end=date(2026, 5, 1)), today=date(2026, 7, 17))

    assert quality.freshness == 0.0
    assert quality.score < 60


def test_absent_news_lowers_the_score_but_keeps_prices_usable() -> None:
    end = date(2026, 7, 17)
    with_news = assess_data_quality(
        generate_demo_prices("AAPL", 90, end_date=end),
        generate_demo_news("AAPL", 90, end_date=end),
        today=end,
    )
    without_news = assess_data_quality(generate_demo_prices("AAPL", 90, end_date=end), today=end)

    assert without_news.news_coverage == 0.0
    assert without_news.score < with_news.score
    assert without_news.session_coverage == 1.0


def test_summary_reports_the_components_behind_the_score() -> None:
    end = date(2026, 7, 17)
    quality = assess_data_quality(generate_demo_prices("AAPL", 30, end_date=end), today=end)

    assert "sessions" in quality.summary
    assert "fields" in quality.summary
    assert set(quality.components) == {
        "Session coverage",
        "Field completeness",
        "News coverage",
        "Freshness",
    }


def test_seven_day_symbols_are_not_described_as_missing_sessions() -> None:
    """Crypto trades daily, so exceeding the business-day count is not a gap."""

    end = date(2026, 7, 17)
    daily = pd.DataFrame(
        {
            "date": pd.date_range(end=pd.Timestamp(end), periods=120, freq="D"),
            "Open": np.linspace(100, 110, 120),
            "High": np.linspace(101, 111, 120),
            "Low": np.linspace(99, 109, 120),
            "Close": np.linspace(100, 110, 120),
            "Volume": np.full(120, 1_000, dtype="int64"),
            "ticker": "BTC-USD",
        }
    )

    quality = assess_data_quality(daily, today=end)

    assert quality.session_coverage == 1.0
    assert "no gaps" in quality.summary
    assert "of 0 expected" not in quality.summary
