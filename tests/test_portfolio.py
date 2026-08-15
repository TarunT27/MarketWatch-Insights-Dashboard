"""Portfolio analytics and local persistence contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from signalglass.portfolio import analyze_portfolio
from signalglass.store import LocalResearchStore


def _prices(start: float, returns: list[float]) -> pd.DataFrame:
    values = [start]
    for daily_return in returns:
        values.append(values[-1] * (1 + daily_return))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=len(values)),
            "Close": values,
        }
    )


def test_portfolio_analysis_normalizes_weights_and_reports_risk() -> None:
    prices = {
        "AAPL": _prices(100, [0.01, -0.01, 0.02, 0.005]),
        "MSFT": _prices(200, [0.005, 0.01, -0.005, 0.015]),
    }

    result = analyze_portfolio(prices, {"AAPL": 70, "MSFT": 30})

    assert result.weights == {"AAPL": pytest.approx(0.7), "MSFT": pytest.approx(0.3)}
    assert list(result.timeline.columns) == ["date", "portfolio_return", "portfolio_equity"]
    assert result.observation_count == 4
    assert result.annualized_volatility >= 0
    assert result.max_drawdown <= 0
    assert set(result.risk_contributions) == {"AAPL", "MSFT"}
    assert sum(result.risk_contributions.values()) == pytest.approx(1.0)


def test_portfolio_analysis_rejects_missing_or_invalid_allocations() -> None:
    prices = {"AAPL": _prices(100, [0.01, 0.02])}

    with pytest.raises(ValueError, match="positive"):
        analyze_portfolio(prices, {"AAPL": 0})
    with pytest.raises(ValueError, match="available"):
        analyze_portfolio(prices, {"MSFT": 1})


def test_local_research_store_persists_validated_watchlists_and_allocations(tmp_path) -> None:
    store = LocalResearchStore(tmp_path / "signalglass.db")

    store.replace_watchlist(["aapl", "MSFT", "AAPL"])
    store.replace_allocations({"AAPL": 70, "MSFT": 30})

    reopened = LocalResearchStore(tmp_path / "signalglass.db")
    assert reopened.load_watchlist() == ("AAPL", "MSFT")
    assert reopened.load_allocations() == {"AAPL": pytest.approx(0.7), "MSFT": pytest.approx(0.3)}


def test_local_research_store_rejects_unsafe_symbols(tmp_path) -> None:
    store = LocalResearchStore(tmp_path / "signalglass.db")

    with pytest.raises(ValueError):
        store.replace_watchlist(["AAPL<script>"])
