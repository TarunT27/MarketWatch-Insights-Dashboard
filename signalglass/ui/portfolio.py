"""Persistent watchlist and portfolio-risk workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from signalglass.ui._data import frame, value
from signalglass.ui.shell import render_page_title


@dataclass(frozen=True, slots=True)
class PortfolioRequest:
    watchlist: tuple[str, ...]
    allocations: dict[str, float]
    save_requested: bool = False


def _percent(raw: Any) -> float:
    try:
        return float(raw) * 100
    except (TypeError, ValueError):
        return 0.0


def render_portfolio(
    bundle: Any,
    derived: Any = None,
    *,
    saved_watchlist: tuple[str, ...] = (),
    saved_allocations: dict[str, float] | None = None,
) -> PortfolioRequest:
    """Render portfolio controls and return an explicit persistence request."""

    render_page_title("Portfolio risk", "Save a research list and inspect diversified risk before acting.")
    comparison = value(derived, "comparison", default={})
    available = tuple(str(symbol) for symbol in comparison) or (str(value(bundle, "ticker", default="AAPL")),)
    default_watchlist = tuple(symbol for symbol in saved_watchlist if symbol in available) or available
    selected = tuple(
        st.multiselect(
            "Research watchlist",
            available,
            default=default_watchlist,
            help="Use the global symbol search to bring another asset into this workspace.",
        )
    )
    if not selected:
        st.info("Select at least one symbol to calculate portfolio risk.")
        return PortfolioRequest((), {}, False)

    existing = saved_allocations or {}
    allocation_columns = st.columns(len(selected))
    raw_allocations: dict[str, float] = {}
    equal_weight = 1.0 / len(selected)
    for column, symbol in zip(allocation_columns, selected, strict=True):
        with column:
            raw_allocations[symbol] = float(
                st.number_input(
                    f"{symbol} allocation (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(existing.get(symbol, equal_weight) * 100),
                    step=5.0,
                    key=f"sg_allocation_{symbol}",
                )
            )
    save_requested = st.button("Save research workspace", type="primary")
    total = sum(raw_allocations.values())
    allocations = (
        {symbol: weight / total for symbol, weight in raw_allocations.items() if weight > 0}
        if total > 0
        else {}
    )

    analysis = value(derived, "portfolio_analysis", default=None)
    if analysis is not None:
        st.markdown(
            f'<section class="sg-scorebar"><div class="sg-score"><div class="sg-score-value">{_percent(value(analysis, "total_return", default=0)):+.1f}%</div><div class="sg-score-label">Total return</div></div><div class="sg-score"><div class="sg-score-value">{_percent(value(analysis, "annualized_volatility", default=0)):.1f}%</div><div class="sg-score-label">Annualized volatility</div></div><div class="sg-score"><div class="sg-score-value">{float(value(analysis, "sharpe_ratio", default=0)):.2f}</div><div class="sg-score-label">Sharpe ratio</div></div><div class="sg-score"><div class="sg-score-value">{_percent(value(analysis, "max_drawdown", default=0)):.1f}%</div><div class="sg-score-label">Maximum drawdown</div></div></section>',
            unsafe_allow_html=True,
        )
        timeline = frame(analysis, "timeline")
        if not timeline.empty:
            chart_data = timeline.set_index("date")[["portfolio_equity"]].sub(1.0).mul(100)
            st.line_chart(chart_data, y_label="Cumulative return (%)")
        contributions = value(analysis, "risk_contributions", default={})
        if isinstance(contributions, dict) and contributions:
            contribution_frame = pd.DataFrame(
                {
                    "Symbol": list(contributions),
                    "Allocation": [allocations.get(symbol, 0) for symbol in contributions],
                    "Risk contribution": list(contributions.values()),
                }
            )
            st.dataframe(
                contribution_frame.style.format({"Allocation": "{:.1%}", "Risk contribution": "{:.1%}"}),
                hide_index=True,
                width="stretch",
            )
    st.caption("Local SQLite persistence stores symbols and weights only—never brokerage credentials.")
    return PortfolioRequest(selected, allocations, save_requested)


__all__ = ["PortfolioRequest", "render_portfolio"]
