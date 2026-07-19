"""Comparison page renderer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from signalglass.charts import make_compare_chart
from signalglass.ui._data import company_name, frame, html, value
from signalglass.ui.shell import render_page_title


def _series(bundle: Any, derived: Any) -> dict[str, pd.DataFrame]:
    candidate = value(derived, "comparison", "series", "price_series")
    if isinstance(candidate, Mapping):
        return {
            str(key): item.copy(deep=True)
            for key, item in candidate.items()
            if isinstance(item, pd.DataFrame)
        }
    ticker = str(value(bundle, "ticker", "symbol", default="AAPL")).upper()
    prices = frame(bundle, "prices", "price_history", "stock_data")
    return {ticker: prices} if not prices.empty else {}


def render_compare(bundle: Any, derived: Any = None) -> None:
    """Render relative performance and a compact comparison table."""

    render_page_title("Compare", "Put price performance and market context on the same baseline.")
    series = _series(bundle, derived)
    source_map = value(derived, "comparison_sources", default={})
    if not isinstance(source_map, Mapping):
        source_map = {}
    unique_sources = {str(source_map.get(symbol, "Unspecified source")) for symbol in series}
    source_label = unique_sources.pop() if len(unique_sources) == 1 else "Mixed price sources"
    with st.container(border=True):
        st.markdown(
            f'<div class="sg-section-heading"><h2>Relative performance</h2><span class="sg-muted">{html(source_label)} · Rebased to 0%</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_compare_chart(series), width="stretch", config={"displayModeBar": False})

    rows: list[str] = []
    for symbol, prices in series.items():
        close_key = "Close" if "Close" in prices else "close" if "close" in prices else None
        close = (
            pd.to_numeric(prices[close_key], errors="coerce").dropna()
            if close_key
            else pd.Series(dtype=float)
        )
        latest = float(close.iloc[-1]) if not close.empty else 0
        period = (latest / float(close.iloc[0]) - 1) * 100 if len(close) > 1 and close.iloc[0] else 0
        daily = (latest / float(close.iloc[-2]) - 1) * 100 if len(close) > 1 and close.iloc[-2] else 0
        rows.append(
            f'<div class="sg-list-row"><div><div class="sg-list-title">{html(symbol)}</div><div class="sg-list-subtitle">{html(company_name(symbol))} · {html(source_map.get(symbol, "Unspecified source"))}</div></div><span>${latest:,.2f}</span><span class="{"sg-positive" if daily >= 0 else "sg-negative"}">{daily:+.2f}% · {period:+.1f}% period</span></div>'
        )
    with st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Snapshot</h2><span class="sg-muted">Last price · Daily move · Period return</span></div>'
            + ("".join(rows) or '<p class="sg-muted">No comparable series available.</p>'),
            unsafe_allow_html=True,
        )


__all__ = ["render_compare"]
