"""Overview page renderer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from signalglass.charts import make_overview_chart
from signalglass.ui._data import company_name, format_compact, format_date, frame, html, records, value


def _snapshot(bundle: Any, derived: Any) -> dict[str, Any]:
    prices = frame(derived, "market", "combined", "combined_data", "market_frame")
    if prices.empty:
        prices = frame(bundle, "prices", "price_history", "stock_data")
    ticker = str(value(bundle, "ticker", "symbol", default=value(derived, "ticker", default="AAPL"))).upper()
    close = pd.to_numeric(
        prices.get("Close", prices.get("close", pd.Series(dtype=float))), errors="coerce"
    ).dropna()
    volume = pd.to_numeric(
        prices.get("Volume", prices.get("volume", pd.Series(dtype=float))), errors="coerce"
    ).dropna()
    latest = float(close.iloc[-1]) if not close.empty else 0.0
    previous = float(close.iloc[-2]) if len(close) > 1 else latest
    change = latest - previous
    percent = change / previous * 100 if previous else 0.0
    returns = close.pct_change(fill_method=None).dropna()
    volatility = float(returns.std() * np.sqrt(252) * 100) if not returns.empty else 0.0

    news = frame(bundle, "news", "articles", "news_data")
    score = pd.to_numeric(news.get("sentiment_score", pd.Series(dtype=float)), errors="coerce").dropna()
    pulse = (
        float(score.tail(8).mean())
        if not score.empty
        else float(
            pd.to_numeric(prices.get("avg_sentiment", pd.Series([0.0])), errors="coerce")
            .fillna(0)
            .tail(8)
            .mean()
        )
    )
    quality = float(value(derived, "data_quality", "quality", default=96 if not prices.empty else 0))
    return {
        "ticker": ticker,
        "company": str(value(derived, "company", "company_name", default=company_name(ticker))),
        "prices": prices,
        "news": news,
        "latest": latest,
        "change": change,
        "percent": percent,
        "volume": float(volume.iloc[-1]) if not volume.empty else 0,
        "volatility": volatility,
        "pulse": pulse,
        "quality": quality,
    }


def _hero(snapshot: dict[str, Any]) -> None:
    direction_class = "sg-positive" if snapshot["change"] >= 0 else "sg-negative"
    st.markdown(
        f"""
        <section class="sg-hero" aria-label="Market snapshot">
          <div>
            <div class="sg-symbol-line"><h1 class="sg-symbol">{html(snapshot["ticker"])}</h1><span class="sg-company">{html(snapshot["company"])}</span><span class="sg-company" aria-label="Watchlist">☆</span></div>
            <div class="sg-price">${snapshot["latest"]:,.2f}</div>
            <div class="sg-change {direction_class}">{snapshot["change"]:+.2f}&nbsp;&nbsp;{snapshot["percent"]:+.2f}%</div>
          </div>
          <div class="sg-period" aria-label="Time range"><span>1M</span><span class="is-active">3M</span><span>6M</span><span>1Y</span></div>
        </section>
        <section class="sg-metrics" aria-label="Key market metrics">
          <div class="sg-metric"><div class="sg-metric-label">Volume</div><div class="sg-metric-value">{format_compact(snapshot["volume"])}</div></div>
          <div class="sg-metric"><div class="sg-metric-label">Volatility</div><div class="sg-metric-value">{snapshot["volatility"]:.1f}%</div></div>
          <div class="sg-metric"><div class="sg-metric-label">News pulse</div><div class="sg-metric-value {"sg-positive" if snapshot["pulse"] >= 0 else "sg-negative"}">{snapshot["pulse"]:+.2f}</div></div>
          <div class="sg-metric"><div class="sg-metric-label">Data quality</div><div class="sg-metric-value">{snapshot["quality"]:.0f}%</div></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _article_rows(items: list[dict[str, Any]], *, compact: bool = False) -> str:
    if not items:
        return '<p class="sg-muted">No evidence is available for this range.</p>'
    rendered: list[str] = []
    for item in items:
        source = item.get("source", "SignalGlass")
        if isinstance(source, dict):
            source = source.get("name", "SignalGlass")
        published = item.get("publishedAt", item.get("date", ""))
        title = item.get("title", "Market update")
        if compact:
            label = str(item.get("sentiment_label", "Neutral")).title()
            rendered.append(
                f'<div class="sg-list-row"><div><div class="sg-list-title">{html(title)}</div><div class="sg-list-subtitle">{html(source)} · {html(format_date(published))}</div></div><span class="sg-muted">{html(item.get("description", "")[:72])}</span><span class="sg-tag {"neutral" if label == "Neutral" else ""}">{html(label)}</span></div>'
            )
        else:
            initial = str(source)[:1].upper()
            icon_class = "reuters" if "reuters" in str(source).lower() else ""
            rendered.append(
                f'<article class="sg-evidence"><div class="sg-source-icon {icon_class}" aria-hidden="true">{html(initial)}</div><div><div class="sg-evidence-source">{html(source)}</div><div class="sg-evidence-title">{html(title)}</div><span class="sg-link">View story&nbsp; ›</span></div><time class="sg-evidence-date">{html(format_date(published))}</time></article>'
            )
    return "".join(rendered)


def render_overview(bundle: Any, derived: Any = None) -> None:
    """Render the network-free overview using a ``MarketBundle`` and derived data."""

    snapshot = _snapshot(bundle, derived)
    _hero(snapshot)

    evidence = records(bundle, "news", "articles", "news_data", limit=3)
    explanation = value(derived, "why_it_moved", "movement_summary", "insight")
    if isinstance(explanation, dict):
        explanation = explanation.get("summary") or explanation.get("headline")
    if not explanation:
        direction = "positive" if snapshot["percent"] >= 0 else "negative"
        explanation = f"Price action and the recent news pulse point to {direction} short-term momentum."

    chart_col, story_col = st.columns([2.15, 1], gap="small")
    with chart_col, st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Price</h2><span class="sg-muted">Volume · Sentiment · Events</span></div>',
            unsafe_allow_html=True,
        )
        events = value(derived, "events", default=None)
        st.plotly_chart(
            make_overview_chart(snapshot["prices"], events), width="stretch", config={"displayModeBar": False}
        )
    with story_col, st.container(border=True):
        st.markdown(
            f'<div class="sg-section-heading"><h2>Why it moved</h2></div><p>{html(explanation)}</p><div class="sg-kicker">Evidence</div>{_article_rows(evidence)}',
            unsafe_allow_html=True,
        )

    watchlist = records(derived, "watchlist", "quotes", limit=6)
    if not watchlist:
        watchlist = [
            {
                "ticker": snapshot["ticker"],
                "company": snapshot["company"],
                "price": snapshot["latest"],
                "change_pct": snapshot["percent"],
            }
        ]
    watch_rows = []
    for item in watchlist:
        symbol = str(item.get("ticker", item.get("symbol", "—"))).upper()
        price = float(item.get("price", item.get("Close", 0)) or 0)
        percent = float(item.get("change_pct", item.get("percent", 0)) or 0)
        watch_rows.append(
            f'<div class="sg-list-row"><div><div class="sg-list-title">{html(symbol)} &nbsp; <span class="sg-muted">{html(item.get("company", company_name(symbol)))}</span></div><div class="sg-list-subtitle">{html(item.get("source", "Unspecified source"))}</div></div><span>${price:,.2f}</span><span class="{"sg-positive" if percent >= 0 else "sg-negative"}">{percent:+.2f}%</span></div>'
        )
    lower_left, lower_right = st.columns([0.95, 1.2], gap="small")
    with lower_left, st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Watchlist</h2><span class="sg-muted">Recent · Search above</span></div>'
            + "".join(watch_rows),
            unsafe_allow_html=True,
        )
    with lower_right, st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Latest intelligence</h2><span class="sg-link">View all&nbsp; ›</span></div>'
            + _article_rows(evidence, compact=True),
            unsafe_allow_html=True,
        )


__all__ = ["render_overview"]
