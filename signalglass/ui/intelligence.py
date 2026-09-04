"""News intelligence page renderer."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from signalglass.ui._data import format_date, frame, html, link, value
from signalglass.ui.shell import render_page_title


def render_intelligence(bundle: Any, derived: Any = None) -> None:
    """Render source-attributed news intelligence without fetching external data."""

    render_page_title("Intelligence", "Trace the headlines, tone, and evidence behind the market narrative.")
    news = frame(bundle, "news", "articles", "news_data")
    summary = value(derived, "intelligence_summary", "summary", "insight")
    if not summary:
        summary = (
            "Recent coverage is balanced; use the evidence trail below to inspect the underlying headlines."
        )

    if news.empty:
        st.info("No news intelligence is available for this range. Demo price data remains fully usable.")
        return

    working = news.copy(deep=True)
    if "publishedAt" in working:
        working["publishedAt"] = pd.to_datetime(working["publishedAt"], errors="coerce")
        working = working.sort_values("publishedAt", ascending=False)
    labels = working.get("sentiment_label", pd.Series("Neutral", index=working.index)).astype(str).str.title()
    positive = int(labels.eq("Positive").sum())
    negative = int(labels.eq("Negative").sum())
    neutral = int(len(labels) - positive - negative)
    st.markdown(
        f'<section class="sg-scorebar"><div class="sg-score"><div class="sg-score-label">Coverage pulse</div><div class="sg-score-value">{html(value(derived, "news_pulse", default="Live"))}</div></div><div class="sg-score"><div class="sg-score-label">Positive</div><div class="sg-score-value sg-positive">{positive}</div></div><div class="sg-score"><div class="sg-score-label">Neutral</div><div class="sg-score-value">{neutral}</div></div><div class="sg-score"><div class="sg-score-label">Negative</div><div class="sg-score-value sg-negative">{negative}</div></div><div class="sg-score"><div class="sg-score-label">Sources</div><div class="sg-score-value">{working.get("source", pd.Series(dtype=str)).nunique()}</div></div></section>',
        unsafe_allow_html=True,
    )

    narrative_col, evidence_col = st.columns([0.72, 1.28], gap="small")
    with narrative_col, st.container(border=True):
        st.markdown(
            f'<div class="sg-section-heading"><h2>Market narrative</h2></div><p>{html(summary)}</p><p class="sg-muted">SignalGlass separates observed headlines from model-derived interpretation so the evidence stays auditable.</p>',
            unsafe_allow_html=True,
        )
    with evidence_col, st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Evidence stream</h2><span class="sg-muted">Newest first</span></div>',
            unsafe_allow_html=True,
        )
        parts: list[str] = []
        for article in working.head(12).to_dict(orient="records"):
            source = article.get("source", "SignalGlass")
            if isinstance(source, dict):
                source = source.get("name", "SignalGlass")
            label = str(article.get("sentiment_label", "Neutral")).title()
            evidence = str(article.get("sentiment_evidence", "")).strip()
            evidence_copy = f" · Evidence: {evidence}" if evidence else ""
            parts.append(
                f'<article class="sg-list-row"><div>{link(article.get("title", "Market update"), article.get("url"))}<div class="sg-list-subtitle">{html(source)} · {html(format_date(article.get("publishedAt")))}</div></div><span class="sg-muted">{html(str(article.get("description", ""))[:96] + evidence_copy)}</span><span class="sg-tag {"neutral" if label == "Neutral" else ""}">{html(label)}</span></article>'
            )
        st.markdown("".join(parts), unsafe_allow_html=True)


__all__ = ["render_intelligence"]
