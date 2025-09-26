"""Data fetching utilities for the MarketWatch Insights Dashboard."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException


DEFAULT_TICKERS = ["AAPL", "TSLA", "MSFT"]
_NEWS_COLUMNS = ["title", "description", "url", "publishedAt", "source", "ticker"]


def _empty_news_frame(message: Optional[str] = None) -> pd.DataFrame:
    """Return a consistent empty news DataFrame with optional warning metadata."""

    df = pd.DataFrame({col: pd.Series(dtype="object") for col in _NEWS_COLUMNS})
    if message:
        df.attrs["warning"] = message
    return df


def get_stock_data(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download stock price data for a ticker using yfinance."""
    ticker = ticker.upper()
    data = yf.download(ticker, start=start, end=end + timedelta(days=1), interval=interval)
    if data.empty:
        return data
    data = data.reset_index().rename(columns={"Date": "date"})
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data["ticker"] = ticker
    return data


def _safe_newsapi_client(api_key: Optional[str]) -> Optional[NewsApiClient]:
    if not api_key:
        return None
    return NewsApiClient(api_key=api_key)


def get_news_articles(
    ticker: str,
    start: datetime,
    end: datetime,
    api_key: Optional[str],
    language: str = "en",
    page_size: int = 100,
) -> pd.DataFrame:
    """Fetch news articles for a ticker within a date range using NewsAPI."""
    client = _safe_newsapi_client(api_key)
    if client is None:
        return _empty_news_frame("NewsAPI key not provided. Headlines are unavailable.")

    all_articles: List[dict] = []
    # NewsAPI limits the date range to the most recent 30 days and total results.
    now = datetime.utcnow()
    effective_end = min(end, now)
    window_start = now - timedelta(days=30)
    effective_start = max(start, window_start)

    if effective_end < window_start:
        return _empty_news_frame(
            "NewsAPI free tier only supports the last 30 days of headlines. Try a more recent date range."
        )

    if effective_start > effective_end:
        effective_start = effective_end

    page = 1
    end_date = (effective_end + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = effective_start.strftime("%Y-%m-%d")

    while True:
        try:
            response = client.get_everything(
                q=f"{ticker}",
                from_param=start_date,
                to=end_date,
                language=language,
                sort_by="publishedAt",
                page=page,
                page_size=page_size,
            )
        except NewsAPIException as exc:  # pragma: no cover - depends on remote API behaviour
            message = getattr(exc, "get_message", None)
            if callable(message):
                message = message()
            if not message:
                message = str(exc)
            if message:
                message = f"NewsAPI request failed: {message}"
            return _empty_news_frame(message)
        except Exception as exc:  # pragma: no cover - depends on remote API behaviour
            return _empty_news_frame(f"Unexpected error from NewsAPI: {exc}")
        articles = response.get("articles", [])
        if not articles:
            break
        for article in articles:
            all_articles.append(
                {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "publishedAt": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name"),
                }
            )
        total_results = response.get("totalResults", 0)
        if len(all_articles) >= total_results:
            break
        page += 1
        if page > 5:
            # Avoid exceeding API rate limits
            break

    df = pd.DataFrame(all_articles)
    if df.empty:
        return _empty_news_frame()

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.dropna(subset=["publishedAt"])  # Drop rows without publication date
    df["publishedAt"] = df["publishedAt"].dt.tz_localize(None)
    df["ticker"] = ticker.upper()
    return df


__all__ = ["get_stock_data", "get_news_articles", "DEFAULT_TICKERS"]
