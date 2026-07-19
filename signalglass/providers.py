"""Market providers with strict normalization and deterministic fallback."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf

from .demo_data import NEWS_COLUMNS, PRICE_COLUMNS, generate_demo_news, generate_demo_prices
from .models import MarketBundle
from .validation import normalize_date_range, validate_ticker

try:
    from newsapi import NewsApiClient
except ImportError:  # pragma: no cover - dependency is optional for demo mode
    NewsApiClient = None  # type: ignore[assignment,misc]


_REQUIRED_PRICE_FIELDS = ("Open", "High", "Low", "Close", "Volume")


def _select_multiindex_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    structural_tokens = {
        "",
        "ADJ CLOSE",
        "CLOSE",
        "DATE",
        "DATETIME",
        "HIGH",
        "LOW",
        "OPEN",
        "PRICE",
        "TICKER",
        "VOLUME",
    }
    value_tokens = {
        str(part).upper()
        for column in frame.columns
        for part in column
        if str(part).upper() not in structural_tokens
    }
    level_names = {str(name).upper() for name in frame.columns.names if name is not None}
    has_ticker_dimension = "TICKER" in level_names or bool(value_tokens)
    selected: dict[str, pd.Series[Any]] = {}
    for field in _REQUIRED_PRICE_FIELDS:
        candidates = [column for column in frame.columns if field in tuple(str(part) for part in column)]
        ticker_matches = [
            column for column in candidates if ticker in tuple(str(part).upper() for part in column)
        ]
        can_use_unlabeled_candidate = not has_ticker_dimension and len(candidates) == 1
        choice = (
            ticker_matches[0] if ticker_matches else (candidates[0] if can_use_unlabeled_candidate else None)
        )
        if choice is not None:
            selected[field] = frame.loc[:, choice].copy()
    return pd.DataFrame(selected, index=frame.index.copy())


def _find_date_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if str(column).lower() in {"date", "datetime", "timestamp"}:
            return str(column)
    return None


def normalize_price_frame(raw_frame: pd.DataFrame, ticker: object) -> pd.DataFrame:
    """Return one stable OHLCV schema from flat or yfinance MultiIndex data."""

    if not isinstance(raw_frame, pd.DataFrame):
        raise TypeError("price data must be a pandas DataFrame")

    symbol = validate_ticker(ticker)
    working = raw_frame.copy(deep=True)
    if isinstance(working.columns, pd.MultiIndex):
        working = _select_multiindex_columns(working, symbol)

    date_column = _find_date_column(working)
    if date_column is None:
        index_name = working.index.name or "date"
        working = working.reset_index().rename(columns={index_name: "date", "index": "date"})
        date_column = "date"

    missing = [field for field in _REQUIRED_PRICE_FIELDS if field not in working.columns]
    if missing:
        raise ValueError(f"price schema is missing required columns: {', '.join(missing)}")

    normalized = working.loc[:, [date_column, *_REQUIRED_PRICE_FIELDS]].copy(deep=True)
    normalized = normalized.rename(columns={date_column: "date"})
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", utc=True).dt.tz_convert(None)
    for field in _REQUIRED_PRICE_FIELDS:
        normalized[field] = pd.to_numeric(normalized[field], errors="coerce")
    normalized = normalized.dropna(subset=["date", *_REQUIRED_PRICE_FIELDS])
    normalized = normalized.sort_values("date").drop_duplicates(subset="date", keep="last")
    normalized["ticker"] = symbol
    normalized["Volume"] = normalized["Volume"].astype("int64")
    return normalized.loc[:, PRICE_COLUMNS].reset_index(drop=True)


@lru_cache(maxsize=64)
def _cached_demo_bundle(ticker: str, days: int, end_date: date) -> MarketBundle:
    return MarketBundle(
        ticker=ticker,
        prices=generate_demo_prices(ticker, days, end_date=end_date),
        news=generate_demo_news(ticker, days, end_date=end_date),
        is_demo=True,
        source="SignalGlass deterministic demo",
        notice="Demo data is synthetic and designed for product exploration.",
        price_source="SignalGlass deterministic demo prices",
        news_source="SignalGlass deterministic demo news",
    )


def load_demo_bundle(
    ticker: object,
    days: int = 90,
    *,
    end_date: date | None = None,
) -> MarketBundle:
    """Load an isolated copy of deterministic demo data."""

    symbol = validate_ticker(ticker)
    resolved_end = end_date or date(2026, 7, 17)
    cached = _cached_demo_bundle(symbol, days, resolved_end)
    return MarketBundle(
        ticker=cached.ticker,
        prices=cached.prices.copy(deep=True),
        news=cached.news.copy(deep=True),
        is_demo=True,
        source=cached.source,
        notice=cached.notice,
        price_source=cached.price_source,
        news_source=cached.news_source,
    )


def _fetch_live_news(ticker: str, start_date: date, end_date: date, api_key: str) -> pd.DataFrame:
    if NewsApiClient is None:
        raise RuntimeError("NewsAPI client is unavailable")
    response = NewsApiClient(api_key=api_key).get_everything(
        q=ticker,
        from_param=start_date.isoformat(),
        to=end_date.isoformat(),
        language="en",
        sort_by="publishedAt",
        page_size=50,
    )
    records = []
    for index, article in enumerate(response.get("articles", [])):
        published_at = pd.to_datetime(article.get("publishedAt"), errors="coerce", utc=True)
        if pd.isna(published_at):
            continue
        title = str(article.get("title") or "")
        description = str(article.get("description") or "")
        lowered = f"{title} {description}".lower()
        positive_hits = sum(word in lowered for word in ("growth", "gain", "beat", "strong", "record"))
        negative_hits = sum(word in lowered for word in ("loss", "fall", "miss", "risk", "weak"))
        score = float(max(-1.0, min(1.0, (positive_hits - negative_hits) * 0.2)))
        label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        records.append(
            {
                "title": title,
                "description": description,
                "url": article.get("url") or "",
                "publishedAt": published_at.tz_convert("America/New_York").tz_localize(None),
                "source": (article.get("source") or {}).get("name") or "NewsAPI",
                "ticker": ticker,
                "sentiment_label": label,
                "sentiment_score": score,
                "_order": index,
            }
        )
    news = pd.DataFrame.from_records(records)
    if news.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return (
        news.sort_values(["publishedAt", "_order"])
        .drop(columns="_order")
        .loc[:, NEWS_COLUMNS]
        .reset_index(drop=True)
    )


def fetch_market_bundle(
    ticker: object,
    start_date: object,
    end_date: object,
    *,
    newsapi_key: str | None,
    prefer_live_prices: bool = False,
) -> MarketBundle:
    """Fetch a market bundle with explicit source-level provenance.

    The backwards-compatible default remains fully offline when no NewsAPI key
    is configured. ``prefer_live_prices`` opts into Yahoo Finance prices while
    retaining deterministic demo headlines when the news provider is unavailable.
    """

    symbol = validate_ticker(ticker)
    start, end = normalize_date_range((start_date, end_date))
    demo_days = max(2, len(pd.bdate_range(start=start, end=end)))
    if not isinstance(prefer_live_prices, bool):
        raise TypeError("prefer_live_prices must be a boolean")
    normalized_news_key = newsapi_key.strip() if isinstance(newsapi_key, str) else ""
    if not normalized_news_key and not prefer_live_prices:
        return load_demo_bundle(symbol, demo_days, end_date=end)

    try:
        raw_prices = yf.download(
            symbol,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        prices = normalize_price_frame(raw_prices, symbol)
        if prices.empty:
            raise ValueError("live price provider returned no usable data")
    except Exception as error:
        fallback = load_demo_bundle(symbol, demo_days, end_date=end)
        return MarketBundle(
            ticker=fallback.ticker,
            prices=fallback.prices.copy(deep=True),
            news=fallback.news.copy(deep=True),
            is_demo=True,
            source=fallback.source,
            notice=f"Live providers were unavailable ({type(error).__name__}); showing deterministic demo data.",
            price_source=fallback.price_source,
            news_source=fallback.news_source,
        )

    if normalized_news_key:
        try:
            news = _fetch_live_news(symbol, start, end, normalized_news_key)
        except Exception as error:
            demo = load_demo_bundle(symbol, demo_days, end_date=end)
            news = demo.news.copy(deep=True)
            source = "Yahoo Finance prices + SignalGlass deterministic demo news"
            notice = (
                f"Prices are live from Yahoo Finance. NewsAPI was unavailable "
                f"({type(error).__name__}); headlines and sentiment are deterministic demo data."
            )
            news_source = "SignalGlass deterministic demo news"
        else:
            source = "Yahoo Finance + NewsAPI"
            notice = None
            news_source = "NewsAPI live headlines"
    else:
        demo = load_demo_bundle(symbol, demo_days, end_date=end)
        news = demo.news.copy(deep=True)
        source = "Yahoo Finance prices + SignalGlass deterministic demo news"
        notice = (
            "Prices are live from Yahoo Finance. Headlines and sentiment are "
            "deterministic demo data because no NewsAPI key is configured."
        )
        news_source = "SignalGlass deterministic demo news"
    return MarketBundle(
        ticker=symbol,
        prices=prices.copy(deep=True),
        news=news.copy(deep=True),
        is_demo=False,
        source=source,
        notice=notice,
        price_source="Yahoo Finance live prices",
        news_source=news_source,
    )


__all__ = [
    "fetch_market_bundle",
    "load_demo_bundle",
    "normalize_price_frame",
    "yf",
]
