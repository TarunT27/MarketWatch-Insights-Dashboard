"""Deterministic, realistic demo data for an offline-first experience."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .sentiment import score_financial_text

PRICE_COLUMNS = ["date", "Open", "High", "Low", "Close", "Volume", "ticker"]
NEWS_COLUMNS = [
    "title",
    "description",
    "url",
    "publishedAt",
    "source",
    "ticker",
    "sentiment_label",
    "sentiment_score",
    "sentiment_evidence",
]
DEFAULT_DEMO_END = date(2026, 7, 17)


@dataclass(frozen=True, slots=True)
class _TickerProfile:
    company: str
    starting_price: float
    drift: float
    volatility: float
    base_volume: int
    seed: int


_PROFILES = {
    "AAPL": _TickerProfile("Apple", 218.0, 0.00045, 0.012, 54_000_000, 101),
    "MSFT": _TickerProfile("Microsoft", 438.0, 0.00052, 0.011, 24_000_000, 202),
    "NVDA": _TickerProfile("NVIDIA", 136.0, 0.00078, 0.020, 188_000_000, 303),
    "TSLA": _TickerProfile("Tesla", 338.0, 0.00025, 0.025, 112_000_000, 404),
}

# Demo headlines are written in real finance language so the shipped sentiment
# engine scores them for real. Nothing here carries a hardcoded label: the
# labels, scores, and evidence in demo mode come from ``score_financial_text``,
# which is the same code path live headlines take.
_HEADLINES = (
    "tops quarterly estimates as demand strengthens",
    "raises guidance after a record quarter",
    "shares steady as the market reviews fresh data",
    "misses expectations as growth slows",
    "announces share repurchase program and dividend increase",
    "faces regulatory scrutiny over platform practices",
    "upgraded by analysts on margin expansion",
    "warns of supply constraints and margin pressure",
    "board schedules its annual shareholder meeting",
    "rallies after better than expected results",
)

_DESCRIPTIONS = (
    "Synthetic headline scored by the SignalGlass finance-domain sentiment engine.",
    "Curated synthetic coverage for the SignalGlass offline product tour.",
)


# Display names known without a network call. Anything outside this map is
# resolved live by the provider boundary and falls back to the ticker itself.
_COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "TSLA": "Tesla, Inc.",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "AMD": "Advanced Micro Devices, Inc.",
    "NFLX": "Netflix, Inc.",
    "BRK.B": "Berkshire Hathaway Inc.",
    "^GSPC": "S&P 500 Index",
    "^IXIC": "NASDAQ Composite",
    "^DJI": "Dow Jones Industrial Average",
    "BTC-USD": "Bitcoin USD",
    "ETH-USD": "Ethereum USD",
}


def demo_company_name(ticker: str) -> str:
    """Return a known display name, or an empty string when none is known."""

    return _COMPANY_NAMES.get(str(ticker).strip().upper(), "")


def _fallback_profile(ticker: str) -> _TickerProfile:
    digest = hashlib.sha256(ticker.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    return _TickerProfile(ticker, 100.0, 0.00035, 0.016, 15_000_000, seed)


def generate_demo_prices(
    ticker: str,
    days: int = 90,
    *,
    end_date: date = DEFAULT_DEMO_END,
) -> pd.DataFrame:
    """Create a repeatable OHLCV business-day series for ``ticker``."""

    if not isinstance(days, int) or isinstance(days, bool) or days < 2:
        raise ValueError("demo days must be an integer of at least two")

    profile = _PROFILES.get(ticker, _fallback_profile(ticker))
    dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=days)
    rng = np.random.default_rng(profile.seed + days + end_date.toordinal())
    cycle = np.sin(np.linspace(0, 3.5 * np.pi, days)) * profile.volatility * 0.28
    daily_returns = profile.drift + cycle + rng.normal(0.0, profile.volatility, days)
    closes = profile.starting_price * np.exp(np.cumsum(daily_returns))
    overnight = rng.normal(0.0, profile.volatility * 0.24, days)
    opens = closes * (1.0 + overnight)
    intraday_range = np.abs(rng.normal(profile.volatility * 0.7, profile.volatility * 0.2, days))
    highs = np.maximum(opens, closes) * (1.0 + intraday_range)
    lows = np.minimum(opens, closes) * np.maximum(0.01, 1.0 - intraday_range)
    volume_noise = rng.lognormal(mean=0.0, sigma=0.22, size=days)
    volumes = np.maximum(1, np.rint(profile.base_volume * volume_noise)).astype("int64")

    return pd.DataFrame(
        {
            "date": dates,
            "Open": np.round(opens, 2),
            "High": np.round(highs, 2),
            "Low": np.round(lows, 2),
            "Close": np.round(closes, 2),
            "Volume": volumes,
            "ticker": ticker,
        },
        columns=PRICE_COLUMNS,
    )


def generate_demo_news(
    ticker: str,
    days: int = 90,
    *,
    end_date: date = DEFAULT_DEMO_END,
) -> pd.DataFrame:
    """Create stable scored headlines spread across the demo price window."""

    if not isinstance(days, int) or isinstance(days, bool) or days < 2:
        raise ValueError("demo days must be an integer of at least two")

    profile = _PROFILES.get(ticker, _fallback_profile(ticker))
    sessions = pd.bdate_range(end=pd.Timestamp(end_date), periods=days)
    article_count = max(8, min(24, days // 3))
    positions = np.linspace(0, len(sessions) - 1, article_count, dtype=int)
    records: list[dict[str, object]] = []

    for index, position in enumerate(positions):
        headline = _HEADLINES[(index + profile.seed) % len(_HEADLINES)]
        description = _DESCRIPTIONS[index % len(_DESCRIPTIONS)]
        title = f"{profile.company} {headline}"
        published_at = sessions[position] + pd.Timedelta(hours=9 + (index % 8))
        # Score demo headlines through the production engine so the demo
        # demonstrates the real scorer rather than a canned label.
        sentiment = score_financial_text(title, description)
        records.append(
            {
                "title": title,
                "description": description,
                "url": f"https://example.com/signalglass/{ticker.lower()}/{index + 1}",
                "publishedAt": published_at,
                "source": "SignalGlass Demo Wire",
                "ticker": ticker,
                "sentiment_label": sentiment.label,
                "sentiment_score": sentiment.score,
                "sentiment_evidence": ", ".join(sentiment.evidence),
            }
        )

    return pd.DataFrame.from_records(records, columns=NEWS_COLUMNS)


__all__ = [
    "DEFAULT_DEMO_END",
    "NEWS_COLUMNS",
    "PRICE_COLUMNS",
    "demo_company_name",
    "generate_demo_news",
    "generate_demo_prices",
]
