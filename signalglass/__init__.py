"""SignalGlass: offline-first market and news intelligence primitives."""

from .analytics import evaluate_directional_signal, merge_market_and_sentiment, prepare_sentiment_summary
from .models import MarketBundle, SignalEvaluation
from .providers import fetch_market_bundle, load_demo_bundle, normalize_price_frame
from .validation import normalize_date_range, validate_ticker

__all__ = [
    "MarketBundle",
    "SignalEvaluation",
    "evaluate_directional_signal",
    "fetch_market_bundle",
    "load_demo_bundle",
    "merge_market_and_sentiment",
    "normalize_date_range",
    "normalize_price_frame",
    "prepare_sentiment_summary",
    "validate_ticker",
]
