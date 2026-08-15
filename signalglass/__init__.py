"""SignalGlass: offline-first market and news intelligence primitives."""

from .analytics import (
    evaluate_directional_signal,
    evaluate_model_suite,
    merge_market_and_sentiment,
    prepare_sentiment_summary,
)
from .backtesting import BacktestConfig, BacktestResult, run_backtest
from .models import MarketBundle, ModelSuite, SignalEvaluation
from .portfolio import PortfolioAnalysis, analyze_portfolio
from .providers import fetch_market_bundle, load_demo_bundle, normalize_price_frame
from .validation import normalize_date_range, validate_ticker

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "MarketBundle",
    "ModelSuite",
    "PortfolioAnalysis",
    "SignalEvaluation",
    "analyze_portfolio",
    "evaluate_directional_signal",
    "evaluate_model_suite",
    "fetch_market_bundle",
    "load_demo_bundle",
    "merge_market_and_sentiment",
    "normalize_date_range",
    "normalize_price_frame",
    "prepare_sentiment_summary",
    "run_backtest",
    "validate_ticker",
]
