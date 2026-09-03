"""SignalGlass — an explainable market-intelligence cockpit."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st

from signalglass.analytics import (
    evaluate_model_suite,
    merge_market_and_sentiment,
    prepare_sentiment_summary,
)
from signalglass.backtesting import BacktestConfig, run_backtest
from signalglass.portfolio import analyze_portfolio
from signalglass.providers import fetch_market_bundle, load_demo_bundle
from signalglass.quality import assess_data_quality
from signalglass.store import LocalResearchStore
from signalglass.symbols import (
    DEFAULT_SYMBOL,
    FEATURED_SYMBOLS,
    build_symbol_choices,
    record_recent_symbol,
    resolve_symbol,
)
from signalglass.ui import (
    PAGES,
    render_app_header,
    render_compare,
    render_footer,
    render_intelligence,
    render_overview,
    render_portfolio,
    render_signals_lab,
)
from signalglass.windows import DEFAULT_RANGE, RANGE_LABELS, resolve_range

APP_ICON = "assets/brand/signalglass-app-icon.png"

st.set_page_config(
    page_title="SignalGlass", page_icon=APP_ICON, layout="wide", initial_sidebar_state="collapsed"
)


def optional_secret(name: str) -> str:
    """Read an optional secret without crashing when secrets.toml is absent."""

    try:
        # ``st.secrets.get`` parses the backing file and raises when it is absent.
        value = st.secrets[name] if name in st.secrets else ""  # noqa: SIM401
    except Exception:
        value = ""
    return str(value or os.getenv(name.upper(), "")).strip()


@st.cache_data(ttl=900, max_entries=128, show_spinner=False)
def load_bundle(symbol: str, mode: str, api_key: str, range_label: str = DEFAULT_RANGE):
    window = resolve_range(range_label)
    end = date.today()
    if mode == "Demo":
        # Demo history ends today so the tour never looks stale next to the clock.
        return load_demo_bundle(symbol, days=window.trading_days, end_date=end)
    return fetch_market_bundle(
        symbol,
        end - timedelta(days=window.calendar_days),
        end,
        newsapi_key=api_key,
        prefer_live_prices=True,
    )


def bundle_fingerprint(bundle, mode: str) -> str:
    """Identify the exact price/news inputs behind a saved evaluation."""

    digest = hashlib.sha256()
    digest.update(mode.encode("utf-8"))
    digest.update(bundle.price_source.encode("utf-8"))
    digest.update(bundle.news_source.encode("utf-8"))
    digest.update(bundle.prices.to_csv(index=False).encode("utf-8"))
    digest.update(bundle.news.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def derived_view(bundle, comparison_bundles: dict[str, object]) -> dict[str, object]:
    sentiment = prepare_sentiment_summary(bundle.news)
    market = merge_market_and_sentiment(bundle.prices, sentiment)
    latest_return = market["Close"].pct_change(fill_method=None).iloc[-1] if len(market) > 1 else 0.0
    pulse = float(market["avg_sentiment"].tail(10).mean()) if not market.empty else 0.0
    tone = "constructive" if pulse >= 0.08 else "cautious" if pulse <= -0.08 else "balanced"
    movement = "gained" if latest_return >= 0 else "softened"
    comparisons = {ticker: item.prices for ticker, item in comparison_bundles.items()}
    comparison_sources = {ticker: item.price_source for ticker, item in comparison_bundles.items()}
    watchlist = [
        {
            "ticker": ticker,
            "price": float(frame["Close"].iloc[-1]),
            "change_pct": float(frame["Close"].pct_change(fill_method=None).iloc[-1] * 100),
            "source": comparison_sources[ticker],
        }
        for ticker, frame in comparisons.items()
    ]
    quality = assess_data_quality(bundle.prices, bundle.news)
    return {
        "company": bundle.company or bundle.ticker,
        "data_quality": quality.score,
        "data_quality_summary": quality.summary,
        "market": market,
        "comparison": comparisons,
        "comparison_sources": comparison_sources,
        "watchlist": watchlist,
        "news_pulse": f"{pulse:+.2f}",
        "why_it_moved": f"Shares {movement} as the latest evidence mix remained {tone}.",
        "intelligence_summary": (
            f"The recent evidence mix is {tone}. SignalGlass keeps every narrative tied to its source headlines."
        ),
    }


environment_mode = os.getenv("SIGNALGLASS_DATA_MODE", "demo").strip().lower()
default_mode = "Live" if environment_mode == "live" else "Demo"

requested_page = st.query_params.get("page", "Overview")
page = requested_page if requested_page in PAGES else "Overview"
header_slot = st.empty()

last_valid_symbol = st.session_state.get("sg_last_valid_symbol", DEFAULT_SYMBOL)
requested_symbol = st.query_params.get("symbol")
requested_symbol_error = None
if requested_symbol:
    query_symbol, requested_symbol_error = resolve_symbol(requested_symbol, fallback=last_valid_symbol)
    if requested_symbol_error is None:
        last_valid_symbol = query_symbol
        if st.session_state.get("sg_applied_query_symbol") != query_symbol:
            st.session_state["sg_symbol_picker"] = query_symbol
            st.session_state["sg_applied_query_symbol"] = query_symbol
recent_symbols = tuple(st.session_state.get("sg_recent_symbols", FEATURED_SYMBOLS[:4]))
symbol_choices = build_symbol_choices(last_valid_symbol, recent_symbols)

control_left, control_mid, control_right = st.columns([1.25, 0.85, 0.7], vertical_alignment="center")
with control_left:
    symbol_candidate = st.selectbox(
        "Search symbol",
        symbol_choices,
        key="sg_symbol_picker",
        label_visibility="collapsed",
        accept_new_options=True,
        placeholder="Search any ticker — AMD, SPY, BRK.B, BTC-USD",
        help="Enter any Yahoo Finance-compatible stock, ETF, index, or crypto symbol.",
    )
with control_mid:
    range_label = (
        st.segmented_control(
            "History range",
            RANGE_LABELS,
            default=DEFAULT_RANGE,
            key="sg_range",
            label_visibility="collapsed",
            help="How much price history to load and evaluate.",
        )
        or DEFAULT_RANGE
    )
with control_right:
    mode = st.segmented_control(
        "Data source",
        ("Demo", "Live"),
        default=default_mode,
        key="sg_data_mode",
        label_visibility="collapsed",
    )

symbol, symbol_error = resolve_symbol(symbol_candidate, fallback=last_valid_symbol)
valid_picker_override = (
    requested_symbol_error is not None
    and symbol_error is None
    and symbol != st.session_state.get("sg_last_valid_symbol", DEFAULT_SYMBOL)
)
selection_error = symbol_error or (None if valid_picker_override else requested_symbol_error)
if selection_error:
    st.error(selection_error)
else:
    updated_recent_symbols = record_recent_symbol(symbol, recent_symbols)
    st.session_state["sg_last_valid_symbol"] = symbol
    st.session_state["sg_recent_symbols"] = updated_recent_symbols
    if requested_symbol != symbol:
        st.query_params["symbol"] = symbol
    if symbol not in symbol_choices:
        st.rerun()

newsapi_key = optional_secret("newsapi_key")
bundle = load_bundle(symbol, mode, newsapi_key, range_label)
portfolio_store = None
saved_watchlist: tuple[str, ...] = ()
saved_allocations: dict[str, float] = {}
if page == "Portfolio":
    try:
        store_path = Path(os.getenv("SIGNALGLASS_DB_PATH", ".signalglass/signalglass.db"))
        portfolio_store = LocalResearchStore(store_path)
        saved_watchlist = portfolio_store.load_watchlist()
        saved_allocations = portfolio_store.load_allocations()
    except (OSError, ValueError, sqlite3.Error) as error:
        st.warning(f"Local research persistence is unavailable ({type(error).__name__}).")

comparison_bundles = {symbol: bundle}
if page in {"Overview", "Compare", "Portfolio"}:
    recent_for_page = (
        (*saved_watchlist, *st.session_state.get("sg_recent_symbols", ()))
        if page == "Portfolio"
        else st.session_state.get("sg_recent_symbols", ())
    )
    comparison_symbols = build_symbol_choices(
        symbol,
        recent_for_page,
        featured=FEATURED_SYMBOLS,
        limit=4,
    )
    comparison_bundles = {
        ticker: bundle if ticker == symbol else load_bundle(ticker, mode, "", range_label)
        for ticker in comparison_symbols
    }
derived = derived_view(bundle, comparison_bundles)
if page == "Portfolio":
    available_allocations = {
        ticker: weight for ticker, weight in saved_allocations.items() if ticker in comparison_bundles
    }
    if not available_allocations:
        available_allocations = {ticker: 1.0 / len(comparison_bundles) for ticker in comparison_bundles}
    try:
        derived["portfolio_analysis"] = analyze_portfolio(
            {ticker: item.prices for ticker, item in comparison_bundles.items()},
            available_allocations,
        )
    except ValueError as error:
        st.warning(f"Portfolio analysis is unavailable: {error}")
comparison_is_mixed = mode == "Live" and any(item.is_demo for item in comparison_bundles.values())
if mode == "Demo":
    data_label = "Demo data"
elif bundle.is_demo:
    data_label = "Demo fallback"
elif comparison_is_mixed:
    data_label = "Mixed price data"
else:
    data_label = "Live prices"
with header_slot.container():
    render_app_header(
        page,
        updated_at=datetime.now().strftime("%I:%M %p").lstrip("0"),
        data_label=data_label,
        symbol=symbol,
    )

if bundle.notice and mode == "Live":
    st.warning(bundle.notice)
if comparison_is_mixed and not bundle.is_demo:
    st.warning("Some comparison symbols could not load live prices and are labeled as demo fallbacks.")

if page == "Signals Lab":
    saved_run = st.session_state.get("sg_completed_signal_run")
    data_fingerprint = bundle_fingerprint(bundle, mode)
    if isinstance(saved_run, dict) and saved_run.get("data_fingerprint") != data_fingerprint:
        st.session_state.pop("sg_completed_signal_run", None)
        saved_run = None
    completed_config = saved_run.get("config") if isinstance(saved_run, dict) else None
    lab_derived = dict(derived)
    if isinstance(saved_run, dict):
        lab_derived["evaluation"] = saved_run.get("evaluation")
        lab_derived["model_suite"] = saved_run.get("model_suite")
        lab_derived["backtest"] = saved_run.get("backtest")
    request = render_signals_lab(bundle, lab_derived, completed_run=completed_config)
    if request.run_requested:
        selected_market = derived["market"].tail(request.coverage_days).copy(deep=True)
        min_train_size = min(20, max(10, len(selected_market) // 3))
        model_suite = evaluate_model_suite(selected_market, min_train_size=min_train_size)
        if model_suite is None:
            evaluation = None
            backtest = None
        else:
            evaluation = (
                model_suite.best_evaluation
                if request.model == "auto"
                else next(item for item in model_suite.evaluations if item.model_name == request.model)
            )
            backtest = run_backtest(
                evaluation.predictions,
                config=BacktestConfig(
                    transaction_cost_bps=request.transaction_cost_bps,
                    signal_threshold=request.signal_threshold,
                    allow_short=request.allow_short,
                ),
            )
        st.session_state["sg_completed_signal_run"] = {
            "config": request,
            "evaluation": evaluation,
            "model_suite": model_suite,
            "backtest": backtest,
            "data_fingerprint": data_fingerprint,
        }
        st.rerun()
elif page == "Portfolio":
    request = render_portfolio(
        bundle,
        derived,
        saved_watchlist=saved_watchlist,
        saved_allocations=saved_allocations,
    )
    if request.save_requested and request.allocations and portfolio_store is not None:
        portfolio_store.replace_watchlist(request.watchlist)
        portfolio_store.replace_allocations(request.allocations)
        st.rerun()
else:
    renderers = {
        "Overview": render_overview,
        "Compare": render_compare,
        "Intelligence": render_intelligence,
    }
    renderers[page](bundle, derived)
render_footer()
