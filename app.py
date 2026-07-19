"""SignalGlass — an explainable market-intelligence cockpit."""

from __future__ import annotations

import hashlib
import os
from datetime import date, datetime, timedelta

import streamlit as st

from signalglass.analytics import (
    evaluate_directional_signal,
    merge_market_and_sentiment,
    prepare_sentiment_summary,
)
from signalglass.providers import fetch_market_bundle, load_demo_bundle
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
    render_signals_lab,
)

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
def load_bundle(symbol: str, mode: str, api_key: str):
    if mode == "Demo":
        return load_demo_bundle(symbol, days=90)
    end = date.today()
    return fetch_market_bundle(
        symbol,
        end - timedelta(days=135),
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
    return {
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

control_left, control_right = st.columns([1, 1], vertical_alignment="center")
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
bundle = load_bundle(symbol, mode, newsapi_key)
comparison_bundles = {symbol: bundle}
if page in {"Overview", "Compare"}:
    comparison_symbols = build_symbol_choices(
        symbol,
        st.session_state.get("sg_recent_symbols", ()),
        featured=FEATURED_SYMBOLS,
        limit=4,
    )
    comparison_bundles = {
        ticker: bundle if ticker == symbol else load_bundle(ticker, mode, "") for ticker in comparison_symbols
    }
derived = derived_view(bundle, comparison_bundles)
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
    request = render_signals_lab(bundle, lab_derived, completed_run=completed_config)
    if request.run_requested:
        selected_market = derived["market"].tail(request.coverage_days).copy(deep=True)
        min_train_size = min(20, max(10, len(selected_market) // 3))
        evaluation = evaluate_directional_signal(selected_market, min_train_size=min_train_size)
        st.session_state["sg_completed_signal_run"] = {
            "config": request,
            "evaluation": evaluation,
            "data_fingerprint": data_fingerprint,
        }
        st.rerun()
else:
    renderers = {
        "Overview": render_overview,
        "Compare": render_compare,
        "Intelligence": render_intelligence,
    }
    renderers[page](bundle, derived)
render_footer()
