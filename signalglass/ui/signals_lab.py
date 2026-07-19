"""Signals Lab page renderer and its explicit run-request contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from signalglass.charts import make_signal_evaluation_chart
from signalglass.ui._data import frame, html, value
from signalglass.ui.shell import render_page_title


@dataclass(frozen=True, slots=True)
class SignalsLabConfig:
    """Immutable inputs selected for a Signals Lab evaluation."""

    ticker: str
    coverage_days: int
    model: str
    run_requested: bool = False

    def matches(self, other: SignalsLabConfig) -> bool:
        """Compare evaluation inputs while ignoring transient button state."""

        return (
            self.ticker == other.ticker
            and self.coverage_days == other.coverage_days
            and self.model == other.model
        )


_COVERAGE_OPTIONS = {"90 trading days": 90, "60 trading days": 60, "30 trading days": 30}
_MODEL_OPTIONS = {"Directional baseline": "directional_baseline"}


def _percent(raw: Any) -> float:
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return number * 100 if abs(number) <= 1 else number


def calculate_directional_baseline(predictions: pd.DataFrame) -> float:
    """Calculate majority-direction accuracy from actual test outcomes."""

    if not isinstance(predictions, pd.DataFrame) or "actual_return" not in predictions:
        return 0.0
    actual = pd.to_numeric(predictions["actual_return"], errors="coerce").dropna()
    if actual.empty:
        return 0.0
    non_negative_share = float(actual.ge(0).mean())
    return max(non_negative_share, 1.0 - non_negative_share)


def _ticker_options(bundle: Any, derived: Any) -> list[str]:
    current = str(value(bundle, "ticker", "symbol", default="AAPL")).upper()
    supplied = value(derived, "available_tickers", default=None)
    if supplied is None:
        comparison = value(derived, "comparison", "series", default=None)
        supplied = comparison.keys() if isinstance(comparison, Mapping) else []
    options = [str(ticker).upper() for ticker in supplied if str(ticker).strip()]
    return list(dict.fromkeys([current, *options]))


def _feature_rows(bundle: Any, derived: Any) -> str:
    snapshot = value(derived, "feature_snapshot", default=None)
    if isinstance(snapshot, dict):
        values = snapshot
    else:
        prices = frame(derived, "market", "combined", "market_frame")
        if prices.empty:
            prices = frame(bundle, "prices")
        close = pd.to_numeric(prices.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
        volume = pd.to_numeric(prices.get("Volume", pd.Series(dtype=float)), errors="coerce").dropna()
        sentiment = pd.to_numeric(
            prices.get("avg_sentiment", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        momentum = float(close.pct_change(20, fill_method=None).iloc[-1]) if len(close) > 20 else 0.0
        volume_shift = (
            float(volume.iloc[-1] / volume.tail(20).mean() - 1)
            if not volume.empty and volume.tail(20).mean()
            else 0.0
        )
        values = {
            "Price momentum": momentum,
            "News pulse": float(sentiment.tail(10).mean()) if not sentiment.empty else 0.0,
            "Volume shift": volume_shift,
            "Volatility": float(close.pct_change(fill_method=None).tail(20).std()) if len(close) > 2 else 0.0,
        }
    descriptions = {
        "Price momentum": "20-day price change",
        "News pulse": "Average tone of recent news",
        "Volume shift": "Latest volume vs 20-day average",
        "Volatility": "20-day realized volatility",
    }
    icons = {"Price momentum": "M", "News pulse": "N", "Volume shift": "V", "Volatility": "R"}
    rows: list[str] = []
    for name, raw in list(values.items())[:4]:
        try:
            observed = float(raw)
        except (TypeError, ValueError):
            observed = 0.0
        color = "sg-positive" if observed >= 0 else "sg-negative"
        rows.append(
            f'<div class="sg-driver"><div class="sg-driver-icon" aria-hidden="true">{icons.get(str(name), "F")}</div><div><div class="sg-driver-name">{html(name)}</div><div class="sg-driver-copy">{html(descriptions.get(str(name), "Observed feature value"))}</div></div><div class="sg-driver-value {color}">{observed:+.2f}</div></div>'
        )
    return "".join(rows)


def _render_methodology() -> None:
    how_col, limit_col = st.columns([2.05, 1], gap="small")
    with how_col, st.container(border=True):
        st.markdown(
            """<div class="sg-section-heading"><h2>How it works</h2></div><div class="sg-process">
            <div class="sg-process-step"><div class="sg-process-number">01</div><div><div class="sg-process-title">Market data + News</div><div class="sg-process-copy">Prices, volume, and news signals collected daily.</div></div></div>
            <div class="sg-process-step"><div class="sg-process-number">02</div><div><div class="sg-process-title">Features</div><div class="sg-process-copy">Raw data transformed into model features.</div></div></div>
            <div class="sg-process-step"><div class="sg-process-number">03</div><div><div class="sg-process-title">Time split</div><div class="sg-process-copy">Chronological train, validate, and test windows.</div></div></div>
            <div class="sg-process-step"><div class="sg-process-number">04</div><div><div class="sg-process-title">Evaluation</div><div class="sg-process-copy">Walk-forward test measures out-of-sample performance.</div></div></div></div>""",
            unsafe_allow_html=True,
        )
    with limit_col, st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Limitations</h2></div><p>Experimental research signal.<br>Evaluated chronologically with no look-ahead.</p><p>Not financial advice.</p>',
            unsafe_allow_html=True,
        )


def render_signals_lab(
    bundle: Any,
    derived: Any = None,
    *,
    completed_run: SignalsLabConfig | None = None,
) -> SignalsLabConfig:
    """Render controls/results and return a configuration for the app to execute."""

    render_page_title("Signals Lab", "Evaluate the signal before you trust it.")
    controls, actions = st.columns([1.15, 0.85], gap="large")
    with controls:
        first, second, third = st.columns(3)
        with first:
            selected_ticker = st.selectbox(
                "Company", _ticker_options(bundle, derived), label_visibility="collapsed", key="sg_lab_ticker"
            )
        with second:
            coverage_label = st.selectbox(
                "Coverage", list(_COVERAGE_OPTIONS), label_visibility="collapsed", key="sg_lab_window"
            )
        with third:
            model_label = st.selectbox(
                "Model",
                list(_MODEL_OPTIONS),
                label_visibility="collapsed",
                key="sg_lab_model",
                disabled=True,
            )
    with actions:
        run_col, method_col = st.columns(2)
        with run_col:
            run_requested = st.button("Run evaluation", type="primary", width="stretch")
        with method_col:
            st.button(
                "Methodology", width="stretch", help="Expanding-window linear regression with no look-ahead."
            )

    request = SignalsLabConfig(
        ticker=str(selected_ticker),
        coverage_days=_COVERAGE_OPTIONS[str(coverage_label)],
        model=_MODEL_OPTIONS[str(model_label)],
        run_requested=run_requested,
    )
    if completed_run is None or not completed_run.matches(request):
        st.info("Run an evaluation to see out-of-sample results.")
        _render_methodology()
        return request

    evaluation = value(derived, "evaluation", "signal_evaluation", default=derived)
    predictions = frame(evaluation, "predictions", "walk_forward", "results")
    if predictions.empty:
        st.warning("The completed evaluation did not return enough test samples.")
        _render_methodology()
        return request

    accuracy = _percent(value(evaluation, "directional_accuracy", "accuracy", default=0))
    baseline = calculate_directional_baseline(predictions) * 100
    mae = _percent(value(evaluation, "mean_absolute_error", "mae", default=0))
    samples = int(value(evaluation, "sample_size", "samples", default=len(predictions)) or 0)
    st.markdown(
        f'<section class="sg-scorebar" aria-label="Evaluation metrics"><div class="sg-score"><div class="sg-score-value">{accuracy:.1f}%</div><div class="sg-score-label">directional accuracy</div></div><div class="sg-score"><div class="sg-score-label">Majority-direction baseline</div><div class="sg-score-value">{baseline:.1f}%</div></div><div class="sg-score"><div class="sg-score-label">MAE</div><div class="sg-score-value">{mae:.2f}%</div></div><div class="sg-score"><div class="sg-score-label">Test samples</div><div class="sg-score-value">{samples}</div></div><div class="sg-score"><div class="sg-score-label">Coverage</div><div class="sg-score-value">{completed_run.coverage_days} days</div></div></section>',
        unsafe_allow_html=True,
    )

    chart_col, feature_col = st.columns([2.05, 1], gap="small")
    with chart_col, st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Walk-forward evaluation</h2><span class="sg-muted">Out-of-sample</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_signal_evaluation_chart(predictions), width="stretch", config={"displayModeBar": False}
        )
        st.caption(
            "Bars show actual daily return. The line shows predicted return. Training data always precedes each test sample."
        )
    with feature_col, st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Market feature snapshot</h2></div>'
            + _feature_rows(bundle, derived),
            unsafe_allow_html=True,
        )
        st.caption(
            "Descriptive feature values from the selected market window. They are context, not model attribution."
        )
    _render_methodology()
    return request


__all__ = ["SignalsLabConfig", "calculate_directional_baseline", "render_signals_lab"]
