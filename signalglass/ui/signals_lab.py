"""Signals Lab page renderer and its explicit run-request contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from signalglass.charts import make_equity_curve_chart, make_signal_evaluation_chart
from signalglass.signal_export import build_execution_signal_frame, serialize_execution_signals
from signalglass.ui._data import frame, html, value
from signalglass.ui.shell import render_page_title


@dataclass(frozen=True, slots=True)
class SignalsLabConfig:
    """Immutable inputs selected for a Signals Lab evaluation."""

    ticker: str
    coverage_days: int
    model: str
    run_requested: bool = False
    transaction_cost_bps: float = 5.0
    signal_threshold: float = 0.0
    allow_short: bool = False

    def matches(self, other: SignalsLabConfig) -> bool:
        """Compare evaluation inputs while ignoring transient button state."""

        return (
            self.ticker == other.ticker
            and self.coverage_days == other.coverage_days
            and self.model == other.model
            and self.transaction_cost_bps == other.transaction_cost_bps
            and self.signal_threshold == other.signal_threshold
            and self.allow_short == other.allow_short
        )


_COVERAGE_OPTIONS = {"90 trading days": 90, "60 trading days": 60, "30 trading days": 30}
_MODEL_OPTIONS = {
    "Model comparison (auto)": "auto",
    "Linear regression": "linear",
    "Ridge regression": "ridge",
    "Random forest": "random_forest",
}


_VERDICT_CLASSES = {
    "Significant": "sg-positive",
    "Not significant": "sg-neutral-metric",
    "Below baseline": "sg-negative",
}


def _render_verdict_note(
    verdict: str,
    accuracy: float,
    baseline: float,
    low: float,
    high: float,
    samples: int,
) -> None:
    """State plainly what the headline accuracy does and does not establish."""

    if verdict == "Below baseline":
        st.warning(
            f"This model is **worse than the naive baseline**. Always predicting the more common "
            f"direction would have scored {baseline:.1f}% against this model's {accuracy:.1f}%. "
            f"Treat the signal as unproven."
        )
    elif verdict == "Not significant":
        st.info(
            f"Accuracy of {accuracy:.1f}% over {samples} sessions has a 95% confidence interval of "
            f"{low:.1f}–{high:.1f}%, which includes 50%. This result is **statistically "
            f"indistinguishable from a coin flip** — it is not yet evidence of an edge."
        )
    else:
        st.success(
            f"The 95% confidence interval ({low:.1f}–{high:.1f}%) excludes 50% and the model clears "
            f"the {baseline:.1f}% baseline, so this edge is statistically meaningful over "
            f"{samples} sessions. It remains a backtest, not a forecast."
        )


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
            <div class="sg-process-step"><div class="sg-process-number">03</div><div><div class="sg-process-title">Time split</div><div class="sg-process-copy">Expanding train window; the next unseen session is the test.</div></div></div>
            <div class="sg-process-step"><div class="sg-process-number">04</div><div><div class="sg-process-title">Evaluation</div><div class="sg-process-copy">Walk-forward test measures out-of-sample performance.</div></div></div></div>""",
            unsafe_allow_html=True,
        )
    with limit_col, st.container(border=True):
        st.markdown(
            '<div class="sg-section-heading"><h2>Limitations</h2></div><p>Experimental research signal.<br>Evaluated chronologically with no look-ahead.</p><p>Accuracy over a few dozen sessions carries a wide confidence interval, and the winning model is chosen on the same windows it is scored on.</p><p>Not financial advice.</p>',
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
            )
    with actions:
        run_col, method_col = st.columns(2)
        with run_col:
            run_requested = st.button("Run evaluation", type="primary", width="stretch")
        with method_col:
            st.button(
                "Methodology", width="stretch", help="Expanding-window model comparison with no look-ahead."
            )
    assumption_one, assumption_two, assumption_three = st.columns(3)
    with assumption_one:
        transaction_cost_bps = float(
            st.number_input("Transaction cost (bps)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        )
    with assumption_two:
        signal_threshold = float(
            st.number_input(
                "Signal threshold", min_value=0.0, max_value=0.10, value=0.0, step=0.001, format="%.3f"
            )
        )
    with assumption_three:
        allow_short = bool(st.toggle("Allow short positions", value=False))

    request = SignalsLabConfig(
        ticker=str(selected_ticker),
        coverage_days=_COVERAGE_OPTIONS[str(coverage_label)],
        model=_MODEL_OPTIONS[str(model_label)],
        run_requested=run_requested,
        transaction_cost_bps=transaction_cost_bps,
        signal_threshold=signal_threshold,
        allow_short=allow_short,
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
    low = _percent(value(evaluation, "accuracy_low", default=0.0))
    high = _percent(value(evaluation, "accuracy_high", default=1.0))
    verdict = str(value(evaluation, "verdict", default="Not significant"))
    # The headline number is only green when it actually clears the naive
    # baseline; a model losing to "always guess up" must not read as a win.
    accuracy_class = _VERDICT_CLASSES.get(verdict, "sg-neutral-metric")
    st.markdown(
        f'<section class="sg-scorebar" aria-label="Evaluation metrics">'
        f'<div class="sg-score"><div class="sg-score-value sg-score-headline {accuracy_class}">{accuracy:.1f}%</div>'
        f'<div class="sg-score-label">directional accuracy · 95% CI {low:.1f}–{high:.1f}%</div></div>'
        f'<div class="sg-score"><div class="sg-score-label">Majority-direction baseline</div>'
        f'<div class="sg-score-value">{baseline:.1f}%</div></div>'
        f'<div class="sg-score"><div class="sg-score-label">Verdict</div>'
        f'<div class="sg-score-value sg-score-verdict {accuracy_class}">{html(verdict)}</div></div>'
        f'<div class="sg-score"><div class="sg-score-label">MAE</div><div class="sg-score-value">{mae:.2f}%</div></div>'
        f'<div class="sg-score"><div class="sg-score-label">Test samples · {completed_run.coverage_days} days</div>'
        f'<div class="sg-score-value">{samples}</div></div></section>',
        unsafe_allow_html=True,
    )
    _render_verdict_note(verdict, accuracy, baseline, low, high, samples)
    suite = value(derived, "model_suite", default=None)
    if suite is not None and not bool(value(suite, "selection_is_separable", default=True)):
        st.info(
            "Model leaderboard: the top models' confidence intervals overlap, so the ranking "
            "reflects sampling noise rather than a demonstrated difference in skill. The winner "
            "is also selected on the same windows it is scored on, which biases its reported "
            "accuracy upward."
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

    suite = value(derived, "model_suite", default=None)
    leaderboard = frame(suite, "leaderboard")
    if not leaderboard.empty:
        with st.container(border=True):
            st.markdown(
                '<div class="sg-section-heading"><h2>Model comparison</h2><span class="sg-muted">Same walk-forward windows</span></div>',
                unsafe_allow_html=True,
            )
            display = leaderboard.copy(deep=True)
            display["model"] = display["model"].str.replace("_", " ").str.title()
            st.dataframe(
                display.style.format({"directional_accuracy": "{:.1%}", "mean_absolute_error": "{:.2%}"}),
                hide_index=True,
                width="stretch",
            )

    backtest = value(derived, "backtest", default=None)
    timeline = frame(backtest, "timeline")
    metrics = value(backtest, "metrics", default=None)
    if metrics is not None and not timeline.empty:
        st.markdown(
            f'<div class="sg-section-heading"><h2>Cost-aware backtest</h2><span class="sg-muted">{completed_run.transaction_cost_bps:.0f} bps · {"long/short" if completed_run.allow_short else "long/cash"}</span></div><section class="sg-scorebar"><div class="sg-score"><div class="sg-score-value">{_percent(value(metrics, "total_return", default=0)):+.1f}%</div><div class="sg-score-label">Strategy return</div></div><div class="sg-score"><div class="sg-score-value">{_percent(value(metrics, "benchmark_return", default=0)):+.1f}%</div><div class="sg-score-label">Buy and hold</div></div><div class="sg-score"><div class="sg-score-value">{float(value(metrics, "sharpe_ratio", default=0)):.2f}</div><div class="sg-score-label">Sharpe ratio</div></div><div class="sg-score"><div class="sg-score-value">{_percent(value(metrics, "max_drawdown", default=0)):.1f}%</div><div class="sg-score-label">Max drawdown</div></div><div class="sg-score"><div class="sg-score-value">{int(value(metrics, "trade_count", default=0))}</div><div class="sg-score-label">Position changes</div></div></section>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_equity_curve_chart(timeline), width="stretch", config={"displayModeBar": False})

    model_name = str(value(evaluation, "model_name", default=completed_run.model))
    export_frame = build_execution_signal_frame(
        predictions,
        ticker=completed_run.ticker,
        model_name=model_name,
        threshold=completed_run.signal_threshold,
    )
    st.download_button(
        "Download C++ signal JSON",
        data=serialize_execution_signals(export_frame),
        file_name=f"signalglass-{completed_run.ticker.lower()}-signals.json",
        mime="application/json",
        width="stretch",
        help="Exports predictions only; realized returns are deliberately excluded.",
    )
    _render_methodology()
    return request


__all__ = ["SignalsLabConfig", "calculate_directional_baseline", "render_signals_lab"]
