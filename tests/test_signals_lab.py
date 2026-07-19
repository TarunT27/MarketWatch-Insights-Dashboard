"""Signals Lab presentation-state contracts."""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from signalglass.ui.signals_lab import calculate_directional_baseline


def _copy(app: AppTest) -> str:
    values: list[str] = []
    for element_type in ("markdown", "caption", "info", "text"):
        for element in getattr(app, element_type, []):
            values.append(str(getattr(element, "value", "")))
    return "\n".join(values)


def test_directional_baseline_uses_majority_direction_in_test_predictions() -> None:
    predictions = pd.DataFrame({"actual_return": [0.03, 0.0, -0.01, 0.02]})

    assert calculate_directional_baseline(predictions) == pytest.approx(0.75)
    assert calculate_directional_baseline(pd.DataFrame()) == 0.0


def test_signal_lab_returns_selected_configuration_and_gates_results_until_completed() -> None:
    script = """
import pandas as pd
import streamlit as st
from signalglass.ui.signals_lab import render_signals_lab

prices = pd.DataFrame({
    "date": pd.date_range("2025-01-02", periods=90, freq="B"),
    "Close": range(100, 190),
    "Volume": range(1000, 1090),
})
predictions = pd.DataFrame({
    "date": pd.date_range("2025-04-01", periods=4, freq="B"),
    "actual_return": [0.01, 0.02, -0.01, 0.03],
    "predicted_return": [0.02, 0.01, -0.02, 0.01],
})
bundle = {"ticker": "AAPL", "prices": prices, "news": pd.DataFrame()}
derived = {
    "available_tickers": ["AAPL", "MSFT"],
    "evaluation": {
        "predictions": predictions,
        "directional_accuracy": .75,
        "mean_absolute_error": .01,
        "sample_size": 4,
    },
}
request = render_signals_lab(bundle, derived)
st.write(f"REQUEST={request.ticker}|{request.coverage_days}|{request.model}|{request.run_requested}")
"""
    app = AppTest.from_string(script, default_timeout=10).run(timeout=10)

    initial_copy = _copy(app)
    assert "Run an evaluation to see out-of-sample results." in initial_copy
    assert "directional accuracy" not in initial_copy
    assert "REQUEST=AAPL|90|directional_baseline|False" in initial_copy

    app.selectbox[0].set_value("MSFT")
    app.selectbox[1].set_value("30 trading days")
    app.button[0].click().run(timeout=10)

    assert "REQUEST=MSFT|30|directional_baseline|True" in _copy(app)


def test_signal_lab_renders_results_only_for_matching_completed_configuration() -> None:
    script = """
import pandas as pd
from signalglass.ui.signals_lab import SignalsLabConfig, render_signals_lab

prices = pd.DataFrame({
    "date": pd.date_range("2025-01-02", periods=90, freq="B"),
    "Close": range(100, 190),
    "Volume": range(1000, 1090),
})
predictions = pd.DataFrame({
    "date": pd.date_range("2025-04-01", periods=4, freq="B"),
    "actual_return": [0.01, 0.02, -0.01, 0.03],
    "predicted_return": [0.02, 0.01, -0.02, 0.01],
})
bundle = {"ticker": "AAPL", "prices": prices, "news": pd.DataFrame()}
derived = {"evaluation": {"predictions": predictions, "directional_accuracy": .75, "mean_absolute_error": .01, "sample_size": 4}}
completed = SignalsLabConfig(ticker="AAPL", coverage_days=90, model="directional_baseline")
render_signals_lab(bundle, derived, completed_run=completed)
"""
    app = AppTest.from_string(script, default_timeout=10).run(timeout=10)
    rendered = _copy(app)

    assert "directional accuracy" in rendered
    assert "75.0%" in rendered
    assert "Market feature snapshot" in rendered
    assert "75.0%" in rendered  # observed up/non-negative majority baseline
    assert "standardized model contribution" not in rendered.lower()
