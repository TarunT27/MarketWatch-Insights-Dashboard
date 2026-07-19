"""Behavior tests for the pure Plotly chart builders."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

from signalglass.charts import (
    make_compare_chart,
    make_overview_chart,
    make_signal_evaluation_chart,
)


@pytest.mark.parametrize(
    ("builder", "expected_message"),
    [
        (make_overview_chart, "Price history is unavailable for this range."),
        (make_signal_evaluation_chart, "Run an evaluation to inspect the signal."),
        (make_compare_chart, "Choose at least one company to compare."),
    ],
)
def test_chart_builders_return_readable_empty_states(
    builder: Callable[..., go.Figure], expected_message: str
) -> None:
    figure = builder(None, height=240)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 0
    assert figure.layout.height == 240
    assert figure.layout.annotations[0].text == expected_message
    assert figure.layout.xaxis.visible is False
    assert figure.layout.yaxis.visible is False


def test_overview_chart_combines_price_volume_sentiment_and_valid_events(price_frame_factory) -> None:
    market = price_frame_factory(5).assign(avg_sentiment=[0.2, -0.4, 0.0, 0.5, -0.1])
    original = market.copy(deep=True)
    events = [
        {"date": "2025-01-06", "label": "Earnings call"},
        {"date": "not-a-date", "label": "Ignored event"},
    ]

    figure = make_overview_chart(market, events)

    assert [trace.name for trace in figure.data] == [
        "Price (USD)",
        "Volume",
        "Sentiment (News Pulse)",
        "Earnings call",
    ]
    assert figure.data[0].type == "scatter"
    assert figure.data[1].type == "bar"
    assert figure.data[2].marker.color[0] != figure.data[2].marker.color[1]
    assert figure.data[3].text == ("E",)
    pd.testing.assert_frame_equal(market, original)


def test_overview_chart_rejects_rows_that_cannot_form_a_price_series() -> None:
    unusable = pd.DataFrame({"date": ["not-a-date"], "Close": ["not-a-price"]})

    figure = make_overview_chart(unusable)

    assert len(figure.data) == 0
    assert figure.layout.annotations[0].text == "Price history is unavailable for this range."


def test_signal_evaluation_chart_scales_actual_and_predicted_returns() -> None:
    evaluation = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-02", periods=3),
            "actual_return": [0.01, -0.025, 0.0],
            "predicted_return": [0.02, -0.02, 0.0025],
        }
    )

    figure = make_signal_evaluation_chart(evaluation)

    assert [trace.name for trace in figure.data] == ["Actual return", "Predicted return"]
    assert list(figure.data[0].y) == pytest.approx([1.0, -2.5, 0.0])
    assert list(figure.data[1].y) == pytest.approx([2.0, -2.0, 0.25])
    assert figure.layout.yaxis.ticksuffix == "%"
    assert figure.data[0].marker.color[0] != figure.data[0].marker.color[1]


def test_compare_chart_rebases_each_usable_series_without_mutating_inputs(price_frame_factory) -> None:
    apple = price_frame_factory(4)
    microsoft = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-02", periods=4, freq="B"),
            "close": [200.0, 202.0, 198.0, 210.0],
        }
    )
    original_apple = apple.copy(deep=True)
    original_microsoft = microsoft.copy(deep=True)

    figure = make_compare_chart(
        {
            "AAPL": apple,
            "MSFT": microsoft,
            "EMPTY": pd.DataFrame({"date": ["2025-01-02"], "Close": [0.0]}),
        }
    )

    assert [trace.name for trace in figure.data] == ["AAPL", "MSFT"]
    assert figure.data[0].y[0] == pytest.approx(0.0)
    assert figure.data[1].y[-1] == pytest.approx(5.0)
    pd.testing.assert_frame_equal(apple, original_apple)
    pd.testing.assert_frame_equal(microsoft, original_microsoft)


def test_compare_chart_groups_a_long_form_dataframe_by_ticker() -> None:
    series = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"] * 2),
            "Close": [100.0, 110.0, 50.0, 45.0],
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    figure = make_compare_chart(series)

    assert {trace.name for trace in figure.data} == {"AAPL", "MSFT"}
    endpoints = {trace.name: trace.y[-1] for trace in figure.data}
    assert endpoints == pytest.approx({"AAPL": 10.0, "MSFT": -10.0})
