"""Pure analytics contracts for sessions and leakage-free signal evaluation."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def analytics_module():
    return import_module("signalglass.analytics")


def test_prepare_sentiment_summary_returns_stable_daily_schema() -> None:
    news = pd.DataFrame(
        {
            "publishedAt": pd.to_datetime(["2025-01-04 10:00", "2025-01-04 14:00", "2025-01-05 09:00"]),
            "sentiment_score": [0.8, -0.4, 0.0],
            "sentiment_label": ["Positive", "Negative", "Neutral"],
        }
    )

    result = analytics_module().prepare_sentiment_summary(news)

    assert list(result.columns) == [
        "date",
        "avg_sentiment",
        "positive_count",
        "negative_count",
        "neutral_count",
        "headline_count",
    ]
    assert result["headline_count"].sum() == 3
    assert result.loc[
        result["date"] == pd.Timestamp("2025-01-04").date(), "avg_sentiment"
    ].item() == pytest.approx(0.2)


def test_merge_market_and_sentiment_rolls_weekend_news_into_next_session(price_frame_factory) -> None:
    prices = price_frame_factory(3)
    prices["date"] = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"])
    summary = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-04").date(), pd.Timestamp("2025-01-05").date()],
            "avg_sentiment": [0.8, -0.2],
            "positive_count": [2, 0],
            "negative_count": [0, 1],
            "neutral_count": [0, 1],
            "headline_count": [2, 2],
        }
    )

    result = analytics_module().merge_market_and_sentiment(prices, summary)
    monday = result.loc[result["date"] == pd.Timestamp("2025-01-06")].iloc[0]

    assert len(result) == len(prices)
    assert monday["headline_count"] == 4
    assert monday["positive_count"] == 2
    assert monday["negative_count"] == 1
    assert monday["neutral_count"] == 1
    assert monday["avg_sentiment"] == pytest.approx(0.3)


def test_merge_market_and_sentiment_fills_sessions_without_news_with_zeros(price_frame_factory) -> None:
    prices = price_frame_factory(3)
    summary = pd.DataFrame(
        columns=[
            "date",
            "avg_sentiment",
            "positive_count",
            "negative_count",
            "neutral_count",
            "headline_count",
        ]
    )

    result = analytics_module().merge_market_and_sentiment(prices, summary)

    assert result["headline_count"].tolist() == [0, 0, 0]
    assert result["avg_sentiment"].tolist() == [0.0, 0.0, 0.0]


def test_signal_evaluation_is_chronological_and_prefix_invariant(combined_market_frame) -> None:
    evaluate = analytics_module().evaluate_directional_signal

    prefix = combined_market_frame.iloc[:11].copy()
    prefix_evaluation = evaluate(prefix, min_train_size=5)
    full_evaluation = evaluate(combined_market_frame, min_train_size=5)

    assert prefix_evaluation is not None
    assert full_evaluation is not None
    prefix_predictions = prefix_evaluation.predictions.reset_index(drop=True)
    matching_full_predictions = full_evaluation.predictions.iloc[: len(prefix_predictions)].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(prefix_predictions, matching_full_predictions)


def test_signal_evaluation_reports_honest_out_of_sample_metrics(combined_market_frame) -> None:
    evaluation = analytics_module().evaluate_directional_signal(combined_market_frame, min_train_size=5)

    assert evaluation is not None
    predictions = evaluation.predictions
    assert list(predictions.columns) == ["date", "actual_return", "predicted_return"]
    assert predictions["date"].is_monotonic_increasing
    assert predictions["date"].is_unique
    assert evaluation.sample_size == len(predictions)
    assert evaluation.sample_size > 0

    expected_mae = np.mean(np.abs(predictions["actual_return"] - predictions["predicted_return"]))
    expected_accuracy = np.mean(
        np.sign(predictions["actual_return"].to_numpy())
        == np.sign(predictions["predicted_return"].to_numpy())
    )
    assert evaluation.mean_absolute_error == pytest.approx(expected_mae)
    assert evaluation.directional_accuracy == pytest.approx(expected_accuracy)
    assert 0.0 <= evaluation.directional_accuracy <= 1.0
    assert not hasattr(evaluation, "confidence"), "In-sample R-squared must not be presented as confidence"


def test_signal_evaluation_returns_none_when_history_is_insufficient(combined_market_frame) -> None:
    result = analytics_module().evaluate_directional_signal(
        combined_market_frame.iloc[:5],
        min_train_size=5,
    )

    assert result is None


def test_model_suite_compares_multiple_walk_forward_models_without_leakage(
    combined_market_frame,
) -> None:
    original = combined_market_frame.copy(deep=True)

    suite = analytics_module().evaluate_model_suite(combined_market_frame, min_train_size=5)

    pd.testing.assert_frame_equal(combined_market_frame, original)
    assert suite is not None
    assert set(suite.leaderboard["model"]) == {"linear", "ridge", "random_forest"}
    assert suite.best_model in set(suite.leaderboard["model"])
    assert suite.leaderboard["directional_accuracy"].between(0, 1).all()
    assert suite.leaderboard["mean_absolute_error"].ge(0).all()
    assert suite.leaderboard["sample_size"].nunique() == 1
    assert tuple(suite.leaderboard["model"]) == tuple(
        suite.leaderboard.sort_values(
            ["directional_accuracy", "mean_absolute_error", "model"],
            ascending=[False, True, True],
        )["model"]
    )
