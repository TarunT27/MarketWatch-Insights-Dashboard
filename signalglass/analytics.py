"""Pure market/sentiment transforms and leakage-free signal evaluation."""

from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import ModelSuite, SignalEvaluation

SENTIMENT_COLUMNS = [
    "date",
    "avg_sentiment",
    "positive_count",
    "negative_count",
    "neutral_count",
    "headline_count",
]
_COUNT_COLUMNS = ["positive_count", "negative_count", "neutral_count", "headline_count"]
_FEATURE_COLUMNS = [
    "avg_sentiment",
    "positive_count",
    "negative_count",
    "neutral_count",
    "headline_count",
    "session_return",
    "sentiment_change",
    "volume_change",
]


def _empty_sentiment_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=SENTIMENT_COLUMNS)


def _effective_news_date(value: object) -> date | None:
    """Resolve publication time to the first date when its signal is tradable."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("America/New_York").tz_localize(None)
    publication_date = timestamp.date()
    if timestamp.time() >= time(16):
        return publication_date + timedelta(days=1)
    return publication_date


def prepare_sentiment_summary(news_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headlines by tradable date, rolling post-close news forward."""

    if not isinstance(news_frame, pd.DataFrame):
        raise TypeError("news data must be a pandas DataFrame")
    if news_frame.empty:
        return _empty_sentiment_summary()

    required = {"publishedAt", "sentiment_score", "sentiment_label"}
    missing = sorted(required.difference(news_frame.columns))
    if missing:
        raise ValueError(f"news schema is missing required columns: {', '.join(missing)}")

    working = news_frame.loc[:, sorted(required)].copy(deep=True)
    working["date"] = working["publishedAt"].map(_effective_news_date)
    working["sentiment_score"] = pd.to_numeric(working["sentiment_score"], errors="coerce")
    working = working.dropna(subset=["date", "sentiment_score"])
    if working.empty:
        return _empty_sentiment_summary()

    working = working.assign(
        sentiment_label=working["sentiment_label"].astype(str).str.strip().str.title(),
    )
    labels = working["sentiment_label"]
    working = working.assign(
        positive_count=labels.eq("Positive").astype("int64"),
        negative_count=labels.eq("Negative").astype("int64"),
        neutral_count=(~labels.isin(["Positive", "Negative"])).astype("int64"),
        headline_count=1,
    )
    summary = (
        working.groupby("date", as_index=False, sort=True)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            positive_count=("positive_count", "sum"),
            negative_count=("negative_count", "sum"),
            neutral_count=("neutral_count", "sum"),
            headline_count=("headline_count", "sum"),
        )
        .loc[:, SENTIMENT_COLUMNS]
    )
    return summary.reset_index(drop=True)


def merge_market_and_sentiment(
    price_frame: pd.DataFrame,
    sentiment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Map calendar-day news to its same or next available trading session."""

    if not isinstance(price_frame, pd.DataFrame) or not isinstance(sentiment_summary, pd.DataFrame):
        raise TypeError("market and sentiment data must be pandas DataFrames")
    if "date" not in price_frame.columns:
        raise ValueError("price schema requires a date column")

    prices = price_frame.copy(deep=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce", utc=True).dt.tz_convert(None)
    prices = prices.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in SENTIMENT_COLUMNS[1:]:
        prices[column] = 0.0 if column == "avg_sentiment" else 0
    if prices.empty or sentiment_summary.empty:
        return prices

    missing = sorted(set(SENTIMENT_COLUMNS).difference(sentiment_summary.columns))
    if missing:
        raise ValueError(f"sentiment schema is missing required columns: {', '.join(missing)}")

    summary = sentiment_summary.loc[:, SENTIMENT_COLUMNS].copy(deep=True)
    summary["date"] = (
        pd.to_datetime(summary["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    )
    summary["avg_sentiment"] = pd.to_numeric(summary["avg_sentiment"], errors="coerce").fillna(0.0)
    for column in _COUNT_COLUMNS:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).clip(lower=0)
    summary = summary.dropna(subset=["date"])

    session_days = prices["date"].dt.normalize().to_numpy(dtype="datetime64[ns]")
    news_days = summary["date"].to_numpy(dtype="datetime64[ns]")
    session_positions = np.searchsorted(session_days, news_days, side="left")
    summary = summary.loc[session_positions < len(prices)].copy()
    if summary.empty:
        return prices
    summary["session_position"] = session_positions[session_positions < len(prices)]
    summary["sentiment_total"] = summary["avg_sentiment"] * summary["headline_count"]
    rolled = summary.groupby("session_position", sort=True).agg(
        sentiment_total=("sentiment_total", "sum"),
        positive_count=("positive_count", "sum"),
        negative_count=("negative_count", "sum"),
        neutral_count=("neutral_count", "sum"),
        headline_count=("headline_count", "sum"),
    )
    rolled["avg_sentiment"] = np.divide(
        rolled["sentiment_total"],
        rolled["headline_count"],
        out=np.zeros(len(rolled), dtype="float64"),
        where=rolled["headline_count"].to_numpy() != 0,
    )

    for position, row in rolled.iterrows():
        integer_position = int(position)
        prices.loc[integer_position, "avg_sentiment"] = float(row["avg_sentiment"])
        for column in _COUNT_COLUMNS:
            prices.loc[integer_position, column] = int(row[column])
    for column in _COUNT_COLUMNS:
        prices[column] = prices[column].astype("int64")
    return prices


def _build_signal_dataset(market_frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = {"date", "Close", "Volume"}
    missing = sorted(required.difference(market_frame.columns))
    if missing:
        raise ValueError(f"market schema is missing required columns: {', '.join(missing)}")

    data = market_frame.copy(deep=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_convert(None)
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    for column in ("avg_sentiment", *_COUNT_COLUMNS):
        if column not in data.columns:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["date", "Close", "Volume"])
    data = data.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    data["session_return"] = data["Close"].pct_change(fill_method=None).fillna(0.0)
    data["sentiment_change"] = data["avg_sentiment"].diff().fillna(0.0)
    data["volume_change"] = (
        data["Volume"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    target_returns = data["Close"].shift(-1).div(data["Close"]).sub(1.0)

    pair_data = data.iloc[:-1].copy()
    pair_data["prediction_date"] = data["date"].shift(-1).iloc[:-1].to_numpy()
    pair_data["target_return"] = target_returns.iloc[:-1].to_numpy()
    finite_mask = np.isfinite(pair_data[_FEATURE_COLUMNS + ["target_return"]].to_numpy(dtype="float64")).all(
        axis=1
    )
    pair_data = pair_data.loc[finite_mask].reset_index(drop=True)
    features = pair_data[_FEATURE_COLUMNS].to_numpy(dtype="float64")
    targets = pair_data["target_return"].to_numpy(dtype="float64")
    return pair_data, features, targets


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval, which stays valid at small sample sizes.

    Directional accuracy is a proportion measured over few dozen sessions, where
    the normal approximation is unreliable and can run past 0 or 1.
    """

    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = z / denominator * np.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4 * trials**2))
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def majority_direction_baseline(actual_returns: np.ndarray) -> float:
    """Accuracy of always predicting whichever direction occurred more often."""

    if actual_returns.size == 0:
        return 0.5
    non_negative_share = float(np.mean(actual_returns >= 0))
    return max(non_negative_share, 1.0 - non_negative_share)


def _model_pipeline(model_name: str):
    if model_name == "linear":
        return Pipeline([("scale", StandardScaler()), ("regression", LinearRegression())])
    if model_name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("regression", Ridge(alpha=1.0))])
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=32,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"unsupported model: {model_name}")


def evaluate_directional_signal(
    market_frame: pd.DataFrame,
    *,
    min_train_size: int = 20,
    model_name: str = "linear",
) -> SignalEvaluation | None:
    """Evaluate next-session returns using expanding-window, out-of-sample fits."""

    if not isinstance(market_frame, pd.DataFrame):
        raise TypeError("market data must be a pandas DataFrame")
    if not isinstance(min_train_size, int) or isinstance(min_train_size, bool) or min_train_size < 2:
        raise ValueError("min_train_size must be an integer of at least two")
    if market_frame.empty:
        return None

    paired, features, targets = _build_signal_dataset(market_frame)
    if len(paired) <= min_train_size:
        return None

    records: list[dict[str, object]] = []
    for test_position in range(min_train_size, len(paired)):
        model = _model_pipeline(model_name)
        model.fit(features[:test_position], targets[:test_position])
        predicted_return = float(model.predict(features[test_position : test_position + 1])[0])
        records.append(
            {
                "date": paired.loc[test_position, "prediction_date"],
                "actual_return": float(targets[test_position]),
                "predicted_return": predicted_return,
            }
        )

    predictions = pd.DataFrame.from_records(
        records,
        columns=["date", "actual_return", "predicted_return"],
    )
    absolute_errors = np.abs(predictions["actual_return"] - predictions["predicted_return"])
    actual_returns = predictions["actual_return"].to_numpy()
    direction_matches = np.sign(actual_returns) == np.sign(predictions["predicted_return"].to_numpy())
    hits = int(direction_matches.sum())
    accuracy_low, accuracy_high = wilson_interval(hits, len(predictions))
    return SignalEvaluation(
        predictions=predictions.reset_index(drop=True),
        mean_absolute_error=float(absolute_errors.mean()),
        directional_accuracy=float(direction_matches.mean()),
        sample_size=len(predictions),
        latest_predicted_return=float(predictions["predicted_return"].iloc[-1]),
        model_name=model_name,
        accuracy_low=accuracy_low,
        accuracy_high=accuracy_high,
        majority_baseline=majority_direction_baseline(actual_returns),
    )


def evaluate_model_suite(
    market_frame: pd.DataFrame,
    *,
    min_train_size: int = 20,
    model_names: tuple[str, ...] = ("linear", "ridge", "random_forest"),
) -> ModelSuite | None:
    """Rank multiple models using identical chronological out-of-sample windows."""

    if not model_names:
        raise ValueError("model suite must contain at least one model")
    evaluations = tuple(
        evaluation
        for name in model_names
        if (
            evaluation := evaluate_directional_signal(
                market_frame,
                min_train_size=min_train_size,
                model_name=name,
            )
        )
        is not None
    )
    if not evaluations:
        return None
    leaderboard = (
        pd.DataFrame(
            [
                {
                    "model": item.model_name,
                    "directional_accuracy": item.directional_accuracy,
                    "accuracy_low": item.accuracy_low,
                    "accuracy_high": item.accuracy_high,
                    "majority_baseline": item.majority_baseline,
                    "verdict": item.verdict,
                    "mean_absolute_error": item.mean_absolute_error,
                    "sample_size": item.sample_size,
                }
                for item in evaluations
            ]
        )
        .sort_values(
            ["directional_accuracy", "mean_absolute_error", "model"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    # The winner is chosen on the same windows it is scored on, so a lead that
    # sits inside the runner-up's interval is selection noise, not skill.
    selection_is_separable = bool(
        len(leaderboard) < 2
        or leaderboard.loc[0, "accuracy_low"] > leaderboard.loc[1, "directional_accuracy"]
    )
    return ModelSuite(
        evaluations=evaluations,
        leaderboard=leaderboard,
        best_model=str(leaderboard.loc[0, "model"]),
        selection_is_separable=selection_is_separable,
    )


__all__ = [
    "SENTIMENT_COLUMNS",
    "evaluate_directional_signal",
    "evaluate_model_suite",
    "majority_direction_baseline",
    "merge_market_and_sentiment",
    "prepare_sentiment_summary",
    "wilson_interval",
]
