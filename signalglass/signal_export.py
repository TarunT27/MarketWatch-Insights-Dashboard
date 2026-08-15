"""Stable research-signal export for downstream execution simulators."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .validation import validate_ticker

_EXPORT_COLUMNS = ["date", "symbol", "model", "action", "score"]


def build_execution_signal_frame(
    predictions: pd.DataFrame,
    *,
    ticker: object,
    model_name: str,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Create an execution-safe signal frame without realized outcomes."""

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    required = {"date", "predicted_return"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"prediction schema is missing required columns: {', '.join(missing)}")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model name must be a non-empty string")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    symbol = validate_ticker(ticker)
    frame = predictions.loc[:, ["date", "predicted_return"]].copy(deep=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
    frame["score"] = pd.to_numeric(frame["predicted_return"], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if not np.isfinite(frame["score"].to_numpy(dtype="float64")).all():
        raise ValueError("predicted returns must be finite")
    frame["symbol"] = symbol
    frame["model"] = model_name.strip().lower()
    frame["action"] = np.where(
        frame["score"] > threshold,
        "BUY",
        np.where(frame["score"] < -threshold, "SELL", "HOLD"),
    )
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.loc[:, _EXPORT_COLUMNS]


def serialize_execution_signals(signal_frame: pd.DataFrame) -> str:
    """Serialize the documented SignalGlass-to-C++ interchange format."""

    if not isinstance(signal_frame, pd.DataFrame) or list(signal_frame.columns) != _EXPORT_COLUMNS:
        raise ValueError("signal frame does not match the execution export schema")
    payload = {
        "schema_version": "signalglass.execution.v1",
        "signals": signal_frame.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = ["build_execution_signal_frame", "serialize_execution_signals"]
