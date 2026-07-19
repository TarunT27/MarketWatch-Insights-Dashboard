"""Small, side-effect-free presentation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from html import escape
from typing import Any

import pandas as pd


def value(source: Any, *names: str, default: Any = None) -> Any:
    """Read the first available key/attribute from a mapping or result object."""

    if source is None:
        return default
    for name in names:
        if isinstance(source, Mapping) and name in source:
            candidate = source[name]
        elif hasattr(source, name):
            candidate = getattr(source, name)
        else:
            continue
        if candidate is not None:
            return candidate
    return default


def frame(source: Any, *names: str) -> pd.DataFrame:
    candidate = value(source, *names, default=source if isinstance(source, pd.DataFrame) else None)
    return candidate.copy(deep=True) if isinstance(candidate, pd.DataFrame) else pd.DataFrame()


def records(source: Any, *names: str, limit: int | None = None) -> list[dict[str, Any]]:
    candidate = value(source, *names, default=[])
    if isinstance(candidate, pd.DataFrame):
        items = candidate.to_dict(orient="records")
    elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        items = []
        for item in candidate:
            if isinstance(item, Mapping):
                items.append(dict(item))
            elif is_dataclass(item):
                items.append(asdict(item))
            elif hasattr(item, "__dict__"):
                items.append(vars(item).copy())
    else:
        items = []
    return items[:limit] if limit is not None else items


def html(value_to_escape: Any) -> str:
    return escape(str(value_to_escape or ""), quote=True)


def company_name(ticker: str) -> str:
    return {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "NVDA": "NVIDIA Corp.",
        "TSLA": "Tesla, Inc.",
        "AMZN": "Amazon.com, Inc.",
        "GOOGL": "Alphabet Inc.",
        "META": "Meta Platforms, Inc.",
    }.get(ticker.upper(), ticker.upper())


def format_compact(number: float | int | None) -> str:
    if number is None or pd.isna(number):
        return "—"
    amount = float(number)
    for divisor, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(amount) >= divisor:
            return f"{amount / divisor:.1f}{suffix}"
    return f"{amount:,.0f}"


def format_date(raw: Any) -> str:
    timestamp = pd.to_datetime(raw, errors="coerce")
    return f"{timestamp.strftime('%b')} {timestamp.day}, {timestamp.year}" if not pd.isna(timestamp) else ""
