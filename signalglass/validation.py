"""Validation helpers for values crossing the application boundary."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime

_MAX_TICKER_LENGTH = 20
_TICKER_PATTERN = re.compile(r"\^?[A-Z0-9][A-Z0-9.=-]*(?:-[A-Z0-9.=-]+)*\Z")


def validate_ticker(raw_ticker: object) -> str:
    """Normalize one Yahoo Finance-compatible symbol or reject unsafe input."""

    if not isinstance(raw_ticker, str):
        raise TypeError("ticker symbol must be a string")

    ticker = raw_ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker symbol is required")
    if len(ticker) > _MAX_TICKER_LENGTH or _TICKER_PATTERN.fullmatch(ticker) is None:
        raise ValueError("ticker symbol is not valid")
    return ticker


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError("date range values must be date or datetime objects")


def normalize_date_range(selection: object) -> tuple[date, date]:
    """Normalize Streamlit's single or pair date selection into an ordered pair."""

    if isinstance(selection, (date, datetime)):
        selected_date = _as_date(selection)
        return selected_date, selected_date

    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
        values = tuple(selection)
        if len(values) not in (1, 2):
            raise ValueError("date range must contain one or two dates")
        start_date = _as_date(values[0])
        end_date = _as_date(values[-1])
        if start_date > end_date:
            raise ValueError("date range start must not be after end")
        return start_date, end_date

    raise TypeError("date range must be a date or a sequence of dates")


__all__ = ["normalize_date_range", "validate_ticker"]
