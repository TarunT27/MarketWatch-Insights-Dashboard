"""Tests for user-input validation at the app boundary."""

from __future__ import annotations

from datetime import date, datetime
from importlib import import_module

import pytest


def validation_module():
    return import_module("signalglass.validation")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" aapl ", "AAPL"),
        ("brk.b", "BRK.B"),
        ("btc-usd", "BTC-USD"),
        ("^gspc", "^GSPC"),
    ],
)
def test_validate_ticker_normalizes_supported_market_symbols(raw: str, expected: str) -> None:
    assert validation_module().validate_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "AAPL<script>", "../AAPL", "AAPL,MSFT", "A" * 25, None])
def test_validate_ticker_rejects_empty_or_unsafe_values(raw: object) -> None:
    with pytest.raises((TypeError, ValueError), match="ticker|symbol|valid|required"):
        validation_module().validate_ticker(raw)


def test_normalize_date_range_accepts_dates_and_datetimes() -> None:
    normalize = validation_module().normalize_date_range

    assert normalize((datetime(2025, 1, 2, 9, 30), datetime(2025, 1, 31, 16, 0))) == (
        date(2025, 1, 2),
        date(2025, 1, 31),
    )


def test_normalize_date_range_handles_partial_streamlit_selection_without_unpacking_error() -> None:
    selected = date(2025, 1, 15)

    assert validation_module().normalize_date_range(selected) == (selected, selected)


def test_normalize_date_range_rejects_reversed_or_malformed_ranges() -> None:
    normalize = validation_module().normalize_date_range

    with pytest.raises(ValueError, match="date|range|start|end"):
        normalize((date(2025, 2, 1), date(2025, 1, 1)))
    with pytest.raises((TypeError, ValueError), match="date|range|two"):
        normalize(())
