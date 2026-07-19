"""Contracts for SignalGlass' arbitrary-symbol picker."""

from __future__ import annotations

import pytest

from signalglass.symbols import build_symbol_choices, record_recent_symbol, resolve_symbol


def test_build_symbol_choices_keeps_selected_and_recent_symbols_first() -> None:
    choices = build_symbol_choices(
        "AMD",
        ("SPY", "AMD", "AAPL"),
        featured=("AAPL", "MSFT", "NVDA", "TSLA"),
        limit=6,
    )

    assert choices == ("AMD", "SPY", "AAPL", "MSFT", "NVDA", "TSLA")


def test_record_recent_symbol_is_normalized_bounded_and_immutable() -> None:
    original = ("AAPL", "MSFT", "NVDA")

    updated = record_recent_symbol(" brk.b ", original, limit=3)

    assert updated == ("BRK.B", "AAPL", "MSFT")
    assert original == ("AAPL", "MSFT", "NVDA")


def test_resolve_symbol_falls_back_with_a_user_facing_error() -> None:
    symbol, error = resolve_symbol("<script>", fallback="AAPL")

    assert symbol == "AAPL"
    assert error == "Enter a valid Yahoo Finance ticker, such as AMD, SPY, BRK.B, or BTC-USD."


def test_symbol_helpers_reject_invalid_limits() -> None:
    with pytest.raises(ValueError, match="limit"):
        record_recent_symbol("AAPL", (), limit=0)


def test_symbol_helpers_honor_a_single_item_limit() -> None:
    assert record_recent_symbol("AAPL", ("MSFT",), limit=1) == ("AAPL",)
    assert build_symbol_choices("AAPL", ("MSFT",), limit=1) == ("AAPL",)
