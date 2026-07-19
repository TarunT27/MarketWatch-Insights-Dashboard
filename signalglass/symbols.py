"""Pure helpers for arbitrary Yahoo-compatible symbol selection."""

from __future__ import annotations

from collections.abc import Iterable

from .validation import validate_ticker

DEFAULT_SYMBOL = "AAPL"
FEATURED_SYMBOLS = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "SPY")
INVALID_SYMBOL_MESSAGE = "Enter a valid Yahoo Finance ticker, such as AMD, SPY, BRK.B, or BTC-USD."


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("symbol limit must be a positive integer")


def resolve_symbol(candidate: object, *, fallback: object = DEFAULT_SYMBOL) -> tuple[str, str | None]:
    """Return a normalized symbol or a validated fallback with display-safe copy."""

    fallback_symbol = validate_ticker(fallback)
    try:
        return validate_ticker(candidate), None
    except (TypeError, ValueError):
        return fallback_symbol, INVALID_SYMBOL_MESSAGE


def record_recent_symbol(symbol: object, recent: Iterable[object], *, limit: int = 6) -> tuple[str, ...]:
    """Return a new bounded MRU sequence without mutating caller-owned state."""

    _validate_limit(limit)
    normalized = validate_ticker(symbol)
    ordered = [normalized]
    for candidate in recent:
        if len(ordered) >= limit:
            break
        try:
            candidate_symbol = validate_ticker(candidate)
        except (TypeError, ValueError):
            continue
        if candidate_symbol not in ordered:
            ordered.append(candidate_symbol)
    return tuple(ordered)


def build_symbol_choices(
    selected: object,
    recent: Iterable[object],
    *,
    featured: Iterable[object] = FEATURED_SYMBOLS,
    limit: int = 8,
) -> tuple[str, ...]:
    """Build deduplicated picker choices with the active symbol first."""

    _validate_limit(limit)
    selected_symbol = validate_ticker(selected)
    ordered = [selected_symbol]
    for candidate in (*tuple(recent), *tuple(featured)):
        if len(ordered) >= limit:
            break
        try:
            candidate_symbol = validate_ticker(candidate)
        except (TypeError, ValueError):
            continue
        if candidate_symbol not in ordered:
            ordered.append(candidate_symbol)
    return tuple(ordered)


__all__ = [
    "DEFAULT_SYMBOL",
    "FEATURED_SYMBOLS",
    "INVALID_SYMBOL_MESSAGE",
    "build_symbol_choices",
    "record_recent_symbol",
    "resolve_symbol",
]
