"""Small local SQLite store for user-controlled research preferences."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path

from .portfolio import normalize_allocations
from .validation import validate_ticker


class LocalResearchStore:
    """Persist a local watchlist and allocation without storing credentials."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=3)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS watchlist (symbol TEXT PRIMARY KEY, position INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS allocations (symbol TEXT PRIMARY KEY, weight REAL NOT NULL CHECK(weight > 0 AND weight <= 1))"
            )

    def replace_watchlist(self, symbols: Iterable[object]) -> None:
        validated = tuple(dict.fromkeys(validate_ticker(symbol) for symbol in symbols))
        with self._connect() as connection:
            connection.execute("DELETE FROM watchlist")
            connection.executemany(
                "INSERT INTO watchlist(symbol, position) VALUES (?, ?)",
                ((symbol, position) for position, symbol in enumerate(validated)),
            )

    def load_watchlist(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT symbol FROM watchlist ORDER BY position").fetchall()
        return tuple(str(row[0]) for row in rows)

    def replace_allocations(self, allocations: Mapping[object, object]) -> None:
        normalized = normalize_allocations(allocations)
        with self._connect() as connection:
            connection.execute("DELETE FROM allocations")
            connection.executemany(
                "INSERT INTO allocations(symbol, weight) VALUES (?, ?)", normalized.items()
            )

    def load_allocations(self) -> dict[str, float]:
        with self._connect() as connection:
            rows = connection.execute("SELECT symbol, weight FROM allocations ORDER BY symbol").fetchall()
        return {str(symbol): float(weight) for symbol, weight in rows}


__all__ = ["LocalResearchStore"]
