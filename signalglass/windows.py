"""Time-range selection shared by the data loader and the workspaces.

The 1M/3M/6M/1Y control is a real input: the label chosen here decides how much
history is fetched, not just how much of a fixed window is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RANGE = "3M"


@dataclass(frozen=True, slots=True)
class RangeWindow:
    """One selectable history window."""

    label: str
    calendar_days: int
    trading_days: int


_WINDOWS = (
    RangeWindow("1M", 45, 22),
    RangeWindow("3M", 135, 65),
    RangeWindow("6M", 260, 130),
    RangeWindow("1Y", 500, 252),
)

RANGE_LABELS = tuple(window.label for window in _WINDOWS)
_BY_LABEL = {window.label: window for window in _WINDOWS}


def resolve_range(label: object) -> RangeWindow:
    """Return the requested window, falling back to the default label."""

    if isinstance(label, str):
        window = _BY_LABEL.get(label.strip().upper())
        if window is not None:
            return window
    return _BY_LABEL[DEFAULT_RANGE]


__all__ = ["DEFAULT_RANGE", "RANGE_LABELS", "RangeWindow", "resolve_range"]
