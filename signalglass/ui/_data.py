"""Small, side-effect-free presentation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from html import escape
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from signalglass.demo_data import demo_company_name


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
    """Display name for a symbol, falling back to the ticker when unknown."""

    return demo_company_name(ticker) or ticker.upper()


def safe_url(raw: Any) -> str:
    """Return an http(s) URL safe to place in an href, or an empty string.

    Headline URLs arrive from an external news provider, so anything that is not
    plainly http/https is dropped rather than rendered into the page.
    """

    candidate = str(raw or "").strip()
    if not candidate:
        return ""
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return candidate


def link(label: Any, raw_url: Any, *, css_class: str = "sg-list-title") -> str:
    """Render a headline as a real link when the source URL is usable."""

    destination = safe_url(raw_url)
    text = html(label)
    if not destination:
        return f'<div class="{css_class}">{text}</div>'
    return (
        f'<a class="{css_class} sg-headline-link" href="{html(destination)}" '
        f'target="_blank" rel="noopener noreferrer nofollow">{text}</a>'
    )


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
