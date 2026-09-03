"""Range selection, safe outbound links, and responsive-shell invariants."""

from __future__ import annotations

import re

import pytest

from signalglass.theme import build_styles
from signalglass.ui import PAGES
from signalglass.ui._data import company_name, link, safe_url
from signalglass.windows import DEFAULT_RANGE, RANGE_LABELS, resolve_range


@pytest.mark.parametrize("label", RANGE_LABELS)
def test_every_offered_range_resolves_to_a_distinct_window(label: str) -> None:
    window = resolve_range(label)

    assert window.label == label
    assert window.trading_days >= 22
    assert window.calendar_days > window.trading_days


def test_ranges_are_ordered_from_shortest_to_longest() -> None:
    spans = [resolve_range(label).trading_days for label in RANGE_LABELS]

    assert spans == sorted(spans)


@pytest.mark.parametrize("bad", ["", "5Y", None, 3, "  "])
def test_unknown_range_labels_fall_back_to_the_default(bad: object) -> None:
    assert resolve_range(bad).label == DEFAULT_RANGE


def test_range_labels_are_case_insensitive() -> None:
    assert resolve_range("1y").label == "1Y"


@pytest.mark.parametrize(
    "raw",
    [
        "javascript:alert(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "file:///etc/passwd",
        "",
        None,
        "not a url",
    ],
)
def test_unsafe_or_missing_urls_are_dropped(raw: object) -> None:
    assert safe_url(raw) == ""


@pytest.mark.parametrize(
    "raw",
    ["https://example.com/story", "http://news.example.org/a/b?c=d"],
)
def test_http_urls_survive(raw: str) -> None:
    assert safe_url(raw) == raw


def test_link_renders_an_anchor_for_a_safe_url() -> None:
    rendered = link("Headline", "https://example.com/story")

    assert 'href="https://example.com/story"' in rendered
    assert 'rel="noopener noreferrer nofollow"' in rendered
    assert "Headline" in rendered


def test_link_degrades_to_plain_text_for_an_unsafe_url() -> None:
    rendered = link("Headline", "javascript:alert(1)")

    assert "<a" not in rendered
    assert "javascript" not in rendered
    assert "Headline" in rendered


def test_link_escapes_markup_in_the_headline() -> None:
    rendered = link("<script>bad()</script>", "https://example.com")

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_mobile_navigation_has_a_column_for_every_page() -> None:
    """A 4-column grid with 5 destinations silently hides the last one."""

    styles = build_styles()
    match = re.search(r"\.sg-nav \{[^}]*grid-template-columns:repeat\((\d+),1fr\)", styles)

    assert match is not None, "mobile nav should use an explicit column count"
    assert int(match.group(1)) == len(PAGES)


def test_headline_metric_colour_is_not_hardcoded_to_positive() -> None:
    """The accuracy figure must be coloured by verdict, not by position."""

    styles = build_styles()

    assert ".sg-score:first-child .sg-score-value" not in styles
    assert ".sg-score-headline" in styles


def test_company_name_falls_back_to_the_ticker() -> None:
    assert company_name("AAPL") == "Apple Inc."
    assert company_name("zzzz") == "ZZZZ"
