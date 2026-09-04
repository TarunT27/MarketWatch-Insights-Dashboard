"""SignalGlass app shell components."""

from __future__ import annotations

from html import escape
from urllib.parse import quote

import streamlit as st

from signalglass.theme import apply_theme

PAGES = ("Overview", "Compare", "Intelligence", "Portfolio", "Signals Lab")


def _wave_mark() -> str:
    return """<svg viewBox="0 0 46 30" role="img" aria-label="SignalGlass waveform mark"><path d="M1 16h5l3-7 4 16 4-23 5 27 5-24 4 19 4-12 4 7h6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""


def _nav_item(page: str, active: str, symbol: str | None = None) -> str:
    class_name = "sg-nav-item is-active" if page == active else "sg-nav-item"
    current = ' aria-current="page"' if page == active else ""
    symbol_query = f"&symbol={quote(symbol)}" if symbol else ""
    return f'<a class="{class_name}" href="?page={quote(page)}{symbol_query}" target="_self"{current}>{escape(page)}</a>'


def render_app_header(
    active_page: str = "Overview",
    *,
    updated_at: str | None = None,
    data_label: str = "Demo data",
    symbol: str | None = None,
) -> None:
    """Render the responsive application header from the approved concepts."""

    apply_theme()
    active = active_page if active_page in PAGES else "Overview"
    nav = "".join(_nav_item(page, active, symbol) for page in PAGES)
    updated = f'<span class="sg-updated">Updated {escape(updated_at)}</span>' if updated_at else ""
    st.markdown(
        f"""
        <header class="sg-header">
          <div class="sg-brand"><span class="sg-brand-mark">{_wave_mark()}</span><span>SignalGlass</span></div>
          <nav class="sg-nav" aria-label="Primary navigation">{nav}</nav>
          <div class="sg-status"><span class="sg-status-dot" aria-hidden="true"></span><span>{escape(data_label)}</span>{updated}</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<footer class="sg-footer">Research prototype · Not financial advice</footer>', unsafe_allow_html=True
    )


def render_page_title(title: str, description: str | None = None) -> None:
    description_html = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f'<div class="sg-lab-heading"><div><h1>{escape(title)}</h1>{description_html}</div></div>',
        unsafe_allow_html=True,
    )


__all__ = ["PAGES", "render_app_header", "render_footer", "render_page_title"]
