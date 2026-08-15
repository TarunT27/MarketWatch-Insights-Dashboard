"""Public Streamlit presentation API for SignalGlass."""

from .compare import render_compare
from .intelligence import render_intelligence
from .overview import render_overview
from .portfolio import PortfolioRequest, render_portfolio
from .shell import PAGES, render_app_header, render_footer, render_page_title
from .signals_lab import SignalsLabConfig, render_signals_lab

__all__ = [
    "PAGES",
    "PortfolioRequest",
    "SignalsLabConfig",
    "render_app_header",
    "render_compare",
    "render_footer",
    "render_intelligence",
    "render_overview",
    "render_portfolio",
    "render_page_title",
    "render_signals_lab",
]
