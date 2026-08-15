"""Streamlit smoke and recruiter-facing navigation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from signalglass import providers

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SECRET_ENV_VARS = (
    "NEWSAPI_KEY",
    "SMTP_SERVER",
    "SMTP_PORT",
    "EMAIL_USERNAME",
    "EMAIL_PASSWORD",
    "EMAIL_SENDER",
)


def _rendered_copy(app: AppTest) -> str:
    """Collect user-visible values, labels, and options without CSS selectors."""

    rendered: list[str] = []
    element_types = (
        "title",
        "header",
        "subheader",
        "caption",
        "markdown",
        "text",
        "info",
        "warning",
        "success",
        "error",
        "tabs",
        "button",
        "button_group",
        "segmented_control",
        "radio",
        "selectbox",
        "pills",
    )
    for element_type in element_types:
        for element in getattr(app, element_type):
            for attribute in ("value", "label"):
                value = getattr(element, attribute, None)
                if value is not None:
                    rendered.append(str(value))
            options = getattr(element, "options", None)
            if options:
                rendered.extend(str(option) for option in options)
    return "\n".join(rendered)


@pytest.fixture
def keyless_app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    st.cache_data.clear()
    for name in SECRET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SIGNALGLASS_DATA_MODE", "demo")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20)
    return app.run(timeout=20)


def test_app_starts_without_secrets_or_uncaught_exceptions(keyless_app: AppTest) -> None:
    assert not keyless_app.exception
    assert not any("Traceback" in str(error.value) for error in keyless_app.error)


def test_app_renders_signalglass_navigation_and_trust_copy(keyless_app: AppTest) -> None:
    copy = _rendered_copy(keyless_app)

    for required_copy in (
        "SignalGlass",
        "Overview",
        "Compare",
        "Intelligence",
        "Portfolio",
        "Signals Lab",
        "Demo data",
        "Why it moved",
        "Research prototype · Not financial advice",
    ):
        assert required_copy in copy


def test_user_can_enter_any_valid_yahoo_symbol_and_keep_it_in_recent_choices(
    keyless_app: AppTest,
) -> None:
    app = keyless_app
    app.query_params["symbol"] = "AMD"
    app.run(timeout=20)

    assert not app.exception
    rendered = _rendered_copy(app)
    assert "AMD" in rendered
    assert "symbol=AMD" in rendered
    refreshed_picker = next(control for control in app.selectbox if control.label == "Search symbol")
    assert "AMD" in refreshed_picker.options


def test_invalid_custom_symbol_is_rejected_without_replacing_last_valid_selection(
    keyless_app: AppTest,
) -> None:
    app = keyless_app
    app.query_params["symbol"] = "../AAPL"
    app.run(timeout=20)

    assert not app.exception
    rendered = _rendered_copy(app)
    assert "Enter a valid Yahoo Finance ticker" in rendered
    assert "AAPL" in rendered

    symbol_picker = next(control for control in app.selectbox if control.label == "Search symbol")
    symbol_picker.set_value("MSFT")
    app.run(timeout=20)

    recovered_copy = _rendered_copy(app)
    assert "Enter a valid Yahoo Finance ticker" not in recovered_copy
    assert app.query_params["symbol"] == ["MSFT"]


def test_custom_symbol_uses_keyless_yahoo_prices_and_dynamic_comparison_peers(
    keyless_app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_symbols: list[str] = []
    live_prices = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-06-01", periods=30),
            "Open": [100.0 + index for index in range(30)],
            "High": [102.0 + index for index in range(30)],
            "Low": [99.0 + index for index in range(30)],
            "Close": [101.0 + index for index in range(30)],
            "Volume": [1_000_000 + index * 1000 for index in range(30)],
        }
    )

    def fake_download(symbol: str, *args: object, **kwargs: object) -> pd.DataFrame:
        requested_symbols.append(symbol)
        return live_prices.copy(deep=True)

    monkeypatch.setattr(providers.yf, "download", fake_download)
    app = keyless_app
    app.query_params["symbol"] = "AMD"
    app.run(timeout=20)
    data_source = next(control for control in app.segmented_control if control.label == "Data source")
    data_source.set_value("Live")
    app.run(timeout=20)

    assert not app.exception
    assert set(requested_symbols) == {"AMD", "AAPL", "MSFT", "NVDA"}
    assert "Live prices" in _rendered_copy(app)


def test_navigation_renders_compare_intelligence_and_signals_lab_user_journeys(keyless_app: AppTest) -> None:
    journeys = {
        "Compare": ("Relative performance", "Snapshot"),
        "Intelligence": ("Market narrative", "Evidence stream"),
        "Portfolio": ("Portfolio risk", "Save research workspace"),
        "Signals Lab": ("Run an evaluation to see out-of-sample results.", "Limitations"),
    }
    app = keyless_app

    for page, expected_copy in journeys.items():
        app.query_params["page"] = page
        app.run(timeout=20)

        assert not app.exception
        rendered = _rendered_copy(app)
        assert page in rendered
        for phrase in expected_copy:
            assert phrase in rendered

    assert [button.label for button in app.button] == ["Run evaluation", "Methodology"]
    app.button[0].click().run(timeout=20)
    assert not app.exception
    completed_copy = _rendered_copy(app)
    assert "directional accuracy" in completed_copy
    assert "Majority-direction baseline" in completed_copy
    assert "Market feature snapshot" in completed_copy
    assert "predicted return" in completed_copy

    coverage = next(control for control in app.selectbox if control.label == "Coverage")
    coverage.set_value("30 trading days")
    app.run(timeout=20)
    assert "Run an evaluation to see out-of-sample results." in _rendered_copy(app)
    app.button[0].click().run(timeout=20)
    assert "30 days" in _rendered_copy(app)


def test_live_mode_uses_yahoo_prices_without_requiring_newsapi_key(
    keyless_app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_symbols: list[str] = []
    live_prices = {
        "Date": ["2026-07-16", "2026-07-17"],
        "Open": [210.0, 212.0],
        "High": [214.0, 216.0],
        "Low": [209.0, 211.0],
        "Close": [213.0, 215.0],
        "Volume": [1_500_000, 1_650_000],
    }

    def fake_download(symbol: str, *args: object, **kwargs: object) -> pd.DataFrame:
        requested_symbols.append(symbol)
        return pd.DataFrame(live_prices)

    monkeypatch.setattr(providers.yf, "download", fake_download)
    app = keyless_app
    data_source = next(control for control in app.segmented_control if control.label == "Data source")

    data_source.set_value("Live")
    app.run(timeout=20)

    assert not app.exception
    copy = _rendered_copy(app)
    assert "Live prices" in copy
    assert "Prices are live from Yahoo Finance" in copy
    assert "demo data because no NewsAPI key is configured" in copy

    app.query_params["page"] = "Compare"
    app.run(timeout=20)
    compare_copy = _rendered_copy(app)

    assert set(requested_symbols) == {"AAPL", "MSFT", "NVDA", "TSLA"}
    assert "Yahoo Finance live prices" in compare_copy


def test_signals_lab_invalidates_completed_run_when_data_mode_changes(
    keyless_app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = keyless_app
    app.query_params["page"] = "Signals Lab"
    app.run(timeout=20)
    app.button[0].click().run(timeout=20)
    assert "directional accuracy" in _rendered_copy(app)

    live_prices = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-05-01", periods=45),
            "Open": [200.0 + index for index in range(45)],
            "High": [202.0 + index for index in range(45)],
            "Low": [199.0 + index for index in range(45)],
            "Close": [201.0 + index for index in range(45)],
            "Volume": [1_000_000 + index * 1000 for index in range(45)],
        }
    )
    monkeypatch.setattr(providers.yf, "download", lambda *args, **kwargs: live_prices.copy(deep=True))
    data_source = next(control for control in app.segmented_control if control.label == "Data source")
    data_source.set_value("Live")
    app.run(timeout=20)

    live_copy = _rendered_copy(app)
    assert "Run an evaluation to see out-of-sample results." in live_copy
    assert "directional accuracy" not in live_copy

    data_source = next(control for control in app.segmented_control if control.label == "Data source")
    data_source.set_value("Demo")
    app.run(timeout=20)
    demo_copy = _rendered_copy(app)
    assert "Run an evaluation to see out-of-sample results." in demo_copy
    assert "directional accuracy" not in demo_copy


@pytest.mark.parametrize(
    ("module", "renderer", "expected_copy"),
    [
        ("compare", "render_compare", "No comparable series available."),
        (
            "intelligence",
            "render_intelligence",
            "No news intelligence is available for this range. Demo price data remains fully usable.",
        ),
        ("signals_lab", "render_signals_lab", "Run an evaluation to see out-of-sample results."),
    ],
)
def test_secondary_pages_render_useful_empty_states(module: str, renderer: str, expected_copy: str) -> None:
    script = f"""\
import pandas as pd
from signalglass.ui.{module} import {renderer}

bundle = {{"ticker": "AAPL", "prices": pd.DataFrame(), "news": pd.DataFrame()}}
{renderer}(bundle)
"""

    app = AppTest.from_string(script, default_timeout=10).run(timeout=10)

    assert not app.exception
    assert expected_copy in _rendered_copy(app)
