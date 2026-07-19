"""Plotly chart primitives for SignalGlass.

Every builder is pure: inputs are copied, no Streamlit state is touched, and no data is
fetched.  That makes the figures useful in both the app and isolated tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signalglass.theme import TOKENS, plotly_layout_defaults


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value).copy(deep=True)
    except (TypeError, ValueError):
        return pd.DataFrame()


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    return next((lookup[name.lower()] for name in names if name.lower() in lookup), None)


def _empty_figure(message: str, *, height: int) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": TOKENS.text_muted, "size": 13},
    )
    figure.update_layout(**plotly_layout_defaults(height=height))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def make_overview_chart(
    data: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
    events: Iterable[Mapping[str, Any]] | pd.DataFrame | None = None,
    *,
    height: int = 430,
) -> go.Figure:
    """Build the concept's combined price, volume, and news-pulse chart."""

    frame = _frame(data)
    date_col = _first_column(frame, ("date", "datetime", "timestamp"))
    close_col = _first_column(frame, ("Close", "close", "price", "adj close"))
    if frame.empty or date_col is None or close_col is None:
        return _empty_figure("Price history is unavailable for this range.", height=height)

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, close_col]).sort_values(date_col)
    if frame.empty:
        return _empty_figure("Price history is unavailable for this range.", height=height)

    volume_col = _first_column(frame, ("Volume", "volume"))
    sentiment_col = _first_column(frame, ("avg_sentiment", "sentiment", "news_pulse", "news pulse"))
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.64, 0.19, 0.17],
    )
    figure.add_trace(
        go.Scatter(
            x=frame[date_col],
            y=frame[close_col],
            name="Price (USD)",
            mode="lines",
            line={"color": TOKENS.cobalt, "width": 2.5},
            hovertemplate="$%{y:,.2f}<extra>Price</extra>",
        ),
        row=1,
        col=1,
    )
    if volume_col:
        volumes = pd.to_numeric(frame[volume_col], errors="coerce").fillna(0)
        figure.add_trace(
            go.Bar(
                x=frame[date_col],
                y=volumes,
                name="Volume",
                marker={"color": "rgba(126,143,164,.58)"},
                hovertemplate="%{y:,.3s}<extra>Volume</extra>",
            ),
            row=2,
            col=1,
        )
    if sentiment_col:
        sentiment = pd.to_numeric(frame[sentiment_col], errors="coerce").fillna(0)
        colors = np.where(sentiment >= 0, TOKENS.positive, TOKENS.negative)
        figure.add_trace(
            go.Bar(
                x=frame[date_col],
                y=sentiment,
                name="Sentiment (News Pulse)",
                marker={"color": colors.tolist()},
                hovertemplate="%{y:+.2f}<extra>News pulse</extra>",
            ),
            row=3,
            col=1,
        )

    event_frame = _frame(events)
    if not event_frame.empty:
        event_date = _first_column(event_frame, ("date", "publishedAt", "timestamp"))
        event_label = _first_column(event_frame, ("label", "type", "event", "title"))
        if event_date:
            for _index, event in event_frame.head(8).iterrows():
                when = pd.to_datetime(event[event_date], errors="coerce")
                if pd.isna(when):
                    continue
                nearest = (frame[date_col] - when).abs().idxmin()
                label = str(event[event_label]) if event_label else "Event"
                glyph = "E" if "earn" in label.lower() else "N"
                color = "#7e838b" if glyph == "E" else TOKENS.purple
                figure.add_trace(
                    go.Scatter(
                        x=[when],
                        y=[frame.loc[nearest, close_col]],
                        mode="markers+text",
                        text=[glyph],
                        textposition="middle center",
                        marker={"size": 20, "color": color, "line": {"color": "#d6dce5", "width": 1}},
                        textfont={"color": TOKENS.text, "size": 10},
                        name=label,
                        showlegend=False,
                        hovertemplate=f"{label}<br>%{{x|%b %d, %Y}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    layout = plotly_layout_defaults(height=height)
    layout.update(
        {
            "showlegend": True,
            "bargap": 0.26,
            "margin": {"l": 44, "r": 22, "t": 30, "b": 22},
            "legend": {"orientation": "h", "y": -0.14, "x": 0, "font": {"size": 11}},
        }
    )
    figure.update_layout(**layout)
    figure.update_xaxes(showgrid=False, linecolor=TOKENS.border_soft, tickfont={"color": TOKENS.text_muted})
    figure.update_yaxes(
        gridcolor="rgba(139,161,184,.10)",
        zerolinecolor="rgba(139,161,184,.24)",
        tickfont={"color": TOKENS.text_muted},
    )
    figure.update_yaxes(tickprefix="$", row=1, col=1)
    figure.update_yaxes(showticklabels=False, row=2, col=1)
    figure.update_yaxes(range=[-1, 1], row=3, col=1)
    return figure


def make_signal_evaluation_chart(
    evaluation: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
    *,
    height: int = 340,
) -> go.Figure:
    """Render actual returns against walk-forward predicted returns."""

    frame = _frame(evaluation)
    date_col = _first_column(frame, ("date", "timestamp", "datetime"))
    actual_col = _first_column(frame, ("actual_return", "return", "target_return", "actual"))
    predicted_col = _first_column(
        frame, ("predicted_return", "predicted_direction", "prediction", "predicted", "signal")
    )
    if frame.empty or date_col is None or actual_col is None or predicted_col is None:
        return _empty_figure("Run an evaluation to inspect the signal.", height=height)

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    actual = pd.to_numeric(frame[actual_col], errors="coerce").fillna(0)
    predicted = pd.to_numeric(frame[predicted_col], errors="coerce").fillna(0)
    if actual.abs().max() <= 1:
        actual = actual * 100
    if predicted.abs().max() <= 1:
        predicted = predicted * 100
    colors = np.where(actual >= 0, TOKENS.positive, TOKENS.negative)

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=dates,
            y=actual,
            name="Actual return",
            marker={"color": colors.tolist()},
            hovertemplate="%{y:+.2f}%<extra>Actual return</extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=predicted,
            name="Predicted return",
            mode="lines",
            line={"color": TOKENS.cobalt, "width": 2},
            hovertemplate="%{y:+.2f}%<extra>Predicted return</extra>",
        ),
        secondary_y=False,
    )
    figure.update_layout(**plotly_layout_defaults(height=height))
    figure.update_layout(
        margin={"l": 40, "r": 40, "t": 32, "b": 34},
        bargap=0.34,
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    figure.update_xaxes(showgrid=False, tickfont={"color": TOKENS.text_muted})
    figure.update_yaxes(
        gridcolor="rgba(139,161,184,.10)",
        zerolinecolor="rgba(139,161,184,.25)",
        ticksuffix="%",
        secondary_y=False,
    )
    return figure


def make_compare_chart(
    series: Mapping[str, pd.DataFrame] | pd.DataFrame | None,
    *,
    height: int = 420,
) -> go.Figure:
    """Plot rebased performance for multiple symbols."""

    if isinstance(series, pd.DataFrame):
        symbol_col = _first_column(series, ("ticker", "symbol"))
        groups = (
            {str(symbol): group.copy() for symbol, group in series.groupby(symbol_col)}
            if symbol_col
            else {"Asset": series}
        )
    else:
        groups = dict(series or {})
    if not groups:
        return _empty_figure("Choose at least one company to compare.", height=height)

    colors = (TOKENS.cobalt, TOKENS.positive, TOKENS.purple, TOKENS.warning, "#52c7d9")
    figure = go.Figure()
    for (symbol, values), color in zip(groups.items(), colors, strict=False):
        frame = _frame(values)
        date_col = _first_column(frame, ("date", "timestamp", "datetime"))
        close_col = _first_column(frame, ("Close", "close", "price", "adj close"))
        if frame.empty or date_col is None or close_col is None:
            continue
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        close = pd.to_numeric(frame[close_col], errors="coerce")
        valid = dates.notna() & close.notna()
        close = close.loc[valid]
        if close.empty or close.iloc[0] == 0:
            continue
        rebased = (close / close.iloc[0] - 1) * 100
        figure.add_trace(
            go.Scatter(
                x=dates.loc[valid],
                y=rebased,
                name=str(symbol),
                mode="lines",
                line={"width": 2.25, "color": color},
                hovertemplate="%{y:+.2f}%<extra>" + str(symbol) + "</extra>",
            )
        )
    if not figure.data:
        return _empty_figure("Comparable price history is unavailable.", height=height)
    figure.update_layout(**plotly_layout_defaults(height=height))
    figure.update_layout(margin={"l": 46, "r": 24, "t": 34, "b": 35})
    figure.update_xaxes(showgrid=False, tickfont={"color": TOKENS.text_muted})
    figure.update_yaxes(gridcolor="rgba(139,161,184,.10)", ticksuffix="%", zerolinecolor=TOKENS.border)
    return figure


# Short aliases make app composition pleasant and preserve a stable public API.
overview_chart = make_overview_chart
signal_evaluation_chart = make_signal_evaluation_chart
compare_chart = make_compare_chart


__all__ = [
    "compare_chart",
    "make_compare_chart",
    "make_overview_chart",
    "make_signal_evaluation_chart",
    "overview_chart",
    "signal_evaluation_chart",
]
