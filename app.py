"""Streamlit application for the MarketWatch Insights Dashboard."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.errors import StreamlitSecretNotFoundError

from alerts import build_email_body, send_email_summary, smtp_credentials_available
from data_fetcher import DEFAULT_TICKERS, get_news_articles, get_stock_data
from sentiment_analyzer import SentimentAnalyzer, attach_sentiment

st.set_page_config(
    page_title="MarketWatch Insights Dashboard",
    layout="wide",
    page_icon="📈",
)


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
            .hero-card {
                background: linear-gradient(120deg, #0f172a, #1e293b);
                color: #f8fafc;
                border-radius: 14px;
                padding: 1.1rem 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.18);
            }
            .hero-subtitle { color: #cbd5e1; font-size: 0.95rem; margin-top: 0.35rem; }
            .metric-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 0.9rem 1rem;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
            }
            .metric-title { color: #475569; font-size: 0.85rem; }
            .metric-value { color: #0f172a; font-size: 1.35rem; font-weight: 700; }
            .metric-delta { color: #2563eb; font-size: 0.85rem; }
            .section-note {
                background: #eff6ff;
                border-left: 4px solid #3b82f6;
                padding: 0.65rem 0.85rem;
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, delta: str = "") -> None:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-delta'>{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    return get_stock_data(ticker, start_date, end_date)


@st.cache_data(show_spinner=False)
def load_news_data(ticker: str, start_date: datetime, end_date: datetime, api_key: str) -> pd.DataFrame:
    return get_news_articles(ticker, start_date, end_date, api_key=api_key)


def prepare_sentiment_summary(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(
            columns=["date", "avg_sentiment", "positive_count", "negative_count", "neutral_count", "headline_count"]
        )

    news_df = news_df.copy()
    news_df["date"] = news_df["publishedAt"].dt.date
    grouped = news_df.groupby("date")

    summary = pd.DataFrame(
        {
            "avg_sentiment": grouped["sentiment_score"].mean(),
            "positive_count": grouped.apply(lambda g: (g["sentiment_label"] == "Positive").sum()),
            "negative_count": grouped.apply(lambda g: (g["sentiment_label"] == "Negative").sum()),
            "neutral_count": grouped.apply(lambda g: (g["sentiment_label"] == "Neutral").sum()),
            "headline_count": grouped.size(),
        }
    ).reset_index()

    return summary


def merge_stock_and_sentiment(stock_df: pd.DataFrame, sentiment_summary: pd.DataFrame) -> pd.DataFrame:
    if stock_df.empty:
        return stock_df

    stock_df = stock_df.copy()
    stock_df["date_only"] = stock_df["date"].dt.date

    merged = stock_df.merge(
        sentiment_summary,
        how="left",
        left_on="date_only",
        right_on="date",
        suffixes=("", "_sentiment"),
    )
    merged = merged.drop(columns=["date_only", "date_sentiment", "date_y"], errors="ignore")
    merged = merged.rename(columns={"date_x": "date"})
    merged[["avg_sentiment", "positive_count", "negative_count", "neutral_count", "headline_count"]] = (
        merged[["avg_sentiment", "positive_count", "negative_count", "neutral_count", "headline_count"]].fillna(0)
    )
    return merged


def build_predictive_features(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    df = data.sort_values("date").copy()
    df["return"] = df["Close"].pct_change().fillna(0)
    df["sentiment_change"] = df["avg_sentiment"].diff().fillna(0)
    df["volume_change"] = df["Volume"].pct_change().fillna(0)
    df["target_return"] = df["return"].shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["return", "sentiment_change", "volume_change", "target_return"])
    return df


def run_predictive_model(data: pd.DataFrame) -> dict[str, float | str] | None:
    if data.empty or data["headline_count"].sum() == 0 or len(data) < 6:
        return None

    features_df = build_predictive_features(data)
    if features_df.empty:
        return None

    feature_cols = [
        "avg_sentiment",
        "positive_count",
        "negative_count",
        "neutral_count",
        "headline_count",
        "return",
        "sentiment_change",
        "volume_change",
    ]

    if any(col not in features_df.columns for col in feature_cols):
        return None

    X = features_df[feature_cols].to_numpy()
    y = features_df["target_return"].to_numpy()

    if len(features_df) < 3 or np.allclose(y, y[0]):
        return None

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X, y)
    r2_score = float(model.score(X, y))

    latest_features = features_df[feature_cols].iloc[-1].to_numpy().reshape(1, -1)
    predicted_return = float(model.predict(latest_features)[0])

    return {
        "predicted_return": predicted_return,
        "predicted_direction": "Up" if predicted_return >= 0 else "Down",
        "confidence": r2_score,
    }


def correlation_insight_text(data: pd.DataFrame) -> str:
    if data.empty or data["headline_count"].sum() == 0:
        return "No sentiment data available for the selected range."

    if data["avg_sentiment"].nunique() <= 1:
        return "Sentiment scores show minimal variation, limiting correlation insights."

    corr = data["avg_sentiment"].corr(data["Close"].pct_change().fillna(0))
    if pd.isna(corr):
        return "Unable to compute a reliable correlation between sentiment and price movements."

    if corr > 0.3:
        return "Positive sentiment appears to align with upward stock movements during the selected period."
    if corr < -0.3:
        return "Negative sentiment tends to coincide with downward price action in this range."
    return "Sentiment and price movements show a weak correlation for the chosen window."


inject_custom_styles()

with st.sidebar:
    st.header("⚙️ Analysis Options")
    sentiment_mode = st.radio(
        "Sentiment Engine",
        options=["simple", "advanced"],
        format_func=lambda x: "TextBlob (Fast)" if x == "simple" else "Hugging Face (Deep)",
        help="Use TextBlob for quick analysis or Hugging Face for deeper insights.",
    )
    st.caption("Advanced mode loads a transformer model on demand and may take longer on first run.")

    export_enabled = st.checkbox("Enable CSV export", value=True)

    st.markdown("---")
    st.header("📨 Daily Email Summary")
    credentials_ready = smtp_credentials_available()
    if not credentials_ready:
        st.caption(
            "Configure SMTP via secrets or environment variables: SMTP_SERVER, SMTP_PORT, EMAIL_USERNAME, "
            "EMAIL_PASSWORD, EMAIL_SENDER."
        )

    recipient_email = st.text_input("Recipient Email", key="recipient_email", placeholder="name@example.com")
    attach_csv = st.checkbox("Attach CSV snapshot", value=True)
    email_button = st.button("Send Daily Email", use_container_width=True, disabled=not credentials_ready)

st.markdown(
    """
    <div class='hero-card'>
        <h2 style='margin:0;'>MarketWatch Insights Dashboard</h2>
        <p class='hero-subtitle'>
            Professional market intelligence view combining stock trends, news sentiment, and predictive analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Filter controls
min_date = datetime.now() - timedelta(days=90)
default_start = datetime.now() - timedelta(days=30)
default_end = datetime.now()

col1, col2, col3 = st.columns([1.4, 1.8, 1])
with col1:
    ticker = st.selectbox("Select Company", DEFAULT_TICKERS, index=0)
with col2:
    start_date, end_date = st.date_input(
        "Date Range",
        value=(default_start.date(), default_end.date()),
        min_value=min_date.date(),
        max_value=datetime.now().date(),
    )
    if isinstance(start_date, tuple):
        start_date, end_date = start_date
with col3:
    refresh_requested = st.button("🔄 Refresh Data", use_container_width=True)

if refresh_requested:
    load_stock_data.clear()
    load_news_data.clear()
    st.success("Cache cleared. Fetching fresh data...")

start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.max.time())

newsapi_key = os.getenv("NEWSAPI_KEY", "")
if not newsapi_key:
    try:
        newsapi_key = st.secrets.get("newsapi_key", "")
    except (StreamlitSecretNotFoundError, FileNotFoundError, KeyError, AttributeError):
        newsapi_key = ""
if not newsapi_key:
    st.warning("NewsAPI key not found. Add it via secrets or NEWSAPI_KEY to load headlines.")

with st.spinner("Fetching latest market data..."):
    stock_df = load_stock_data(ticker, start_dt, end_dt)

with st.spinner("Collecting news and sentiment scores..."):
    news_df = load_news_data(ticker, start_dt, end_dt, api_key=newsapi_key) if newsapi_key else pd.DataFrame()
    analyzer = SentimentAnalyzer(mode=sentiment_mode)
    news_df = attach_sentiment(news_df, analyzer)
    sentiment_summary = prepare_sentiment_summary(news_df)

combined_df = merge_stock_and_sentiment(stock_df, sentiment_summary)
model_results = run_predictive_model(combined_df)

if stock_df.empty:
    st.error("No stock data available for the selected range. Try adjusting the filters.")
    st.stop()

latest_close = combined_df["Close"].iloc[-1] if not combined_df.empty else 0
window_return = 0.0
if len(combined_df) > 1 and combined_df["Close"].iloc[0] != 0:
    window_return = ((combined_df["Close"].iloc[-1] - combined_df["Close"].iloc[0]) / combined_df["Close"].iloc[0]) * 100

headline_total = int(combined_df["headline_count"].sum()) if "headline_count" in combined_df else 0
avg_sentiment = float(combined_df["avg_sentiment"].mean()) if headline_total > 0 else 0.0

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
with kpi_1:
    metric_card("Latest Close", f"${latest_close:,.2f}", f"{ticker}")
with kpi_2:
    metric_card("Window Return", f"{window_return:+.2f}%", f"{start_date} → {end_date}")
with kpi_3:
    metric_card("Headlines Analyzed", f"{headline_total}", "NewsAPI + sentiment model")
with kpi_4:
    sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
    metric_card("Average Sentiment", f"{avg_sentiment:+.2f}", sentiment_label)

st.markdown("<div class='section-note'><strong>Tip:</strong> Use the Overview tab for trend analysis, News tab for evidence, and Model tab for directional signal checks.</div>", unsafe_allow_html=True)

overview_tab, news_tab, model_tab = st.tabs(["📊 Overview", "📰 News & Sentiment", "🔮 Predictive Model"])

with overview_tab:
    left_col, right_col = st.columns(2)

    with left_col:
        price_fig = px.line(
            combined_df,
            x="date",
            y="Close",
            title=f"{ticker} Closing Price Trend",
            labels={"date": "Date", "Close": "Close Price (USD)"},
        )
        price_fig.update_traces(line=dict(width=3, color="#2563eb"))
        price_fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(price_fig, use_container_width=True)

    with right_col:
        if sentiment_summary.empty:
            st.info("No sentiment data available for this date range.")
        else:
            melted = sentiment_summary.melt(
                id_vars="date",
                value_vars=["positive_count", "negative_count", "neutral_count"],
                var_name="sentiment",
                value_name="count",
            )
            sentiment_fig = px.bar(
                melted,
                x="date",
                y="count",
                color="sentiment",
                color_discrete_map={
                    "positive_count": "#22c55e",
                    "negative_count": "#ef4444",
                    "neutral_count": "#64748b",
                },
                title="Daily Headline Sentiment Mix",
                labels={"date": "Date", "count": "Headline Count", "sentiment": "Sentiment"},
            )
            sentiment_fig.update_layout(height=420)
            st.plotly_chart(sentiment_fig, use_container_width=True)

    st.subheader("Correlation Insights")
    if combined_df["headline_count"].sum() == 0:
        st.info("Sentiment data is unavailable; correlation chart requires news headlines.")
    else:
        corr_fig = go.Figure()
        corr_fig.add_trace(
            go.Scatter(
                x=combined_df["date"],
                y=combined_df["Close"],
                name="Close Price",
                yaxis="y1",
                mode="lines",
                line=dict(color="#1d4ed8", width=3),
            )
        )
        corr_fig.add_trace(
            go.Scatter(
                x=combined_df["date"],
                y=combined_df["avg_sentiment"],
                name="Average Sentiment",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#64748b"),
            )
        )
        corr_fig.update_layout(
            height=420,
            yaxis=dict(title="Close Price (USD)", showgrid=False),
            yaxis2=dict(title="Average Sentiment", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(corr_fig, use_container_width=True)

    st.markdown(f"**Insight:** {correlation_insight_text(combined_df)}")

with news_tab:
    if news_df.empty:
        st.info("No headlines were fetched for this time range.")
    else:
        st.dataframe(
            news_df[["publishedAt", "title", "sentiment_label", "sentiment_score", "source", "url"]]
            .sort_values("publishedAt", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        sentiment_avg = (
            news_df.groupby("sentiment_label")["sentiment_score"]
            .mean()
            .reindex(["Positive", "Neutral", "Negative"])
            .fillna(0)
        )
        st.bar_chart(sentiment_avg)

with model_tab:
    st.subheader("Predictive Signal")
    if model_results is None:
        st.info(
            "Not enough historical sentiment data to build a reliable predictive signal. Try expanding the date range "
            "or enabling more headlines."
        )
    else:
        predicted_pct = model_results["predicted_return"] * 100
        direction = model_results["predicted_direction"]
        confidence = model_results["confidence"]
        st.metric("Projected Next-Day Direction", direction, f"{predicted_pct:+.2f}%")
        st.caption(
            "Linear regression over sentiment + market features. Confidence is in-sample R² and should be treated as "
            f"diagnostic only ({confidence:.2f})."
        )

csv_bytes = b""
if not combined_df.empty:
    csv_bytes = combined_df.to_csv(index=False).encode("utf-8")

action_col1, action_col2 = st.columns([1, 1])
with action_col1:
    if export_enabled and csv_bytes:
        st.download_button(
            label="Download Daily Summary CSV",
            data=csv_bytes,
            file_name=f"{ticker}_marketwatch_summary.csv",
            mime="text/csv",
        )
with action_col2:
    if email_button and recipient_email:
        try:
            email_body = build_email_body(combined_df, news_df, ticker)
            attachments = [(f"{ticker}_marketwatch_summary.csv", csv_bytes)] if attach_csv and csv_bytes else None
            send_email_summary(
                to=[recipient_email],
                subject=f"{ticker} Daily MarketWatch Summary",
                body=email_body,
                attachments=attachments,
            )
            st.success(f"Email summary sent to {recipient_email}.")
        except Exception as exc:  # pragma: no cover - depends on remote SMTP configuration
            st.error(f"Unable to send email: {exc}")
    elif email_button and not recipient_email:
        st.warning("Please provide a recipient email address before sending.")

st.markdown(
    """
---
**How to interpret this dashboard**
- Blue line highlights closing-price trend for your selected range.
- Sentiment bars expose how headline tone changes day-to-day.
- Use correlation + predictive tabs together for directional hypothesis generation, not guaranteed forecasts.
    """
)
