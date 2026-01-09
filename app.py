"""Streamlit application for the MarketWatch Insights Dashboard."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from alerts import build_email_body, send_email_summary, smtp_credentials_available
from data_fetcher import DEFAULT_TICKERS, get_news_articles, get_stock_data
from sentiment_analyzer import SentimentAnalyzer, attach_sentiment

st.set_page_config(
    page_title="MarketWatch Insights Dashboard",
    layout="wide",
    page_icon="📈",
)


@st.cache_data(show_spinner=False)
def load_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    data = get_stock_data(ticker, start_date, end_date)
    data = data.copy()
    data.attrs["fetched_at"] = datetime.now()
    return data


@st.cache_data(show_spinner=False)
def load_news_data(ticker: str, start_date: datetime, end_date: datetime, api_key: str) -> pd.DataFrame:
    try:
        data = get_news_articles(ticker, start_date, end_date, api_key=api_key)
        data = data.copy()
        data.attrs["fetched_at"] = datetime.now()
        return data
    except Exception as exc:  # pragma: no cover - defensive cache wrapper
        fallback = pd.DataFrame(
            {col: pd.Series(dtype="object") for col in ["title", "description", "url", "publishedAt", "source", "ticker"]}
        )
        fallback.attrs["warning"] = f"Unable to load headlines: {exc}"
        fallback.attrs["fetched_at"] = datetime.now()
        return fallback


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
    )
    merged = merged.drop(columns=["date"])
    merged = merged.rename(columns={"date_only": "date"})
    merged[["avg_sentiment", "positive_count", "negative_count", "neutral_count", "headline_count"]] = (
        merged[["avg_sentiment", "positive_count", "negative_count", "neutral_count", "headline_count"]].fillna(0)
    )
    return merged


def build_predictive_features(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    df = data.sort_values("date").copy()
    df["return"] = df["Close"].pct_change()
    df["sentiment_change"] = df["avg_sentiment"].diff().fillna(0)
    df["volume_change"] = df["Volume"].pct_change().fillna(0)
    df["target_return"] = df["return"].shift(-1)
    df = df.dropna(subset=["return", "sentiment_change", "volume_change", "target_return", "avg_sentiment"])
    return df


def _fit_linear_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Return (weights, intercept, r2) using a simple least-squares fit."""

    if len(X) != len(y) or len(X) == 0:
        return None

    X_augmented = np.column_stack([np.ones(len(X)), X])
    try:
        coeffs, residuals, rank, _ = np.linalg.lstsq(X_augmented, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < X_augmented.shape[1]:
        return None

    intercept = float(coeffs[0])
    weights = coeffs[1:].astype(float)
    predictions = X_augmented @ coeffs
    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return weights, intercept, r2


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

    # Standardize features manually to keep the regression numerically stable.
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std

    fit = _fit_linear_regression(X_scaled, y)
    if fit is None:
        return None

    weights, intercept, r2_score = fit

    latest_features = features_df[feature_cols].iloc[-1].to_numpy()
    latest_scaled = (latest_features - X_mean) / X_std
    predicted_return = float(np.dot(weights, latest_scaled) + intercept)

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


def format_refresh_timestamp(timestamp: datetime | None) -> str:
    if not timestamp:
        return "Last refreshed: unavailable"
    return f"Last refreshed: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"


# Sidebar controls
with st.sidebar:
    st.header("Analysis Options")
    sentiment_mode = st.radio(
        "Sentiment Engine",
        options=["simple", "advanced"],
        format_func=lambda x: "TextBlob" if x == "simple" else "Hugging Face Transformers",
        help="Use TextBlob for quick analysis or Hugging Face for deeper insights.",
    )
    st.write(
        "Advanced mode loads a transformer model on demand. Ensure your environment has sufficient resources."
    )

    export_enabled = st.checkbox("Enable CSV export", value=True)

    st.markdown("---")
    st.header("Daily Email Summary")
    credentials_ready = smtp_credentials_available()
    if not credentials_ready:
        st.caption(
            "Configure SMTP settings in Streamlit secrets or environment variables (SMTP_SERVER, SMTP_PORT, "
            "EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_SENDER)."
        )

    recipient_email = st.text_input("Recipient Email", key="recipient_email")
    attach_csv = st.checkbox("Attach CSV snapshot", value=True)
    email_button = st.button(
        "Send Daily Email",
        use_container_width=True,
        disabled=not credentials_ready,
    )

st.title("MarketWatch Insights Dashboard")
st.caption("Real-time stock trends and news sentiment analysis for data-driven decisions.")

# Filter controls
min_date = datetime.now() - timedelta(days=90)
default_start = datetime.now() - timedelta(days=30)
default_end = datetime.now()

col1, col2, col3 = st.columns([1.5, 1.5, 1])
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

try:
    newsapi_key = st.secrets.get("newsapi_key")
except Exception:
    newsapi_key = None

newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY", "")
if not newsapi_key:
    st.warning(
        "NewsAPI key not found. Add it to Streamlit secrets or the NEWSAPI_KEY environment variable to load headlines."
    )

with st.spinner("Fetching latest market data..."):
    stock_df = load_stock_data(ticker, start_dt, end_dt)
stock_refreshed_at = getattr(stock_df, "attrs", {}).get("fetched_at")

with st.spinner("Collecting news and sentiment scores..."):
    news_df = load_news_data(ticker, start_dt, end_dt, api_key=newsapi_key) if newsapi_key else pd.DataFrame()
    analyzer = SentimentAnalyzer(mode=sentiment_mode)
    news_df = attach_sentiment(news_df, analyzer)
    sentiment_summary = prepare_sentiment_summary(news_df)
news_refreshed_at = getattr(news_df, "attrs", {}).get("fetched_at")

news_warning = getattr(news_df, "attrs", {}).get("warning") if isinstance(news_df, pd.DataFrame) else None
if news_warning:
    st.error(f"News alert: {news_warning}")

combined_df = merge_stock_and_sentiment(stock_df, sentiment_summary)

model_results = run_predictive_model(combined_df)

if stock_df.empty:
    st.error("No stock data available for the selected range. Try adjusting the filters.")
    st.stop()

# Layout: two columns for main charts
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Stock Price Trend")
    st.caption(format_refresh_timestamp(stock_refreshed_at))
    price_fig = px.line(
        combined_df,
        x="date",
        y="Close",
        title=f"{ticker} Closing Prices",
        labels={"date": "Date", "Close": "Close Price (USD)"},
    )
    price_fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(price_fig, use_container_width=True)

with right_col:
    st.subheader("Daily Sentiment Breakdown")
    st.caption(format_refresh_timestamp(news_refreshed_at))
    if sentiment_summary.empty:
        st.info("No news sentiment data available for this range.")
    else:
        melted = sentiment_summary.melt(
            id_vars="date",
            value_vars=["positive_count", "negative_count", "neutral_count"],
            var_name="sentiment",
            value_name="count",
        )
        color_map = {
            "positive_count": "#1f77b4",  # Blue
            "negative_count": "#d62728",  # Red
            "neutral_count": "#7f7f7f",  # Gray
        }
        sentiment_fig = px.bar(
            melted,
            x="date",
            y="count",
            color="sentiment",
            color_discrete_map=color_map,
            title="Headline Sentiment Counts",
            labels={"date": "Date", "count": "Headline Count", "sentiment": "Sentiment"},
        )
        sentiment_fig.update_layout(height=400)
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
            line=dict(color="#1f77b4"),
        )
    )
    corr_fig.add_trace(
        go.Scatter(
            x=combined_df["date"],
            y=combined_df["avg_sentiment"],
            name="Average Sentiment",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#7f7f7f"),
        )
    )

    corr_fig.update_layout(
        height=420,
        yaxis=dict(title="Close Price (USD)", showgrid=False),
        yaxis2=dict(
            title="Average Sentiment",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(corr_fig, use_container_width=True)

insight = correlation_insight_text(combined_df)
st.markdown(f"**Insight:** {insight}")

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
    delta_text = f"{predicted_pct:+.2f}% expected next-day return"
    st.metric(
        label="Projected Movement",
        value=direction,
        delta=delta_text,
    )
    st.caption(
        "Simple linear regression model using sentiment and price-derived features. Confidence reflects in-sample "
        f"R² of {confidence:.2f}."
    )

if not news_df.empty:
    st.subheader("Latest Headlines")
    st.dataframe(
        news_df[["publishedAt", "title", "sentiment_label", "sentiment_score", "source", "url"]]
        .sort_values("publishedAt", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )

csv_bytes = b""
if not combined_df.empty:
    csv_bytes = combined_df.to_csv(index=False).encode("utf-8")

if export_enabled and csv_bytes:
    st.download_button(
        label="Download Daily Summary CSV",
        data=csv_bytes,
        file_name=f"{ticker}_marketwatch_summary.csv",
        mime="text/csv",
    )

if email_button and recipient_email:
    try:
        email_body = build_email_body(combined_df, news_df, ticker)
        attachments = [(f"{ticker}_marketwatch_summary.csv", csv_bytes)] if attach_csv and csv_bytes else None
        send_email_summary(to=[recipient_email], subject=f"{ticker} Daily MarketWatch Summary", body=email_body, attachments=attachments)
        st.success(f"Email summary sent to {recipient_email}.")
    except Exception as exc:  # pragma: no cover - depends on remote SMTP configuration
        st.error(f"Unable to send email: {exc}")
elif email_button and not recipient_email:
    st.warning("Please provide a recipient email address before sending.")

st.markdown(
    """
---
**How to interpret this dashboard:**
- Blue lines indicate positive price performance, while grey sentiment markers reveal headline tone.
- Compare daily sentiment spikes against price changes to spot potential leading indicators.
- Use the sidebar to toggle advanced sentiment analysis for deeper NLP models.
    """
)

st.markdown(
    """
### Deployment Notes
1. Create a `secrets.toml` file on Streamlit Cloud and add your `newsapi_key`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Launch locally using `streamlit run app.py`.
4. Deploy by connecting your repository to [Streamlit Cloud](https://streamlit.io/cloud) and selecting `app.py` as the entry point.
    """
)
