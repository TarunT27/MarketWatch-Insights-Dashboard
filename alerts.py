"""Utility helpers for sending dashboard email alerts."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Iterable, Optional

import pandas as pd


def _get_secret(name: str, *, default: str = "") -> str:
    """Read a secret from Streamlit or environment variables lazily."""

    try:  # Lazy import to avoid hard dependency on streamlit outside the app
        import streamlit as st  # type: ignore

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # pragma: no cover - streamlit not available in non-app contexts
        pass

    return os.getenv(name.upper(), default)


def smtp_credentials_available() -> bool:
    """Return True when the required SMTP credentials are configured."""

    required = ["smtp_server", "smtp_port", "email_username", "email_password", "email_sender"]
    return all(_get_secret(key) for key in required)


def send_email_summary(
    *,
    to: Iterable[str],
    subject: str,
    body: str,
    attachments: Optional[Iterable[tuple[str, bytes]]] = None,
) -> None:
    """Send an email using configured SMTP credentials.

    Parameters
    ----------
    to:
        Iterable of recipient email addresses.
    subject:
        Email subject line.
    body:
        Plain-text email body.
    attachments:
        Optional iterable of (filename, binary_data) tuples to attach.
    """

    smtp_server = _get_secret("smtp_server")
    smtp_port = int(_get_secret("smtp_port", default="587") or 587)
    username = _get_secret("email_username")
    password = _get_secret("email_password")
    sender = _get_secret("email_sender")

    if not (smtp_server and username and password and sender):
        raise RuntimeError("SMTP credentials are not fully configured.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(list(to))
    message.set_content(body)

    if attachments:
        for filename, data in attachments:
            message.add_attachment(data, maintype="application", subtype="octet-stream", filename=filename)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)


def build_email_body(summary_df: pd.DataFrame, headlines: pd.DataFrame, ticker: str) -> str:
    """Generate a friendly email body summarizing sentiment and price insights."""

    lines = [f"MarketWatch Insights Dashboard summary for {ticker}", ""]

    if summary_df.empty:
        lines.append("No combined sentiment or price data was available for the requested range.")
    else:
        latest = summary_df.sort_values("date").iloc[-1]
        avg_sentiment = latest.get("avg_sentiment", 0.0)
        pos = int(latest.get("positive_count", 0))
        neg = int(latest.get("negative_count", 0))
        neu = int(latest.get("neutral_count", 0))
        close_price = latest.get("Close", float("nan"))

        lines.extend(
            [
                f"Latest close price: ${close_price:,.2f}",
                f"Average sentiment score: {avg_sentiment:.2f}",
                f"Headline sentiment counts — Positive: {pos}, Negative: {neg}, Neutral: {neu}.",
                "",
            ]
        )

    if not headlines.empty:
        lines.append("Top headlines:")
        for _, row in headlines.sort_values("publishedAt", ascending=False).head(5).iterrows():
            title = row.get("title")
            label = row.get("sentiment_label")
            url = row.get("url")
            published = row.get("publishedAt")
            timestamp = published.strftime("%Y-%m-%d %H:%M") if pd.notna(published) else ""
            lines.append(f"- [{timestamp}] ({label}) {title} — {url}")

    return "\n".join(lines)


__all__ = ["send_email_summary", "smtp_credentials_available", "build_email_body"]
