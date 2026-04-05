# MarketWatch Insights Dashboard

A production-style Streamlit dashboard for tracking market performance, real-time headline sentiment, and lightweight predictive signals for popular tickers (AAPL, TSLA, MSFT, etc.).

## ✨ What’s New
- Refreshed, more professional UI (hero panel, KPI cards, tabbed analytics layout).
- Clear separation of **Overview**, **News & Sentiment**, and **Predictive Model** workflows.
- More robust feature engineering and merge logic for stable runtime behavior.
- Improved operational guidance for local setup, cloud deployment, and troubleshooting.

## Core Features
- 📈 Interactive price analytics with Plotly.
- 📰 News ingestion from NewsAPI with per-headline sentiment scoring.
- 🤖 Switchable sentiment engines:
  - **TextBlob** (fast)
  - **Hugging Face Transformer** (deeper context)
- 🔗 Correlation view between average sentiment and market price.
- 🔮 Predictive directional signal using scikit-learn linear regression.
- 📤 CSV export and SMTP-based email snapshot delivery.

## UI Preview Snippets
> Add screenshots to `docs/screenshots/` and keep these links updated.

![Dashboard Overview](docs/screenshots/dashboard-overview.png)
![News and Sentiment Tab](docs/screenshots/news-sentiment-tab.png)
![Predictive Model Tab](docs/screenshots/predictive-model-tab.png)

## Project Structure
```text
.
├── app.py
├── alerts.py
├── data_fetcher.py
├── sentiment_analyzer.py
├── requirements.txt
├── README.md
└── docs/
    └── screenshots/
```

## Quick Start

### 1) Clone + install
```bash
git clone <your-fork-url>
cd MarketWatch-Insights-Dashboard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure secrets or environment variables
Create `.streamlit/secrets.toml`:
```toml
newsapi_key = "YOUR_NEWSAPI_KEY"

# Optional (email alerts)
smtp_server = "smtp.gmail.com"
smtp_port = 587
email_username = "you@example.com"
email_password = "APP_PASSWORD"
email_sender = "MarketWatch Dashboard <you@example.com>"
```

Or use environment variables:
```bash
export NEWSAPI_KEY="YOUR_NEWSAPI_KEY"
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export EMAIL_USERNAME="you@example.com"
export EMAIL_PASSWORD="APP_PASSWORD"
export EMAIL_SENDER="MarketWatch Dashboard <you@example.com>"
```

### 3) Run locally
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

## Recommended GitHub Repo Improvements
To make this project look more professional on GitHub:
1. Add a deployed app URL near the top of this README.
2. Record a short demo GIF (15–30s) and place it under `docs/screenshots/`.
3. Add badges for Python version, Streamlit, CI, and license.
4. Add a `CONTRIBUTING.md` and issue templates.
5. Add GitHub Actions for lint + smoke test (`python -m compileall .`).

## Deployment (Streamlit Cloud)
1. Push your repo to GitHub.
2. Create a new Streamlit app and set `app.py` as entrypoint.
3. Add `newsapi_key` in Streamlit **App settings → Secrets**.
4. Deploy and verify logs for API-key and dependency issues.

## Troubleshooting
- **No headlines shown**: Verify `NEWSAPI_KEY` / `newsapi_key` is configured.
- **Model not available**: Expand date range to include more rows/headlines.
- **Email disabled**: Confirm SMTP variables are set correctly.
- **Advanced sentiment slow on first run**: Transformer model load/warmup is expected.

## Resume-Ready Description
> Built a professional market-intelligence dashboard in Streamlit that combines stock-time-series analysis, NLP sentiment on financial headlines, and regression-based predictive signals. Implemented interactive visualizations, export/email workflows, and cloud deployment readiness for portfolio demonstrations.

## License
Distributed for educational use. Review Yahoo Finance and NewsAPI terms before commercial usage.
