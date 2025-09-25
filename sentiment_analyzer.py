"""Sentiment analysis helpers for the MarketWatch Insights Dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional

import pandas as pd
from textblob import TextBlob

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - transformers is optional at runtime
    pipeline = None


@dataclass
class SentimentResult:
    label: str
    score: float


class SentimentAnalyzer:
    """Performs sentiment analysis using TextBlob or Hugging Face."""

    def __init__(self, mode: str = "simple", model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        self.mode = mode
        self.model_name = model_name
        self._hf_pipeline = None

    @property
    def uses_transformers(self) -> bool:
        return self.mode == "advanced" and pipeline is not None

    @lru_cache(maxsize=1)
    def _load_transformer_pipeline(self):
        if not self.uses_transformers:
            return None
        return pipeline("sentiment-analysis", model=self.model_name)

    def _analyze_textblob(self, text: str) -> SentimentResult:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        label = sentiment_label_from_score(polarity)
        return SentimentResult(label=label, score=float(polarity))

    def _analyze_transformer(self, text: str) -> SentimentResult:
        nlp = self._load_transformer_pipeline()
        if nlp is None:
            return self._analyze_textblob(text)
        prediction = nlp(text[:512])[0]
        raw_label = prediction["label"]
        score = float(prediction["score"])
        if raw_label.upper().startswith("NEG"):
            label = "Negative"
            score = -score
        elif raw_label.upper().startswith("POS"):
            label = "Positive"
        else:
            label = "Neutral"
            score = 0.0
        return SentimentResult(label=label, score=score)

    def analyze(self, text: Optional[str]) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(label="Neutral", score=0.0)
        if self.uses_transformers:
            return self._analyze_transformer(text)
        return self._analyze_textblob(text)

    def bulk_analyze(self, texts: Iterable[Optional[str]]) -> List[SentimentResult]:
        return [self.analyze(text) for text in texts]


def sentiment_label_from_score(score: float, threshold: float = 0.05) -> str:
    if score > threshold:
        return "Positive"
    if score < -threshold:
        return "Negative"
    return "Neutral"


def attach_sentiment(df: pd.DataFrame, analyzer: SentimentAnalyzer) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["sentiment_label"] = []
        df["sentiment_score"] = []
        return df

    sentiments = analyzer.bulk_analyze(df["title"].fillna("") + " " + df["description"].fillna(""))
    df = df.copy()
    df["sentiment_label"] = [result.label for result in sentiments]
    df["sentiment_score"] = [result.score for result in sentiments]
    return df


__all__ = ["SentimentAnalyzer", "SentimentResult", "attach_sentiment", "sentiment_label_from_score"]
