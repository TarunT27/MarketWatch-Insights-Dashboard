"""Explainable finance-domain sentiment scoring for market headlines."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FinancialSentiment:
    score: float
    label: str
    evidence: tuple[str, ...]


_POSITIVE_TERMS = {
    "beats estimates": 0.35,
    "beat estimates": 0.35,
    "raises guidance": 0.35,
    "raised guidance": 0.35,
    "record revenue": 0.25,
    "strong growth": 0.25,
    "expanding margins": 0.20,
    "margin expansion": 0.20,
    "upgrade": 0.20,
    "outperform": 0.18,
    "buyback": 0.15,
}
_NEGATIVE_TERMS = {
    "does not beat estimates": 0.45,
    "not beat estimates": 0.45,
    "misses estimates": 0.35,
    "missed estimates": 0.35,
    "cuts guidance": 0.40,
    "cut guidance": 0.40,
    "guidance cut": 0.40,
    "margins declined": 0.25,
    "margin pressure": 0.22,
    "regulatory risk": 0.25,
    "downgrade": 0.20,
    "investigation": 0.20,
    "default": 0.35,
    "fraud": 0.40,
    "layoffs": 0.18,
}


def _normalized_text(*parts: object) -> str:
    joined = " ".join(str(part or "") for part in parts).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%$]+", " ", joined)).strip()


def score_financial_text(title: object, description: object = "") -> FinancialSentiment:
    """Score finance-specific phrases and return the exact matched evidence."""

    text = _normalized_text(title, description)
    positive_matches = [term for term in _POSITIVE_TERMS if term in text]
    negative_matches = [term for term in _NEGATIVE_TERMS if term in text]

    # Do not reward an affirmative phrase when it is contained inside an explicit negation.
    if any(term in text for term in ("does not beat estimates", "not beat estimates")):
        positive_matches = [term for term in positive_matches if term not in {"beat estimates"}]

    raw_score = sum(_POSITIVE_TERMS[term] for term in positive_matches) - sum(
        _NEGATIVE_TERMS[term] for term in negative_matches
    )
    score = float(np.clip(raw_score, -1.0, 1.0))
    label = "Positive" if score > 0.10 else "Negative" if score < -0.10 else "Neutral"
    evidence = tuple(dict.fromkeys([*positive_matches, *negative_matches]))
    return FinancialSentiment(score=score, label=label, evidence=evidence)


__all__ = ["FinancialSentiment", "score_financial_text"]
