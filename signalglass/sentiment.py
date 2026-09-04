"""Explainable finance-domain sentiment scoring for market headlines.

The scorer is deliberately a transparent lexicon rather than a black-box
classifier: every point of the returned score traces back to a phrase that
appears in the headline, and negated phrases are reported as such. Matching is
token-based with longest-phrase-wins, so "beats estimates" is scored as one
piece of evidence instead of double counting the word "beats".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# How many tokens before a match are scanned for a negation cue.
_NEGATION_WINDOW = 3
# Negated evidence is real but weaker than a direct statement.
_NEGATION_DAMPING = 0.8
_POSITIVE_LABEL_THRESHOLD = 0.10


@dataclass(frozen=True, slots=True)
class SentimentMatch:
    """One phrase that contributed to a headline's score."""

    term: str
    weight: float
    negated: bool

    @property
    def contribution(self) -> float:
        """Signed points this match added to the headline score."""

        if self.negated:
            return -self.weight * _NEGATION_DAMPING
        return self.weight

    def describe(self) -> str:
        """Render the match the way it is surfaced in the product."""

        return f"no {self.term}" if self.negated else self.term


@dataclass(frozen=True, slots=True)
class FinancialSentiment:
    score: float
    label: str
    evidence: tuple[str, ...]
    matches: tuple[SentimentMatch, ...] = ()


# Weights are points on a [-1, 1] scale. Multi-word phrases outrank the single
# words they contain, which is what makes the evidence readable.
_POSITIVE_TERMS: dict[str, float] = {
    # Results versus expectations.
    "beats estimates": 0.35,
    "beat estimates": 0.35,
    "beats expectations": 0.35,
    "beat expectations": 0.35,
    "tops estimates": 0.35,
    "tops expectations": 0.35,
    "topped estimates": 0.35,
    "topped expectations": 0.35,
    "exceeds expectations": 0.32,
    "exceeded expectations": 0.32,
    "ahead of expectations": 0.28,
    "better than expected": 0.30,
    "beats": 0.24,
    "beat": 0.20,
    "tops": 0.24,
    "topped": 0.24,
    # Guidance and outlook.
    "raises guidance": 0.35,
    "raised guidance": 0.35,
    "raises outlook": 0.32,
    "raised outlook": 0.32,
    "raises forecast": 0.32,
    "lifts guidance": 0.32,
    "boosts outlook": 0.30,
    "upbeat forecast": 0.28,
    "upbeat outlook": 0.28,
    "guidance raised": 0.32,
    # Fundamentals.
    "record revenue": 0.25,
    "record profit": 0.28,
    "record quarter": 0.26,
    "record sales": 0.25,
    "record high": 0.24,
    "all time high": 0.26,
    "strong growth": 0.25,
    "strong demand": 0.26,
    "strong results": 0.25,
    "robust demand": 0.24,
    "expanding margins": 0.20,
    "margin expansion": 0.20,
    "profit rose": 0.22,
    "profits rose": 0.22,
    "revenue growth": 0.18,
    "returns to growth": 0.22,
    # Capital returns and corporate actions.
    "buyback": 0.15,
    "share repurchase": 0.18,
    "repurchase program": 0.18,
    "dividend increase": 0.20,
    "raises dividend": 0.22,
    "wins contract": 0.24,
    "won contract": 0.24,
    "wins approval": 0.26,
    "approval": 0.15,
    "approved": 0.14,
    "partnership": 0.14,
    "breakthrough": 0.24,
    "expansion": 0.12,
    # Analyst and market reaction.
    "upgrade": 0.20,
    "upgrades": 0.20,
    "upgraded": 0.20,
    "outperform": 0.18,
    "price target raised": 0.24,
    "raises price target": 0.24,
    "overweight": 0.14,
    "bullish": 0.22,
    "optimism": 0.18,
    "rally": 0.20,
    "rallies": 0.20,
    "rallied": 0.20,
    "surge": 0.24,
    "surges": 0.24,
    "surged": 0.24,
    "soars": 0.26,
    "soared": 0.26,
    "jumps": 0.22,
    "jumped": 0.22,
    "climbs": 0.18,
    "climbed": 0.18,
    "gains": 0.14,
    "rebound": 0.18,
    "recovery": 0.16,
}

_NEGATIVE_TERMS: dict[str, float] = {
    # Results versus expectations.
    "misses estimates": 0.35,
    "missed estimates": 0.35,
    "misses expectations": 0.35,
    "missed expectations": 0.35,
    "falls short": 0.28,
    "fell short": 0.28,
    "worse than expected": 0.32,
    "disappointing": 0.28,
    "disappoints": 0.28,
    "misses": 0.22,
    "missed": 0.20,
    # Guidance and outlook.
    "cuts guidance": 0.40,
    "cut guidance": 0.40,
    "guidance cut": 0.40,
    "lowers guidance": 0.38,
    "lowered guidance": 0.38,
    "slashes guidance": 0.42,
    "cuts outlook": 0.36,
    "lowers outlook": 0.34,
    "weak forecast": 0.34,
    "weak outlook": 0.34,
    "weak guidance": 0.34,
    "warns": 0.28,
    "warning": 0.26,
    "profit warning": 0.40,
    # Fundamentals.
    "margins declined": 0.25,
    "margin pressure": 0.22,
    "weak demand": 0.30,
    "weak results": 0.28,
    "weak sales": 0.28,
    "slowing growth": 0.26,
    "growth slows": 0.26,
    "sales decline": 0.26,
    "revenue decline": 0.26,
    "loss widened": 0.30,
    "losses widened": 0.30,
    "writedown": 0.28,
    "write down": 0.26,
    "impairment": 0.26,
    "shortage": 0.20,
    "supply constraints": 0.22,
    "oversupply": 0.20,
    # Legal, regulatory, and governance.
    "regulatory risk": 0.25,
    "regulatory scrutiny": 0.26,
    "investigation": 0.20,
    "probe": 0.22,
    "lawsuit": 0.24,
    "sued": 0.22,
    "litigation": 0.20,
    "antitrust": 0.24,
    "fined": 0.24,
    "penalty": 0.20,
    "fraud": 0.40,
    "bankruptcy": 0.45,
    "default": 0.35,
    "delisting": 0.35,
    "recall": 0.26,
    "recalls": 0.26,
    "recalled": 0.26,
    "steps down": 0.22,
    "resignation": 0.20,
    "ousted": 0.26,
    # Workforce and analyst reaction.
    "layoffs": 0.18,
    "job cuts": 0.20,
    "cuts jobs": 0.20,
    "downgrade": 0.20,
    "downgrades": 0.20,
    "downgraded": 0.20,
    "underperform": 0.18,
    "price target cut": 0.24,
    "cuts price target": 0.24,
    "underweight": 0.14,
    "bearish": 0.22,
    "selloff": 0.24,
    "plunges": 0.32,
    "plunged": 0.32,
    "plummets": 0.32,
    "plummeted": 0.32,
    "tumbles": 0.28,
    "tumbled": 0.28,
    "slumps": 0.26,
    "slumped": 0.26,
    "sinks": 0.26,
    "sank": 0.26,
    "slides": 0.20,
    "slid": 0.20,
    "falls": 0.18,
    "fell": 0.16,
    "drops": 0.18,
    "dropped": 0.18,
    "declines": 0.18,
    "declined": 0.18,
    "cautious": 0.14,
    "concerns": 0.16,
    "headwinds": 0.18,
    "volatility": 0.12,
}

# Cues that invert the meaning of a following phrase.
_NEGATION_CUES = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "fails",
        "fail",
        "failed",
        "failing",
        "denies",
        "denied",
        "unlikely",
        "avoids",
        "avoided",
        "halts",
        "halted",
        "stops",
        "stopped",
    }
)

_TOKEN_SPLIT_PATTERN = re.compile(r"[^a-z0-9%$]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _tokenize(*parts: object) -> list[str]:
    joined = " ".join(str(part or "") for part in parts).lower()
    collapsed = _WHITESPACE_PATTERN.sub(" ", _TOKEN_SPLIT_PATTERN.sub(" ", joined)).strip()
    return collapsed.split() if collapsed else []


def _build_term_index() -> tuple[tuple[tuple[str, ...], str, float], ...]:
    """Pre-tokenize every lexicon entry, longest phrase first."""

    entries: list[tuple[tuple[str, ...], str, float]] = []
    for term, weight in _POSITIVE_TERMS.items():
        entries.append((tuple(term.split()), term, weight))
    for term, weight in _NEGATIVE_TERMS.items():
        entries.append((tuple(term.split()), term, -weight))
    entries.sort(key=lambda entry: (-len(entry[0]), entry[1]))
    return tuple(entries)


_TERM_INDEX = _build_term_index()


def _is_negated(tokens: list[str], start: int) -> bool:
    window_start = max(0, start - _NEGATION_WINDOW)
    return any(token in _NEGATION_CUES for token in tokens[window_start:start])


def _find_matches(tokens: list[str]) -> list[SentimentMatch]:
    """Match lexicon phrases, longest first, without reusing a token."""

    consumed = [False] * len(tokens)
    matches: list[SentimentMatch] = []
    for term_tokens, term, signed_weight in _TERM_INDEX:
        span = len(term_tokens)
        if span > len(tokens):
            continue
        for start in range(len(tokens) - span + 1):
            end = start + span
            if any(consumed[start:end]):
                continue
            if tuple(tokens[start:end]) != term_tokens:
                continue
            for position in range(start, end):
                consumed[position] = True
            matches.append(
                SentimentMatch(
                    term=term,
                    weight=abs(signed_weight) if signed_weight > 0 else -abs(signed_weight),
                    negated=_is_negated(tokens, start),
                )
            )
    return matches


def score_financial_text(title: object, description: object = "") -> FinancialSentiment:
    """Score finance-specific phrases and return the exact matched evidence."""

    tokens = _tokenize(title, description)
    matches = _find_matches(tokens)
    raw_score = sum(match.contribution for match in matches)
    score = float(np.clip(raw_score, -1.0, 1.0))
    label = (
        "Positive"
        if score > _POSITIVE_LABEL_THRESHOLD
        else "Negative"
        if score < -_POSITIVE_LABEL_THRESHOLD
        else "Neutral"
    )
    # Order evidence by how much it moved the score so the top reason reads first.
    ordered = sorted(matches, key=lambda match: (-abs(match.contribution), match.term))
    evidence = tuple(dict.fromkeys(match.describe() for match in ordered))
    return FinancialSentiment(
        score=score,
        label=label,
        evidence=evidence,
        matches=tuple(ordered),
    )


def lexicon_size() -> int:
    """Total number of scored phrases, surfaced in the product as provenance."""

    return len(_POSITIVE_TERMS) + len(_NEGATIVE_TERMS)


__all__ = ["FinancialSentiment", "SentimentMatch", "lexicon_size", "score_financial_text"]
