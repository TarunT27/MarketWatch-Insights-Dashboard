"""Sentiment behaviour on headlines written the way wires actually write them.

The original suite only fed the scorer sentences assembled from its own lexicon,
which is why an engine that scored every real headline as Neutral still passed.
These tests use naturally worded headlines and assert the engine is not inert.
"""

from __future__ import annotations

import pytest

from signalglass.demo_data import generate_demo_news
from signalglass.sentiment import score_financial_text

# Headlines phrased the way Reuters/Bloomberg/AP write them, not lexicon dumps.
_POSITIVE_HEADLINES = (
    "Apple tops Wall Street expectations as iPhone sales surge",
    "Microsoft announces $60 billion share repurchase program",
    "Meta raises guidance and reports record revenue",
    "Bank shares rally after upgrade from analysts",
    "Chipmaker soars on better than expected results",
)
_NEGATIVE_HEADLINES = (
    "Nvidia stock plunges after weak forecast spooks investors",
    "Tesla recalls 1.2 million vehicles over software flaw",
    "Amazon misses revenue estimates as growth slows",
    "Alphabet faces antitrust probe in Europe",
    "Retailer announces layoffs amid weak demand",
)
_NEUTRAL_HEADLINES = (
    "Board schedules annual shareholder meeting",
    "Company to present at industry conference next month",
    "Chief executive to speak at the investor day in June",
)


@pytest.mark.parametrize("headline", _POSITIVE_HEADLINES)
def test_realistic_positive_headlines_score_positive(headline: str) -> None:
    result = score_financial_text(headline)

    assert result.label == "Positive", f"{headline!r} scored {result.score}"
    assert result.score > 0
    assert result.evidence, "a scored headline must name the phrase that moved it"


@pytest.mark.parametrize("headline", _NEGATIVE_HEADLINES)
def test_realistic_negative_headlines_score_negative(headline: str) -> None:
    result = score_financial_text(headline)

    assert result.label == "Negative", f"{headline!r} scored {result.score}"
    assert result.score < 0
    assert result.evidence


@pytest.mark.parametrize("headline", _NEUTRAL_HEADLINES)
def test_headlines_without_finance_signal_stay_neutral(headline: str) -> None:
    result = score_financial_text(headline)

    assert result.label == "Neutral"
    assert result.evidence == ()


def test_negation_flips_an_otherwise_positive_phrase() -> None:
    plain = score_financial_text("Chipmaker beats expectations")
    negated = score_financial_text("Chipmaker fails to beat expectations")

    assert plain.label == "Positive"
    assert negated.label == "Negative"
    assert any(term.startswith("no ") for term in negated.evidence)


def test_longest_phrase_wins_so_evidence_is_not_double_counted() -> None:
    result = score_financial_text("Company beats estimates")

    # "beats estimates" must be reported once, not as "beats" plus "estimates".
    assert result.evidence == ("beats estimates",)


def test_demo_headlines_are_scored_by_the_engine_not_hardcoded() -> None:
    """Demo mode must exercise the real scorer, or the demo proves nothing."""

    news = generate_demo_news("AAPL", 90)

    for record in news.to_dict(orient="records"):
        recomputed = score_financial_text(record["title"], record["description"])
        assert record["sentiment_label"] == recomputed.label
        assert record["sentiment_score"] == pytest.approx(recomputed.score)
        assert record["sentiment_evidence"] == ", ".join(recomputed.evidence)


def test_demo_headlines_span_every_label() -> None:
    """A demo that is all one tone cannot demonstrate the scorer."""

    labels = set(generate_demo_news("AAPL", 90)["sentiment_label"])

    assert labels == {"Positive", "Negative", "Neutral"}


def test_scores_stay_within_bounds_when_many_phrases_match() -> None:
    result = score_financial_text(
        "record revenue strong growth beats estimates raises guidance upgrade record profit"
    )

    assert -1.0 <= result.score <= 1.0
