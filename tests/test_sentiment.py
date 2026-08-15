"""Explainable finance-domain sentiment contracts."""

from __future__ import annotations

import pytest

from signalglass.sentiment import score_financial_text


def test_financial_sentiment_recognizes_positive_results_and_guidance() -> None:
    result = score_financial_text(
        "Company beats estimates and raises guidance",
        "Record revenue and expanding margins supported the outlook.",
    )

    assert result.label == "Positive"
    assert result.score > 0.5
    assert "beats estimates" in result.evidence
    assert "raises guidance" in result.evidence


def test_financial_sentiment_handles_negation_and_downside_language() -> None:
    result = score_financial_text(
        "Company does not beat estimates and cuts guidance",
        "Margins declined while regulatory risk increased.",
    )

    assert result.label == "Negative"
    assert result.score < -0.4
    assert any(term in result.evidence for term in ("cuts guidance", "regulatory risk"))


def test_financial_sentiment_is_bounded_and_neutral_without_evidence() -> None:
    neutral = score_financial_text("Board schedules annual shareholder meeting")
    extreme = score_financial_text("record revenue strong growth beats estimates raises guidance upgrade")

    assert neutral.label == "Neutral"
    assert neutral.score == 0
    assert neutral.evidence == ()
    assert extreme.score == pytest.approx(1.0)
