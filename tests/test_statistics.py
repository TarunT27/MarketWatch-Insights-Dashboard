"""Statistical honesty around the headline directional-accuracy number."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalglass.analytics import majority_direction_baseline, wilson_interval
from signalglass.models import SignalEvaluation


def _evaluation(accuracy: float, samples: int, baseline: float = 0.5) -> SignalEvaluation:
    hits = round(accuracy * samples)
    low, high = wilson_interval(hits, samples)
    return SignalEvaluation(
        predictions=pd.DataFrame(columns=["date", "actual_return", "predicted_return"]),
        mean_absolute_error=0.01,
        directional_accuracy=accuracy,
        sample_size=samples,
        accuracy_low=low,
        accuracy_high=high,
        majority_baseline=baseline,
    )


def test_wilson_interval_brackets_the_observed_proportion() -> None:
    low, high = wilson_interval(43, 74)

    assert low < 43 / 74 < high
    assert 0.0 <= low < high <= 1.0


def test_wilson_interval_is_wider_for_smaller_samples() -> None:
    narrow_low, narrow_high = wilson_interval(600, 1000)
    wide_low, wide_high = wilson_interval(6, 10)

    assert (wide_high - wide_low) > (narrow_high - narrow_low)


def test_wilson_interval_stays_inside_zero_and_one_at_the_extremes() -> None:
    low, high = wilson_interval(10, 10)

    assert low >= 0.0
    assert high <= 1.0


def test_wilson_interval_handles_no_trials() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_majority_baseline_picks_the_more_common_direction() -> None:
    mostly_up = np.array([0.01, 0.02, -0.01, 0.03, 0.01])

    assert majority_direction_baseline(mostly_up) == pytest.approx(0.8)


def test_majority_baseline_is_never_below_a_coin_flip() -> None:
    balanced = np.array([0.01, -0.01, 0.02, -0.02])

    assert majority_direction_baseline(balanced) >= 0.5


def test_majority_baseline_defaults_to_half_without_observations() -> None:
    assert majority_direction_baseline(np.array([])) == 0.5


def test_a_typical_74_session_result_is_reported_as_not_significant() -> None:
    """58% over 74 sessions is inside the coin-flip band and must say so."""

    evaluation = _evaluation(0.581, 74, baseline=0.55)

    assert evaluation.beats_baseline is True
    assert evaluation.beats_coin_flip is False
    assert evaluation.verdict == "Not significant"


def test_a_model_losing_to_the_naive_baseline_is_flagged() -> None:
    evaluation = _evaluation(0.435, 69, baseline=0.58)

    assert evaluation.beats_baseline is False
    assert evaluation.verdict == "Below baseline"


def test_a_genuinely_strong_result_is_reported_as_significant() -> None:
    evaluation = _evaluation(0.70, 400, baseline=0.55)

    assert evaluation.beats_coin_flip is True
    assert evaluation.beats_baseline is True
    assert evaluation.verdict == "Significant"
