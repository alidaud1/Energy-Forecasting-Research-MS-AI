"""Regression checks for the equal-hour 2020 comparison."""

from pathlib import Path

import numpy as np
import pytest

from elsevier_paper.experiments.compare_opsd_2020 import hac_test, hourly_pinball
from elsevier_paper.experiments.verify_opsd_baseline_2020 import verify


def test_hourly_pinball_exact_at_median():
    y = np.array([10.0, 20.0])
    q = np.repeat(y[:, None], 99, axis=1)
    assert np.array_equal(hourly_pinball(y, q), np.zeros(2))


def test_hac_rejects_too_few_hours():
    with pytest.raises(ValueError, match="invalid loss"):
        hac_test(np.ones(10), 6)


def test_hac_constant_offset():
    result = hac_test(np.full(400, 3.0), 24)
    assert result["mean_loss_A_minus_B_mw"] == 3.0
    assert result["standard_error_mw"] == 0.0
    assert result["ci95_mw"] == [3.0, 3.0]


def test_saved_2020_baseline_recomputes():
    root = Path(__file__).resolve().parent / "opsd_de_baseline_2020"
    result = verify(root)
    assert result["verified_hours"] == 4367
    assert result["monthly_decision_reports"] == 12
