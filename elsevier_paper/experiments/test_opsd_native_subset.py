"""Regression check for the independently saved measured native-subset run."""

from pathlib import Path

from elsevier_paper.experiments.verify_opsd_native_subset import verify


def test_native_subset_artifacts_recompute_exactly():
    root = Path(__file__).resolve().parent / "opsd_native_subset"
    result = verify(root)
    assert result["verified_hours"] == 8760
    assert result["monthly_decisions"] == 12
    assert result["scores"]["copula"]["coverage_percent"] == 97.28
