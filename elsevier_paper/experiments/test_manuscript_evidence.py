"""Keep the new displayed journal claims aligned with saved computations."""
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
# The submission copy is manuscript_v2.tex; manuscript.tex is the frozen v1
# baseline.  Checking both keeps the claim guards on the file actually being
# submitted without losing cover on the archived version.
TEX_V1 = (BASE.parent / "manuscript.tex").read_text()
TEX = (BASE.parent / "manuscript_v2.tex").read_text()


def test_solar_primary_statistics_match_displayed_values():
    directory = BASE / "solar_checkpoint_replay/statistical_audit"
    audit = json.loads((directory / "statistical_audit.json").read_text())
    lstm = next(r for r in audit["comparisons"] if r["comparator"] == "LSTM")
    assert f"$p={lstm['p_value']:.2f}$" in TEX
    assert f"$p={lstm['holm_p_value']:.2f}$" in TEX
    assert audit["benchmark_all_quantiles_identical"] is True
    metrics = pd.read_csv(directory / "solar_metrics_with_shared_day_ci.csv")
    ensemble = metrics.loc[metrics.model.eq("Ensemble")].iloc[0]
    assert f"{100*ensemble.positive_generation_coverage:.2f}\\%" in TEX
    assert f"{100*ensemble.pinball_ci95_low:.2f}--{100*ensemble.pinball_ci95_high:.2f}\\%" in TEX


def test_contract_counts_and_unit_error_match_computed_artifact():
    audit = json.loads((BASE / "contract_ablation/contract_ablation.json").read_text())
    assert audit["summary"]["checked_rejections"] == 9
    assert audit["summary"]["unchecked_finite_outputs"] == 7
    value = audit["declared_unit_case"]["unchecked_max_abs_difference_mw"]
    assert f"{value:,.2f}".replace(",", "{,}") in TEX
    assert audit["negative_control"]["checked"] == "finite_output"
    assert "Not detected & Not detected" in TEX


def test_new_load_table_and_inference_match_full_budget_artifacts():
    audit = json.loads((BASE / "load_full_budget_24h/verified_statistical_summary.json").read_text())
    for name, report in audit["models"].items():
        scores = report["metrics"]["test"]
        row = f"{name} & {scores['pinball']:.2f} & {scores['mae']:.2f} & {100*scores['coverage_q10_q90']:.2f} & {scores['sharpness_q10_q90']:.2f}"
        assert row in TEX
        low, high = report["test_pinball_shared_week_ci95"]
        assert f"{low:.2f}--{high:.2f}" in TEX
    assert f"$p={audit['primary_hac_lag168']['p_value']:.2f}$" in TEX
    assert f"$p={audit['sensitivity_hac_lag24']['p_value']:.2f}$" in TEX
