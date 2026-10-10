import json
from pathlib import Path

from elsevier_paper.experiments.export_reproducibility import literal_assignment, source_tabular_recipe

ROOT = Path(__file__).resolve().parents[2]


def test_source_recipe_extraction_does_not_import_training_code():
    solar = ROOT / "agentic_solar/solar-forecasting-GEFCom2014"
    recipe = literal_assignment(solar / "colab/solar_gpu_training.py", "MODEL_CONFIG")
    assert recipe["LSTM"]["lookback"] == 168
    assert recipe["iTransformer"]["lr"] == .0003
    assert source_tabular_recipe(solar / "pipeline/model_training.py", "train_xgboost")["n_estimators"] == 300


def test_archived_load_budget_is_not_full_training():
    configurations = ROOT / "elsevier_paper/experiments/reproducibility"
    load = json.loads((configurations / "load_GBR_recorded_configuration.json").read_text())
    linear = json.loads((configurations / "load_Linear_recorded_configuration.json").read_text())
    assert load["training_row_limit"] == linear["training_row_limit"] == 5000
    assert load["training"]["n_estimators"] == 5
    assert linear["training"]["max_epochs"] == 10
