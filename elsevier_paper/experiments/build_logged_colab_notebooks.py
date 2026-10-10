"""Generate separate journal rerun notebooks; never rewrite legacy notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "elsevier_paper/colab_reproducibility"
MODELS = ("LSTM", "TCN", "PatchTST", "iTransformer", "TFT")


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build(output=OUTPUT):
    original = (ROOT / "agentic_solar/solar-forecasting-GEFCom2014/colab/solar_gpu_training.py").read_text()
    wrapper = Path(__file__).with_name("colab_provenance_runner.py").read_text()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in MODELS:
        intro = f"""# {model}: isolated, logged solar research rerun

Optional fresh experiment. Original results remain unchanged. One model per notebook/session.
Uses the original model-specific configuration and seed 42, not a reduced smoke-test budget.
The environment is observed at execution; it is not assigned to historical training.

Before running, upload `pipeline/outputs/splits/{model}/` (all three zone folders)
and `pipeline/outputs/scalers/{model}_scaler.pkl` into
`MyDrive/GEFCom2014-Solar/` with the same relative paths.
Select a GPU runtime. Run all cells. Do not change settings to improve a test score.
An unsuccessful run is retained and marked failed/interrupted; start a new attempt
rather than relabeling a partial checkpoint as completed training.
"""
        embedded = ("import sys, types\nORIGINAL_RUNNER_SOURCE = "+repr(original)+"\n"
                    "original_runner = types.ModuleType('solar_original_runner')\n"
                    "sys.modules[original_runner.__name__] = original_runner\n"
                    "exec(ORIGINAL_RUNNER_SOURCE, original_runner.__dict__)\n"
                    "PROVENANCE_WRAPPER_SOURCE = "+repr(wrapper)+"\n"
                    "logged_runner = types.ModuleType('solar_logged_runner')\n"
                    "sys.modules[logged_runner.__name__] = logged_runner\n"
                    "exec(PROVENANCE_WRAPPER_SOURCE, logged_runner.__dict__)\n"
                    "print(logged_runner.environment())\n")
        launch = f'''DRIVE_BASE = "/content/drive/MyDrive/GEFCom2014-Solar"
ARCHIVE_DIR = "/content/drive/MyDrive/GEFCom2014-Solar/journal_repro_archives"

output = logged_runner.run_logged_solar(
    original_runner, "{model}", ORIGINAL_RUNNER_SOURCE,
    drive_base=DRIVE_BASE, archive_dir=ARCHIVE_DIR,
)
print("Return the completed ZIP in:", ARCHIVE_DIR)
print("Keep the executed notebook with its cell outputs as additional evidence.")
'''
        notebook = {"cells": [cell("markdown", intro),
            cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"),
            cell("code", embedded), cell("code", launch),
            cell("markdown", "Return the completed run ZIP. It contains environment, source/settings, split hashes, scaler, epoch history, checkpoint, keyed forecasts, metrics and a verified artifact manifest. No original result is overwritten. Free Colab GPU access/time is not guaranteed.")],
            "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        path = output / f"{model}_logged_research.ipynb"
        path.write_text(json.dumps(notebook, indent=1)+"\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for path in build():
        print(path)
