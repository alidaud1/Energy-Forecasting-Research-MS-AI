"""Measure the CPU subnormal arithmetic optimization on real TCN inputs."""
import importlib.util
import json
import time

import numpy as np
import pandas as pd
import torch

from .replay_solar_checkpoints import SOLAR, OUTPUT, sha


def run():
    torch.set_num_threads(1)
    path = SOLAR / "colab/solar_gpu_training.py"
    spec = importlib.util.spec_from_file_location("tcn_cpu_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    splits = SOLAR / "pipeline/outputs/splits/TCN"
    features = module.get_feature_cols(pd.read_csv(splits / "zone1/train.csv", nrows=2))
    cfg = module.MODEL_CONFIG["TCN"]
    data = module.make_dataset("TCN", module.load_zone_frame(splits, 1, "val", cfg["lookback"]), features, cfg)
    # First validation target in zone 1: the full observed weather window.
    x = data[0][0][None]
    model = module.build_model("TCN", features, cfg).eval()
    checkpoint = SOLAR / "pipeline/saved_models/TCN_best.pt"
    torch.set_flush_denormal(False)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    count = sum(int(((v.abs()>0)&(v.abs()<torch.finfo(v.dtype).tiny)).sum()) for v in weights.values())
    model.load_state_dict(weights, strict=True)
    with torch.no_grad():
        started = time.monotonic()
        ordinary = model(x).numpy()
        ordinary_seconds = time.monotonic()-started
        torch.set_flush_denormal(True)
        started = time.monotonic()
        optimized = model(x).numpy()
        optimized_seconds = time.monotonic()-started
    difference = float(np.abs(ordinary-optimized).max())
    np.testing.assert_allclose(ordinary, optimized, atol=1e-7, rtol=0)
    report = {"checkpoint_sha256": sha(checkpoint), "subnormal_weight_count": count,
              "control": "first actual zone1 validation window; raw 99 outputs; one CPU thread",
              "max_absolute_output_difference": difference,
              "ordinary_seconds": ordinary_seconds, "flush_subnormals_seconds": optimized_seconds,
              "limit": "one-window arithmetic check, not a claim of identical hardware on every input"}
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "tcn_cpu_numerics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
