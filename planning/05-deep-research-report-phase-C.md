# Phase C Precheck: Verifying Phase A SlidingWindowDataset and LeakageSafeScaler for Transformer Training

## Executive summary

Your Phase A data objects already contain the essential ingredients needed for transformer forecasters: **chronologically ordered sequences**, **fixed-length windows**, and **leakage-safe normalization**. The key to Phase C is ensuring that the **tensor shapes match what PatchTST/iTransformer expect** and that your preprocessing does not accidentally leak future information through imputation or scaling.

The two core invariants you must preserve for Transformer training are:

- **Invariant A (Temporal integrity):** for every sample, history ends strictly before the forecast horizon begins (no time “peeking”). Your SlidingWindowDataset design satisfies this by taking `X_hist` from `[t … t+W-1]` and `Y` from `[t+W … t+W+H-1]`.
- **Invariant B (Feature dimensional integrity):** the order and meaning of channels/features must remain consistent across train/val/test, and your scalers must be fit on train only (your LeakageSafeScaler does this). StandardScaler explicitly stores per-feature mean/std learned from the training set and applies them to later data via `transform`. citeturn12search3

The main Phase C risk is not the windowing logic, but **when** you handle missing timestamps and interpolation—if interpolation is performed on the full time range before splitting, the interpolated values around split boundaries can indirectly use future values (a subtle leakage vector). If your dataset has no missing timestamps after reindexing, this is moot; otherwise it’s worth fixing.

## What Transformers expect vs what Phase A provides

### PatchTST input shape expectations

The Hugging Face PatchTST interface documents that `past_values` must be a tensor shaped:

- **`(batch_size, sequence_length, num_input_channels)`** citeturn0search1

This matches your Phase A window return shape naturally (after DataLoader batching):

- `X_hist` becomes **`[B, W, D_hist]`**

Thus, from a shape standpoint, **PatchTST-ready batching is already supported** so long as `X_hist` is a float tensor and your DataLoader uses default collation (which adds the batch dimension). PyTorch DataLoader docs describe that automatic batching yields tensors with an added batch dimension. citeturn12search6

### iTransformer shape implications

iTransformer’s conceptual change is that it treats **variates (channels) as tokens** by “inverting” dimensions relative to classical temporal-token transformers. Even if you store data as `[B, L, N]`, the model typically transposes internally or expects a transpose to `[B, N, L]` at some stage. High-level references describe this inversion as applying attention/processing on inverted dimensions to capture multivariate correlations. citeturn9search0turn2search0

The implication for Phase A is: **your `[B, W, D]` format is perfectly usable** for iTransformer too—either transpose inside the model or add a preprocessing layer that rearranges to the model’s preferred internal layout.

## SlidingWindowDataset: temporal dependency preservation and shape contract

### Shape contract produced by your dataset class

Your Phase A `SlidingWindowDataset` returns three tensors per sample:

- `X_hist`: shape **`[W, D_hist]`**
- `X_future`: shape **`[H, D_future]`**
- `Y`: shape **`[H, D_y]`** (typically `D_y=1` for load-only or solar-only)

When wrapped in a PyTorch `DataLoader(batch_size=B)`, default collation stacks them to:

- `X_hist_batch`: **`[B, W, D_hist]`**
- `X_future_batch`: **`[B, H, D_future]`**
- `Y_batch`: **`[B, H, D_y]`**

PyTorch’s data loading documentation notes that the default automatic batching collates samples into batched tensors with a batch dimension (usually the first). citeturn12search6

This is a strong match for transformer cores like PatchTST. citeturn0search1

### Temporal integrity guarantee (no leakage inside a sample)

Your dataset logic enforces the standard supervised forecasting structure:

- History covers indices `[start … start+W-1]`
- Forecast horizon covers `[start+W … start+W+H-1]`

This ensures:

- You never use future targets in the model input.
- You preserve causal order even if you later shuffle windows for SGD.

Your unit test (Phase A) checking `hist_end < future_start` is exactly the kind of invariant test you should keep as you expand to transformers.

### Feature dimensional integrity (the easy-to-miss requirement)

Transformers are extremely sensitive to “channel meaning drift.” Your manifest already locks down:

- `hist_cols`
- `future_cols`
- `target_cols`

For Phase C, **never infer columns on the fly**. Always load from `dataset_manifest.json` so that:

- The same features appear in the same order for all splits.
- The same `D_hist`/`D_future` shapes occur for Train/Val/Test.

If you later add “hour-of-day embeddings” or drop predictors, that must be done by creating a *new* manifest variant (or at least bumping a manifest version field) to avoid silent shape mismatches.

## LeakageSafeScaler: scaling correctness and how it interacts with transformer training

### Why your scaler strategy is correct for time series

scikit-learn’s StandardScaler is documented to compute mean/std on the training set and store them for later transformations. citeturn12search3  
This is exactly what you want for time series to avoid data leakage.

Your LeakageSafeScaler design also keeps a clean separation:

- `_x_scaler` fits/transforms **feature columns**
- `_y_scaler` fits/transforms **target columns**

### Important implication: the target appears in the history inputs

Your `hist_cols` include the past target values (e.g., past `LOAD`), and those values will be transformed by `_y_scaler` if the column is in `target_cols`. Meanwhile, weather/NWP/time features are transformed by `_x_scaler`.

This is logically consistent, but it creates a subtle reality:

- The target channel in `X_hist` is scaled with **target statistics**
- Other channels are scaled with **feature statistics**

This is not “wrong,” and many forecasting pipelines do this. The transformer can learn with mixed-scaled channels. But it becomes important when you implement:
- per-channel normalization inside the model,
- or channel-independence assumptions (PatchTST often uses channel-independence concepts at the model level).

If you later observe training instability, one fix is to use a **single scaler** for all input channels (including lagged target) and keep a separate scaler only for loss/metrics. For now, keep your current approach—it’s defensible and reproducible.

### Inverse scaling for metrics and plots

For Phase C evaluation, you’ll compute loss in scaled space but report MAE/RMSE in original units. That requires `inverse_transform_targets`.

Be careful about shapes:
- scikit-learn scalers expect 2D arrays: `[n_samples, n_targets]`
- your transformer outputs are typically 3D: `[B, H, D_y]`

So you’ll need to reshape for inverse transform:

- `(B*H, D_y)` → inverse transform → reshape back to `(B, H, D_y)`

This preserves temporal meaning while enabling correct reporting.

### Artifact safety note for joblib

When you load scalers/models from `.joblib`, joblib notes explicitly that `joblib.load()` uses pickle and can execute arbitrary code; it should not be used on untrusted files. citeturn12search0  
This is fine in your controlled thesis environment but should be documented in your methodology if you claim “production-grade” behavior.

## Critical issues to resolve before plugging in PatchTST/iTransformer

### Interpolation timing can create boundary leakage

In your Phase A loader scripts, you suggested reindexing to a full hourly grid and interpolating missing timestamps. Interpolating the entire timeline *before splitting* can leak information near train/val/test boundaries because interpolation can use future points to fill past gaps.

If your dataset has no missing timestamps after extraction, this won’t matter. If it does, a safer “research-grade” modification is:

- Reindex the full grid to detect gaps
- Split chronologically per zone
- Interpolate within each split independently (or use forward-fill only in train)

This is one of the few ways Phase A can silently violate causal evaluation while still passing shape checks.

### Known-future covariates vs PatchTST interface

Your dataset yields `(X_hist, X_future, Y)`. PatchTST’s Hugging Face API accepts `past_values` and optionally `future_values`, where `future_values` is described as **future target values associated with past_values** (useful during training for some tasks), not general exogenous covariates. citeturn0search1

So, for Phase C you must decide one of these modeling strategies:

- **Strategy A (simplest):** ignore `X_future` and train a transformer on `X_hist → Y`.  
  This works for load reasonably well but can underuse solar NWP predictors.
- **Strategy B (recommended for your thesis):** implement your own PatchTST-like model that accepts **known-future covariates** (encoder inputs + separate future covariate stream or decoder conditioning).
- **Strategy C:** “mask-and-concatenate” sequence modeling: create an input length `W+H` where you provide covariates for all `W+H` steps but mask the target for the horizon portion.

Your Phase A objects already support all three strategies; the difference is in the transformer core and how you assemble tensors.

### Missing-value mask support (useful for robustness)

PatchTST in HF supports a `past_observed_mask` to tell the model which past values were observed vs missing (e.g., NaNs replaced by zeros). citeturn0search1  
Your current pipeline interpolates instead of keeping NaNs, so you may not need masks. But if you later choose to avoid interpolation, adding a mask becomes helpful and aligns with PatchTST’s documented interface. citeturn0search1

## Recommended transformer input mapping per track

### Load track

Your manifest likely defines:

- `target_cols = ["LOAD"]`
- `hist_cols = ["LOAD", "w1..w25", time_features...]`
- `future_cols = [time_features...]` (recommended default to avoid assuming future weather is known)

Transformer mapping:

- `past_values = X_hist` → `[B, W, D_hist]` (PatchTST-compatible) citeturn0search1
- `labels/targets = Y` → `[B, H, 1]`

### Solar track

Solar is exogenous-heavy:

- `target_cols = ["POWER"]`
- `hist_cols = ["POWER", VAR78.., time_features...]`
- `future_cols = ["VAR...", time_features...]` (NWP variables are known for forecast horizon in the benchmark setting)

Transformer mapping options:

- If using Strategy A: `past_values = X_hist` only.
- If using Strategy B/C: pass `X_future` explicitly into the model as known future covariates.

Either way, your dataset preserves the temporal alignment between predictors and future targets—so long as decumulation and shift conventions were applied consistently during Phase A (which is why your solar-specific preprocessing exists).

## Concrete “Phase C readiness checklist” for your two Phase A classes

Your Phase A design is transformer-ready if all of the following are true:

- `SlidingWindowDataset` returns float tensors and DataLoader batches them to `[B, W, D]` (default batching behavior supports this). citeturn12search6  
- `hist_cols`, `future_cols`, `target_cols` come from `dataset_manifest.json` and are used consistently.
- Scalers are fit on train only (consistent with StandardScaler design storing training statistics for later `transform`). citeturn12search3  
- Any interpolation is either absent or done in a way that does not leak validation/test information into train.
- For PatchTST: you can map `X_hist` to `past_values` shape `(bs, sequence_length, num_input_channels)` as documented. citeturn0search1  

If you confirm these invariants, you can proceed safely to Phase C agent and transformer integration (SolarForecastingAgent + transformer core + ForecastIntegrationAgent) with high confidence that the data interface will not break or silently leak future information.