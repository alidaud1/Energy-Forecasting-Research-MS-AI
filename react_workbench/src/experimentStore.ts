import React from "react";

export type ModelFamily =
  | "linear"
  | "svr"
  | "mlp"
  | "lstm"
  | "gru"
  | "cnn"
  | "xgboost"
  | "patchtst"
  | "itransformer";

export type Track = "load" | "solar";

export type ExperimentConfig = {
  description: string;
  tracks: Track[];
  models: ModelFamily[]; // selected families (baseline + transformer)

  W: number;
  H: number;
  stride: number;

  interactive: boolean;
  use_llm: boolean;
  with_baselines: boolean;

  max_train_windows: number;
  max_test_windows: number;
  seed: number;

  epochs: number;
  batch_size: number;
  lr: number;
  patience: number;

  // HPO (baseline only). Backend support is incremental; UI still captures it.
  enable_hpo: boolean;
  hpo_track: Track;
  n_trials: number;
  init_batch_K: number;
  batch_B: number;
  constant_liar: boolean;
};

const STORAGE_KEY = "uaw_experiment_config_v1";

export const defaultExperimentConfig: ExperimentConfig = {
  description: "",
  tracks: ["load", "solar"],
  models: ["xgboost", "patchtst"],

  W: 168,
  H: 24,
  stride: 24,

  interactive: true,
  use_llm: false,
  with_baselines: true,

  max_train_windows: 20000,
  max_test_windows: 5000,
  seed: 42,

  epochs: 3,
  batch_size: 128,
  lr: 0.001,
  patience: 2,

  enable_hpo: false,
  hpo_track: "load",
  n_trials: 10,
  init_batch_K: 2,
  batch_B: 2,
  constant_liar: true,
};

function safeParse(raw: string | null): ExperimentConfig | null {
  if (!raw) return null;
  try {
    const obj = JSON.parse(raw);
    if (!obj || typeof obj !== "object") return null;
    return obj as ExperimentConfig;
  } catch {
    return null;
  }
}

export function useExperimentStore(): [ExperimentConfig, (next: ExperimentConfig) => void] {
  const [cfg, setCfg] = React.useState<ExperimentConfig>(() => {
    const saved = safeParse(localStorage.getItem(STORAGE_KEY));
    return saved ?? defaultExperimentConfig;
  });

  React.useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
    } catch {
      // ignore storage issues
    }
  }, [cfg]);

  return [cfg, setCfg];
}

