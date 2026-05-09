import React from "react";
import type { ExperimentConfig, ModelFamily, Track } from "../experimentStore";

const MODEL_LABELS: Record<ModelFamily, string> = {
  linear: "Linear",
  svr: "SVR",
  mlp: "MLP",
  lstm: "LSTM",
  gru: "GRU",
  cnn: "CNN",
  xgboost: "XGBoost",
  patchtst: "PatchTST",
  itransformer: "iTransformer",
};

const BASELINES: ModelFamily[] = ["linear", "svr", "mlp", "lstm", "gru", "cnn", "xgboost"];
const TRANSFORMERS: ModelFamily[] = ["patchtst", "itransformer"];

function toggleInList<T>(xs: T[], x: T): T[] {
  return xs.includes(x) ? xs.filter((v) => v !== x) : [...xs, x];
}

function clampInt(v: any, fallback: number): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.floor(n));
}

function clampFloat(v: any, fallback: number): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return n;
}

export default function ExperimentSetupTab(props: {
  cfg: ExperimentConfig;
  setCfg: (c: ExperimentConfig) => void;
  onStart: () => void;
  startDisabled: boolean;
  startError: string;
}) {
  const { cfg, setCfg, onStart, startDisabled, startError } = props;

  const set = (patch: Partial<ExperimentConfig>) => setCfg({ ...cfg, ...patch });

  const hasAllModels = cfg.models.length === BASELINES.length + TRANSFORMERS.length;

  const selectAll = () => set({ models: [...BASELINES, ...TRANSFORMERS] });
  const clearAll = () => set({ models: [] });

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-neutral-200 overflow-hidden">
        <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">Experiment Setup</div>
        <div className="p-4 grid grid-cols-1 xl:grid-cols-3 gap-4">
          <div className="space-y-3">
            <div className="text-sm font-semibold">Run Metadata</div>
            <label className="block">
              <div className="text-xs text-neutral-600 mb-1">Run name / description</div>
              <input
                className="w-full border border-neutral-300 rounded-md px-3 py-2 text-sm"
                value={cfg.description}
                onChange={(e) => set({ description: e.target.value })}
                placeholder="e.g., Peak-demand tuning sweep (trial 1)"
              />
            </label>

            <div className="text-sm font-semibold pt-2">Tracks</div>
            <div className="flex gap-3">
              {(["load", "solar"] as Track[]).map((t) => (
                <label key={t} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={cfg.tracks.includes(t)}
                    onChange={() => set({ tracks: toggleInList(cfg.tracks, t) })}
                  />
                  {t}
                </label>
              ))}
            </div>

            <div className="text-sm font-semibold pt-2">Global Window</div>
            <div className="grid grid-cols-3 gap-2">
              <label className="block">
                <div className="text-xs text-neutral-600 mb-1">W</div>
                <input
                  className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm"
                  value={cfg.W}
                  onChange={(e) => set({ W: clampInt(e.target.value, cfg.W) })}
                />
              </label>
              <label className="block">
                <div className="text-xs text-neutral-600 mb-1">H</div>
                <input
                  className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm"
                  value={cfg.H}
                  onChange={(e) => set({ H: clampInt(e.target.value, cfg.H) })}
                />
              </label>
              <label className="block">
                <div className="text-xs text-neutral-600 mb-1">stride</div>
                <input
                  className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm"
                  value={cfg.stride}
                  onChange={(e) => set({ stride: clampInt(e.target.value, cfg.stride) })}
                />
              </label>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">Model Configuration Grid</div>
              <div className="flex gap-2">
                <button className="text-xs px-2 py-1 rounded-md border border-neutral-300" onClick={hasAllModels ? clearAll : selectAll}>
                  {hasAllModels ? "Clear All" : "Select All"}
                </button>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              {[...BASELINES, ...TRANSFORMERS].map((m) => {
                const on = cfg.models.includes(m);
                return (
                  <button
                    key={m}
                    onClick={() => set({ models: toggleInList(cfg.models, m) })}
                    className={`text-sm px-3 py-2 rounded-lg border ${
                      on ? "border-neutral-900 bg-neutral-900 text-white" : "border-neutral-300 bg-white hover:bg-neutral-50"
                    }`}
                  >
                    {MODEL_LABELS[m]}
                  </button>
                );
              })}
            </div>

            <div className="text-xs text-neutral-600">
              Note: PatchTST/iTransformer run in research mode; baselines run as additional commands in the same run.
            </div>
          </div>

          <div className="space-y-3">
            <div className="text-sm font-semibold">Agentic & Training Controls</div>
            <div className="grid grid-cols-2 gap-3">
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={cfg.interactive} onChange={(e) => set({ interactive: e.target.checked })} />
                Interactive mode (gates)
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={cfg.use_llm} onChange={(e) => set({ use_llm: e.target.checked })} />
                Use LLM control layer
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={cfg.with_baselines} onChange={(e) => set({ with_baselines: e.target.checked })} />
                Include baselines
              </label>
            </div>

            <div className="grid grid-cols-4 gap-2 pt-1">
              <label className="block col-span-2">
                <div className="text-xs text-neutral-600 mb-1">epochs</div>
                <input className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm" value={cfg.epochs} onChange={(e) => set({ epochs: clampInt(e.target.value, cfg.epochs) })} />
              </label>
              <label className="block col-span-2">
                <div className="text-xs text-neutral-600 mb-1">batch_size</div>
                <input className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm" value={cfg.batch_size} onChange={(e) => set({ batch_size: clampInt(e.target.value, cfg.batch_size) })} />
              </label>
              <label className="block col-span-2">
                <div className="text-xs text-neutral-600 mb-1">lr</div>
                <input className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm" value={cfg.lr} onChange={(e) => set({ lr: clampFloat(e.target.value, cfg.lr) })} />
              </label>
              <label className="block col-span-2">
                <div className="text-xs text-neutral-600 mb-1">patience</div>
                <input className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm" value={cfg.patience} onChange={(e) => set({ patience: clampInt(e.target.value, cfg.patience) })} />
              </label>
            </div>

            <div className="rounded-lg border border-neutral-200 p-3 space-y-2">
              <div className="text-sm font-semibold">HPO & Optimization (Baseline)</div>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={cfg.enable_hpo} onChange={(e) => set({ enable_hpo: e.target.checked })} />
                Enable Bayesian Optimization (HPO)
              </label>
              {cfg.enable_hpo ? (
                <div className="grid grid-cols-2 gap-2">
                  <label className="block">
                    <div className="text-xs text-neutral-600 mb-1">target track</div>
                    <select className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm" value={cfg.hpo_track} onChange={(e) => set({ hpo_track: e.target.value as Track })}>
                      <option value="load">load</option>
                      <option value="solar">solar</option>
                    </select>
                  </label>
                  <label className="block">
                    <div className="text-xs text-neutral-600 mb-1">trials</div>
                    <input className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm" value={cfg.n_trials} onChange={(e) => set({ n_trials: clampInt(e.target.value, cfg.n_trials) })} />
                  </label>
                  <label className="block">
                    <div className="text-xs text-neutral-600 mb-1">init batch K</div>
                    <input className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm" value={cfg.init_batch_K} onChange={(e) => set({ init_batch_K: clampInt(e.target.value, cfg.init_batch_K) })} />
                  </label>
                  <label className="block">
                    <div className="text-xs text-neutral-600 mb-1">batch B</div>
                    <input className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm" value={cfg.batch_B} onChange={(e) => set({ batch_B: clampInt(e.target.value, cfg.batch_B) })} />
                  </label>
                  <label className="flex items-center gap-2 text-sm col-span-2">
                    <input type="checkbox" checked={cfg.constant_liar} onChange={(e) => set({ constant_liar: e.target.checked })} />
                    Constant liar batching
                  </label>
                  <div className="text-xs text-neutral-600 col-span-2">
                    Backend support for baseline-HPO from the workbench is incremental; CLI baseline mode remains the reference.
                  </div>
                </div>
              ) : null}
            </div>

            <div className="pt-2 flex items-center gap-3">
              <button
                className="px-4 py-2 rounded-md bg-ieeeGreen hover:brightness-95 font-semibold text-sm disabled:opacity-50"
                disabled={startDisabled}
                onClick={onStart}
              >
                Start Master Run
              </button>
              {startError ? <div className="text-sm text-red-700">{startError}</div> : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

