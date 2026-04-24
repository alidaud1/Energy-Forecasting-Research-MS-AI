import React from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { staticUrl } from "../api";
import { PoolMessage } from "../types";

function topicType(topic: string): string {
  const i = topic.indexOf("/");
  return i >= 0 ? topic.slice(0, i) : topic;
}

function findLatestIntegrate(messages: PoolMessage[], runId: string): PoolMessage | undefined {
  const rows = messages.filter((m) => m.run_id === runId && topicType(m.topic) === "integrate" && m.kind === "result");
  rows.sort((a, b) => a.seq - b.seq);
  return rows[rows.length - 1];
}

export default function VisualizationPanel(props: { runId: string; messages: PoolMessage[]; loadChartData: any[]; solarChartData: any[] }) {
  const { runId, messages, loadChartData } = props;

  const integrateMsg = React.useMemo(() => findLatestIntegrate(messages, runId), [messages, runId]);
  const net = (integrateMsg?.payload || {}).net_load || {};
  const netComputed = Boolean(net?.computed);
  const triplePlot = (integrateMsg?.payload?.net_load || {}).plot_path || "";

  const lossPoints = React.useMemo(() => {
    const pts: any[] = [];
    for (const m of messages) {
      if (m.run_id !== runId) continue;
      if (topicType(m.topic) !== "train") continue;
      const p = m.payload || {};
      if (p.event !== "epoch_metrics") continue;
      pts.push({
        epoch: Number(p.epoch ?? 0),
        train_mse: Number(p.train_mse ?? NaN),
        val_mse: Number(p.val_mse ?? NaN),
        track: String(p.track || ""),
        model: String(p.model_name || ""),
        variant: String(p.variant || "")
      });
    }
    pts.sort((a, b) => a.epoch - b.epoch);
    return pts;
  }, [messages, runId]);

  const [lossTrack, setLossTrack] = React.useState<"load" | "solar">("load");
  const lossFiltered = React.useMemo(() => lossPoints.filter((p) => p.track === lossTrack), [lossPoints, lossTrack]);

  const llmUsage = React.useMemo(() => {
    const rows: any[] = [];
    for (const m of messages) {
      if (m.run_id !== runId) continue;
      if (topicType(m.topic) !== "llm_usage") continue;
      const p = m.payload || {};
      if (p.event !== "llm_usage") continue;
      const usage = p.usage || {};
      // Support multiple usage shapes; keep it best-effort.
      const input = Number(usage.input_tokens ?? usage.prompt_tokens ?? usage.input ?? 0);
      const output = Number(usage.output_tokens ?? usage.completion_tokens ?? usage.output ?? 0);
      const total = Number(usage.total_tokens ?? usage.total ?? input + output);
      rows.push({ gate: String(p.gate || "unknown"), input, output, total });
    }
    // aggregate by gate
    const agg = new Map<string, any>();
    for (const r of rows) {
      const cur = agg.get(r.gate) || { gate: r.gate, input: 0, output: 0, total: 0 };
      cur.input += r.input;
      cur.output += r.output;
      cur.total += r.total;
      agg.set(r.gate, cur);
    }
    return Array.from(agg.values()).sort((a, b) => String(a.gate).localeCompare(String(b.gate)));
  }, [messages, runId]);

  const hpImportance = React.useMemo(() => {
    const rows: any[] = [];
    for (const m of messages) {
      if (m.run_id !== runId) continue;
      if (topicType(m.topic) !== "hyperparameter_importance") continue;
      const p = m.payload || {};
      if (p.event !== "hyperparameter_importance") continue;
      const imp = p.importances || {};
      for (const k of Object.keys(imp)) {
        rows.push({ param: String(k), importance: Number(imp[k] ?? 0), track: String(p.track || "") });
      }
      break; // use the latest published importances snapshot
    }
    rows.sort((a, b) => b.importance - a.importance);
    return rows.slice(0, 20);
  }, [messages, runId]);

  const errorByHour = React.useMemo(() => {
    // Use the merged load chart series (primary pred) for a real error-vs-hour scatter.
    const pts: any[] = [];
    for (const r of loadChartData || []) {
      const t = String(r.t || "");
      const yt = Number(r.y_true);
      const yp = Number(r.y_pred_a);
      if (!t || !Number.isFinite(yt) || !Number.isFinite(yp)) continue;
      const d = new Date(t);
      const hour = Number.isFinite(d.getHours()) ? d.getHours() : 0;
      pts.push({ hour, abs_err: Math.abs(yt - yp) });
    }
    return pts.slice(0, 5000);
  }, [loadChartData]);

  return (
    <div className="rounded-xl border border-neutral-200 overflow-hidden">
      <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">Visualization Panel</div>
      <div className="p-3 grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="rounded-xl border border-neutral-200 overflow-hidden">
          <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">Net-Load</div>
          <div className="p-3 text-sm text-neutral-700">
            {netComputed ? (
              <>
                <div className="mb-2">
                  computed rows={String(net.n_rows ?? "")} MAE={String(net.metrics?.mae ?? "")} RMSE={String(net.metrics?.rmse ?? "")}
                </div>
                {triplePlot ? (
                  <img className="w-full rounded-md border border-neutral-200" src={staticUrl(String(triplePlot))} alt="Net-load triple axis plot" />
                ) : (
                  <div className="text-neutral-600">No plot path found in artifacts.</div>
                )}
              </>
            ) : (
              <div className="text-neutral-600">Not computed: {String(net.reason || "no overlap / no artifact")}</div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-neutral-200 overflow-hidden">
          <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold flex items-center justify-between">
            <div>Loss Curve (Train vs Val)</div>
            <select className="border border-neutral-300 rounded-md px-2 py-1 text-xs" value={lossTrack} onChange={(e) => setLossTrack(e.target.value as any)}>
              <option value="load">load</option>
              <option value="solar">solar</option>
            </select>
          </div>
          <div className="h-[260px] p-2">
            {lossFiltered.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossFiltered}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis width={56} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="train_mse" name="train_mse" stroke="#ADD8E6" dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="val_mse" name="val_mse" stroke="#90EE90" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-sm text-neutral-600 p-3">No epoch telemetry yet (run a transformer training).</div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-neutral-200 overflow-hidden">
          <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">LLM Token Usage (TaskManager)</div>
          <div className="h-[260px] p-2">
            {llmUsage.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={llmUsage}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="gate" />
                  <YAxis width={56} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="input" name="input_tokens" fill="#ADD8E6" />
                  <Bar dataKey="output" name="output_tokens" fill="#90EE90" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-sm text-neutral-600 p-3">No LLM usage messages (enable `--use_llm` and submit a free-text command).</div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-neutral-200 overflow-hidden">
          <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">Hyperparameter Importance (Optuna)</div>
          <div className="h-[260px] p-2">
            {hpImportance.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hpImportance} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="param" type="category" width={90} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="importance" name="importance" fill="#ADD8E6" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-sm text-neutral-600 p-3">
                No importance yet (run baseline with <code>--hpo --n_trials &gt;= 2</code>).
              </div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-neutral-200 overflow-hidden">
          <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">Load Error vs Hour-of-Day (H=selected)</div>
          <div className="h-[260px] p-2">
            {errorByHour.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" type="number" domain={[0, 23]} ticks={[0, 6, 12, 18, 23]} />
                  <YAxis dataKey="abs_err" type="number" width={56} />
                  <Tooltip />
                  <Scatter data={errorByHour} fill="#2563eb" />
                </ScatterChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-sm text-neutral-600 p-3">No prediction series loaded yet (run a model, then Refresh charts).</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
