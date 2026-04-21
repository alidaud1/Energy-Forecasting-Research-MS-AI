import React from "react";
import ReactMarkdown from "react-markdown";
import GateModal from "./components/GateModal";
import LogConsole from "./components/LogConsole";
import ForecastChart from "./components/ForecastChart";
import { getJson, postJson, staticUrl, wsBase } from "./api";
import { PredictionsResponse, PoolMessage, WsEnvelope } from "./types";

type RunsResponse = { runs: { run_id: string; status: string }[] };
type StartResp = { run_id: string };

function topicType(topic: string): string {
  const i = topic.indexOf("/");
  return i >= 0 ? topic.slice(0, i) : topic;
}

export default function App() {
  const [runs, setRuns] = React.useState<RunsResponse["runs"]>([]);
  const [runId, setRunId] = React.useState<string>("");
  const [status, setStatus] = React.useState<string>("");
  const [messages, setMessages] = React.useState<PoolMessage[]>([]);

  const [gateOpen, setGateOpen] = React.useState(false);
  const [gateStage, setGateStage] = React.useState("");
  const [gateId, setGateId] = React.useState("");
  const [gatePrompt, setGatePrompt] = React.useState("");

  const [loadRows, setLoadRows] = React.useState<any[]>([]);
  const [solarRows, setSolarRows] = React.useState<any[]>([]);
  const [reportMd, setReportMd] = React.useState<string>("");
  const [variant, setVariant] = React.useState<string>(""); // "" = best available
  const [horizon, setHorizon] = React.useState<number>(1);
  const [overlayVariant, setOverlayVariant] = React.useState<boolean>(false);
  const [overlayFamily, setOverlayFamily] = React.useState<boolean>(false);

  const [loadChartData, setLoadChartData] = React.useState<any[]>([]);
  const [solarChartData, setSolarChartData] = React.useState<any[]>([]);

  const wsRef = React.useRef<WebSocket | null>(null);

  async function refreshRuns() {
    const r = await getJson<RunsResponse>("/api/runs");
    setRuns(r.runs);
  }

  React.useEffect(() => {
    refreshRuns().catch(() => {});
    const t = setInterval(() => refreshRuns().catch(() => {}), 2000);
    return () => clearInterval(t);
  }, []);

  function connectWs(rid: string) {
    wsRef.current?.close();
    setMessages([]);
    const url = `${wsBase()}/ws?run_id=${encodeURIComponent(rid)}&last_seq=-1`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const env = JSON.parse(ev.data) as WsEnvelope;
        if (env.type === "message") {
          const m = env.data;
          setMessages((prev) => {
            const next = [...prev, m];
            next.sort((a, b) => a.seq - b.seq);
            return next.slice(-2000);
          });

          if (topicType(m.topic) === "gate") {
            const p = m.payload || {};
            if (p.event === "gate_requested") {
              setGateStage(String(p.stage || ""));
              setGateId(String(p.gate_id || ""));
              setGatePrompt(String(p.prompt || ""));
              setGateOpen(true);
            }
          }

          if (topicType(m.topic) === "integrate") {
            const p = m.payload || {};
            if (p.report_path) {
              fetch(staticUrl(String(p.report_path)))
                .then((r) => r.text())
                .then((t) => setReportMd(t))
                .catch(() => {});
            }
          }

          if (topicType(m.topic) === "evaluate") {
            const p = m.payload || {};
            if (p.event === "evaluation_complete") {
              const track = String(p.track || "");
              if (track === "load") {
                const v = variant ? `&variant=${encodeURIComponent(variant)}` : "";
                getJson<PredictionsResponse>(`/api/runs/${rid}/predictions?track=load&horizon=${horizon}&max_points=2000${v}`)
                  .then((r) => setLoadRows(r.rows))
                  .catch(() => {});
              }
              if (track === "solar") {
                const v = variant ? `&variant=${encodeURIComponent(variant)}` : "";
                getJson<PredictionsResponse>(`/api/runs/${rid}/predictions?track=solar&horizon=${horizon}&max_points=2000${v}`)
                  .then((r) => setSolarRows(r.rows))
                  .catch(() => {});
              }
            }
          }
        }
      } catch {
        // ignore
      }
    };
  }

  async function startRun() {
    const r = await postJson<StartResp>("/api/runs/start", {
      manifest: "data/processed/dataset_manifest.json",
      W: 168,
      H: 24,
      stride: 24,
      with_baselines: true,
      epochs: 3,
      batch_size: 128,
      lr: 0.001,
      patience: 2,
      load_model: "patchtst",
      solar_model: "patchtst",
      use_llm: false
    });
    setRunId(r.run_id);
    setStatus("running");
    connectWs(r.run_id);
  }

  function mergeForChart(primary?: PredictionsResponse, secondary?: PredictionsResponse, third?: PredictionsResponse) {
    const idx = new Map<string, any>();
    const add = (r: any, prefix: string) => {
      const t = String(r.timestamp);
      const row = idx.get(t) || { t };
      row[`${prefix}_true`] = r.y_true;
      row[`${prefix}_pred`] = r.y_pred;
      idx.set(t, row);
    };
    if (primary) primary.rows.forEach((r) => add(r, "a"));
    if (secondary) secondary.rows.forEach((r) => add(r, "b"));
    if (third) third.rows.forEach((r) => add(r, "c"));
    const out = Array.from(idx.values());
    out.sort((x, y) => (x.t < y.t ? -1 : x.t > y.t ? 1 : 0));
    // Choose one "actual" reference if present
    for (const r of out) {
      r.y_true = r.a_true ?? r.b_true ?? r.c_true;
      r.y_pred_a = r.a_pred;
      r.y_pred_b = r.b_pred;
      r.y_pred_c = r.c_pred;
    }
    return out;
  }

  async function refreshCharts(rid: string) {
    const v = variant ? `&variant=${encodeURIComponent(variant)}` : "";
    const baseLoad = await getJson<PredictionsResponse>(`/api/runs/${rid}/predictions?track=load&horizon=${horizon}&max_points=2000${v}`);
    const baseSolar = await getJson<PredictionsResponse>(`/api/runs/${rid}/predictions?track=solar&horizon=${horizon}&max_points=2000${v}`);

    let load2: PredictionsResponse | undefined;
    let solar2: PredictionsResponse | undefined;
    let load3: PredictionsResponse | undefined;
    let solar3: PredictionsResponse | undefined;

    if (overlayVariant) {
      const other = variant === "human_adjusted" ? "proposed" : "human_adjusted";
      load2 = await getJson<PredictionsResponse>(
        `/api/runs/${rid}/predictions?track=load&variant=${encodeURIComponent(other)}&horizon=${horizon}&max_points=2000`
      ).catch(() => undefined as any);
      solar2 = await getJson<PredictionsResponse>(
        `/api/runs/${rid}/predictions?track=solar&variant=${encodeURIComponent(other)}&horizon=${horizon}&max_points=2000`
      ).catch(() => undefined as any);
    }

    if (overlayFamily) {
      // Overlay baseline vs transformer for the selected variant (or best available when variant is empty).
      load3 = await getJson<PredictionsResponse>(
        `/api/runs/${rid}/predictions?track=load&horizon=${horizon}&max_points=2000${v}&model_family=baseline`
      ).catch(() => undefined as any);
      solar3 = await getJson<PredictionsResponse>(
        `/api/runs/${rid}/predictions?track=solar&horizon=${horizon}&max_points=2000${v}&model_family=baseline`
      ).catch(() => undefined as any);
    }

    setLoadChartData(mergeForChart(baseLoad, load2, load3));
    setSolarChartData(mergeForChart(baseSolar, solar2, solar3));
  }

  function sendFeedback(text: string, gid?: string) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "feedback", text, gate_id: gid || gateId }));
  }

  return (
    <div className="min-h-screen">
      <header className="px-4 py-3 bg-ieeeBlue border-b border-neutral-200 flex items-center justify-between">
        <div className="font-bold">Unified Agent Workbench</div>
        <div className="flex items-center gap-2">
          <button onClick={() => refreshRuns()} className="text-sm px-3 py-1.5 rounded-md bg-white/70 hover:bg-white">
            Refresh
          </button>
          <button onClick={() => startRun()} className="text-sm px-3 py-1.5 rounded-md bg-ieeeGreen hover:brightness-95 font-semibold">
            Start Run
          </button>
        </div>
      </header>

      <GateModal
        open={gateOpen}
        stage={gateStage}
        gateId={gateId}
        prompt={gatePrompt}
        onClose={() => setGateOpen(false)}
        onSend={(t) => sendFeedback(t, gateId)}
      />

      <main className="p-4 grid grid-cols-12 gap-4">
        <section className="col-span-12 lg:col-span-3 space-y-3">
          <div className="rounded-xl border border-neutral-200 overflow-hidden">
            <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">Runs</div>
            <div className="p-3 space-y-2">
              <select
                className="w-full border border-neutral-300 rounded-md px-2 py-2 text-sm"
                value={runId}
                onChange={(e) => {
                  const v = e.target.value;
                  setRunId(v);
                  if (v) connectWs(v);
                }}
              >
                <option value="">Select run_id</option>
                {runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} ({r.status})
                  </option>
                ))}
              </select>
              <div className="text-xs text-neutral-600">status: {status || (runs.find((r) => r.run_id === runId)?.status ?? "")}</div>
            </div>
          </div>

          <LogConsole messages={messages.filter((m) => m.run_id === runId)} />
        </section>

        <section className="col-span-12 lg:col-span-9 space-y-4">
          <div className="rounded-xl border border-neutral-200 overflow-hidden">
            <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">View Controls</div>
            <div className="p-3 flex flex-wrap gap-3 items-end">
              <div>
                <div className="text-xs text-neutral-600 mb-1">variant</div>
                <select className="border border-neutral-300 rounded-md px-2 py-2 text-sm" value={variant} onChange={(e) => setVariant(e.target.value)}>
                  <option value="">best available</option>
                  <option value="proposed">proposed</option>
                  <option value="human_adjusted">human_adjusted</option>
                </select>
              </div>
              <div>
                <div className="text-xs text-neutral-600 mb-1">horizon</div>
                <select
                  className="border border-neutral-300 rounded-md px-2 py-2 text-sm"
                  value={horizon}
                  onChange={(e) => setHorizon(Number(e.target.value))}
                >
                  {Array.from({ length: 24 }).map((_, i) => (
                    <option key={i + 1} value={i + 1}>
                      {i + 1}
                    </option>
                  ))}
                </select>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={overlayVariant} onChange={(e) => setOverlayVariant(e.target.checked)} />
                Overlay variants
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={overlayFamily} onChange={(e) => setOverlayFamily(e.target.checked)} />
                Overlay baseline
              </label>
              <button
                className="px-4 py-2 rounded-md bg-neutral-900 text-white text-sm"
                disabled={!runId}
                onClick={() => {
                  if (!runId) return;
                  refreshCharts(runId).catch(() => {});
                }}
              >
                Refresh charts
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <ForecastChart
              title={`Load (horizon=${horizon}${variant ? `, ${variant}` : ""})`}
              data={loadChartData}
              lines={[
                { key: "y_true", label: "Actual", color: "#111827", width: 1.5 },
                { key: "y_pred_a", label: "Pred (primary)", color: "#2563eb", width: 2.5 },
                ...(overlayVariant ? [{ key: "y_pred_b", label: "Pred (other variant)", color: "#6b7280", dashed: true, width: 2 }] : []),
                ...(overlayFamily ? [{ key: "y_pred_c", label: "Pred (baseline)", color: "#10b981", dashed: true, width: 2 }] : [])
              ]}
            />
            <ForecastChart
              title={`Solar (horizon=${horizon}${variant ? `, ${variant}` : ""})`}
              data={solarChartData}
              lines={[
                { key: "y_true", label: "Actual", color: "#111827", width: 1.5 },
                { key: "y_pred_a", label: "Pred (primary)", color: "#2563eb", width: 2.5 },
                ...(overlayVariant ? [{ key: "y_pred_b", label: "Pred (other variant)", color: "#6b7280", dashed: true, width: 2 }] : []),
                ...(overlayFamily ? [{ key: "y_pred_c", label: "Pred (baseline)", color: "#10b981", dashed: true, width: 2 }] : [])
              ]}
            />
          </div>

          <div className="rounded-xl border border-neutral-200 overflow-hidden">
            <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">Report</div>
            <div className="p-3 prose prose-sm max-w-none">
              {reportMd ? <ReactMarkdown>{reportMd}</ReactMarkdown> : <div className="text-neutral-600 text-sm">No report yet.</div>}
            </div>
          </div>

          <div className="rounded-xl border border-neutral-200 overflow-hidden">
            <div className="px-3 py-2 bg-neutral-50 text-sm font-semibold">Chat / Feedback</div>
            <ChatBox onSend={(t) => sendFeedback(t)} />
          </div>
        </section>
      </main>
    </div>
  );
}

function ChatBox(props: { onSend: (text: string) => void }) {
  const [text, setText] = React.useState("");
  return (
    <div className="p-3 flex gap-2">
      <input
        className="flex-1 border border-neutral-300 rounded-md px-3 py-2 text-sm"
        placeholder="Type a command (approve / set epochs=5 / scale solar factor=0.9)"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        className="px-4 py-2 rounded-md bg-neutral-900 text-white text-sm"
        onClick={() => {
          const t = text.trim();
          if (!t) return;
          props.onSend(t);
          setText("");
        }}
      >
        Send
      </button>
    </div>
  );
}
