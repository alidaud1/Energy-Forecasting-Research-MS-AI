import React from "react";
import { wsBase } from "../api";
import { PoolMessage, WsEnvelope } from "../types";

type Edge = {
  key: string;
  seq: number;
  from: string;
  to: string;
  label: string;
  kind: string;
  level: string;
  tooltip: string;
  dashed?: boolean;
  stroke?: string;
};

function topicType(topic: string): string {
  const i = topic.indexOf("/");
  return i >= 0 ? topic.slice(0, i) : topic;
}

function prettyRole(role: string): string {
  const r = (role || "").toLowerCase();
  if (r === "task_manager") return "TaskManager";
  if (r === "preparation") return "Preparation";
  if (r === "load_forecasting") return "LoadAgent";
  if (r === "solar_forecasting") return "SolarAgent";
  if (r === "model_developer") return "ModelDev";
  if (r === "forecast_integration") return "Integration";
  if (r === "human") return "Human";
  if (r === "system") return "System";
  return role || "Unknown";
}

function safeJson(x: any, limit = 1400): string {
  let s = "";
  try {
    s = JSON.stringify(x ?? {}, null, 2);
  } catch {
    s = String(x ?? "");
  }
  if (s.length > limit) return s.slice(0, limit) + "…";
  return s;
}

function buildRoleOrder(messages: PoolMessage[]): string[] {
  // Stable left-to-right ordering (lifelines).
  const preferred = [
    "human",
    "task_manager",
    "preparation",
    "load_forecasting",
    "solar_forecasting",
    "model_developer",
    "forecast_integration",
    "system"
  ];

  const seen = new Set<string>();
  for (const m of messages) {
    if (m.sender_role) seen.add(m.sender_role);
    if (Array.isArray((m as any).to_roles)) {
      for (const r of (m as any).to_roles) seen.add(String(r));
    }
  }

  const out: string[] = [];
  for (const r of preferred) {
    if (seen.has(r) || r === "human" || r === "system") out.push(r);
  }
  // include any other roles that appear
  for (const r of Array.from(seen).sort()) {
    if (!out.includes(r)) out.push(r);
  }
  return out;
}

function inferRecipients(
  msg: PoolMessage,
  opts: { roleOrder: string[]; directiveById: Map<string, PoolMessage> }
): string[] {
  const { roleOrder, directiveById } = opts;
  const tt = topicType(msg.topic);
  const payload = msg.payload || {};

  // Explicit routing if present.
  const anyMsg: any = msg as any;
  if (Array.isArray(anyMsg.to_roles) && anyMsg.to_roles.length > 0) {
    return anyMsg.to_roles.map((r: any) => String(r));
  }

  if (tt === "prepare") {
    if (msg.kind === "command") return ["preparation"];
    return ["task_manager"];
  }
  if (tt === "user_feedback") return ["task_manager"];
  if (tt === "gate") return ["human"];

  if (tt === "train" && msg.kind === "command") {
    const trk = String(payload.track || "");
    if (trk === "load") return ["load_forecasting"];
    if (trk === "solar") return ["solar_forecasting"];
    // Baseline mode commands effectively target model_developer.
    return ["model_developer"];
  }
  if (tt === "train_execute") return ["model_developer"];

  if (tt === "evaluate") return ["forecast_integration"];
  if (tt === "integrate") return ["human"];

  if (tt === "system_directive") {
    const targets = payload.target_roles;
    if (Array.isArray(targets)) {
      const t = targets.map((x: any) => String(x));
      if (t.includes("*")) {
        // Broadcast to forecasting agents by default (the only ones that currently apply training overrides).
        return roleOrder.filter((r) => r === "load_forecasting" || r === "solar_forecasting");
      }
      return t;
    }
    return roleOrder.filter((r) => r === "load_forecasting" || r === "solar_forecasting");
  }

  if (tt === "directive_ack" || tt === "directive_rejected") {
    const did = String(payload.directive_id || "");
    const origin = did ? directiveById.get(did) : undefined;
    if (origin?.sender_role) return [origin.sender_role];
    return ["task_manager"];
  }

  // Default: treat as broadcast-to-integration for traceability.
  if (tt === "state" || tt === "system" || tt === "directive" || tt === "human_gate_decision") return ["forecast_integration"];

  return [];
}

function labelFor(msg: PoolMessage): string {
  const tt = topicType(msg.topic);
  const p = msg.payload || {};
  const ev = String(p.event || "");
  const cmd = String(p.command || "");
  if (ev) return `${tt}:${ev}`;
  if (cmd) return `${tt}:${cmd}`;
  return `${tt}:${msg.kind}`;
}

function strokeFor(edge: Edge): string {
  if (edge.stroke) return edge.stroke;
  if (edge.kind === "error") return "#ef4444"; // red-500
  if (edge.label.startsWith("directive_ack")) return "#16a34a"; // green-600
  if (edge.label.startsWith("directive_rejected")) return "#ca8a04"; // yellow-600
  if (edge.kind === "result") return "#16a34a";
  if (edge.kind === "command") return "#111827"; // neutral-900
  return "#2563eb"; // blue-600 (events)
}

export default function AgentFlowDiagram(props: { runId: string }) {
  const { runId } = props;
  const [connected, setConnected] = React.useState(false);
  const [messages, setMessages] = React.useState<PoolMessage[]>([]);
  const [lastSeq, setLastSeq] = React.useState<number>(-1);
  const [reconnectNonce, setReconnectNonce] = React.useState(0);

  const wsRef = React.useRef<WebSocket | null>(null);
  const cursorRef = React.useRef<number>(-1);

  React.useEffect(() => {
    setMessages([]);
    const k = `agentflow:last_seq:${runId}`;
    const v = Number(localStorage.getItem(k) ?? "-1");
    const cur = Number.isFinite(v) ? v : -1;
    cursorRef.current = cur;
    setLastSeq(cur);
  }, [runId]);

  React.useEffect(() => {
    if (!runId) return;
    wsRef.current?.close();
    setConnected(false);

    const url = `${wsBase()}/ws?run_id=${encodeURIComponent(runId)}&last_seq=${encodeURIComponent(String(cursorRef.current))}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (ev) => {
      let env: WsEnvelope | null = null;
      try {
        env = JSON.parse(ev.data) as WsEnvelope;
      } catch {
        return;
      }
      if (!env || (env as any).type !== "message") return;
      const m = (env as any).data as PoolMessage;
      if (!m || m.run_id !== runId) return;
      setMessages((prev) => {
        // Keep it deterministic; de-dupe by seq.
        const seen = new Set<number>(prev.map((x) => x.seq));
        const next = seen.has(m.seq) ? prev : [...prev, m];
        next.sort((a, b) => a.seq - b.seq);
        return next.slice(-4000);
      });
      const ns = Math.max(Number(cursorRef.current ?? -1), Number(m.seq ?? -1));
      cursorRef.current = ns;
      setLastSeq(ns);
      const k = `agentflow:last_seq:${runId}`;
      localStorage.setItem(k, String(ns));
    };

    return () => ws.close();
  }, [runId, reconnectNonce]);

  const roleOrder = React.useMemo(() => buildRoleOrder(messages), [messages]);
  const roleX = React.useMemo(() => {
    const m = new Map<string, number>();
    roleOrder.forEach((r, i) => m.set(r, i));
    return m;
  }, [roleOrder]);

  const directiveById = React.useMemo(() => {
    const m = new Map<string, PoolMessage>();
    for (const msg of messages) {
      const mid = String((msg as any).msg_id || "");
      if (topicType(msg.topic) === "system_directive" && mid) m.set(mid, msg);
    }
    return m;
  }, [messages]);

  const edges = React.useMemo(() => {
    const out: Edge[] = [];
    for (const msg of messages) {
      const from = msg.sender_role || "unknown";
      const recips = inferRecipients(msg, { roleOrder, directiveById });
      if (!recips.length) continue;
      for (const to of recips) {
        const payload = msg.payload || {};
        const tt = topicType(msg.topic);
        const label = labelFor(msg);
        const tooltip =
          tt === "directive_ack" && payload?.applied_overrides
            ? safeJson(payload.applied_overrides)
            : safeJson({ topic: msg.topic, kind: msg.kind, sender: msg.sender, payload });
        const dashed = tt === "directive_ack" || tt === "directive_rejected";
        out.push({
          key: `${msg.seq}:${from}->${to}:${tt}`,
          seq: msg.seq,
          from,
          to,
          label,
          kind: msg.kind,
          level: msg.level,
          tooltip,
          dashed,
          stroke: undefined
        });
      }
    }

    // Integration connection: if postprocess exists and we have a final integrate result, add a highlight arrow.
    const lastPost = [...messages]
      .reverse()
      .find((m) => topicType(m.topic) === "directive" && (m.payload || {}).stage === "postprocess");
    const lastIntegrate = [...messages]
      .reverse()
      .find((m) => topicType(m.topic) === "integrate" && m.kind === "result");
    if (lastPost && lastIntegrate) {
      out.push({
        key: `postprocess_to_integrate:${lastIntegrate.seq}`,
        seq: lastIntegrate.seq,
        from: lastPost.sender_role || "task_manager",
        to: "forecast_integration",
        label: "postprocess->integrate",
        kind: "event",
        level: "info",
        tooltip: "Postprocess directive followed by integration completion.",
        dashed: true,
        stroke: "#90EE90"
      });
    }

    out.sort((a, b) => a.seq - b.seq);
    return out;
  }, [messages, roleOrder, directiveById]);

  const columnWidth = 170;
  const rowHeight = 22;
  const leftPad = 40;
  const topPad = 50;
  const w = leftPad + roleOrder.length * columnWidth + 60;
  const h = topPad + Math.max(1, edges.length) * rowHeight + 80;

  function xFor(role: string): number {
    const idx = roleX.get(role) ?? roleX.get("system") ?? 0;
    return leftPad + idx * columnWidth;
  }

  return (
    <div className="rounded-xl border border-neutral-200 overflow-hidden">
      <div className="px-3 py-2 bg-neutral-50 flex items-center justify-between">
        <div className="text-sm font-semibold">Agent Flow (Sequence Diagram)</div>
        <div className="text-xs text-neutral-600 flex items-center gap-3">
          <div>
            ws: <span className={connected ? "text-ieeeGreen font-semibold" : "text-red-600 font-semibold"}>{connected ? "connected" : "disconnected"}</span>
          </div>
          <div>last_seq={lastSeq}</div>
          <button
            className="px-2 py-1 rounded-md border border-neutral-300 bg-white hover:bg-neutral-50"
            onClick={() => {
              const k = `agentflow:last_seq:${runId}`;
              localStorage.setItem(k, "-1");
              setMessages([]);
              cursorRef.current = -1;
              setLastSeq(-1);
              setReconnectNonce((n) => n + 1);
            }}
          >
            Reset Cursor
          </button>
        </div>
      </div>

      <div className="bg-white" style={{ height: "70vh", overflow: "auto" }}>
        <svg width={w} height={h} role="img" aria-label="Agent Flow Sequence Diagram">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
            </marker>
          </defs>

          {/* Lifelines */}
          {roleOrder.map((r) => {
            const x = xFor(r);
            const isLoad = r === "load_forecasting";
            const isSolar = r === "solar_forecasting";
            const stroke = isLoad ? "#ADD8E6" : isSolar ? "#90EE90" : "#d4d4d8";
            return (
              <g key={r}>
                <rect x={x - 55} y={10} width={110} height={22} rx={6} fill="#ffffff" stroke="#e5e7eb" />
                <text x={x} y={26} textAnchor="middle" fontSize="11" fill="#111827">
                  {prettyRole(r)}
                </text>
                <line x1={x} y1={35} x2={x} y2={h - 20} stroke={stroke} strokeWidth={2} strokeDasharray="4 4" />
              </g>
            );
          })}

          {/* Message arrows */}
          {edges.map((e, idx) => {
            const y = topPad + idx * rowHeight;
            const x1 = xFor(e.from);
            const x2 = xFor(e.to);
            const dy = 0;
            const dir = x2 >= x1 ? 1 : -1;
            const pad = 12;
            const sx = x1 + dir * pad;
            const tx = x2 - dir * pad;
            const stroke = strokeFor(e);
            const labelX = (sx + tx) / 2;
            const labelY = y - 4;
            return (
              <g key={e.key}>
                <line
                  x1={sx}
                  y1={y + dy}
                  x2={tx}
                  y2={y + dy}
                  stroke={stroke}
                  strokeWidth={1.5}
                  strokeDasharray={e.dashed ? "4 3" : undefined}
                  markerEnd="url(#arrow)"
                  style={{ color: stroke }}
                >
                  <title>{e.tooltip}</title>
                </line>
                <text x={labelX} y={labelY} textAnchor="middle" fontSize="10" fill="#111827">
                  {e.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
