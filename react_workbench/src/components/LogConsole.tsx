import { PoolMessage } from "../types";

function topicType(topic: string): string {
  const i = topic.indexOf("/");
  return i >= 0 ? topic.slice(0, i) : topic;
}

export default function LogConsole(props: { messages: PoolMessage[] }) {
  const msgs = props.messages.slice(-250);
  return (
    <div className="bg-neutral-900 text-neutral-100 rounded-xl border border-neutral-800 overflow-hidden">
      <div className="px-3 py-2 text-xs font-semibold bg-neutral-950">Live Log Console</div>
      <div className="h-[420px] overflow-auto p-2 font-mono text-[11px] leading-4">
        {msgs.map((m) => (
          <div key={m.seq} className="py-0.5">
            <span className="text-neutral-400 mr-2">#{m.seq}</span>
            <span className="mr-2 px-1.5 py-0.5 rounded bg-neutral-800">{topicType(m.topic)}</span>
            <span className="mr-2 text-neutral-300">{m.sender}</span>
            <span className={m.level === "error" ? "text-red-300" : m.level === "warning" ? "text-yellow-300" : "text-green-300"}>
              {m.kind}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

