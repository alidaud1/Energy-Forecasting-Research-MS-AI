import React from "react";

export default function GateModal(props: {
  open: boolean;
  stage: string;
  gateId: string;
  prompt: string;
  onClose: () => void;
  onSend: (text: string) => void;
}) {
  const { open, stage, gateId, prompt, onClose, onSend } = props;
  const [text, setText] = React.useState("");

  React.useEffect(() => {
    if (open) setText("");
  }, [open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/30 flex items-center justify-center p-6">
      <div className="w-full max-w-3xl bg-white rounded-xl border border-neutral-200 shadow-xl overflow-hidden">
        <div className="px-4 py-3 bg-ieeeBlue flex items-center justify-between">
          <div className="font-semibold">Human Gate: {stage}</div>
          <button onClick={onClose} className="text-sm px-2 py-1 rounded-md bg-white/70 hover:bg-white">
            Close
          </button>
        </div>
        <div className="p-4">
          <div className="text-xs text-neutral-600 mb-2">
            gate_id: <span className="font-mono">{gateId}</span>
          </div>
          <pre className="text-xs bg-neutral-50 border border-neutral-200 rounded-lg p-3 overflow-auto max-h-72 whitespace-pre-wrap">
            {prompt}
          </pre>
          <div className="mt-3 flex gap-2">
            <input
              className="flex-1 border border-neutral-300 rounded-md px-3 py-2 text-sm"
              placeholder="Type approve / set ... / accept / scale ..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <button
              className="px-4 py-2 rounded-md bg-ieeeGreen hover:brightness-95 text-sm font-semibold"
              onClick={() => onSend((text || "").trim() || "approve")}
            >
              Send
            </button>
            <button className="px-4 py-2 rounded-md bg-neutral-900 text-white text-sm" onClick={() => onSend("approve")}>
              Approve
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

