from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running as a script from repo root on Windows/Unix.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Export message_pool.jsonl to a readable appendix format.")
    p.add_argument("--log", type=str, default="logs/message_pool.jsonl")
    p.add_argument("--run_id", type=str, required=True, help="Run id, e.g. run_20260417_074056")
    p.add_argument("--out_dir", type=str, default="reports")
    p.add_argument("--csv", action="store_true", help="Also export CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("run_id") != args.run_id:
                continue

            payload = obj.get("payload", {}) or {}
            preview_keys = ["event", "command", "track", "model_name", "model_type"]
            preview = {k: payload.get(k) for k in preview_keys if k in payload}

            rows.append(
                {
                    "created_at": obj.get("created_at"),
                    "seq": obj.get("seq"),
                    "topic": obj.get("topic"),
                    "kind": obj.get("kind"),
                    "sender": obj.get("sender"),
                    "sender_role": obj.get("sender_role"),
                    "correlation_id": obj.get("correlation_id"),
                    "payload_preview": json.dumps(preview, ensure_ascii=False),
                }
            )

    md_path = out_dir / f"message_trace_{args.run_id}.md"
    lines = []
    lines.append(f"# Message Trace Appendix ({args.run_id})\n\n")
    lines.append("| time | seq | topic | kind | sender | role | correlation | payload preview |\n")
    lines.append("|---|---:|---|---|---|---|---|---|\n")
    for r in rows:
        lines.append(
            f"| {r['created_at']} | {r['seq']} | {r['topic']} | {r['kind']} | {r['sender']} | {r['sender_role']} | {r['correlation_id']} | {r['payload_preview']} |\n"
        )
    md_path.write_text("".join(lines), encoding="utf-8")
    print("Wrote:", md_path)

    if args.csv:
        csv_path = out_dir / f"message_trace_{args.run_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) if rows else ["created_at"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("Wrote:", csv_path)


if __name__ == "__main__":
    main()

