"""Create a plain-language supervisor summary from numeric joint results.

The Fusion and Decision Agents remain deterministic.  OpenAI is used only to
explain their already-computed JSON outputs.  The API key is loaded from
``fusion/.env`` (or ``OPENAI_API_KEY`` in the environment) and is never
written to an output file or printed.

Usage::

    agentic/.venv/bin/python -m fusion.openai_summary \
      --run joint_outputs/synthetic_benchmark

If no key is available, the command exits successfully with a clear message;
the numeric reports remain usable without the optional narrative summary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is optional for numeric runs
    load_dotenv = None


def _load_key(run_dir: Path) -> bool:
    env_file = Path(__file__).resolve().parent / ".env"
    if load_dotenv is not None:
        load_dotenv(env_file, override=False)
    return bool(os.environ.get("OPENAI_API_KEY"))


def _round_numbers(value):
    """Keep the supervisor-facing narrative at the thesis' two-decimal format."""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {k: _round_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_numbers(v) for v in value]
    return value


def generate_summary(run_dir: str | Path, *, model: str | None = None) -> dict:
    root = Path(run_dir).resolve()
    evaluation_path = root / "joint_evaluation.json"
    decision_path = root / "joint_run" / "decision_report.json"
    if not evaluation_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("joint_evaluation.json and joint_run/decision_report.json are required")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not _load_key(root):
        result = {"status": "skipped", "reason": "OPENAI_API_KEY is not available"}
        (root / "openai_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the OpenAI Python SDK in the active environment") from exc

    selected_model = model or os.environ.get("OPENAI_MODEL", "gpt-5")
    payload = {"evaluation": evaluation, "decision": decision}
    prompt = (
        "You are an energy forecasting research assistant. Summarize this "
        "controlled synthetic load-solar fusion experiment for an MS thesis "
        "supervisor using exactly 7 concise bullet points. Include: benchmark "
        "scope, forecast metrics, peak/ramp, reserve, risk/recommendation, "
        "solar context, and limitation. Do not use paragraphs or headings. "
        "Do not invent results, change numbers, or call synthetic data real. "
        "Report all numerical values to at most two decimal places. Mention "
        "that numerical decisions were deterministic and you only explain the "
        "supplied results.\n\n"
        + json.dumps(_round_numbers(payload), indent=2)
    )
    client = OpenAI()
    try:
        response = client.responses.create(model=selected_model, input=prompt, store=False)
    except Exception as exc:
        # Numeric artifacts must remain usable when the optional network call
        # is unavailable (offline machine, exhausted quota, or invalid key).
        result = {"status": "failed", "model": selected_model,
                  "reason": f"OpenAI request unavailable: {type(exc).__name__}"}
        (root / "openai_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    text = (response.output_text or "").strip()
    result = {"status": "generated", "model": selected_model, "summary": text}
    (root / "openai_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (root / "openai_summary.txt").write_text(text + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("joint_outputs/synthetic_benchmark"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    result = generate_summary(args.run, model=args.model)
    print(json.dumps({"status": result["status"], "model": result.get("model"),
                      "reason": result.get("reason")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
