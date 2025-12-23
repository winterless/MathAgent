from __future__ import annotations

from typing import Any, Dict, List, Optional

JsonDict = Dict[str, Any]

# Canonical top-level keys we want to match (based on datasets/sample.jsonl).
CANONICAL_KEYS = [
    "source_category",
    "text",
    "uuid",
    "provider",
    "version",
    "xx_version",
    "aigc_modelname",
    "language",
    "raw_source_path",
    "prompt",
    "question",
    "answer",
    "line_number",
    "output",
]


def normalize_output_wrapper(output: Any, *, uuid: Any = None, stage: str = "stage1") -> JsonDict:
    """
    Normalize an "output" wrapper to match datasets/sample.jsonl's shape as closely as possible.
    Unknown fields are mocked with safe defaults.
    """
    # We align keys to sample.jsonl:
    # output: { status, content: { id, object, model, choices, usage, prefill_time } }
    existing = output if isinstance(output, dict) else {}
    existing_content = existing.get("content") if isinstance(existing.get("content"), dict) else {}

    # message.content (preserve if present)
    content_str = ""
    tool_calls = None
    if isinstance(existing_content, dict):
        choices = existing_content.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                if isinstance(msg.get("content"), str):
                    content_str = msg["content"]
                tool_calls = msg.get("tool_calls", None)

    # Build a strictly-shaped wrapper (drop unknown keys).
    out_choices: List[JsonDict] = []
    if isinstance(existing_content.get("choices"), list) and existing_content["choices"]:
        # Keep first choice, but normalize its shape.
        c0 = existing_content["choices"][0]
        if not isinstance(c0, dict):
            c0 = {}
        msg0 = c0.get("message") if isinstance(c0.get("message"), dict) else {}
        out_choices.append(
            {
                "indext": int(c0.get("indext", 0)) if str(c0.get("indext", "0")).isdigit() else 0,
                "message": {
                    "role": msg0.get("role", "assistant"),
                    "content": msg0.get("content", content_str),
                    "tool_calls": msg0.get("tool_calls", tool_calls),
                },
                "logprobs": c0.get("logprobs", None),
                "finish_reason": c0.get("finish_reason", "stop"),
            }
        )
    else:
        out_choices.append(
            {
                "indext": 0,
                "message": {"role": "assistant", "content": content_str, "tool_calls": tool_calls},
                "logprobs": None,
                "finish_reason": "stop",
            }
        )

    usage = existing_content.get("usage") if isinstance(existing_content.get("usage"), dict) else {}
    out_usage = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }

    prefill_time = existing_content.get("prefill_time", 0)
    if not isinstance(prefill_time, (int, float)):
        prefill_time = 0

    out_content: JsonDict = {
        "id": existing_content.get("id", f"endpoint_{stage}_{uuid}" if uuid is not None else f"endpoint_{stage}_unknown"),
        "object": existing_content.get("object", 0),
        "model": existing_content.get("model", ""),
        "choices": out_choices,
        "usage": out_usage,
        "prefill_time": prefill_time,
    }

    status = existing.get("status", "SUCCESS")
    if not isinstance(status, str) or not status:
        status = "SUCCESS"

    return {"status": status, "content": out_content}


def normalize_record(row: JsonDict) -> JsonDict:
    """
    Return a new dict that:
    - Contains exactly CANONICAL_KEYS (no extras)
    - Fills missing keys with ""/None defaults
    - Normalizes row["output"] to the sample-like wrapper
    """
    out: JsonDict = {}
    for k in CANONICAL_KEYS:
        out[k] = row.get(k, "" if k not in ("line_number", "output") else (None if k == "line_number" else {}))

    # Normalize output wrapper *shape* while preserving its content when present.
    # For raw inputs (no output yet), this becomes an empty wrapper.
    out["output"] = normalize_output_wrapper(row.get("output"), uuid=out.get("uuid"), stage="input")
    return out


