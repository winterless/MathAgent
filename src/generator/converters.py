from __future__ import annotations

import os
from typing import Any, Dict, List

from core.stages import append_choice_map_if_any, normalize_for_model
from dataio.jsonl_io import append_jsonl_line, iter_jsonl


def convert_stage1_output_to_stage1_infer(
    *,
    stage1_output_path: str,
    stage1_infer_path: str,
    min_votes_to_accept: int,
) -> int:
    """
    Convert stage1_output.stage1.jsonl -> stage1_infer.stage1.jsonl.

    Motivation:
    - In the "generic data generator" world, we treat artifacts as transformable.
    - Some upstream pipelines only provide stage1_output (judge-like) artifacts.
    - For downstream tooling consistency, we still want a stage1_infer artifact.

    Notes:
    - raw_model_outputs / extracted_answers can be empty (as requested).
    - This is idempotent: will skip uuids already present in stage1_infer_path.
    """
    os.makedirs(os.path.dirname(stage1_infer_path) or ".", exist_ok=True)

    done: set[str] = set()
    if os.path.exists(stage1_infer_path):
        for rr in iter_jsonl(stage1_infer_path, tolerate_errors=True):
            u = rr.get("uuid")
            if u is not None:
                done.add(str(u))

    written = 0
    for r in iter_jsonl(stage1_output_path, tolerate_errors=True):
        uuid = r.get("uuid")
        if uuid is None:
            continue
        uuid_key = str(uuid)
        if uuid_key in done:
            continue

        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        model_q = (
            r.get("model_input")
            if isinstance(r.get("model_input"), str) and str(r.get("model_input")).strip()
            else append_choice_map_if_any(normalize_for_model(q_raw))
        )

        append_jsonl_line(
            stage1_infer_path,
            {
                "uuid": uuid,
                "line_number": r.get("line_number"),
                "stage": "stage1_infer",
                "question": q_raw,
                "answer": gold,
                "gold": gold,
                "model_input": model_q,
                "model_prompt_system": "",
                "model_prompt_user": "",
                # requested: raw can be empty
                "raw_model_outputs": [],
                "extracted_answers": [],
                "min_votes_to_accept": int(min_votes_to_accept),
                "raw_source_path": r.get("raw_source_path"),
            },
        )
        done.add(uuid_key)
        written += 1

    return int(written)

