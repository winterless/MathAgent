"""
MathAgent minimal pipeline (JSONL between stages).

This is the **single entrypoint** for the project.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import Any, Dict, List

from core.prompt_assemble import assemble_stored_prompt
from core.stages import (
    append_choice_map_if_any,
    extract_boxed_counts_from_output,
    extract_choice_map,
    extract_final_answer,
    normalize_for_model,
    standardize_choice_answer,
)
from core.voting import majority_vote
from infra.llm_client import LLMClient
from dataio.jsonl_io import iter_jsonl, write_jsonl_atomic
from dataio.sample_schema import CANONICAL_KEYS, normalize_output_wrapper, normalize_record


def _default_out_dir() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join("datasets", "out", ts)


def _read_all(path: str) -> List[dict]:
    return list(iter_jsonl(path))


def main() -> None:
    p = argparse.ArgumentParser(description="MathAgent minimal pipeline (JSONL between stages).")
    p.add_argument("--input", required=True, help="Input JSONL path (one JSON object per line).")
    p.add_argument("--out", default=_default_out_dir(), help="Output directory (default datasets/out/<timestamp>)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM calls (rate limit)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    raw_input_copy_path = os.path.join(args.out, "example_input.jsonl")
    stage1_output_path = os.path.join(args.out, "stage1_output.jsonl")
    stage1_raw_generations_path = os.path.join(args.out, "stage1_raw_generations.jsonl")
    stage2_output_path = os.path.join(args.out, "stage2_output.jsonl")
    stage3_output_path = os.path.join(args.out, "stage3_output.jsonl")

    llm = LLMClient()

    try:
        shutil.copyfile(args.input, raw_input_copy_path)
    except Exception:
        pass

    input_rows = _read_all(args.input)
    normalized = [normalize_record(r) for r in input_rows]

    # ---- Stage 1: solve 8 times, archive raw, then evaluate once -> stage1_output.jsonl ----
    stage1_raw_archive_rows: List[Dict[str, Any]] = []
    stage1_output_rows: List[Dict[str, Any]] = []

    for r in normalized:
        uuid = r.get("uuid")
        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        if not q_raw.strip():
            raise ValueError(f"Missing question: uuid={uuid}")
        if not gold:
            raise ValueError(f"Missing gold in field 'answer': uuid={uuid}")

        model_q = append_choice_map_if_any(normalize_for_model(q_raw))
        raw_solutions = llm.generate_n(
            stage_name="stage1_solve",
            question=model_q,
            prompt_mode="problem",
            n=8,
            temperature=0.2,
            max_tokens=512,
            sleep_s=args.sleep,
        )

        extracted = [extract_final_answer(x) for x in raw_solutions]
        choice_map = extract_choice_map(model_q)
        standardized = [standardize_choice_answer(a, choice_map=choice_map) for a in extracted]
        vr = majority_vote(standardized)
        majority_answer = vr.majority

        stored_prompt = assemble_stored_prompt(question=q_raw, standard_answer="", stage1_answers=standardized[:8])

        stage1_raw_archive_rows.append(
            {
                "uuid": uuid,
                "line_number": r.get("line_number"),
                "stage": "stage1",
                "model_input": model_q,
                "raw_model_outputs": [str(x) for x in raw_solutions][:8],
                "extracted_answers": [str(x) for x in standardized][:8],
                "majority_answer": majority_answer,
            }
        )

        eval_user = f"{stored_prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
        eval_text = llm.generate_n(
            stage_name="stage1_eval",
            question=eval_user,
            prompt_mode="raw_prompt_eval",
            n=1,
            temperature=0.2,
            max_tokens=2048,
            sleep_s=args.sleep,
        )[0]
        out = normalize_output_wrapper(
            {
                "status": "SUCCESS",
                "content": {"choices": [{"indext": 0, "message": {"role": "assistant", "content": eval_text}}]},
            },
            uuid=uuid,
            stage="stage1",
        )

        clean = {k: r.get(k) for k in CANONICAL_KEYS}
        clean["prompt"] = stored_prompt
        clean["output"] = out
        stage1_output_rows.append(clean)

    write_jsonl_atomic(stage1_raw_generations_path, stage1_raw_archive_rows)
    write_jsonl_atomic(stage1_output_path, stage1_output_rows)

    # ---- Stage 2 (minimal): re-evaluate only rows missing boxed ----
    stage2_rows: List[Dict[str, Any]] = []
    for r in _read_all(stage1_output_path):
        uuid = r.get("uuid")
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        prompt = r.get("prompt") if isinstance(r.get("prompt"), str) else ""
        if extract_boxed_counts_from_output(r.get("output", {})) is not None:
            stage2_rows.append(r)
            continue
        eval_user = f"{prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
        eval_text = llm.generate_n(
            stage_name="stage2_eval",
            question=eval_user,
            prompt_mode="raw_prompt_eval",
            n=1,
            temperature=0.2,
            max_tokens=2048,
            sleep_s=args.sleep,
        )[0]
        out = normalize_output_wrapper(
            {
                "status": "SUCCESS",
                "content": {"choices": [{"indext": 0, "message": {"role": "assistant", "content": eval_text}}]},
            },
            uuid=uuid,
            stage="stage2",
        )
        clean = {k: r.get(k) for k in CANONICAL_KEYS}
        clean["output"] = out
        stage2_rows.append(clean)
    write_jsonl_atomic(stage2_output_path, stage2_rows)

    # ---- Stage 3 (minimal): re-evaluate only rows still missing boxed ----
    stage3_rows: List[Dict[str, Any]] = []
    for r in _read_all(stage2_output_path):
        uuid = r.get("uuid")
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        prompt = r.get("prompt") if isinstance(r.get("prompt"), str) else ""
        if extract_boxed_counts_from_output(r.get("output", {})) is not None:
            stage3_rows.append(r)
            continue
        eval_user = f"{prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
        eval_text = llm.generate_n(
            stage_name="stage3_eval",
            question=eval_user,
            prompt_mode="raw_prompt_eval",
            n=1,
            temperature=0.5,
            max_tokens=2048,
            sleep_s=args.sleep,
        )[0]
        out = normalize_output_wrapper(
            {
                "status": "SUCCESS",
                "content": {"choices": [{"indext": 0, "message": {"role": "assistant", "content": eval_text}}]},
            },
            uuid=uuid,
            stage="stage3",
        )
        clean = {k: r.get(k) for k in CANONICAL_KEYS}
        clean["output"] = out
        stage3_rows.append(clean)
    write_jsonl_atomic(stage3_output_path, stage3_rows)

    print("Done.")
    print(f"- out_dir: {args.out}")
    print(f"- example_input_copy: {raw_input_copy_path}")
    print(f"- stage1_raw_generations: {stage1_raw_generations_path}")
    print(f"- stage1_output: {stage1_output_path}")
    print(f"- stage2_output: {stage2_output_path}")
    print(f"- stage3_output: {stage3_output_path}")


if __name__ == "__main__":
    main()


