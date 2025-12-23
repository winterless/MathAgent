from __future__ import annotations

import argparse
import os
import time
import shutil
from typing import List

from jsonl_io import iter_jsonl, write_jsonl_atomic
from llm_client import LLMClient
from sample_schema import CANONICAL_KEYS, normalize_record, normalize_output_wrapper
from prompt_assemble import assemble_stored_prompt
from stages import (
    StageConfig,
    run_stage,
    output_is_stable,
    stage2_split,
    stage3_split,
    to_final_rows,
)


def _default_out_dir() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    # Keep all data artifacts under datasets/
    return os.path.join("datasets", "out", ts)


def _read_all(path: str) -> List[dict]:
    return list(iter_jsonl(path))


def main() -> None:
    p = argparse.ArgumentParser(description="MathAgent stage2+stage3 pipeline (JSONL between stages).")
    p.add_argument("--input", required=True, help="Input JSONL path (one JSON object per line).")
    p.add_argument("--out", default=_default_out_dir(), help="Output directory (default datasets/out/<timestamp>)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM calls (rate limit)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Paths (every transition is via JSONL files)
    raw_input_copy_path = os.path.join(args.out, "example_input.jsonl")
    stage1_output_path = os.path.join(args.out, "stage1_output.jsonl")
    # Stage1 "solve" artifacts (8x generation + assembled prompt) are useful for debugging,
    # but user expects stage1_output.jsonl to already look like the evaluator output.
    stage1_solve_output_path = os.path.join(args.out, "stage1_solve_output.jsonl")
    stage1_raw_generations_path = os.path.join(args.out, "stage1_raw_generations.jsonl")
    stage2_output_path = os.path.join(args.out, "stage2_output.jsonl")
    stage2_internal_path = os.path.join(args.out, "stage2_internal.jsonl")
    stage3_output_path = os.path.join(args.out, "stage3_output.jsonl")
    stage3_internal_path = os.path.join(args.out, "stage3_internal.jsonl")
    final_stage2_path = os.path.join(args.out, "final_stage2.jsonl")
    final_stage3_path = os.path.join(args.out, "final_stage3.jsonl")
    discarded_path = os.path.join(args.out, "discarded.jsonl")
    final_path = os.path.join(args.out, "final.jsonl")  # stage2+stage3 stable merged

    llm = LLMClient()

    # Treat --input as RAW problem input (no fake stage1/stage2 text).
    # Stage1: call model 8 times to generate 8 answers, then build the evaluator prompt.
    # Also copy the raw input file into the out dir for easy inspection/debugging.
    try:
        shutil.copyfile(args.input, raw_input_copy_path)
    except Exception:
        # Best-effort copy; pipeline should still run.
        pass
    input_rows = _read_all(args.input)
    # Normalize to match datasets/sample.jsonl schema (structure/fields).
    # Here, `answer` is treated as the gold/standard answer (needed by evaluation prompt).
    normalized = [normalize_record(r) for r in input_rows]

    # ---- Stage 1 (generate 8 raw answers) ----
    stage1_rows = []
    for r in normalized:
        q = r.get("question") or r.get("prompt") or ""
        gold = (r.get("answer") or "").strip()
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Missing question for stage1: uuid={r.get('uuid')}")
        if not isinstance(gold, str) or not gold:
            raise ValueError(f"Missing gold/standard answer in field 'answer' for uuid={r.get('uuid')}")
        stage1_rows.append(
            {
                **r,
                "text": "问题",
                "prompt": "",  # will be assembled after we have 8 answers
                "output": normalize_output_wrapper({}, uuid=r.get("uuid"), stage="stage1"),
            }
        )

    # Stage1 only needs a short final answer (A/B/C/D or a short value).
    stage1_cfg = StageConfig(
        name="stage1",
        # Keep stage1 stable: we want standardized final answer tokens.
        temperature=0.2,
        samples=8,
        stable_threshold_n=7,
        sleep_s=args.sleep,
        kind="solve",
        # Allow capturing full raw model output (including long reasoning) for archiving.
        max_tokens=512,
    )
    stage1_out = run_stage(llm=llm, stage=stage1_cfg, rows=stage1_rows)

    # Assemble stored prompt (sample.jsonl style) using stage1 candidates.
    # Also archive raw model generations for stage1 (for debugging).
    stage1_raw_archive_rows = []
    stage1_final_rows = []
    for r in stage1_out:
        debug = (((r.get("output") or {}).get("content") or {}).get("debug") or {})
        answers8 = debug.get("candidates") if isinstance(debug, dict) else None
        raw8 = debug.get("raw_candidates") if isinstance(debug, dict) else None
        model_input = debug.get("model_input") if isinstance(debug, dict) else None
        if not isinstance(raw8, list):
            raw8 = []
        if not isinstance(answers8, list):
            answers8 = []
        q = r.get("question") if isinstance(r.get("question"), str) else ""
        # To match sample.jsonl, we keep [标准解答] empty in stored prompt by default.
        prompt = assemble_stored_prompt(question=q, standard_answer="", stage1_answers=[str(x) for x in answers8][:8])
        majority_answer = ""
        out_obj = r.get("output")
        if isinstance(out_obj, dict):
            content = out_obj.get("content")
            if isinstance(content, dict):
                choices = content.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        majority_answer = msg["content"]

        stage1_raw_archive_rows.append(
            {
                "uuid": r.get("uuid"),
                "line_number": r.get("line_number"),
                "stage": "stage1",
                # Exact input used for model call (normalized newlines + option map).
                "model_input": model_input if isinstance(model_input, str) else "",
                # Full raw model outputs (8 calls).
                "raw_model_outputs": [str(x) for x in raw8][:8],
                # The extracted/standardized outputs we used as 解答1..8.
                "extracted_answers": [str(x) for x in answers8][:8],
                # Majority after voting (what stage1 output wrapper stores in output.content.choices[0].message.content).
                "majority_answer": majority_answer,
            }
        )
        # Stage1 output file should match sample.jsonl schema:
        # - only CANONICAL_KEYS at top-level
        # - output wrapper is strictly-shaped (no debug/stage/etc)
        clean = {k: r.get(k) for k in CANONICAL_KEYS}
        clean["prompt"] = prompt
        clean["output"] = normalize_output_wrapper(r.get("output"), uuid=r.get("uuid"), stage="stage1")
        stage1_final_rows.append(clean)

    write_jsonl_atomic(stage1_raw_generations_path, stage1_raw_archive_rows)
    # Persist stage1 "solve" results (before evaluation) for debugging.
    write_jsonl_atomic(stage1_solve_output_path, stage1_final_rows)

    # ---- Stage 2 (evaluate the assembled prompt) ----
    # NOTE: user expects stage1_output.jsonl to look like evaluator output (sample.jsonl-like content),
    # so we always run stage2 evaluation for all rows, and write that into stage1_output.jsonl.
    stage2_in = _read_all(stage1_solve_output_path)
    stage2_candidates = [{**r, "output_stage1": r.get("output")} for r in stage2_in]

    # Stage2 evaluation should output a boxed summary (can be concise).
    stage2_cfg = StageConfig(
        name="stage2",
        temperature=0.2,
        samples=8,
        stable_threshold_n=7,
        sleep_s=args.sleep,
        kind="eval",
        max_tokens=1024,
    )
    stage2_ran = run_stage(llm=llm, stage=stage2_cfg, rows=stage2_candidates)
    # Internal stage2 artifacts (keep debug for routing/stability decisions).
    write_jsonl_atomic(stage2_internal_path, stage2_ran)

    # Public stage2 results: strictly sample.jsonl-aligned wrapper, no debug/candidates.
    stage2_out_all: List[dict] = []
    for r in stage2_ran:
        clean = {k: r.get(k) for k in CANONICAL_KEYS}
        clean["output"] = normalize_output_wrapper(r.get("output"), uuid=r.get("uuid"), stage="stage2")
        stage2_out_all.append(clean)
    # Persist both:
    # - stage1_output.jsonl: evaluator-style output (what user expects)
    # - stage2_output.jsonl: kept for backward compatibility/debugging
    write_jsonl_atomic(stage1_output_path, stage2_out_all)
    write_jsonl_atomic(stage2_output_path, stage2_out_all)

    # Routing uses the internal file which retains debug.stable.
    stage2_saved = _read_all(stage2_internal_path)
    stable2_rows, to_stage3_rows = stage2_split(stage2_saved, stage2_name="stage2")
    write_jsonl_atomic(final_stage2_path, to_final_rows(stable2_rows))

    # ---- Stage 3 (stronger evaluator) ----
    # Stage3 reads from the persisted stage2 output, and only processes rows that stage2 deemed unstable.
    stage3_in = to_stage3_rows
    # Keep a backup of stage2 output before overwriting `output` with stage3.
    stage3_candidates = []
    for r in stage3_in:
        stage3_candidates.append({**r, "output_stage2": r.get("output")})
    stage3_cfg = StageConfig(
        name="stage3",
        temperature=0.2,
        samples=8,
        stable_threshold_n=7,
        sleep_s=args.sleep,
        kind="eval",
        max_tokens=1024,
    )
    stage3_out = run_stage(llm=llm, stage=stage3_cfg, rows=stage3_candidates)
    # Internal stage3 artifacts (keep debug for routing/stability decisions).
    write_jsonl_atomic(stage3_internal_path, stage3_out)

    # Public stage3 results: strictly sample.jsonl-aligned wrapper, no debug/candidates.
    stage3_out_all: List[dict] = []
    for r in stage3_out:
        clean = {k: r.get(k) for k in CANONICAL_KEYS}
        clean["output"] = normalize_output_wrapper(r.get("output"), uuid=r.get("uuid"), stage="stage3")
        stage3_out_all.append(clean)
    write_jsonl_atomic(stage3_output_path, stage3_out_all)  # persist

    # Routing uses the internal file which retains debug.stable.
    stage3_saved = _read_all(stage3_internal_path)
    stable3_rows, discarded_rows = stage3_split(stage3_saved, stage3_name="stage3")
    write_jsonl_atomic(final_stage3_path, to_final_rows(stable3_rows))
    write_jsonl_atomic(discarded_path, discarded_rows)

    # ---- Merge finals (also via JSONL) ----
    # For experiments, ensure every input row has a final record:
    # - stable stage2 + stable stage3
    # - plus discarded (stage3 still unstable) so final.jsonl is never empty.
    final_all = _read_all(final_stage2_path) + _read_all(final_stage3_path)
    final_all += to_final_rows(discarded_rows)
    write_jsonl_atomic(final_path, final_all)

    print("Done.")
    print(f"- out_dir: {args.out}")
    print(f"- example_input_copy: {raw_input_copy_path}")
    print(f"- stage1_solve_output: {stage1_solve_output_path}")
    print(f"- stage1_output: {stage1_output_path} (evaluator-style, sample.jsonl-like content)")
    print(f"- stage2_output: {stage2_output_path}")
    print(f"- stage2_internal: {stage2_internal_path} (keeps debug for routing)")
    print(f"- stage3_output: {stage3_output_path}")
    print(f"- stage3_internal: {stage3_internal_path} (keeps debug for routing)")
    print(f"- final:  {final_path} (stage2+stage3 stable)")
    print(f"- discarded: {discarded_path}")


if __name__ == "__main__":
    main()


