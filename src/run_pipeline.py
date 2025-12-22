from __future__ import annotations

import argparse
import os
import time
from typing import List

from jsonl_io import iter_jsonl, write_jsonl_atomic
from llm_client import LLMClient
from stages import (
    StageConfig,
    run_stage,
    stage1_decide_easy,
    stage2_split,
    stage3_split,
    to_final_rows,
)


def _default_out_dir() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join("runs", ts)


def _read_all(path: str) -> List[dict]:
    return list(iter_jsonl(path))


def main() -> None:
    p = argparse.ArgumentParser(description="MathAgent minimal multi-stage pipeline (JSONL in/out).")
    p.add_argument("--input", required=True, help="Input JSONL path. Each line: {id, question}")
    p.add_argument("--out", default=_default_out_dir(), help="Output directory (default runs/<timestamp>)")
    p.add_argument("--samples", type=int, default=8, help="Samples per stage (default 8)")
    p.add_argument("--n", type=int, default=7, help="Stability threshold n (stable if majority_count > n)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM calls (rate limit)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Paths (every transition is via JSONL files)
    stage1_path = os.path.join(args.out, "stage1.jsonl")
    stage2_input_path = os.path.join(args.out, "stage2_input.jsonl")
    easy_end_path = os.path.join(args.out, "easy_end.jsonl")

    stage2_path = os.path.join(args.out, "stage2.jsonl")
    stage3_input_path = os.path.join(args.out, "stage3_input.jsonl")
    final_stage2_path = os.path.join(args.out, "final_stage2.jsonl")

    stage3_path = os.path.join(args.out, "stage3.jsonl")
    discarded_path = os.path.join(args.out, "discarded.jsonl")
    final_stage3_path = os.path.join(args.out, "final_stage3.jsonl")

    final_path = os.path.join(args.out, "final.jsonl")

    llm = LLMClient()

    # ---- Stage 1 ----
    input_rows = _read_all(args.input)
    stage1_cfg = StageConfig(name="stage1", temperature=0.7, samples=args.samples, stable_threshold_n=args.n, sleep_s=args.sleep)
    stage1_out = run_stage(llm=llm, stage=stage1_cfg, rows=input_rows)
    write_jsonl_atomic(stage1_path, stage1_out)  # persist

    # Read stage1.jsonl to decide routing
    stage1_saved = _read_all(stage1_path)
    easy_end_rows, to_stage2_rows = stage1_decide_easy(stage1_saved, stage1_name="stage1")
    write_jsonl_atomic(easy_end_path, easy_end_rows)  # persist
    write_jsonl_atomic(stage2_input_path, to_stage2_rows)  # persist (transition file)

    # ---- Stage 2 ----
    stage2_in = _read_all(stage2_input_path)
    stage2_cfg = StageConfig(name="stage2", temperature=0.7, samples=args.samples, stable_threshold_n=args.n, sleep_s=args.sleep)
    stage2_out = run_stage(llm=llm, stage=stage2_cfg, rows=stage2_in)
    write_jsonl_atomic(stage2_path, stage2_out)  # persist

    stage2_saved = _read_all(stage2_path)
    stable2_rows, to_stage3_rows = stage2_split(stage2_saved, stage2_name="stage2")
    write_jsonl_atomic(final_stage2_path, to_final_rows(stable2_rows, majority_field="stage2_majority"))
    write_jsonl_atomic(stage3_input_path, to_stage3_rows)  # persist (transition file)

    # ---- Stage 3 ----
    stage3_in = _read_all(stage3_input_path)
    stage3_cfg = StageConfig(name="stage3", temperature=0.2, samples=args.samples, stable_threshold_n=args.n, sleep_s=args.sleep)
    stage3_out = run_stage(llm=llm, stage=stage3_cfg, rows=stage3_in)
    write_jsonl_atomic(stage3_path, stage3_out)  # persist

    stage3_saved = _read_all(stage3_path)
    stable3_rows, discarded_rows = stage3_split(stage3_saved, stage3_name="stage3")
    write_jsonl_atomic(final_stage3_path, to_final_rows(stable3_rows, majority_field="stage3_majority"))
    write_jsonl_atomic(discarded_path, discarded_rows)

    # ---- Merge finals (also via JSONL) ----
    final_all = _read_all(final_stage2_path) + _read_all(final_stage3_path)
    write_jsonl_atomic(final_path, final_all)

    print("Done.")
    print(f"- out_dir: {args.out}")
    print(f"- stage1: {stage1_path}")
    print(f"- stage2: {stage2_path}")
    print(f"- stage3: {stage3_path}")
    print(f"- final:  {final_path} (stage2+stage3 stable)")
    print(f"- easy_end: {easy_end_path}")
    print(f"- discarded: {discarded_path}")


if __name__ == "__main__":
    main()


