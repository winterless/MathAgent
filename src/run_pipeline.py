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
    extract_boxed_counts,
    extract_boxed_counts_from_output,
    extract_boxed_answer,
    extract_choice_map,
    extract_final_answer,
    normalize_for_model,
    rule_equivalent,
    strip_think,
    standardize_choice_answer,
)
from core.voting import majority_vote
from infra.llm_router import LLMRouter
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
    p.add_argument(
        "--llm-config",
        default="config/llm_models.json",
        help="Path to JSON config describing models + stage routing.",
    )
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    raw_input_copy_path = os.path.join(args.out, "example_input.jsonl")
    stage1_output_path = os.path.join(args.out, "stage1_output.jsonl")
    stage1_raw_generations_path = os.path.join(args.out, "stage1_raw_generations.jsonl")
    stage2_archive_path = os.path.join(args.out, "stage2_archive.jsonl")
    stage3_archive_path = os.path.join(args.out, "stage3_archive.jsonl")
    accepted_bank_path = os.path.join(args.out, "accepted_bank.jsonl")
    discarded_hard_path = os.path.join(args.out, "discarded_hard.jsonl")

    llm = LLMRouter(config_path=args.llm_config)
    min_ok_to_accept = llm.threshold_int("min_ok_to_accept", 5)

    # ---- Shared judge helper (Stage1/2/3) ----
    def _llm_judge_equivalence(*, uuid: Any, question: str, gold: str, pred: str, choice_map: Dict[str, str], stage: str) -> bool | None:
        """
        Ask LLM to judge equivalence only when rules cannot decide.
        Return True/False/None.
        """
        choice_lines = []
        for k in ("A", "B", "C", "D"):
            if k in choice_map:
                choice_lines.append(f"{k} = {choice_map[k]}")
        choice_block = "\n".join(choice_lines)
        user = (
            "你是“答案一致性判定器”，只判断 pred 是否与 gold 含义一致，禁止解题。\n"
            "输出必须是且只能是三者之一：一致 / 不一致 / 不确定\n\n"
            f"[题目]\n{question}\n\n"
            f"[选项映射]\n{choice_block}\n\n"
            f"[gold]\n{gold}\n\n"
            f"[pred]\n{pred}\n"
        ).strip()
        resp = llm.generate_n(
            stage_name=f"{stage}_judge",
            question=user,
            prompt_mode="raw_prompt",
            sleep_s=args.sleep,
        )[0]
        t = (resp or "").strip()
        # Be strict: accept only the 3 allowed outputs (optionally with trivial punctuation).
        first = (t.splitlines()[0] if t else "").strip().strip("。.!！?？")
        if first == "不确定":
            return None
        if first == "不一致":
            return False
        if first == "一致":
            return True
        return None

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
            sleep_s=args.sleep,
        )

        extracted = [extract_final_answer(x) for x in raw_solutions]
        choice_map = extract_choice_map(model_q)
        standardized = [standardize_choice_answer(a, choice_map=choice_map) for a in extracted]
        vr = majority_vote(standardized)
        majority_answer = vr.majority

        # Stage1 uses the same extractor + rule-first judge (LLM fallback) as Stage2/Stage3.
        stage1_attempts: List[Dict[str, Any]] = []
        ok1 = 0
        for raw, extracted_i, pred_i in zip(raw_solutions[:8], extracted[:8], standardized[:8]):
            boxed_i = extract_boxed_answer(raw)
            extracted_final = boxed_i or extracted_i
            pred_final = standardize_choice_answer(extracted_final, choice_map=choice_map)
            eq = rule_equivalent(pred_final, gold, choice_map=choice_map)
            judge_src = "rules"
            if eq is None:
                eq = _llm_judge_equivalence(
                    uuid=uuid,
                    question=q_raw,
                    gold=gold,
                    pred=pred_final,
                    choice_map=choice_map,
                    stage="stage1",
                )
                judge_src = "llm" if eq is not None else "unknown"
            verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
            if eq is True:
                ok1 += 1
            stage1_attempts.append(
                {
                    "raw_text": str(raw),
                    "boxed_answer": boxed_i,
                    "extracted_answer": extracted_final,
                    "normalized_answer": pred_final,
                    "verdict": verdict,
                    "judge_source": judge_src,
                }
            )

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
                "attempts": stage1_attempts,
                "ok": ok1,
                "bad": 8 - ok1,
                "llm_call_counts": {
                    # counts of requests (not tokens)
                    "stage1_solve": llm.stage_params("stage1_solve").n,
                    "stage1_eval": 0,  # filled after eval (may retry once)
                    "stage1_judge_llm_fallback": sum(1 for a in stage1_attempts if a.get("judge_source") == "llm"),
                    "stage1_judge_unknown": sum(1 for a in stage1_attempts if a.get("judge_source") == "unknown"),
                },
            }
        )

        eval_user = f"{stored_prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
        # Evaluate once; if boxed counts missing, retry once immediately so routing/metrics are consistent.
        eval_calls = 0
        eval_text = ""
        for _ in range(2):
            eval_calls += 1
            eval_text = llm.generate_n(
                stage_name="stage1_eval",
                question=eval_user,
                prompt_mode="raw_prompt_eval",
                sleep_s=args.sleep,
            )[0]
            eval_text = strip_think(eval_text)
            if extract_boxed_counts(eval_text) is not None:
                break
        stage1_raw_archive_rows[-1]["llm_call_counts"]["stage1_eval"] = eval_calls
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

    # ---- Stage 2/3 (per Architecture.md) ----

    stage2_archive: List[Dict[str, Any]] = []
    stage3_archive: List[Dict[str, Any]] = []
    accepted_bank: List[Dict[str, Any]] = []
    discarded_hard: List[Dict[str, Any]] = []

    stage1_rows = _read_all(stage1_output_path)

    # Route hard problems by Stage1 eval boxed counts (retry once if missing).
    hard_rows: List[Dict[str, Any]] = []
    for r in stage1_rows:
        uuid = r.get("uuid")
        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""

        counts = extract_boxed_counts_from_output(r.get("output", {}))

        if counts is None:
            continue
        ok, bad = counts
        difficulty = bad
        r["_stage1_ok"] = ok
        r["_stage1_bad"] = bad
        r["_difficulty"] = difficulty
        if ok < min_ok_to_accept:
            hard_rows.append(r)

    # Stage2: only hard problems.
    stage3_candidates: List[Dict[str, Any]] = []
    for r in hard_rows:
        uuid = r.get("uuid")
        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        model_q = append_choice_map_if_any(normalize_for_model(q_raw))
        choice_map = extract_choice_map(model_q)

        raw_outputs = llm.generate_n(
            stage_name="stage2_solve",
            question=model_q,
            prompt_mode="boxed_solve",
            sleep_s=args.sleep,
        )

        attempts: List[Dict[str, Any]] = []
        ok2 = 0
        for raw in raw_outputs[:8]:
            boxed = extract_boxed_answer(raw)
            extracted = boxed or extract_final_answer(raw)
            pred = standardize_choice_answer(extracted, choice_map=choice_map)
            eq = rule_equivalent(pred, gold, choice_map=choice_map)
            judge_src = "rules"
            if eq is None:
                eq = _llm_judge_equivalence(
                    uuid=uuid,
                    question=q_raw,
                    gold=gold,
                    pred=pred,
                    choice_map=choice_map,
                    stage="stage2",
                )
                judge_src = "llm" if eq is not None else "unknown"

            verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
            if eq is True:
                ok2 += 1
            attempts.append(
                {
                    "raw_text": str(raw),
                    "boxed_answer": boxed,
                    "extracted_answer": extracted,
                    "normalized_answer": pred,
                    "verdict": verdict,
                    "judge_source": judge_src,
                }
            )

        entry = {
            "uuid": uuid,
            "difficulty": r.get("_difficulty"),
            "stage1_ok": r.get("_stage1_ok"),
            "stage1_bad": r.get("_stage1_bad"),
            "question": q_raw,
            "gold_answer": gold,
            "attempts": attempts,
            "ok": ok2,
            "bad": 8 - ok2,
            "stage": "stage2",
            "llm_call_counts": {
                "stage2_solve": llm.stage_params("stage2_solve").n,
                "stage2_judge_llm_fallback": sum(1 for a in attempts if a.get("judge_source") == "llm"),
                "stage2_judge_unknown": sum(1 for a in attempts if a.get("judge_source") == "unknown"),
            },
        }
        stage2_archive.append(entry)

        # If wrong count >= 4, enter next stage; otherwise accept.
        if ok2 < min_ok_to_accept:
            stage3_candidates.append(r)
        else:
            accepted_bank.append({**entry, "accepted_from": "stage2"})

    # Stage3: repeat Stage2 logic for stage3 candidates; discard if still hard.
    for r in stage3_candidates:
        uuid = r.get("uuid")
        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        model_q = append_choice_map_if_any(normalize_for_model(q_raw))
        choice_map = extract_choice_map(model_q)

        raw_outputs = llm.generate_n(
            stage_name="stage3_solve",
            question=model_q,
            prompt_mode="boxed_solve",
            sleep_s=args.sleep,
        )

        attempts: List[Dict[str, Any]] = []
        ok3 = 0
        for raw in raw_outputs[:8]:
            boxed = extract_boxed_answer(raw)
            extracted = boxed or extract_final_answer(raw)
            pred = standardize_choice_answer(extracted, choice_map=choice_map)
            eq = rule_equivalent(pred, gold, choice_map=choice_map)
            judge_src = "rules"
            if eq is None:
                eq = _llm_judge_equivalence(
                    uuid=uuid,
                    question=q_raw,
                    gold=gold,
                    pred=pred,
                    choice_map=choice_map,
                    stage="stage3",
                )
                judge_src = "llm" if eq is not None else "unknown"

            verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
            if eq is True:
                ok3 += 1
            attempts.append(
                {
                    "raw_text": str(raw),
                    "boxed_answer": boxed,
                    "extracted_answer": extracted,
                    "normalized_answer": pred,
                    "verdict": verdict,
                    "judge_source": judge_src,
                }
            )

        entry = {
            "uuid": uuid,
            "difficulty": r.get("_difficulty"),
            "stage1_ok": r.get("_stage1_ok"),
            "stage1_bad": r.get("_stage1_bad"),
            "question": q_raw,
            "gold_answer": gold,
            "attempts": attempts,
            "ok": ok3,
            "bad": 8 - ok3,
            "stage": "stage3",
            "llm_call_counts": {
                "stage3_solve": llm.stage_params("stage3_solve").n,
                "stage3_judge_llm_fallback": sum(1 for a in attempts if a.get("judge_source") == "llm"),
                "stage3_judge_unknown": sum(1 for a in attempts if a.get("judge_source") == "unknown"),
            },
        }
        stage3_archive.append(entry)

        # If wrong count >= 4, discard (no further stage); otherwise accept.
        if ok3 < min_ok_to_accept:
            discarded_hard.append({**entry, "discarded": True})
        else:
            accepted_bank.append({**entry, "accepted_from": "stage3"})

    write_jsonl_atomic(stage2_archive_path, stage2_archive)
    write_jsonl_atomic(stage3_archive_path, stage3_archive)
    write_jsonl_atomic(accepted_bank_path, accepted_bank)
    write_jsonl_atomic(discarded_hard_path, discarded_hard)

    print("Done.")
    print(f"- out_dir: {args.out}")
    print(f"- example_input_copy: {raw_input_copy_path}")
    print(f"- stage1_raw_generations: {stage1_raw_generations_path}")
    print(f"- stage1_output: {stage1_output_path}")
    print(f"- stage2_archive: {stage2_archive_path}")
    print(f"- stage3_archive: {stage3_archive_path}")
    print(f"- accepted_bank: {accepted_bank_path}")
    print(f"- discarded_hard: {discarded_hard_path}")


if __name__ == "__main__":
    main()


