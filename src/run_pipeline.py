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
from dataio.jsonl_io import append_jsonl_line, iter_jsonl, write_jsonl_atomic
from dataio.sample_schema import CANONICAL_KEYS, normalize_output_wrapper, normalize_record


def _default_out_dir() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join("datasets", "out", ts)


def _read_all(path: str) -> List[dict]:
    return list(iter_jsonl(path, tolerate_errors=True))


def _load_status_map(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load a status.jsonl file into a map keyed by str(uuid).
    Each line is a JSON object at least containing 'uuid'.
    """
    if not path or not os.path.exists(path):
        return {}
    m: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path, tolerate_errors=True):
        u = row.get("uuid")
        if u is None:
            continue
        m[str(u)] = row
    return m


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

    stage1_dir = os.path.join(args.out, "stage1")
    stage2_dir = os.path.join(args.out, "stage2")
    stage3_dir = os.path.join(args.out, "stage3")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(stage3_dir, exist_ok=True)

    raw_input_copy_path = os.path.join(args.out, "example_input.stage0.jsonl")
    stage1_output_path = os.path.join(stage1_dir, "stage1_output.stage1.jsonl")
    stage1_raw_generations_path = os.path.join(stage1_dir, "stage1_raw_generations.stage1.jsonl")
    stage1_status_path = os.path.join(stage1_dir, "status.stage1.jsonl")

    stage2_archive_path = os.path.join(stage2_dir, "stage2_archive.stage2.jsonl")
    stage2_status_path = os.path.join(stage2_dir, "status.stage2.jsonl")

    stage3_archive_path = os.path.join(stage3_dir, "stage3_archive.stage3.jsonl")
    stage3_status_path = os.path.join(stage3_dir, "status.stage3.jsonl")

    accepted_bank_path = os.path.join(args.out, "accepted_bank.stage_final.jsonl")
    discarded_hard_path = os.path.join(args.out, "discarded_hard.stage_final.jsonl")

    llm = LLMRouter(config_path=args.llm_config)
    min_ok_to_accept = llm.threshold_int("min_ok_to_accept", 5)
    stage1_okbad_by_uuid: Dict[Any, tuple[int, int]] = {}

    # ---- Resume bookkeeping ----
    stage1_done = _load_status_map(stage1_status_path)
    stage2_done = _load_status_map(stage2_status_path)
    stage3_done = _load_status_map(stage3_status_path)
    for u_str, row in stage1_done.items():
        try:
            ok = int(row.get("ok", 0))
            bad = int(row.get("bad", 0))
            stage1_okbad_by_uuid[u_str] = (ok, bad)
        except Exception:
            pass

    # ---- Shared judge helper (Stage1/2/3) ----
    def _llm_judge_equivalence(
        *,
        uuid: Any,
        question: str,
        gold: str,
        pred: str,
        choice_map: Dict[str, str],
        stage: str,
        stats: Dict[str, int] | None = None,
    ) -> bool | None:
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
            stats=stats,
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

    # ---- Stage 1: per-uuid checkpointing (append) ----

    for r in normalized:
        uuid = r.get("uuid")
        uuid_key = str(uuid)
        if uuid_key in stage1_done:
            continue
        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        if not q_raw.strip():
            raise ValueError(f"Missing question: uuid={uuid}")
        if not gold:
            raise ValueError(f"Missing gold in field 'answer': uuid={uuid}")

        model_q = append_choice_map_if_any(normalize_for_model(q_raw))
        s1_solve_stats: Dict[str, int] = {}
        raw_solutions = llm.generate_n(
            stage_name="stage1_solve",
            question=model_q,
            prompt_mode="problem",
            sleep_s=args.sleep,
            stats=s1_solve_stats,
        )

        extracted = [extract_final_answer(x) for x in raw_solutions]
        choice_map = extract_choice_map(model_q)
        standardized = [standardize_choice_answer(a, choice_map=choice_map) for a in extracted]
        vr = majority_vote(standardized)
        majority_answer = vr.majority

        # Stage1 uses the same extractor + rule-first judge (LLM fallback) as Stage2/Stage3.
        stage1_attempts: List[Dict[str, Any]] = []
        ok1 = 0
        s1_judge_stats: Dict[str, int] = {}
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
                    stats=s1_judge_stats,
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

        stage1_raw_entry: Dict[str, Any] = {
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
                "stage1_solve_http_calls": int(s1_solve_stats.get("http_calls", 0)),
                "stage1_solve_retries": int(s1_solve_stats.get("retries", 0)),
                "stage1_solve_timeouts": int(s1_solve_stats.get("timeouts", 0)),
                "stage1_solve_errors": int(s1_solve_stats.get("errors", 0)),
                "stage1_judge_http_calls": int(s1_judge_stats.get("http_calls", 0)),
                "stage1_judge_retries": int(s1_judge_stats.get("retries", 0)),
                "stage1_judge_timeouts": int(s1_judge_stats.get("timeouts", 0)),
                "stage1_judge_errors": int(s1_judge_stats.get("errors", 0)),
            },
        }
        stage1_okbad_by_uuid[uuid_key] = (ok1, 8 - ok1)

        eval_user = f"{stored_prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
        # Evaluate once; if boxed counts missing, retry once immediately so routing/metrics are consistent.
        eval_calls = 0
        eval_text = ""
        s1_eval_stats: Dict[str, int] = {}
        for _ in range(2):
            eval_calls += 1
            eval_text = llm.generate_n(
                stage_name="stage1_eval",
                question=eval_user,
                prompt_mode="raw_prompt_eval",
                sleep_s=args.sleep,
                stats=s1_eval_stats,
            )[0]
            eval_text = strip_think(eval_text)
            if extract_boxed_counts(eval_text) is not None:
                break
        stage1_raw_entry["llm_call_counts"]["stage1_eval"] = eval_calls
        stage1_raw_entry["llm_call_counts"]["stage1_eval_http_calls"] = int(s1_eval_stats.get("http_calls", 0))
        stage1_raw_entry["llm_call_counts"]["stage1_eval_retries"] = int(s1_eval_stats.get("retries", 0))
        stage1_raw_entry["llm_call_counts"]["stage1_eval_timeouts"] = int(s1_eval_stats.get("timeouts", 0))
        stage1_raw_entry["llm_call_counts"]["stage1_eval_errors"] = int(s1_eval_stats.get("errors", 0))
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
        # Write per-uuid outputs first, then status last (so status implies outputs exist).
        append_jsonl_line(stage1_raw_generations_path, stage1_raw_entry)
        append_jsonl_line(stage1_output_path, clean)

        counts = extract_boxed_counts(eval_text)
        route_ok, route_bad = (counts if counts is not None else (ok1, 8 - ok1))
        next_stage = "stage2" if int(route_ok) < int(min_ok_to_accept) else "accepted"
        append_jsonl_line(
            stage1_status_path,
            {
                "uuid": uuid,
                "stage": "stage1",
                "ok": int(route_ok),
                "bad": int(route_bad),
                "eval_ok": int(counts[0]) if counts is not None else None,
                "eval_bad": int(counts[1]) if counts is not None else None,
                "judge_ok": int(ok1),
                "judge_bad": int(8 - ok1),
                "min_ok_to_accept": int(min_ok_to_accept),
                "next_stage": next_stage,
                "paths": {
                    "raw_generations": stage1_raw_generations_path,
                    "output": stage1_output_path,
                },
            },
        )
        stage1_done[uuid_key] = {"uuid": uuid, "ok": int(route_ok), "bad": int(route_bad), "next_stage": next_stage}

    # ---- Stage 2/3 (per Architecture.md) ----

    # Load Stage1 outputs for downstream (we need question/gold text for candidates).
    stage1_rows = _read_all(stage1_output_path) if os.path.exists(stage1_output_path) else []
    stage1_row_by_uuid: Dict[str, Dict[str, Any]] = {str(r.get("uuid")): r for r in stage1_rows if r.get("uuid") is not None}

    # Route hard problems by Stage1 status (resume-safe).
    hard_rows: List[Dict[str, Any]] = []
    for u_str, st in stage1_done.items():
        if (st.get("next_stage") or "") != "stage2":
            continue
        r = stage1_row_by_uuid.get(u_str)
        if not r:
            continue
        ok = int(st.get("ok", 0))
        bad = int(st.get("bad", 0))
        r["_stage1_ok"] = ok
        r["_stage1_bad"] = bad
        r["_difficulty"] = bad
        hard_rows.append(r)

    # Stage2: only hard problems.
    stage3_candidates: List[Dict[str, Any]] = []
    for r in hard_rows:
        uuid = r.get("uuid")
        uuid_key = str(uuid)
        if uuid_key in stage2_done:
            # Already finished Stage2; routing will be handled by Stage2 status.
            continue
        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        model_q = append_choice_map_if_any(normalize_for_model(q_raw))
        choice_map = extract_choice_map(model_q)

        s2_solve_stats: Dict[str, int] = {}
        raw_outputs = llm.generate_n(
            stage_name="stage2_solve",
            question=model_q,
            prompt_mode="boxed_solve",
            sleep_s=args.sleep,
            stats=s2_solve_stats,
        )

        attempts: List[Dict[str, Any]] = []
        ok2 = 0
        s2_judge_stats: Dict[str, int] = {}
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
                    stats=s2_judge_stats,
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
                "stage2_solve_http_calls": int(s2_solve_stats.get("http_calls", 0)),
                "stage2_solve_retries": int(s2_solve_stats.get("retries", 0)),
                "stage2_solve_timeouts": int(s2_solve_stats.get("timeouts", 0)),
                "stage2_solve_errors": int(s2_solve_stats.get("errors", 0)),
                "stage2_judge_http_calls": int(s2_judge_stats.get("http_calls", 0)),
                "stage2_judge_retries": int(s2_judge_stats.get("retries", 0)),
                "stage2_judge_timeouts": int(s2_judge_stats.get("timeouts", 0)),
                "stage2_judge_errors": int(s2_judge_stats.get("errors", 0)),
            },
        }
        # Persist per-uuid artifacts, then status last.
        append_jsonl_line(stage2_archive_path, entry)
        next_stage = "stage3" if int(ok2) < int(min_ok_to_accept) else "accepted"
        if next_stage == "accepted":
            append_jsonl_line(accepted_bank_path, {**entry, "accepted_from": "stage2"})
        append_jsonl_line(
            stage2_status_path,
            {
                "uuid": uuid,
                "stage": "stage2",
                "ok": int(ok2),
                "bad": int(8 - ok2),
                "min_ok_to_accept": int(min_ok_to_accept),
                "next_stage": next_stage,
                "difficulty": r.get("_difficulty"),
                "stage1_ok": r.get("_stage1_ok"),
                "stage1_bad": r.get("_stage1_bad"),
                "paths": {"archive": stage2_archive_path},
            },
        )
        stage2_done[uuid_key] = {"uuid": uuid, "ok": int(ok2), "bad": int(8 - ok2), "next_stage": next_stage}
        if next_stage == "stage3":
            stage3_candidates.append(r)

    # Stage3: repeat Stage2 logic for stage3 candidates; discard if still hard.
    # Also include candidates from previous Stage2 runs (resume).
    for u_str, st in stage2_done.items():
        if (st.get("next_stage") or "") != "stage3":
            continue
        r = stage1_row_by_uuid.get(u_str)
        if r and r not in stage3_candidates:
            r["_difficulty"] = st.get("difficulty")
            r["_stage1_ok"] = st.get("stage1_ok")
            r["_stage1_bad"] = st.get("stage1_bad")
            stage3_candidates.append(r)

    for r in stage3_candidates:
        uuid = r.get("uuid")
        uuid_key = str(uuid)
        if uuid_key in stage3_done:
            continue
        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        model_q = append_choice_map_if_any(normalize_for_model(q_raw))
        choice_map = extract_choice_map(model_q)

        s3_solve_stats: Dict[str, int] = {}
        raw_outputs = llm.generate_n(
            stage_name="stage3_solve",
            question=model_q,
            prompt_mode="boxed_solve",
            sleep_s=args.sleep,
            stats=s3_solve_stats,
        )

        attempts: List[Dict[str, Any]] = []
        ok3 = 0
        s3_judge_stats: Dict[str, int] = {}
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
                    stats=s3_judge_stats,
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
                "stage3_solve_http_calls": int(s3_solve_stats.get("http_calls", 0)),
                "stage3_solve_retries": int(s3_solve_stats.get("retries", 0)),
                "stage3_solve_timeouts": int(s3_solve_stats.get("timeouts", 0)),
                "stage3_solve_errors": int(s3_solve_stats.get("errors", 0)),
                "stage3_judge_http_calls": int(s3_judge_stats.get("http_calls", 0)),
                "stage3_judge_retries": int(s3_judge_stats.get("retries", 0)),
                "stage3_judge_timeouts": int(s3_judge_stats.get("timeouts", 0)),
                "stage3_judge_errors": int(s3_judge_stats.get("errors", 0)),
            },
        }
        append_jsonl_line(stage3_archive_path, entry)
        next_stage = "discarded" if int(ok3) < int(min_ok_to_accept) else "accepted"
        if next_stage == "accepted":
            append_jsonl_line(accepted_bank_path, {**entry, "accepted_from": "stage3"})
        else:
            append_jsonl_line(discarded_hard_path, {**entry, "discarded": True})
        append_jsonl_line(
            stage3_status_path,
            {
                "uuid": uuid,
                "stage": "stage3",
                "ok": int(ok3),
                "bad": int(8 - ok3),
                "min_ok_to_accept": int(min_ok_to_accept),
                "next_stage": next_stage,
                "difficulty": r.get("_difficulty"),
                "stage1_ok": r.get("_stage1_ok"),
                "stage1_bad": r.get("_stage1_bad"),
                "paths": {"archive": stage3_archive_path},
            },
        )
        stage3_done[uuid_key] = {"uuid": uuid, "ok": int(ok3), "bad": int(8 - ok3), "next_stage": next_stage}

    print("Done.")
    print(f"- out_dir: {args.out}")
    print(f"- example_input_copy: {raw_input_copy_path}")
    print(f"- stage1_dir: {stage1_dir}")
    print(f"- stage1_raw_generations: {stage1_raw_generations_path}")
    print(f"- stage1_output: {stage1_output_path}")
    print(f"- stage1_status: {stage1_status_path}")
    print(f"- stage2_dir: {stage2_dir}")
    print(f"- stage2_archive: {stage2_archive_path}")
    print(f"- stage2_status: {stage2_status_path}")
    print(f"- stage3_dir: {stage3_dir}")
    print(f"- stage3_archive: {stage3_archive_path}")
    print(f"- stage3_status: {stage3_status_path}")
    print(f"- accepted_bank: {accepted_bank_path}")
    print(f"- discarded_hard: {discarded_hard_path}")


if __name__ == "__main__":
    main()


