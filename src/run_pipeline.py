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


def _iter_input_jsonl_paths(input_arg: str) -> List[str]:
    """Return a sorted list of *.jsonl files if input_arg is a directory; otherwise [input_arg]."""
    if os.path.isdir(input_arg):
        files: List[str] = []
        for name in os.listdir(input_arg):
            p = os.path.join(input_arg, name)
            if os.path.isfile(p) and name.lower().endswith(".jsonl"):
                files.append(p)
        files.sort()
        return files
    return [input_arg]


def _input_prefix(path: str) -> str:
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    return stem or "input"


def _iter_stage1_output_paths(stage1_dir: str) -> List[str]:
    """
    Find stage1 output files under a stage1 directory.
    Supports:
    - stage1_output.stage1.jsonl
    - <prefix>.stage1_output.stage1.jsonl
    """
    if not stage1_dir or not os.path.isdir(stage1_dir):
        return []
    outs: List[str] = []
    for name in os.listdir(stage1_dir):
        if not name.endswith(".jsonl"):
            continue
        if name.endswith("stage1_output.stage1.jsonl"):
            outs.append(os.path.join(stage1_dir, name))
    outs.sort()
    return outs


def _stage1_output_prefix(path: str) -> str:
    """Infer prefix from '<prefix>.stage1_output.stage1.jsonl' (or '' for unprefixed)."""
    base = os.path.basename(path)
    suf = ".stage1_output.stage1.jsonl"
    if base == "stage1_output.stage1.jsonl":
        return ""
    if base.endswith(suf):
        return base[: -len(suf)]
    return ""


def _run_one_input(
    *,
    input_path: str,
    prefix: str,
    out_dir: str,
    stage1_dir: str,
    stage2_dir: str,
    stage3_dir: str,
    llm: LLMRouter,
    min_votes_to_accept: int,
    sleep_s: float,
    start_stage: str = "stage1",
) -> None:
    """
    Run the whole pipeline for a single input JSONL file, writing outputs with a prefix
    (derived from filename) to avoid collisions when --input is a directory.
    """
    if start_stage not in ("stage1", "stage2"):
        raise ValueError(f"start_stage must be 'stage1' or 'stage2', got: {start_stage}")
    pfx = f"{prefix}." if prefix else ""
    raw_input_copy_path = os.path.join(out_dir, f"{pfx}stage0.jsonl")

    stage1_output_path = os.path.join(stage1_dir, f"{pfx}stage1_output.stage1.jsonl")
    stage1_raw_generations_path = os.path.join(stage1_dir, f"{pfx}stage1_raw_generations.stage1.jsonl")
    stage1_status_path = os.path.join(stage1_dir, f"{pfx}status.stage1.jsonl")

    stage2_archive_path = os.path.join(stage2_dir, f"{pfx}stage2_archive.stage2.jsonl")
    stage2_status_path = os.path.join(stage2_dir, f"{pfx}status.stage2.jsonl")

    stage3_archive_path = os.path.join(stage3_dir, f"{pfx}stage3_archive.stage3.jsonl")
    stage3_status_path = os.path.join(stage3_dir, f"{pfx}status.stage3.jsonl")

    accepted_bank_path = os.path.join(out_dir, f"{pfx}accepted_bank.stage_final.jsonl")

    result_dir = os.path.join(out_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{pfx}result.stage_final.jsonl")
    # Ensure each input file (prefix) produces a corresponding result file (even if empty).
    try:
        open(result_path, "a", encoding="utf-8").close()
    except Exception:
        pass
    # Sanitize legacy result files:
    # - result should only contain stage2/stage3 vote-based results.
    # - upgrade legacy schema (voted_answer/selected_*) to the current schema (final_answer/final_*),
    #   and recompute attempt verdicts as (boxed_answer == final_answer) when possible.
    try:
        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
            kept: List[Dict[str, Any]] = []
            changed = False
            def _choice_letter_for_upgrade(s: str) -> str:
                ss = (s or "").strip()
                if not ss:
                    return ""
                c0 = ss[0].upper()
                if c0 in ("A", "B", "C", "D"):
                    return c0
                return ss
            for row in iter_jsonl(result_path, tolerate_errors=True):
                st = str(row.get("stage") or "")
                if st in ("stage2", "stage3"):
                    if isinstance(row, dict):
                        # Upgrade legacy top-level field name.
                        if "final_answer" not in row and "voted_answer" in row:
                            row["final_answer"] = row.get("voted_answer")
                            try:
                                del row["voted_answer"]
                            except Exception:
                                pass
                            changed = True

                        # Upgrade attempts field name and recompute verdicts if possible.
                        attempts = row.get("attempts")
                        if isinstance(attempts, list):
                            fa = str(row.get("final_answer") or "").strip()
                            for a in attempts:
                                if not isinstance(a, dict):
                                    continue
                                if "final_answer" not in a and "voted_answer" in a:
                                    a["final_answer"] = a.get("voted_answer")
                                    try:
                                        del a["voted_answer"]
                                    except Exception:
                                        pass
                                    changed = True
                                boxed = a.get("boxed_answer")
                                boxed = boxed.strip() if isinstance(boxed, str) else str(boxed).strip()
                                if boxed and fa:
                                    verdict = "正确" if (_choice_letter_for_upgrade(boxed) == _choice_letter_for_upgrade(fa) or boxed == fa) else "错误"
                                else:
                                    verdict = "不确定"
                                if a.get("verdict") != verdict:
                                    a["verdict"] = verdict
                                    changed = True

                    kept.append(row)
                else:
                    changed = True
            if changed:
                write_jsonl_atomic(result_path, kept)
    except Exception:
        # Best-effort; do not fail pipeline for result cleanup.
        pass

    # ---- Resume bookkeeping ----
    stage1_done = _load_status_map(stage1_status_path)
    stage2_done = _load_status_map(stage2_status_path)
    stage3_done = _load_status_map(stage3_status_path)

    # ---- Result bookkeeping (per uuid, per input/prefix) ----
    result_done: set[str] = set()
    if os.path.exists(result_path):
        for row in iter_jsonl(result_path, tolerate_errors=True):
            u = row.get("uuid")
            if u is not None:
                result_done.add(str(u))

    # ---- Shared judge helper (Stage1/2/3) ----
    def _as_choice_letter(s: str) -> str:
        """If s looks like 'A' or 'A.xxx', return 'A'; else return original trimmed string."""
        if not isinstance(s, str):
            return ""
        ss = s.strip()
        if not ss:
            return ""
        c0 = ss[0].upper()
        if c0 in ("A", "B", "C", "D"):
            return c0
        return ss

    def _select_answer(*, gold: str, majority: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the final answer for this stage:
        - If voting is confident (majority_count >= min_votes_to_accept): use voting result
        - Else: fall back to provided gold answer
        """
        maj_raw = str((majority or {}).get("majority") or "").strip()
        maj_cnt = int((majority or {}).get("majority_count") or 0)
        gold_letter = _as_choice_letter(gold)
        maj_letter = _as_choice_letter(maj_raw)
        if maj_cnt >= int(min_votes_to_accept) and maj_letter:
            return {"final_answer": maj_letter, "final_source": "majority", "final_vote_count": maj_cnt}
        return {"final_answer": gold_letter or gold.strip(), "final_source": "answer_fallback", "final_vote_count": maj_cnt}

    def _to_result_entry(*, stage: str, final_answer: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Canonical result schema (one line per uuid per input file):
        - uuid, question
        - each attempt: raw_text, boxed_answer, verdict, final_answer
        - final_answer is selected by vote strength: majority if confident else input.answer fallback
        """
        # Keep the original input answer (a.k.a. gold) for analysis.
        answer = entry.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = entry.get("gold_answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = entry.get("gold")
        answer = (answer or "").strip() if isinstance(answer, str) else ""

        attempts = entry.get("attempts") if isinstance(entry.get("attempts"), list) else []
        attempts_slim: List[Dict[str, Any]] = []
        for a in attempts:
            if not isinstance(a, dict):
                continue
            boxed = (a.get("boxed_answer") or "")
            boxed = boxed.strip() if isinstance(boxed, str) else str(boxed).strip()
            fa = (final_answer or "").strip()
            # Verdict in result is defined as: whether the current boxed_answer equals final_answer.
            # For MCQ-like outputs, compare by choice letter; otherwise compare by trimmed string.
            if not boxed or not fa:
                verdict = "不确定"
            else:
                verdict = "正确" if (_as_choice_letter(boxed) == _as_choice_letter(fa) or boxed == fa) else "错误"
            attempts_slim.append(
                {
                    "raw_text": a.get("raw_text"),
                    "boxed_answer": a.get("boxed_answer"),
                    "verdict": verdict,
                    "final_answer": final_answer,
                }
            )
        return {
            "uuid": entry.get("uuid"),
            "question": entry.get("question"),
            "answer": answer,
            "stage": stage,
            "final_answer": final_answer,
            "majority_answer": entry.get("majority_answer"),
            "attempts": attempts_slim,
        }

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
            sleep_s=sleep_s,
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
        shutil.copyfile(input_path, raw_input_copy_path)
    except Exception:
        pass

    input_rows = _read_all(input_path)
    normalized = [normalize_record(r) for r in input_rows]

    # If we start from stage2 but there is no stage1 status file, treat everything as stage2 candidates.
    if start_stage == "stage2" and not stage1_done:
        for r in normalized:
            u = r.get("uuid")
            if u is None:
                continue
            stage1_done[str(u)] = {"uuid": u, "ok": 0, "bad": 0, "next_stage": "stage2"}

    # ---- Stage 1: per-uuid checkpointing (append) ----
    if start_stage == "stage1":
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
                sleep_s=sleep_s,
                stats=s1_solve_stats,
            )

            extracted = [extract_final_answer(x) for x in raw_solutions]
            choice_map = extract_choice_map(model_q)
            standardized = [standardize_choice_answer(a, choice_map=choice_map) for a in extracted]
            n1 = int(llm.stage_params("stage1_solve").n)
            majority_answer = majority_vote(standardized[:n1])
            majority_answer_json = {
                "majority": majority_answer.majority,
                "majority_count": int(majority_answer.majority_count),
                "counts": dict(majority_answer.counts),
            }

            stage1_attempts: List[Dict[str, Any]] = []
            ok1 = 0
            s1_judge_stats: Dict[str, int] = {}
            for raw, extracted_i, pred_i in zip(raw_solutions[:n1], extracted[:n1], standardized[:n1]):
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

            stored_prompt = assemble_stored_prompt(question=q_raw, standard_answer="", stage1_answers=standardized[:n1])

            stage1_raw_entry: Dict[str, Any] = {
                "uuid": uuid,
                "line_number": r.get("line_number"),
                "stage": "stage1",
                "question": q_raw,
                "gold": gold,
                "model_input": model_q,
                "raw_model_outputs": [str(x) for x in raw_solutions][:n1],
                "extracted_answers": [str(x) for x in standardized][:n1],
                "majority_answer": majority_answer_json,
                "attempts": stage1_attempts,
                "ok": ok1,
                "bad": n1 - ok1,
                "llm_call_counts": {
                    "stage1_solve": llm.stage_params("stage1_solve").n,
                    "stage1_eval": 0,
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
            eval_user = f"{stored_prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
            eval_calls = 0
            eval_text = ""
            s1_eval_stats: Dict[str, int] = {}
            for _ in range(2):
                eval_calls += 1
                eval_text = llm.generate_n(
                    stage_name="stage1_eval",
                    question=eval_user,
                    prompt_mode="raw_prompt_eval",
                    sleep_s=sleep_s,
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

            append_jsonl_line(stage1_raw_generations_path, stage1_raw_entry)
            append_jsonl_line(stage1_output_path, clean)

            counts = extract_boxed_counts(eval_text)
            route_ok, route_bad = (counts if counts is not None else (ok1, n1 - ok1))
            sel1 = _select_answer(gold=gold, majority=majority_answer_json)
            # Routing uses vote strength (consensus). If not enough votes, go to Stage2.
            next_stage = (
                "stage2" if int(majority_answer_json.get("majority_count", 0)) < int(min_votes_to_accept) else "accepted"
            )
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
                    "judge_bad": int(n1 - ok1),
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "vote_majority": majority_answer_json.get("majority"),
                    "vote_majority_count": int(majority_answer_json.get("majority_count", 0)),
                    **sel1,
                    "next_stage": next_stage,
                    "paths": {"raw_generations": stage1_raw_generations_path, "output": stage1_output_path},
                },
            )
            if next_stage == "accepted":
                append_jsonl_line(accepted_bank_path, {**stage1_raw_entry, **sel1, "accepted_from": "stage1"})
            stage1_done[uuid_key] = {"uuid": uuid, "ok": int(route_ok), "bad": int(route_bad), "next_stage": next_stage}

    # ---- Stage 2/3 ----
    # Stage2/Stage3 do NOT need Stage1 output files. They only need:
    #   - the original input rows (question/answer/uuid)
    #   - Stage1 status routing decisions (status.stage1.jsonl), which may be produced externally.
    input_row_by_uuid: Dict[str, Dict[str, Any]] = {str(r.get("uuid")): r for r in normalized if r.get("uuid") is not None}

    # Ensure stage1-accepted UUIDs are present in accepted_bank, even when resuming or when start_stage="stage2".
    accepted_done: set[str] = set()
    if os.path.exists(accepted_bank_path):
        for row in iter_jsonl(accepted_bank_path, tolerate_errors=True):
            u = row.get("uuid")
            if u is not None:
                accepted_done.add(str(u))

    # Backfill result archive from accepted_bank (useful for resume or re-run with result dir missing).
    if os.path.exists(accepted_bank_path):
        for row in iter_jsonl(accepted_bank_path, tolerate_errors=True):
            u = row.get("uuid")
            if u is None:
                continue
            u_str = str(u)
            if u_str in result_done:
                continue
            if (row.get("final_source") or "") != "majority":
                continue
            accepted_from = str(row.get("accepted_from") or "")
            if accepted_from not in ("stage2", "stage3"):
                continue
            stage = accepted_from
            final_answer = str(row.get("final_answer") or "").strip()
            append_jsonl_line(result_path, _to_result_entry(stage=stage, final_answer=final_answer, entry=row))
            result_done.add(u_str)
    raw_stage1_by_uuid: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(stage1_raw_generations_path):
        for row in iter_jsonl(stage1_raw_generations_path, tolerate_errors=True):
            u = row.get("uuid")
            if u is not None:
                raw_stage1_by_uuid[str(u)] = row
    for u_str, st in stage1_done.items():
        if (st.get("next_stage") or "") != "accepted":
            continue
        if u_str in accepted_done:
            continue
        base = input_row_by_uuid.get(u_str, {})
        entry = raw_stage1_by_uuid.get(u_str, {"uuid": st.get("uuid"), "stage": "stage1"})
        entry = {
            **entry,
            "question": entry.get("question") or base.get("question"),
            "gold": entry.get("gold") or base.get("answer"),
            "final_answer": st.get("final_answer"),
            "final_source": st.get("final_source"),
            "final_vote_count": st.get("final_vote_count"),
            "accepted_from": "stage1_replay",
        }
        append_jsonl_line(accepted_bank_path, entry)
        accepted_done.add(u_str)

    hard_rows: List[Dict[str, Any]] = []
    for u_str, st in stage1_done.items():
        if (st.get("next_stage") or "") != "stage2":
            continue
        r = input_row_by_uuid.get(u_str)
        if not r:
            continue
        ok = int(st.get("ok", 0))
        bad = int(st.get("bad", 0))
        r["_stage1_ok"] = ok
        r["_stage1_bad"] = bad
        r["_difficulty"] = bad
        hard_rows.append(r)

    stage3_candidates: List[Dict[str, Any]] = []
    for r in hard_rows:
        uuid = r.get("uuid")
        uuid_key = str(uuid)
        if uuid_key in stage2_done:
            continue
        q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
        gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
        model_q = append_choice_map_if_any(normalize_for_model(q_raw))
        choice_map = extract_choice_map(model_q)

        s2_solve_stats: Dict[str, int] = {}
        raw_solutions = llm.generate_n(
            stage_name="stage2_solve",
            question=model_q,
            prompt_mode="boxed_solve",
            sleep_s=sleep_s,
            stats=s2_solve_stats,
        )
        n2 = int(llm.stage_params("stage2_solve").n)
        extracted = [extract_final_answer(x) for x in raw_solutions]
        standardized = [standardize_choice_answer(a, choice_map=choice_map) for a in extracted]

        stage2_attempts: List[Dict[str, Any]] = []
        ok2 = 0
        s2_judge_stats: Dict[str, int] = {}
        for raw, extracted_i, pred_i in zip(raw_solutions[:n2], extracted[:n2], standardized[:n2]):
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
                    stage="stage2",
                    stats=s2_judge_stats,
                )
                judge_src = "llm" if eq is not None else "unknown"
            verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
            if eq is True:
                ok2 += 1
            stage2_attempts.append(
                {
                    "raw_text": str(raw),
                    "boxed_answer": boxed_i,
                    "extracted_answer": extracted_final,
                    "normalized_answer": pred_final,
                    "verdict": verdict,
                    "judge_source": judge_src,
                }
            )

        # Majority vote uses the same "boxed-first, then normalize" answers we judge with.
        s2_vote = majority_vote([str(a.get("normalized_answer") or "") for a in stage2_attempts][:n2])
        stage2_majority_answer = {
            "majority": s2_vote.majority,
            "majority_count": int(s2_vote.majority_count),
            "counts": dict(s2_vote.counts),
        }

        sel2 = _select_answer(gold=gold, majority=stage2_majority_answer)
        final_answer2 = str(sel2.get("final_answer") or "").strip()
        # In archive, verdict is defined as whether boxed_answer equals final_answer (MCQ compares by letter).
        ok2_final = 0
        for a in stage2_attempts:
            boxed = a.get("boxed_answer")
            boxed = boxed.strip() if isinstance(boxed, str) else str(boxed).strip()
            if not boxed or not final_answer2:
                a["verdict"] = "不确定"
            else:
                a["verdict"] = "正确" if (_as_choice_letter(boxed) == _as_choice_letter(final_answer2) or boxed == final_answer2) else "错误"
            if a["verdict"] == "正确":
                ok2_final += 1

        entry: Dict[str, Any] = {
            "uuid": uuid,
            "line_number": r.get("line_number"),
            "stage": "stage2",
            "difficulty": r.get("_difficulty"),
            "question": q_raw,
            "answer": gold,
            "gold": gold,
            "model_input": model_q,
            "raw_model_outputs": [str(x) for x in raw_solutions][:n2],
            "extracted_answers": [str(x) for x in standardized][:n2],
            "majority_answer": stage2_majority_answer,
            "attempts": stage2_attempts,
            **sel2,
            "ok": ok2_final,
            "bad": n2 - ok2_final,
            "llm_call_counts": {
                "stage2_solve": llm.stage_params("stage2_solve").n,
                "stage2_judge_llm_fallback": sum(1 for a in stage2_attempts if a.get("judge_source") == "llm"),
                "stage2_judge_unknown": sum(1 for a in stage2_attempts if a.get("judge_source") == "unknown"),
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

        append_jsonl_line(stage2_archive_path, entry)
        # Routing uses vote strength (consensus), not reference-accuracy ok/bad.
        next_stage = "stage3" if int(stage2_majority_answer["majority_count"]) < int(min_votes_to_accept) else "accepted"
        if next_stage == "accepted":
            append_jsonl_line(accepted_bank_path, {**entry, **sel2, "accepted_from": "stage2"})
            if sel2.get("final_source") == "majority" and uuid_key not in result_done:
                append_jsonl_line(result_path, _to_result_entry(stage="stage2", final_answer=str(sel2.get("final_answer") or ""), entry=entry))
                result_done.add(uuid_key)
        append_jsonl_line(
            stage2_status_path,
            {
                "uuid": uuid,
                "stage": "stage2",
                "ok": int(ok2),
                "bad": int(n2 - ok2),
                "min_votes_to_accept": int(min_votes_to_accept),
                "vote_majority": stage2_majority_answer.get("majority"),
                "vote_majority_count": int(stage2_majority_answer.get("majority_count", 0)),
                **_select_answer(gold=gold, majority=stage2_majority_answer),
                "next_stage": next_stage,
                "difficulty": r.get("_difficulty"),
                "stage1_ok": r.get("_stage1_ok"),
                "stage1_bad": r.get("_stage1_bad"),
                "paths": {"archive": stage2_archive_path},
            },
        )
        stage2_done[uuid_key] = {"uuid": uuid, "ok": int(ok2), "bad": int(n2 - ok2), "next_stage": next_stage}
        if next_stage == "stage3":
            stage3_candidates.append(r)

    for u_str, st in stage2_done.items():
        if (st.get("next_stage") or "") != "stage3":
            continue
        r = input_row_by_uuid.get(u_str)
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
        raw_solutions = llm.generate_n(
            stage_name="stage3_solve",
            question=model_q,
            prompt_mode="boxed_solve",
            sleep_s=sleep_s,
            stats=s3_solve_stats,
        )
        n3 = int(llm.stage_params("stage3_solve").n)
        extracted = [extract_final_answer(x) for x in raw_solutions]
        standardized = [standardize_choice_answer(a, choice_map=choice_map) for a in extracted]

        stage3_attempts: List[Dict[str, Any]] = []
        ok3 = 0
        s3_judge_stats: Dict[str, int] = {}
        for raw, extracted_i, pred_i in zip(raw_solutions[:n3], extracted[:n3], standardized[:n3]):
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
                    stage="stage3",
                    stats=s3_judge_stats,
                )
                judge_src = "llm" if eq is not None else "unknown"
            verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
            if eq is True:
                ok3 += 1
            stage3_attempts.append(
                {
                    "raw_text": str(raw),
                    "boxed_answer": boxed_i,
                    "extracted_answer": extracted_final,
                    "normalized_answer": pred_final,
                    "verdict": verdict,
                    "judge_source": judge_src,
                }
            )

        # Majority vote uses the same "boxed-first, then normalize" answers we judge with.
        s3_vote = majority_vote([str(a.get("normalized_answer") or "") for a in stage3_attempts][:n3])
        stage3_majority_answer = {
            "majority": s3_vote.majority,
            "majority_count": int(s3_vote.majority_count),
            "counts": dict(s3_vote.counts),
        }

        sel3 = _select_answer(gold=gold, majority=stage3_majority_answer)
        final_answer3 = str(sel3.get("final_answer") or "").strip()
        ok3_final = 0
        for a in stage3_attempts:
            boxed = a.get("boxed_answer")
            boxed = boxed.strip() if isinstance(boxed, str) else str(boxed).strip()
            if not boxed or not final_answer3:
                a["verdict"] = "不确定"
            else:
                a["verdict"] = "正确" if (_as_choice_letter(boxed) == _as_choice_letter(final_answer3) or boxed == final_answer3) else "错误"
            if a["verdict"] == "正确":
                ok3_final += 1

        entry = {
            "uuid": uuid,
            "line_number": r.get("line_number"),
            "stage": "stage3",
            "difficulty": r.get("_difficulty"),
            "question": q_raw,
            "answer": gold,
            "gold": gold,
            "model_input": model_q,
            "raw_model_outputs": [str(x) for x in raw_solutions][:n3],
            "extracted_answers": [str(x) for x in standardized][:n3],
            "majority_answer": stage3_majority_answer,
            "attempts": stage3_attempts,
            **sel3,
            "ok": ok3_final,
            "bad": n3 - ok3_final,
            "llm_call_counts": {
                "stage3_solve": llm.stage_params("stage3_solve").n,
                "stage3_judge_llm_fallback": sum(1 for a in stage3_attempts if a.get("judge_source") == "llm"),
                "stage3_judge_unknown": sum(1 for a in stage3_attempts if a.get("judge_source") == "unknown"),
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
        # Finalization rule:
        # - If vote has consensus: accept by vote
        # - Else: accept by provided gold fallback (do NOT discard)
        next_stage = "accepted"
        accepted_from = "stage3" if sel3.get("final_source") == "majority" else "stage3_gold_fallback"
        append_jsonl_line(accepted_bank_path, {**entry, **sel3, "accepted_from": accepted_from})
        if sel3.get("final_source") == "majority" and uuid_key not in result_done:
            append_jsonl_line(result_path, _to_result_entry(stage="stage3", final_answer=str(sel3.get("final_answer") or ""), entry=entry))
            result_done.add(uuid_key)
        append_jsonl_line(
            stage3_status_path,
            {
                "uuid": uuid,
                "stage": "stage3",
                "ok": int(ok3),
                "bad": int(n3 - ok3),
                "min_votes_to_accept": int(min_votes_to_accept),
                "vote_majority": stage3_majority_answer.get("majority"),
                "vote_majority_count": int(stage3_majority_answer.get("majority_count", 0)),
                **sel3,
                "next_stage": next_stage,
                "difficulty": r.get("_difficulty"),
                "stage1_ok": r.get("_stage1_ok"),
                "stage1_bad": r.get("_stage1_bad"),
                "paths": {"archive": stage3_archive_path},
            },
        )
        stage3_done[uuid_key] = {"uuid": uuid, "ok": int(ok3), "bad": int(n3 - ok3), "next_stage": next_stage}

    print("Done.")
    print(f"- input: {input_path}")
    print(f"- prefix: {prefix}")
    print(f"- out_dir: {out_dir}")
    print(f"- input_copy: {raw_input_copy_path}")
    print(f"- stage1_output: {stage1_output_path}")
    print(f"- stage2_archive: {stage2_archive_path}")
    print(f"- stage3_archive: {stage3_archive_path}")
    print(f"- accepted_bank: {accepted_bank_path}")
    print(f"- result: {result_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="MathAgent minimal pipeline (JSONL between stages).")
    p.add_argument("--input", default=None, help="Input JSONL file path OR a directory containing JSONL files.")
    p.add_argument(
        "--stage1",
        default=None,
        help="Run Stage2/Stage3 using an existing stage1 directory as input (reads *stage1_output.stage1.jsonl there).",
    )
    p.add_argument("--out", default=_default_out_dir(), help="Output directory (default datasets/out/<timestamp>)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM calls (rate limit)")
    p.add_argument(
        "--llm-config",
        default="config/llm_models.json",
        help="Path to JSON config describing models + stage routing.",
    )
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not args.input and not args.stage1:
        raise ValueError("Must provide either --input or --stage1")
    if args.input and args.stage1:
        raise ValueError("Provide only one of --input or --stage1 (not both)")

    stage1_dir = os.path.join(args.out, "stage1") if not args.stage1 else str(args.stage1)
    stage2_dir = os.path.join(args.out, "stage2")
    stage3_dir = os.path.join(args.out, "stage3")
    if not os.path.isdir(stage1_dir):
        os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(stage3_dir, exist_ok=True)

    llm = LLMRouter(config_path=args.llm_config)
    min_votes_to_accept = llm.threshold_int("min_votes_to_accept", 5)

    # Stage1-dir input mode: run Stage2/Stage3 using existing Stage1 artifacts.
    if args.stage1:
        stage1_outs = _iter_stage1_output_paths(stage1_dir)
        if not stage1_outs:
            raise ValueError(f"--stage1 has no *stage1_output.stage1.jsonl files: {stage1_dir}")
        for stage1_output_path in stage1_outs:
            prefix = _stage1_output_prefix(stage1_output_path)
            _run_one_input(
                input_path=stage1_output_path,
                prefix=prefix,
                out_dir=args.out,
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                stage3_dir=stage3_dir,
                llm=llm,
                min_votes_to_accept=min_votes_to_accept,
                sleep_s=float(args.sleep),
                start_stage="stage2",
            )
        return

    assert args.input
    # Directory input mode: run once per *.jsonl file, prefixing outputs by filename stem.
    if os.path.isdir(args.input):
        input_paths = _iter_input_jsonl_paths(args.input)
        if not input_paths:
            raise ValueError(f"--input is a directory but has no *.jsonl files: {args.input}")
        for input_path in input_paths:
            prefix = _input_prefix(input_path)
            _run_one_input(
                input_path=input_path,
                prefix=prefix,
                out_dir=args.out,
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                stage3_dir=stage3_dir,
                llm=llm,
                min_votes_to_accept=min_votes_to_accept,
                sleep_s=float(args.sleep),
            )
        return

    # Single-file mode: run through the unified implementation.
    # Use empty prefix to keep historical filenames (no '<prefix>.' prefix).
    _run_one_input(
        input_path=args.input,
        prefix="",
        out_dir=args.out,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        stage3_dir=stage3_dir,
        llm=llm,
        min_votes_to_accept=min_votes_to_accept,
        sleep_s=float(args.sleep),
    )
    return


if __name__ == "__main__":
    main()


