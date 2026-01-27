"""
Pipeline v2: only three ops

- infer
- eval (majority-vote)
- result_rebuild

No --stage: stage is inferred from the stage directory name:
  <run_dir>/<stage>/
    infer.jsonl
    status.jsonl
    (optionally) raw_generations.jsonl, archive.jsonl

Legacy stage1/2/3 specific logic lives elsewhere; this module stays generic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import fnmatch
from typing import Any, Dict, Iterable, List, Tuple

from core.stages import normalize_for_model
from dataio.jsonl_io import append_jsonl_line, iter_jsonl, write_jsonl_atomic
from dataio.sample_schema import get_canonical_keys, normalize_record, validate_question_key_priority
from infra.llm_helper import maybe_autostart_vllm, maybe_shutdown_vllm
from infra.llm_router import LLMRouter


def _stage_from_dir(stage_dir: str) -> str:
    return os.path.basename(os.path.normpath(stage_dir))


def _run_dir_from_stage_dir(stage_dir: str) -> str:
    return os.path.dirname(os.path.normpath(stage_dir))


def _out_stage_from_dir(p: str) -> str:
    """
    Stage name is derived ONLY from the output directory name.
    This avoids any implicit "stageK depends on stage(K-1)" assumptions.
    """
    base = os.path.basename(os.path.normpath(str(p or ""))).strip()
    if not base:
        return ""
    return base


def _has_prev_artifacts(dir_path: str) -> bool:
    """A previous-stage directory is considered valid if it contains infer.jsonl + status.jsonl."""
    if not dir_path or not os.path.isdir(dir_path):
        return False
    return os.path.exists(os.path.join(dir_path, "infer.jsonl")) and os.path.exists(os.path.join(dir_path, "status.jsonl"))


def _load_done_uuid_set(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    s: set[str] = set()
    for row in iter_jsonl(path, tolerate_errors=True):
        if isinstance(row, dict) and row.get("uuid") is not None:
            s.add(str(row["uuid"]))
    return s


def _input_opts(llm: LLMRouter) -> Dict[str, Any]:
    """
    Input behavior is configuration-driven via options.input (a dict).

    Expected keys (all optional):
    - raw_input_glob: str glob for directory input (default "*.jsonl")
    - question_key_priority: list[str] (default ["question","prompt","text"])
    - canonical_keys: list[str] (default: predefined list)
    """
    obj = llm.option_any("input", {})
    return obj if isinstance(obj, dict) else {}


def _get_canonical_keys(llm: LLMRouter) -> List[str]:
    """Get canonical keys from config."""
    input_opts = _input_opts(llm)
    canonical_keys = input_opts.get("canonical_keys")
    return get_canonical_keys(canonical_keys)


def _iter_raw_input_rows(*, input_path: str, glob: str) -> Iterable[Dict[str, Any]]:
    """
    Iterate raw input rows from:
    - a JSONL file, or
    - a directory (top-level files matching `glob`, stable order).
    """
    if not input_path:
        raise ValueError("Missing --input")
    pat = str(glob or "").strip() or "*.jsonl"

    # File
    if os.path.isfile(input_path):
        return iter_jsonl(input_path, tolerate_errors=True)

    # Directory
    if os.path.isdir(input_path):
        files = sorted([f for f in os.listdir(input_path) if fnmatch.fnmatch(f, pat)])
        if not files:
            raise ValueError(f"No files matching {pat} under directory: {input_path}")
        def _rows() -> Iterable[Dict[str, Any]]:
            for name in files:
                p = os.path.join(input_path, name)
                if not os.path.isfile(p):
                    continue
                for row in iter_jsonl(p, tolerate_errors=True):
                    yield row
        return _rows()

    raise ValueError(f"--input must be a file or directory for raw infer: {input_path}")


def _question_from_row(*, llm: LLMRouter, row: Dict[str, Any]) -> str:
    """
    Decide which field(s) constitute the effective question text.
    This is configuration-driven via options.input.question_key_priority.
    Validates that question_key_priority keys exist in canonical_keys.
    """
    input_opts = _input_opts(llm)
    keys = input_opts.get("question_key_priority")
    if not isinstance(keys, list) or not keys:
        keys = ["question", "prompt", "text"]
    
    # Validate that all keys in question_key_priority exist in canonical_keys
    canonical_keys = _get_canonical_keys(llm)
    validate_question_key_priority(keys, canonical_keys)
    
    for k in keys:
        if not isinstance(k, str) or not k:
            continue
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_answers(*, llm: LLMRouter, stage_solve: str, raw_outputs: List[str]) -> List[str]:
    """
    Non-LLM answer extraction by keyword (config-driven).
    Uses options.answer_extract_keywords.
    """
    raws = [str(x) for x in (raw_outputs or [])]
    if not raws:
        return []
    # Router provides the generation delimiter keyword (first keyword).
    kw = str(llm.answer_keyword_for_stage(stage_solve) or "").strip() or "FINAL:"
    out: List[str] = []
    for t in raws:
        s = str(t or "")
        i = s.rfind(kw)
        if i < 0:
            out.append("")
            continue
        out.append(str(s[i + len(kw) :]).strip())
    return out


def _vote_majority(
    *,
    llm: LLMRouter,
    stage_eval: str,
    question_raw: str,
    candidates: List[str],
    sleep_s: float,
    stats: Dict[str, int] | None,
) -> Dict[str, Any]:
    """
    Majority vote via LLM. Prompt template is config-driven:
      options.prompts.majority_vote
    Output expects JSON with majority/majority_count/normalized/majority_answer_idxs.
    """
    answers = [str(x or "") for x in (candidates or [])]
    prompt_t = str(llm.prompt_text("majority_vote") or "").strip()
    if not prompt_t:
        raise ValueError("Missing config options.prompts.majority_vote")

    q_payload = str(question_raw or "").strip()
    cand_payload = json.dumps(answers, ensure_ascii=False)
    prompt = (f"{prompt_t}\n\n[题目]\n{q_payload}\n\n[候选答案]\n{cand_payload}\n").strip()

    resp = llm.generate_n(
        stage_name=stage_eval,
        question=prompt,
        prompt_mode="raw_prompt",
        n=1,
        temperature=0.0,
        sleep_s=sleep_s,
        stats=stats,
    )[0]
    txt = (resp or "").strip()

    def _safe_load_json(s: str) -> Dict[str, Any]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        end = s.rfind("}")
        if end < 0:
            return {}
        for i in range(end, -1, -1):
            if s[i] != "{":
                continue
            cand = s[i : end + 1].strip()
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return {}

    obj = _safe_load_json(txt)
    majority = str(obj.get("majority") or "")
    try:
        majority_count = int(obj.get("majority_count") or 0)
    except Exception:
        majority_count = 0
    normalized = obj.get("normalized")
    if not (isinstance(normalized, list) and len(normalized) == len(answers)):
        normalized = answers[:]
    normalized = [str(x or "") for x in normalized]

    raw_idxs = obj.get("majority_answer_idxs")
    keep: List[int] = []
    if isinstance(raw_idxs, list):
        for x in raw_idxs:
            if isinstance(x, int):
                keep.append(int(x))
            elif isinstance(x, str) and x.isdigit():
                keep.append(int(x))
    keep = [i for i in keep if 0 <= i < len(answers)]
    # de-dup, keep order
    if keep:
        seen = set()
        keep2: List[int] = []
        for i in keep:
            if i in seen:
                continue
            seen.add(i)
            keep2.append(i)
        keep = keep2

    return {
        "normalized": normalized,
        "majority": majority,
        "majority_count": int(majority_count),
        "majority_answer_idxs": keep,
        "raw_output": txt,
        "model_input": prompt,
    }


def infer_stage_from_raw(*, raw_input_path: str, out_stage_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> str:
    stage = _out_stage_from_dir(out_stage_dir)
    if not stage:
        raise ValueError("Missing stage name from --out directory")
    stage_solve = f"{stage}_solve"
    outp = os.path.join(out_stage_dir, "infer.jsonl")
    os.makedirs(out_stage_dir, exist_ok=True)

    done = _load_done_uuid_set(outp)
    pat = str(_input_opts(llm).get("raw_input_glob") or "*.jsonl")
    for row0 in _iter_raw_input_rows(input_path=raw_input_path, glob=pat):
        if not isinstance(row0, dict):
            continue
        canonical_keys = _get_canonical_keys(llm)
        row = normalize_record(row0, canonical_keys=canonical_keys)
        uuid = row.get("uuid")
        if uuid is None:
            continue
        uuid_key = str(uuid)
        if uuid_key in done:
            continue

        q_raw = _question_from_row(llm=llm, row=row)
        if not q_raw.strip():
            raise ValueError(f"Missing question: uuid={uuid}")
        gold = (row.get("answer") or "").strip() if isinstance(row.get("answer"), str) else ""
        model_q = normalize_for_model(q_raw)

        stats: Dict[str, int] = {}
        raws = llm.generate_n(stage_name=stage_solve, question=model_q, prompt_mode="problem", sleep_s=sleep_s, stats=stats)
        n = int(llm.stage_params(stage_solve).n)
        raws = [str(x) for x in raws][:n]
        extracted = _extract_answers(llm=llm, stage_solve=stage_solve, raw_outputs=raws)

        append_jsonl_line(
            outp,
            {
                "uuid": uuid,
                "line_number": row.get("line_number"),
                "stage": f"{stage}_infer",
                "question": q_raw,
                "answer": gold,
                "model_input": model_q,
                "raw_model_outputs": raws,
                "extracted_answers": extracted,
                "min_votes_to_accept": int(min_votes_to_accept),
                "llm_call_counts": {
                    stage_solve: int(n),
                    f"{stage_solve}_http_calls": int(stats.get("http_calls", 0)),
                    f"{stage_solve}_retries": int(stats.get("retries", 0)),
                    f"{stage_solve}_timeouts": int(stats.get("timeouts", 0)),
                    f"{stage_solve}_errors": int(stats.get("errors", 0)),
                },
                "raw_source_path": row.get("raw_source_path"),
            },
        )
        done.add(uuid_key)
    return outp


def infer_stage_from_prev(*, input_dir: str, out_stage_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> str:
    """
    Infer one stage by consuming a directory that contains infer+status (authoritative routing):
      <input_dir>/infer.jsonl
      <input_dir>/status.jsonl

    It selects uuids whose status indicates "not accepted" (needs another stage).
    Backward-compatible with older status schema that used `next_stage`.
    """
    stage = _out_stage_from_dir(out_stage_dir)
    if not stage:
        raise ValueError("Missing stage name from --out directory")
    if not _has_prev_artifacts(input_dir):
        raise ValueError(f"--input must point to a directory containing infer.jsonl + status.jsonl: {input_dir}")
    prev_infer = os.path.join(input_dir, "infer.jsonl")
    prev_status = os.path.join(input_dir, "status.jsonl")

    # Build status map
    status_map: Dict[str, Dict[str, Any]] = {}
    for st in iter_jsonl(prev_status, tolerate_errors=True):
        if isinstance(st, dict) and st.get("uuid") is not None:
            status_map[str(st["uuid"])] = st

    stage_solve = f"{stage}_solve"
    outp = os.path.join(out_stage_dir, "infer.jsonl")
    os.makedirs(out_stage_dir, exist_ok=True)
    # Ensure file exists even if nothing to do.
    try:
        with open(outp, "a", encoding="utf-8"):
            pass
    except Exception:
        pass

    done = _load_done_uuid_set(outp)
    for row in iter_jsonl(prev_infer, tolerate_errors=True):
        if not isinstance(row, dict):
            continue
        uuid = row.get("uuid")
        if uuid is None:
            continue
        uuid_key = str(uuid)
        if uuid_key in done:
            continue
        st = status_map.get(uuid_key, {})
        if not isinstance(st, dict) or not st:
            continue
        # Preferred schema: accepted: bool
        if "accepted" in st:
            if bool(st.get("accepted", False)):
                continue
        else:
            # Backward-compat schema: next_stage: str
            ns = st.get("next_stage")
            if isinstance(ns, str) and ns.strip():
                ns_s = ns.strip()
                # Old convention: "accepted" means stop.
                if ns_s == "accepted":
                    continue
                # Old convention: explicit routing by stage name.
                # If next_stage points elsewhere, skip this out stage.
                if ns_s not in ("no_answer",) and ns_s != str(stage):
                    continue
                # ns_s == stage OR ns_s == "no_answer" => continue to next stage (sequential).
            else:
                # Last resort: infer from vote counts.
                try:
                    maj_cnt = int(st.get("vote_majority_count") or 0)
                except Exception:
                    maj_cnt = 0
                try:
                    mv = int(st.get("min_votes_to_accept") or min_votes_to_accept)
                except Exception:
                    mv = int(min_votes_to_accept)
                if maj_cnt >= mv and str(st.get("vote_majority") or "").strip():
                    continue

        q_raw = _question_from_row(llm=llm, row=row)
        if not q_raw.strip():
            raise ValueError(f"Missing question: uuid={uuid}")
        gold = (row.get("answer") or "").strip() if isinstance(row.get("answer"), str) else ""
        model_q = normalize_for_model(q_raw)

        stats: Dict[str, int] = {}
        raws = llm.generate_n(stage_name=stage_solve, question=model_q, prompt_mode="problem", sleep_s=sleep_s, stats=stats)
        n = int(llm.stage_params(stage_solve).n)
        raws = [str(x) for x in raws][:n]
        extracted = _extract_answers(llm=llm, stage_solve=stage_solve, raw_outputs=raws)

        append_jsonl_line(
            outp,
            {
                "uuid": uuid,
                "line_number": row.get("line_number"),
                "stage": f"{stage}_infer",
                "question": q_raw,
                "answer": gold,
                "model_input": model_q,
                "raw_model_outputs": raws,
                "extracted_answers": extracted,
                "min_votes_to_accept": int(min_votes_to_accept),
                "llm_call_counts": {
                    stage_solve: int(n),
                    f"{stage_solve}_http_calls": int(stats.get("http_calls", 0)),
                    f"{stage_solve}_retries": int(stats.get("retries", 0)),
                    f"{stage_solve}_timeouts": int(stats.get("timeouts", 0)),
                    f"{stage_solve}_errors": int(stats.get("errors", 0)),
                },
                "raw_source_path": row.get("raw_source_path"),
            },
        )
        done.add(uuid_key)
    return outp


def eval_stage_dir(*, input_stage_dir: str, out_stage_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> str:
    stage = _out_stage_from_dir(out_stage_dir)
    if not stage:
        raise ValueError("Missing stage name from --out directory")
    stage_eval = f"{stage}_eval"
    infer_path = os.path.join(input_stage_dir, "infer.jsonl")
    status_path = os.path.join(out_stage_dir, "status.jsonl")
    os.makedirs(out_stage_dir, exist_ok=True)

    done = _load_done_uuid_set(status_path)

    for row in iter_jsonl(infer_path, tolerate_errors=True):
        if not isinstance(row, dict):
            continue
        uuid = row.get("uuid")
        if uuid is None:
            continue
        uuid_key = str(uuid)
        if uuid_key in done:
            continue

        q_raw = _question_from_row(llm=llm, row=row)
        extracted = row.get("extracted_answers") if isinstance(row.get("extracted_answers"), list) else []
        extracted_trim = [str(x or "").strip() for x in extracted]

        vote_stats: Dict[str, int] = {}
        vote = _vote_majority(
            llm=llm,
            stage_eval=stage_eval,
            question_raw=q_raw,
            candidates=extracted_trim,
            sleep_s=sleep_s,
            stats=vote_stats,
        )
        maj = str(vote.get("majority") or "").strip()
        maj_cnt = int(vote.get("majority_count") or 0)
        keep = vote.get("majority_answer_idxs") if isinstance(vote.get("majority_answer_idxs"), list) else []

        accepted = bool(maj and maj_cnt >= int(min_votes_to_accept))
        # Backward-compat field: keep next_stage string for older tooling.
        next_stage = "accepted" if accepted else "no_answer"

        append_jsonl_line(
            status_path,
            {
                "uuid": uuid,
                "stage": stage,
                "ok": int(maj_cnt),
                "bad": int(max(0, len(extracted_trim) - maj_cnt)),
                "min_votes_to_accept": int(min_votes_to_accept),
                "vote_majority": maj,
                "vote_majority_count": int(maj_cnt),
                "vote_raw_output": str(vote.get("raw_output") or ""),
                "vote_model_input": str(vote.get("model_input") or ""),
                "vote_candidates": extracted_trim,
                "vote_majority_answer_idxs": [int(x) for x in keep if isinstance(x, int)],
                "final_answer": maj if accepted else "",
                "final_source": "majority" if accepted else "no_majority",
                "final_vote_count": int(maj_cnt),
                # No fixed "next stage" name. Consumers should use accepted=false to decide whether to continue.
                "accepted": bool(accepted),
                "next_stage": next_stage,
                "paths": {"infer": infer_path},
                "llm_call_counts": {
                    **(row.get("llm_call_counts") if isinstance(row.get("llm_call_counts"), dict) else {}),
                    f"{stage}_vote_http_calls": int(vote_stats.get("http_calls", 0)),
                    f"{stage}_vote_retries": int(vote_stats.get("retries", 0)),
                    f"{stage}_vote_timeouts": int(vote_stats.get("timeouts", 0)),
                    f"{stage}_vote_errors": int(vote_stats.get("errors", 0)),
                },
            },
        )
        done.add(uuid_key)
    return status_path


def result_rebuild_run_dir(*, run_dir: str, min_votes_to_accept: int) -> str:
    run_dir = str(run_dir)
    out_dir = os.path.join(run_dir, "result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.stage_final.jsonl")
    done = _load_done_uuid_set(out_path)

    def _build_result_text(question: str, raw: str, pred: str) -> str:
        q = str(question or "").strip()
        r = str(raw or "").strip()
        p = str(pred or "").strip()
        return f"问题：{q}\n\n思考：{r}\n\n答案：{p}"

    for name in sorted(os.listdir(run_dir)):
        stage_dir = os.path.join(run_dir, name)
        if not os.path.isdir(stage_dir):
            continue
        infer_path = os.path.join(stage_dir, "infer.jsonl")
        status_path = os.path.join(stage_dir, "status.jsonl")
        if not (os.path.exists(infer_path) and os.path.exists(status_path)):
            continue

        status_map: Dict[str, Dict[str, Any]] = {}
        for st in iter_jsonl(status_path, tolerate_errors=True):
            if isinstance(st, dict) and st.get("uuid") is not None:
                status_map[str(st["uuid"])] = st

        for row in iter_jsonl(infer_path, tolerate_errors=True):
            if not isinstance(row, dict):
                continue
            uuid = row.get("uuid")
            if uuid is None:
                continue
            uuid_key = str(uuid)
            st = status_map.get(uuid_key, {})
            maj_cnt = int(st.get("vote_majority_count") or 0) if isinstance(st, dict) else 0
            if maj_cnt < int(min_votes_to_accept):
                continue
            keep = st.get("vote_majority_answer_idxs") if isinstance(st, dict) else None
            if not isinstance(keep, list) or not keep:
                continue
            raws = row.get("raw_model_outputs") if isinstance(row.get("raw_model_outputs"), list) else []
            extracted = row.get("extracted_answers") if isinstance(row.get("extracted_answers"), list) else []
            q = row.get("question") or row.get("prompt") or row.get("text") or ""
            for idx in keep:
                if not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= min(len(raws), len(extracted)):
                    continue
                out_uuid = f"{uuid_key}-{idx}"
                if out_uuid in done:
                    continue
                text = _build_result_text(str(q), str(raws[idx]), str(extracted[idx]))
                append_jsonl_line(out_path, {"uuid": out_uuid, "text": text})
                done.add(out_uuid)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="MathAgent pipeline v2 (infer/eval/result_rebuild).")
    p.add_argument("--mode", required=True, choices=["infer", "eval", "result_rebuild"])
    p.add_argument(
        "--input",
        required=True,
        help="For infer: a raw JSONL file/dir OR a directory containing infer.jsonl + status.jsonl. For eval: a stage directory containing infer.jsonl. For result_rebuild: the run_dir.",
    )
    p.add_argument(
        "--out",
        required=False,
        default="",
        help="Required for infer/eval: output stage directory. For result_rebuild: unused.",
    )
    p.add_argument("--llm-config", default="config/llm_models.json")
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    llm = LLMRouter(config_path=args.llm_config)
    min_votes_to_accept = llm.threshold_int("min_votes_to_accept", 5)
    in_opts = _input_opts(llm)

    # Only infer/eval need LLM connectivity.
    if args.mode in ("infer", "eval"):
        maybe_autostart_vllm(llm)
        if llm.option_bool("vllm_shutdown_on_exit", False):
            import atexit

            atexit.register(maybe_shutdown_vllm, llm)

    inp = str(args.input)
    if args.mode in ("infer", "eval"):
        out_dir = str(args.out or "").strip()
        if not out_dir:
            raise ValueError("--out must be provided for mode=infer/eval (output stage directory).")
        # Check for common issues: empty variable expansion (e.g., "$RUN_DIR/stage1" when RUN_DIR is unset)
        if out_dir.startswith("/") and len([p for p in out_dir.split(os.path.sep) if p]) <= 1:
            original_out = str(args.out or "").strip()
            raise ValueError(
                f"Invalid output directory: {out_dir}\n"
                f"  Original argument: --out '{original_out}'\n"
                f"  This looks like an unset environment variable (e.g., $RUN_DIR/stage1 when RUN_DIR is empty).\n"
                f"  Please ensure environment variables are set, or use a relative path like 'datasets/out/demo_3/stage1'."
            )
        # Normalize path: convert relative paths to absolute based on current working directory
        if not os.path.isabs(out_dir):
            out_dir = os.path.abspath(out_dir)
        # Safety check: prevent creating directories directly in root filesystem
        path_parts = [p for p in out_dir.split(os.path.sep) if p]
        if out_dir.startswith(os.path.sep) and len(path_parts) <= 1:
            raise ValueError(
                f"Invalid output directory: {out_dir}. "
                f"Cannot create directories directly in root filesystem. "
                f"Please use a relative path (e.g., 'datasets/out/demo_3/stage1') or an absolute path within your project directory."
            )
        if args.mode == "infer":
            # If input points to a directory with infer+status -> derive tasks from it.
            if _has_prev_artifacts(inp):
                outp = infer_stage_from_prev(
                    input_dir=inp,
                    out_stage_dir=out_dir,
                    llm=llm,
                    min_votes_to_accept=min_votes_to_accept,
                    sleep_s=float(args.sleep),
                )
            else:
                # Otherwise treat input as raw dataset (file/dir).
                outp = infer_stage_from_raw(
                    raw_input_path=inp,
                    out_stage_dir=out_dir,
                    llm=llm,
                    min_votes_to_accept=min_votes_to_accept,
                    sleep_s=float(args.sleep),
                )
        else:
            # Eval reads infer.jsonl from input stage dir and writes status.jsonl to out stage dir.
            if not os.path.isdir(inp):
                raise ValueError(f"--input must be a stage directory for mode=eval: {inp}")
            inf_file = os.path.join(inp, "infer.jsonl")
            if not os.path.exists(inf_file):
                raise ValueError(f"Missing {inf_file}. Run infer first.")
            outp = eval_stage_dir(
                input_stage_dir=inp,
                out_stage_dir=out_dir,
                llm=llm,
                min_votes_to_accept=min_votes_to_accept,
                sleep_s=float(args.sleep),
            )
        print(outp)
        return

    # result_rebuild
    run_dir = inp
    if not os.path.isdir(run_dir):
        raise ValueError(f"--input must be a run_dir for result_rebuild: {inp}")
    outp = result_rebuild_run_dir(run_dir=run_dir, min_votes_to_accept=min_votes_to_accept)
    print(outp)

