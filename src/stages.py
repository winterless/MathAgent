from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from jsonl_io import JsonDict
from llm_client import LLMClient
from voting import majority_vote

import re


@dataclass(frozen=True)
class StageConfig:
    name: str
    temperature: float
    samples: int = 8
    stable_threshold_n: int = 7  # stable if majority_count > n
    sleep_s: float = 0.0
    kind: str = "generic"  # "solve" | "eval"
    max_tokens: int = 256


_ABCD_RE = re.compile(r"(?<![A-Za-z0-9])[ABCD](?![A-Za-z0-9])")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED_ANS_RE = re.compile(r"\\boxed\s*\{\s*(?P<ans>[ABCD])\s*\}", re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(r"(?:最终答案|答案)\s*[:：]?\s*(?P<ans>[ABCD])", re.IGNORECASE)
_CHOICE_LINE_RE = re.compile(r"(?m)^\s*([ABCD])\s*[\.．、]\s*(.+?)\s*$")
_CHOICE_MAP_LINE_RE = re.compile(r"(?m)^\s*([ABCD])\s*=\s*(.+?)\s*$")

# Extra formatting guidance appended to the *model input* for eval stages only.
# This is NOT stored in the JSONL `prompt` field; it only helps the evaluator model
# produce sample.jsonl-like long-form content.
_EVAL_USER_FORMAT_SUFFIX = (
    "\n\n【输出格式（必须严格遵守；禁止输出JSON/代码块/额外说明；最后一行必须是boxed）】\n"
    "你必须输出与 sample.jsonl 类似的一段 Markdown 文本，且顺序固定：\n"
    "1) 开头一句：我们按照题目要求，逐条对比么个解答与**标准答案**是否一致。\n"
    "2) 空行 + ---\n"
    "3) ### 题目回顾：\n"
    "   题目是：\n"
    "   你必须把上面 prompt 中 [题目] 与 [题目] 之间的内容原样粘贴进引号中（包含所有换行与 A/B/C/D 选项），即：\n"
    "   “<这里原样粘贴完整题干+选项>”\n"
    "   标准答案是：**<标准答案>**\n"
    "   其中 <标准答案> 必须用 [GOLD_STANDARD_ANSWER] 与题目选项映射来标准化：\n"
    "   - 若 gold 是 A/B/C/D 且题干包含选项行（如 A.-5），则输出形如 A.-5 / B.2（字母 + '.' + 选项内容）。\n"
    "   - 若无法从题干提取选项内容，则直接输出 gold（可能是字母或数值/表达式）。\n"
    "4) 空行 + ---\n"
    "5) ###解答分析（逐个对比）：\n"
    "   必须输出8行：- **解答i**：<解答i> -> 与标准答案一致 ✅ / 与标准答案不一致 ❌\n"
    "6) 空行 + ---\n"
    "7) ### 判断汇总：\n"
    "   必须输出markdown表格（8行）：| 编号 | 是否正确 | 理由 |\n"
    "8) ### 统计：\n"
    "   - 解题正确数量：<编号列表> -> 共**x个**\n"
    "   - 解题错误数量：<编号列表> -> 共 **y**个\n"
    "9) 空行 + ---\n"
    "10) 最后一行必须且只能是：\\boxed{解答正确：x，解答错误：y}\n"
)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", (text or ""), flags=re.IGNORECASE).strip()

def _normalize_for_model(text: str) -> str:
    """
    Normalize common dataset escaping for *model input only*.
    - Convert literal '\\n' to real newlines
    - Unescape double backslashes once (e.g. '\\\\frac' -> '\\frac')
    - Unescape '\\$' -> '$'
    """
    s = text or ""
    if "\\n" in s:
        s = s.replace("\\n", "\n")
    # Unescape one level for common LaTeX-ish sequences
    if "\\\\" in s:
        s = s.replace("\\\\", "\\")
    if "\\$" in s:
        s = s.replace("\\$", "$")
    return s


def _append_choice_map_if_any(question: str) -> str:
    """
    If question contains A/B/C/D options, append a normalized option map to reduce model confusion.
    """
    s = question or ""
    found = {}
    for m in _CHOICE_LINE_RE.finditer(s):
        key = m.group(1).upper()
        val = m.group(2).strip()
        # Avoid overwriting if repeated
        if key not in found and val:
            found[key] = val
    if len(found) >= 2:
        lines = [s.rstrip(), "", "选项映射（用于确认 A/B/C/D 含义）："]
        for k in ("A", "B", "C", "D"):
            if k in found:
                lines.append(f"{k} = {found[k]}")
        return "\n".join(lines) + "\n"
    return s


def _extract_choice_map(text: str) -> dict[str, str]:
    """
    Extract mapping like {A: -5, B: 5} from either:
    - 'A = -5' (our appended map)
    - 'A.-5' / 'A. -5'
    """
    s = text or ""
    out: dict[str, str] = {}
    for m in _CHOICE_MAP_LINE_RE.finditer(s):
        k = m.group(1).upper()
        v = m.group(2).strip()
        if k not in out and v:
            out[k] = v
    for m in _CHOICE_LINE_RE.finditer(s):
        k = m.group(1).upper()
        v = m.group(2).strip()
        if k not in out and v:
            out[k] = v
    return out


def _try_parse_number(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    ss = s.strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", ss):
        try:
            return float(ss)
        except Exception:
            return None
    return None


def _standardize_choice_answer(extracted: str, *, choice_map: dict[str, str]) -> str:
    """
    Convert extracted answer into 'A.-5' style when possible.
    - If extracted is letter and map exists => 'A.<value>'
    - If extracted is numeric and matches mapped numeric value => '<letter>.<value>'
    Otherwise keep extracted as-is.
    """
    if not extracted:
        return extracted
    ex = extracted.strip()
    if ex in ("A", "B", "C", "D") and ex in choice_map:
        return f"{ex}.{choice_map[ex]}"
    ex_num = _try_parse_number(ex)
    if ex_num is not None:
        for k, v in choice_map.items():
            v_num = _try_parse_number(v)
            if v_num is not None and abs(v_num - ex_num) <= 1e-9:
                return f"{k}.{v}"
    return extracted


def _extract_final_answer(text: str) -> str:
    """
    Extract a "final answer only" from messy solver output.
    Strategy (robust against the model echoing options A/B/C/D inside the question):
    - Prefer boxed answer like \\boxed{A}.
    - Else prefer explicit answer line like "答案：A" / "最终答案A".
    - Else scan the tail of the output for A/B/C/D (standalone).
    - Else scan the tail for a number.
    - Else last non-empty line.
    """
    s = _strip_think(text)
    if not s:
        return ""

    m_box = _BOXED_ANS_RE.search(s)
    if m_box:
        return m_box.group("ans").upper()

    m_line = _ANSWER_LINE_RE.search(s)
    if m_line:
        return m_line.group("ans").upper()

    # Prefer searching near the end; models often repeat the question/options earlier.
    tail = s[-300:]
    m = list(_ABCD_RE.finditer(tail))
    if m:
        return m[-1].group(0).upper()

    nums = list(_NUM_RE.finditer(tail))
    if nums:
        return nums[-1].group(0)

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    last = lines[-1] if lines else s.strip()
    # If last line is long, still try a last-pass tail scan on that line.
    m2 = list(_ABCD_RE.finditer(last))
    if m2:
        return m2[-1].group(0).upper()
    nums2 = list(_NUM_RE.finditer(last))
    if nums2:
        return nums2[-1].group(0)
    return last


def output_is_stable(output: JsonDict) -> bool:
    """
    Routing condition:
    - Prefer output.content.debug.stable == True (our pipeline-produced outputs)
    - Otherwise, treat as unstable (so the next stage will run)
    """
    if not isinstance(output, dict):
        return False
    content = output.get("content")
    if not isinstance(content, dict):
        return False
    debug = content.get("debug")
    if isinstance(debug, dict) and debug.get("stable") is True:
        return True
    # Some upstream producers may put stable directly under content
    if content.get("stable") is True:
        return True
    return False


_BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*解答正确\s*：\s*(?P<ok>\d+)\s*[，,]\s*解答错误\s*：\s*(?P<bad>\d+)\s*\}",
    re.MULTILINE,
)


def extract_boxed_counts(text: str) -> tuple[int, int] | None:
    """
    Parse '\\boxed{解答正确：x，解答错误：y}' from an evaluation output.
    Returns (x,y) or None if not found.
    """
    if not isinstance(text, str):
        return None
    m = _BOXED_RE.search(text)
    if not m:
        return None
    return int(m.group("ok")), int(m.group("bad"))


def pick_task_text(row: JsonDict) -> tuple[str, str]:
    """
    Pick the text we send to the model and which mode to use.

    Prefer `prompt` (common in datasets that already include the full instruction / standard prompt),
    otherwise fallback to `question`, then `text`.
    """
    for k in ("prompt", "question", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            mode = "raw_prompt" if k == "prompt" else "problem"
            return v, mode
    raise ValueError(f"Row missing task text (need one of prompt/question/text): {row}")


def run_stage(
    *,
    llm: LLMClient,
    stage: StageConfig,
    rows: List[JsonDict],
) -> List[JsonDict]:
    """
    Run one stage on all rows.

    Input row must contain at least one of: prompt/question/text.
    This stage overwrites the row's `output` with a sample-like wrapper.
    The next stage decides whether to run by inspecting `output.content.debug.stable`.
    """
    out: List[JsonDict] = []
    for r in rows:
        q, mode = pick_task_text(r)
        # Normalize for model input (keep stored JSON prompt unchanged).
        if isinstance(q, str):
            q = _normalize_for_model(q)
        # For evaluation stages we want a strict system message even if we're sending a raw assembled prompt.
        if (stage.kind or "generic") == "eval" and mode == "raw_prompt":
            mode = "raw_prompt_eval"
        # Model needs gold for eval, but stored prompt may have empty [标准解答].
        # Pass gold via an internal field appended at runtime (not persisted).
        model_q = q
        if (stage.kind or "generic") == "solve":
            model_q = _append_choice_map_if_any(model_q)
        if (stage.kind or "generic") == "eval":
            gold = r.get("answer")
            if isinstance(gold, str) and gold.strip():
                # Provide gold in a machine-readable way while keeping stored JSON prompt unchanged.
                # Combine:
                # - stored prompt (question + answers)
                # - strict formatting suffix (not persisted)
                # - gold (machine-readable, last)
                model_q = f"{q.rstrip()}{_EVAL_USER_FORMAT_SUFFIX}\n\n[GOLD_STANDARD_ANSWER]={gold.strip()}\n"
        candidates = llm.generate_n(
            stage_name=stage.name,
            question=model_q,
            prompt_mode=mode,
            n=stage.samples,
            temperature=stage.temperature,
            max_tokens=stage.max_tokens,
            sleep_s=stage.sleep_s,
        )

        raw_candidates = list(candidates)
        if (stage.kind or "generic") == "solve":
            extracted = [_extract_final_answer(str(x)) for x in candidates]
            choice_map = _extract_choice_map(model_q)
            candidates = [_standardize_choice_answer(a, choice_map=choice_map) for a in extracted]
        else:
            candidates = [_strip_think(str(x)) for x in candidates]

        if (stage.kind or "generic") == "eval":
            # Stability is based on boxed counts agreement.
            parsed = [extract_boxed_counts(x) for x in candidates]
            # Use a canonical token for voting; None is treated as a unique failure.
            canon = [str(p) if p is not None else "__PARSE_FAIL__" for p in parsed]
            vr = majority_vote(canon)
            # If we can't parse boxed counts, treat as unstable (force escalation / avoid false positives).
            stable = vr.majority != "__PARSE_FAIL__" and (vr.majority_count >= stage.stable_threshold_n)
            # Pick a representative full text whose parsed tuple matches the majority.
            maj_tuple = None
            if vr.majority != "__PARSE_FAIL__":
                m = re.match(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$", vr.majority)
                if m:
                    maj_tuple = (int(m.group(1)), int(m.group(2)))
            majority_text = ""
            if maj_tuple is not None:
                for t, p in zip(candidates, parsed):
                    if p == maj_tuple:
                        majority_text = t
                        break
            if not majority_text:
                majority_text = candidates[0] if candidates else ""
            majority_payload = majority_text
            debug_payload = {
                "model_input": model_q,
                "raw_candidates": raw_candidates,
                "candidates": candidates,
                "parsed_boxed": parsed,
                "majority_key": vr.majority,
                "majority_count": vr.majority_count,
                "samples": stage.samples,
                "stable": stable,
            }
        else:
            vr = majority_vote(candidates)
            stable = vr.majority_count >= stage.stable_threshold_n
            majority_payload = vr.majority
            debug_payload = {
                "model_input": model_q,
                "raw_candidates": raw_candidates,
                "candidates": candidates,
                "majority_count": vr.majority_count,
                "samples": stage.samples,
                "stable": stable,
            }

        # Sample-like wrapper (align to datasets/sample.jsonl "output" shape).
        # Unknown fields are mocked with simple defaults.
        stage_output: JsonDict = {
            "status": "SUCCESS",
            "content": {
                "id": f"endpoint_{stage.name}_{r.get('uuid', r.get('id', 'unknown'))}",
                "object": 0,  # mock
                "model": llm.config.model or "",
                "choices": [
                    {
                        # sample.jsonl uses a misspelled key "indext"; keep it for compatibility
                        "indext": 0,
                        "message": {
                            "role": "assistant",
                            "content": majority_payload,
                            "tool_calls": None,  # mock
                        },
                        "logprobs": None,  # mock
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "prefill_time": 0,
                # Extra debug info kept under content, but doesn't affect the standard wrapper fields.
                "debug": debug_payload,
            },
        }
        out.append(
            {
                **r,
                "stage": stage.name,
                "output": stage_output,
            }
        )
    return out


def stage2_split(stage2_rows: List[JsonDict], *, stage2_name: str = "stage2") -> Tuple[List[JsonDict], List[JsonDict]]:
    """Returns: (stable_rows, to_stage3_rows)"""
    stable: List[JsonDict] = []
    to3: List[JsonDict] = []
    for r in stage2_rows:
        ok = output_is_stable(r.get("output", {}))
        if ok:
            stable.append({**r, "decision": "final_from_stage2"})
        else:
            to3.append({**r, "decision": "to_stage3"})
    return stable, to3


def stage3_split(stage3_rows: List[JsonDict], *, stage3_name: str = "stage3") -> Tuple[List[JsonDict], List[JsonDict]]:
    """Returns: (stable_rows, discarded_rows)"""
    stable: List[JsonDict] = []
    disc: List[JsonDict] = []
    for r in stage3_rows:
        ok = output_is_stable(r.get("output", {}))
        if ok:
            stable.append({**r, "decision": "final_from_stage3"})
        else:
            disc.append({**r, "decision": "discarded_unstable"})
    return stable, disc


def to_final_rows(rows: List[JsonDict]) -> List[JsonDict]:
    """
    Convert stage rows into final storage rows with minimal identifiers.
    """
    out: List[JsonDict] = []
    for r in rows:
        ans = ""
        stage_out = r.get("output")
        if isinstance(stage_out, dict):
            content = stage_out.get("content")
            if isinstance(content, dict):
                choices = content.get("choices")
                if isinstance(choices, list) and choices:
                    msg = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else None
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        ans = msg["content"]
        out.append(
            {
                "uuid": r.get("uuid"),
                "id": r.get("id"),
                "line_number": r.get("line_number"),
                "answer": ans,
                "source_decision": r.get("decision", ""),
                "stage": r.get("stage", ""),
            }
        )
    return out


