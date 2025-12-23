from __future__ import annotations

import re
from typing import Any, Dict

JsonDict = Dict[str, Any]

_ABCD_RE = re.compile(r"(?<![A-Za-z0-9])[ABCD](?![A-Za-z0-9])")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED_ANS_RE = re.compile(r"\\boxed\s*\{\s*(?P<ans>[ABCD])\s*\}", re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(r"(?:最终答案|答案)\s*[:：]?\s*(?P<ans>[ABCD])", re.IGNORECASE)
_FINAL_LINE_RE = re.compile(r"(?mi)^\s*FINAL\s*[:：]\s*(?P<ans>.+?)\s*$")
_ANSWER_ANY_LINE_RE = re.compile(r"(?mi)^\s*(?:最终答案|答案)\s*[:：]?\s*(?P<ans>.+?)\s*$")
_CHOICE_LINE_RE = re.compile(r"(?m)^\s*([ABCD])\s*[\.．、]\s*(.+?)\s*$")
_CHOICE_MAP_LINE_RE = re.compile(r"(?m)^\s*([ABCD])\s*=\s*(.+?)\s*$")

_BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*解答正确\s*：\s*(?P<ok>\d+)\s*[，,]\s*解答错误\s*：\s*(?P<bad>\d+)\s*\}",
    re.MULTILINE,
)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>[\s\S]*?</think>", "", (text or ""), flags=re.IGNORECASE).strip()


def normalize_for_model(text: str) -> str:
    """
    Normalize common dataset escaping for *model input only*:
    - Convert literal '\\n' to real newlines
    - Unescape one level of backslashes
    - Unescape '\\$' -> '$'
    """
    s = text or ""
    if "\\n" in s:
        s = s.replace("\\n", "\n")
    if "\\\\" in s:
        s = s.replace("\\\\", "\\")
    if "\\$" in s:
        s = s.replace("\\$", "$")
    return s


def append_choice_map_if_any(question: str) -> str:
    """Append a normalized option map to reduce model confusion for MCQ."""
    s = question or ""
    found: Dict[str, str] = {}
    for m in _CHOICE_LINE_RE.finditer(s):
        key = m.group(1).upper()
        val = m.group(2).strip()
        if key not in found and val:
            found[key] = val
    if len(found) >= 2:
        lines = [s.rstrip(), "", "选项映射（用于确认 A/B/C/D 含义）："]
        for k in ("A", "B", "C", "D"):
            if k in found:
                lines.append(f"{k} = {found[k]}")
        return "\n".join(lines) + "\n"
    return s


def extract_choice_map(text: str) -> Dict[str, str]:
    """Extract mapping like {A: -5, B: 5} from either 'A = -5' or 'A.-5'."""
    s = text or ""
    out: Dict[str, str] = {}
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


def extract_final_answer(text: str) -> str:
    """
    Extract a "final answer only" from solver output.
    Prefers boxed/explicit answer lines, then scans tail for A/B/C/D or a number.
    """
    s = strip_think(text)
    if not s:
        return ""

    m_box = _BOXED_ANS_RE.search(s)
    if m_box:
        return m_box.group("ans").upper()

    finals = list(_FINAL_LINE_RE.finditer(s))
    if finals:
        return (finals[-1].group("ans") or "").strip()

    m_line = _ANSWER_LINE_RE.search(s)
    if m_line:
        return m_line.group("ans").upper()

    ans_lines = list(_ANSWER_ANY_LINE_RE.finditer(s))
    if ans_lines:
        return (ans_lines[-1].group("ans") or "").strip()

    tail = s[-300:]
    m = list(_ABCD_RE.finditer(tail))
    if m:
        return m[-1].group(0).upper()

    nums = list(_NUM_RE.finditer(tail))
    if nums:
        return nums[-1].group(0)

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else s.strip()


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


def standardize_choice_answer(extracted: str, *, choice_map: Dict[str, str]) -> str:
    """Convert extracted answer into 'A.-5' style when possible."""
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


def extract_boxed_counts(text: str) -> tuple[int, int] | None:
    """Parse '\\boxed{解答正确：x，解答错误：y}' from an evaluation output text."""
    if not isinstance(text, str):
        return None
    m = _BOXED_RE.search(text)
    if not m:
        return None
    return int(m.group("ok")), int(m.group("bad"))


def extract_boxed_counts_from_output(output: Any) -> tuple[int, int] | None:
    """Extract boxed counts from a sample-shaped output wrapper."""
    if not isinstance(output, dict):
        return None
    content = output.get("content")
    if not isinstance(content, dict):
        return None
    choices = content.get("choices")
    if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
        return None
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return None
    txt = msg.get("content")
    return extract_boxed_counts(txt) if isinstance(txt, str) else None



