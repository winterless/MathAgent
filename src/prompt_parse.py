from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


JsonDict = Dict[str, Any]


_ANSWER_BLOCK_RE = re.compile(
    r"\[解答(?P<idx>[1-8])\]\s*\n(?P<ans>.*?)\n\s*\[解答(?P=idx)\]",
    re.DOTALL,
)


def extract_stage1_raw_answers(prompt: str) -> List[Optional[str]]:
    """
    Extract the 8 raw answers from the assembled prompt section:
      ###\n[解答1]\n...\n[解答1]\n###\n ... up to 解答8

    Returns a list of length 8, positions 0..7 for answers 1..8.
    Missing ones are None.
    """
    # Some datasets store literal "\\n" characters instead of real newlines.
    s = (prompt or "").replace("\\n", "\n")
    out: List[Optional[str]] = [None] * 8
    for m in _ANSWER_BLOCK_RE.finditer(s):
        idx = int(m.group("idx"))
        ans = (m.group("ans") or "").strip()
        out[idx - 1] = ans if ans else None
    return out


def extract_question_block(prompt: str) -> Optional[str]:
    """
    Try to extract the [题目] block for quick inspection/debug. Not required for routing.
    """
    if not prompt:
        return None
    s = prompt.replace("\\n", "\n")
    m = re.search(r"\[题目\]\s*\n(?P<body>.*?)(\n###|\Z)", s, flags=re.DOTALL)
    if not m:
        return None
    body = (m.group("body") or "").strip()
    return body or None


