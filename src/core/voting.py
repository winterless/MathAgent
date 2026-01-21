from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


def normalize_answer(s: str) -> str:
    # Minimal normalization for majority voting
    return " ".join((s or "").strip().split())


@dataclass(frozen=True)
class VoteResult:
    majority: str
    majority_count: int
    counts: Dict[str, int]


def majority_vote(answers: List[str]) -> VoteResult:
    if not answers:
        return VoteResult(majority="", majority_count=0, counts={})
    normalized = [normalize_answer(a) for a in answers]
    c = Counter(normalized)
    # Deterministic tie-break: if multiple answers share the max count,
    # pick the one that appears first in the input list.
    max_cnt = max(c.values()) if c else 0
    majority = ""
    for x in normalized:
        if c.get(x, 0) == max_cnt:
            majority = x
            break
    return VoteResult(majority=majority, majority_count=int(max_cnt), counts=dict(c))



