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
    majority, majority_count = c.most_common(1)[0]
    return VoteResult(majority=majority, majority_count=majority_count, counts=dict(c))



