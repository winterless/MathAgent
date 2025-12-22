from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from jsonl_io import JsonDict
from llm_client import LLMClient
from voting import majority_vote


@dataclass(frozen=True)
class StageConfig:
    name: str
    temperature: float
    samples: int = 8
    stable_threshold_n: int = 7  # stable if majority_count > n
    sleep_s: float = 0.0


def run_stage(
    *,
    llm: LLMClient,
    stage: StageConfig,
    rows: List[JsonDict],
) -> List[JsonDict]:
    """
    Input rows must include at least:
      - id (str/int)
      - question (str)
    Output rows include:
      - answers (list[str])
      - majority (str)
      - majority_count (int)
      - stable (bool): majority_count > n
    """
    out: List[JsonDict] = []
    for r in rows:
        q = r.get("question")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Row missing 'question': {r}")
        answers = llm.generate_n(
            stage_name=stage.name,
            question=q,
            n=stage.samples,
            temperature=stage.temperature,
            sleep_s=stage.sleep_s,
        )
        vr = majority_vote(answers)
        stable = vr.majority_count > stage.stable_threshold_n
        out.append(
            {
                **r,
                f"{stage.name}_answers": answers,
                f"{stage.name}_majority": vr.majority,
                f"{stage.name}_majority_count": vr.majority_count,
                f"{stage.name}_stable": stable,
            }
        )
    return out


def stage1_decide_easy(stage1_rows: List[JsonDict], *, stage1_name: str = "stage1") -> Tuple[List[JsonDict], List[JsonDict]]:
    """
    Returns: (easy_end_rows, to_stage2_rows)
    easy_end if stage1_stable is True (i.e. majority_count > n)
    """
    easy: List[JsonDict] = []
    to2: List[JsonDict] = []
    for r in stage1_rows:
        stable = r.get(f"{stage1_name}_stable")
        if stable is True:
            easy.append({**r, "decision": "easy_end"})
        else:
            to2.append({**r, "decision": "to_stage2"})
    return easy, to2


def stage2_split(stage2_rows: List[JsonDict], *, stage2_name: str = "stage2") -> Tuple[List[JsonDict], List[JsonDict]]:
    """Returns: (stable_rows, to_stage3_rows)"""
    stable: List[JsonDict] = []
    to3: List[JsonDict] = []
    for r in stage2_rows:
        ok = r.get(f"{stage2_name}_stable") is True
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
        ok = r.get(f"{stage3_name}_stable") is True
        if ok:
            stable.append({**r, "decision": "final_from_stage3"})
        else:
            disc.append({**r, "decision": "discarded_unstable"})
    return stable, disc


def to_final_rows(rows: List[JsonDict], *, majority_field: str) -> List[JsonDict]:
    """
    Convert stage rows into final storage rows: {id, question, answer, source_decision}
    """
    out: List[JsonDict] = []
    for r in rows:
        out.append(
            {
                "id": r.get("id"),
                "question": r.get("question"),
                "answer": r.get(majority_field, ""),
                "source_decision": r.get("decision", ""),
            }
        )
    return out


