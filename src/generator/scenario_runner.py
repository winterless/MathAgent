from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

from generator.scenario_config import ScenarioConfig, render_template


@dataclass(frozen=True)
class ScenarioContext:
    raw_input: str
    out_dir: str


RunOp = Callable[[str], List[str]]


def run_scenario(
    *,
    scenario_path: str,
    ctx: ScenarioContext,
    ops: Mapping[str, RunOp],
) -> List[str]:
    """
    Execute scenario steps sequentially.

    - scenario config describes the ordered list of ops (multi-round is just one scenario).
    - ops provides a generic "bottom" mapping from op-name -> callable(input_path)->outputs.
    """
    cfg = ScenarioConfig.load(scenario_path)
    vars = {
        "RAW_INPUT": ctx.raw_input,
        "OUT_DIR": ctx.out_dir,
    }

    outs: List[str] = []
    for step in cfg.steps:
        op = step.op.strip()
        if op not in ops:
            raise ValueError(f"Unknown scenario op: {op!r}. Available: {sorted(ops.keys())}")
        inp = render_template(step.input, vars=vars).strip()
        if not inp:
            raise ValueError(f"Scenario op {op!r} rendered empty input from: {step.input!r}")
        step_outs = ops[op](inp)
        outs.extend(step_outs or [])
    return outs

