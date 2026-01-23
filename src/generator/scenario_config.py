from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class ScenarioStep:
    op: str
    input: str


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    steps: List[ScenarioStep]

    @staticmethod
    def load(path: str) -> "ScenarioConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Scenario config must be a JSON object: {path}")
        name = str(raw.get("name") or "").strip() or "unnamed"
        steps_raw = raw.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ValueError(f"Scenario config must have non-empty steps[]: {path}")
        steps: List[ScenarioStep] = []
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                raise ValueError(f"Scenario step[{i}] must be an object: {path}")
            op = str(s.get("op") or "").strip()
            inp = str(s.get("input") or "").strip()
            if not op:
                raise ValueError(f"Scenario step[{i}].op is required: {path}")
            if not inp:
                raise ValueError(f"Scenario step[{i}].input is required: {path}")
            steps.append(ScenarioStep(op=op, input=inp))
        return ScenarioConfig(name=name, steps=steps)


def render_template(s: str, *, vars: Mapping[str, str]) -> str:
    """
    Very small templating: replace ${VAR} with vars[VAR] if present.
    Unknown vars are left untouched.
    """
    out = str(s or "")
    for k, v in vars.items():
        out = out.replace("${" + str(k) + "}", str(v))
    return out

