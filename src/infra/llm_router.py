from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from infra.llm_client import LLMClient, LLMConfig


@dataclass(frozen=True)
class StageParams:
    n: int
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class LLMRouterConfig:
    """
    JSON config file schema:

    {
      "models": {
        "base": {"base_url": "...", "api_key": "", "model": "...", "timeout_s": 60},
        "think_fast": {...},
        "think_slow": {...}
      },
      "routes": {
        "default": "base",
        "stage1_solve": "think_fast",
        "stage1_eval": "think_slow",
        "stage2_eval": "base",
        "stage3_eval": "base"
      }
    }
    """

    models: Dict[str, LLMConfig]
    routes: Dict[str, str]
    stage_params: Dict[str, StageParams]
    thresholds: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "LLMRouterConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid llm config (expected object): {path}")
        raw_models = raw.get("models", {})
        raw_routes = raw.get("routes", {})
        raw_stage_params = raw.get("stage_params", {})
        raw_thresholds = raw.get("thresholds", {})
        if not isinstance(raw_models, dict):
            raise ValueError(f"Invalid llm config: models must be an object: {path}")
        if not isinstance(raw_routes, dict):
            raise ValueError(f"Invalid llm config: routes must be an object: {path}")
        if raw_stage_params is not None and not isinstance(raw_stage_params, dict):
            raise ValueError(f"Invalid llm config: stage_params must be an object: {path}")
        if raw_thresholds is not None and not isinstance(raw_thresholds, dict):
            raise ValueError(f"Invalid llm config: thresholds must be an object: {path}")
        models: Dict[str, LLMConfig] = {}
        for name, cfg in raw_models.items():
            if not isinstance(name, str):
                continue
            if not isinstance(cfg, dict):
                raise ValueError(f"Invalid llm config: models.{name} must be an object: {path}")
            models[name] = LLMConfig.from_dict(cfg)
        routes: Dict[str, str] = {}
        for k, v in raw_routes.items():
            if isinstance(k, str) and isinstance(v, str):
                routes[k] = v

        # stage_params
        stage_params: Dict[str, StageParams] = {}
        if isinstance(raw_stage_params, dict):
            for k, v in raw_stage_params.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                n = int(v.get("n") or 1)
                temperature = float(v.get("temperature") if v.get("temperature") is not None else 0.2)
                max_tokens = int(v.get("max_tokens") or 512)
                stage_params[k] = StageParams(n=n, temperature=temperature, max_tokens=max_tokens)

        thresholds: Dict[str, Any] = raw_thresholds if isinstance(raw_thresholds, dict) else {}

        return LLMRouterConfig(models=models, routes=routes, stage_params=stage_params, thresholds=thresholds)


class LLMRouter:
    """
    A thin facade that keeps *all model calls* in one place:
    pipeline uses router.generate_n(...) exactly like LLMClient.generate_n(...),
    but router selects which underlying LLMClient to use by stage_name.
    """

    def __init__(self, *, config_path: str = "config/llm_models.json") -> None:
        if not config_path:
            raise ValueError("LLM config path is empty. Provide --llm-config <path>.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"LLM config file not found: {config_path}")

        cfg = LLMRouterConfig.load(config_path)
        self._routes: Dict[str, str] = cfg.routes
        self._stage_params: Dict[str, StageParams] = cfg.stage_params
        self._thresholds: Dict[str, Any] = cfg.thresholds
        self._clients: Dict[str, LLMClient] = {}

        for name, llm_cfg in cfg.models.items():
            if not llm_cfg.base_url or not llm_cfg.model:
                raise ValueError(f"Invalid llm config: models.{name} must set base_url and model")
            self._clients[name] = LLMClient(config=llm_cfg)

        default_name = self._routes.get("default")
        if not default_name:
            raise ValueError("Invalid llm config: routes.default is required")
        if default_name not in self._clients:
            raise ValueError(f"Invalid llm config: routes.default={default_name} is not defined in models")

    def stage_params(self, stage_name: str) -> StageParams:
        """
        Get per-stage generation params.
        Falls back to stage_params.default, then hard defaults.
        """
        key = (stage_name or "").strip()
        p = self._stage_params.get(key) or self._stage_params.get("default")
        if p is not None:
            return p
        return StageParams(n=1, temperature=0.2, max_tokens=512)

    def threshold_int(self, name: str, default: int) -> int:
        v = self._thresholds.get(name)
        try:
            return int(v)
        except Exception:
            return default

    def _pick_client(self, stage_name: str) -> LLMClient:
        key = (stage_name or "").strip()
        model_name = self._routes.get(key) or self._routes.get("default") or ""
        if not model_name:
            raise RuntimeError(f"LLM route not found for stage={stage_name!r} and no routes.default provided")
        if model_name not in self._clients:
            raise RuntimeError(f"LLM route points to unknown model: stage={stage_name!r} -> {model_name!r}")
        return self._clients[model_name]

    def generate_n(
        self,
        *,
        stage_name: str,
        question: str,
        prompt_mode: str = "problem",
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        sleep_s: float = 0.0,
    ):
        p = self.stage_params(stage_name)
        client = self._pick_client(stage_name)
        return client.generate_n(
            stage_name=stage_name,
            question=question,
            prompt_mode=prompt_mode,
            n=p.n if n is None else int(n),
            temperature=p.temperature if temperature is None else float(temperature),
            max_tokens=p.max_tokens if max_tokens is None else int(max_tokens),
            sleep_s=sleep_s,
        )


