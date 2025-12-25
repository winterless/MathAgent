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
    options: Dict[str, Any]

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
        raw_options = raw.get("options", {})
        if not isinstance(raw_models, dict):
            raise ValueError(f"Invalid llm config: models must be an object: {path}")
        if not isinstance(raw_routes, dict):
            raise ValueError(f"Invalid llm config: routes must be an object: {path}")
        if raw_stage_params is not None and not isinstance(raw_stage_params, dict):
            raise ValueError(f"Invalid llm config: stage_params must be an object: {path}")
        if raw_thresholds is not None and not isinstance(raw_thresholds, dict):
            raise ValueError(f"Invalid llm config: thresholds must be an object: {path}")
        if raw_options is not None and not isinstance(raw_options, dict):
            raise ValueError(f"Invalid llm config: options must be an object: {path}")
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
        options: Dict[str, Any] = raw_options if isinstance(raw_options, dict) else {}

        return LLMRouterConfig(models=models, routes=routes, stage_params=stage_params, thresholds=thresholds, options=options)


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
        self._options: Dict[str, Any] = cfg.options
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

    def option_bool(self, name: str, default: bool) -> bool:
        v = self._options.get(name)
        if v is None:
            return bool(default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "y", "on", "enable", "enabled"):
                return True
            if s in ("0", "false", "no", "n", "off", "disable", "disabled"):
                return False
        return bool(default)

    def option_str(self, name: str, default: str) -> str:
        v = self._options.get(name)
        if v is None:
            return str(default)
        if isinstance(v, str):
            return v
        return str(v)

    def option_int(self, name: str, default: int) -> int:
        v = self._options.get(name)
        try:
            return int(v)
        except Exception:
            return int(default)

    def think_tag_for_stage(self, stage_name: str) -> str:
        """
        Returns a prefix tag to inject into user content for some models (e.g. Qwen /think or /no_think).
        Config keys:
          - options.think_tag_default: string
          - options.think_tag_by_stage: { "<stage_name>": "<tag>" }
        """
        by_stage = self._options.get("think_tag_by_stage")
        if isinstance(by_stage, dict):
            v = by_stage.get((stage_name or "").strip())
            if isinstance(v, str):
                return v
        return self.option_str("think_tag_default", "")

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
        stats: Optional[Dict[str, int]] = None,
        finish_early: Optional[bool] = None,
        think_tag: Optional[str] = None,
        debug_print_prompts: Optional[bool] = None,
        debug_print_prompts_max_chars: Optional[int] = None,
        debug_print_outputs: Optional[bool] = None,
        debug_print_outputs_max_chars: Optional[int] = None,
        debug_stream_outputs: Optional[bool] = None,
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
            stats=stats,
            finish_early=self.option_bool("finish_early", True) if finish_early is None else bool(finish_early),
            think_tag=self.think_tag_for_stage(stage_name) if think_tag is None else str(think_tag),
            debug_print_prompts=self.option_bool("debug_print_prompts", False)
            if debug_print_prompts is None
            else bool(debug_print_prompts),
            debug_print_prompts_max_chars=self.option_int("debug_print_prompts_max_chars", 4000)
            if debug_print_prompts_max_chars is None
            else int(debug_print_prompts_max_chars),
            debug_print_outputs=self.option_bool("debug_print_outputs", False)
            if debug_print_outputs is None
            else bool(debug_print_outputs),
            debug_print_outputs_max_chars=self.option_int("debug_print_outputs_max_chars", 2000)
            if debug_print_outputs_max_chars is None
            else int(debug_print_outputs_max_chars),
            debug_stream_outputs=self.option_bool("debug_stream_outputs", False)
            if debug_stream_outputs is None
            else bool(debug_stream_outputs),
        )


