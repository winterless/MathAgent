from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from infra.llm_client import LLMClient, LLMConfig, RecoveryConfig


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
        "local_script": {"py_script": "/abs/path/to/runner.py", "model": "optional_name", "timeout_s": 300},
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
        # Be robust to occasional non-UTF8 bytes in config files.
        # This prevents crashes like:
        #   'utf-8' codec can't decode byte ...: invalid continuation byte
        with open(path, "rb") as f:
            b = f.read()
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError as e:
            s = b.decode("utf-8", errors="replace")
            print(
                f"[WARN] Non-UTF8 bytes in LLM config {path}: {e}. Decoded with replacement.",
                file=sys.stderr,
                flush=True,
            )
        # tolerate UTF-8 BOM
        if s.startswith("\ufeff"):
            s = s.lstrip("\ufeff")
        raw = json.loads(s)
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
            # A model can be backed either by HTTP (base_url+model) or by a local python script (py_script).
            if (llm_cfg.py_script or "").strip():
                # model name is optional for py_script runner; keep it for logging / downstream usage.
                self._clients[name] = LLMClient(config=llm_cfg)
                continue
            if not llm_cfg.base_url or not llm_cfg.model:
                raise ValueError(
                    f"Invalid llm config: models.{name} must set either (base_url and model) or py_script"
                )
            self._clients[name] = LLMClient(config=llm_cfg)

        default_name = self._routes.get("default")
        if not default_name:
            raise ValueError("Invalid llm config: routes.default is required")
        if default_name not in self._clients:
            raise ValueError(f"Invalid llm config: routes.default={default_name} is not defined in models")

        self._apply_recovery_config()

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

    def option_any(self, name: str, default: Any) -> Any:
        """
        Return the raw option value as-is (supports list/dict from JSON config).
        """
        v = self._options.get(name)
        return default if v is None else v

    def override_options(self, updates: Dict[str, Any]) -> None:
        """
        Override options at runtime (e.g., CLI overrides).
        Re-applies recovery config after updates.
        """
        if not updates:
            return
        for k, v in updates.items():
            if v is None:
                continue
            self._options[k] = v
        self._apply_recovery_config()

    def _apply_recovery_config(self) -> None:
        """
        Build recovery settings from options and apply to all clients.
        This avoids env var dependency.
        """
        start_cmd = self.vllm_start_cmd_resolved()
        stop_cmd = str(self._options.get("vllm_stop_cmd") or "").strip()
        restart_cmd = str(self._options.get("vllm_restart_cmd") or "").strip()
        restart_delay_s = float(self.option_int("vllm_restart_delay_s", 0))
        if not restart_cmd:
            if stop_cmd and start_cmd:
                if restart_delay_s > 0:
                    restart_cmd = f"{stop_cmd}; sleep {restart_delay_s:.1f}; {start_cmd}"
                else:
                    restart_cmd = f"{stop_cmd}; {start_cmd}"
            else:
                restart_cmd = start_cmd

        wait_s = self.option_int("vllm_wait_s", 0)
        health_url = self.vllm_health_url_resolved()
        log_path = str(self._options.get("vllm_log_path") or "").strip() or "/tmp/mathagent_vllm.log"
        log_to_stderr = self.option_bool("vllm_log_to_stderr", True)
        pre_restart_nvidia_smi = self.option_bool("vllm_pre_restart_nvidia_smi", True)
        gpu_reset_on_restart = self.option_bool("vllm_gpu_reset_on_restart", True)
        gpu_reset_ids = str(self._options.get("vllm_gpu_reset_ids") or "all").strip()
        gpu_reset_cmd = str(self._options.get("vllm_gpu_reset_cmd") or "").strip()

        cfg = RecoveryConfig(
            wait_on_connrefused_s=float(wait_s),
            health_url=health_url,
            restart_cmd=restart_cmd,
            connrefused_log=self.option_bool("vllm_connrefused_log", True),
            restart_on_connrefused=self.option_bool("vllm_restart_on_connrefused", False),
            restart_on_runtime_error=self.option_bool("vllm_restart_on_runtime_error", False),
            restart_cooldown_s=float(self.option_int("vllm_restart_cooldown_s", 30)),
            start_cmd=start_cmd,
            restart_fallback_to_start=self.option_bool("vllm_restart_fallback_to_start", True),
            log_path=log_path,
            log_to_stderr=log_to_stderr,
            pre_restart_nvidia_smi=pre_restart_nvidia_smi,
            gpu_reset_on_restart=gpu_reset_on_restart,
            gpu_reset_ids=gpu_reset_ids,
            gpu_reset_cmd=gpu_reset_cmd,
        )
        for c in self._clients.values():
            c.set_recovery_config(cfg)

    def _render_vllm_cmd(self, cmd: str, model_path: str, model_name: str) -> str:
        """
        If placeholders exist, replace them. Otherwise, append model args when provided.
        Supported placeholders:
          - {model_path}
          - {model_name}
        """
        out = str(cmd or "").strip()
        if not out:
            return ""
        has_path = "{model_path}" in out
        has_name = "{model_name}" in out
        if has_path or has_name:
            return out.replace("{model_path}", model_path).replace("{model_name}", model_name)
        # No placeholders: append model args if provided.
        parts = [out]
        if model_path:
            parts.append(f'--model "{model_path}"')
        if model_name:
            parts.append(f"--served-model-name {model_name}")
        return " ".join(parts)

    def vllm_start_cmd_resolved(self) -> str:
        cmd = str(self._options.get("vllm_start_cmd") or "").strip()
        model_path = str(self._options.get("vllm_model_path") or "").strip()
        model_name = str(self._options.get("vllm_model_name") or "").strip()
        return self._render_vllm_cmd(cmd, model_path, model_name)

    def vllm_health_url_resolved(self) -> str:
        health_url = str(self._options.get("vllm_health_url") or "").strip()
        if health_url:
            return health_url
        base = self.default_base_url().strip()
        if base:
            return base.rstrip("/") + "/v1/models"
        return ""

    def default_base_url(self) -> str:
        """
        Return base_url of the default routed model if available (empty string otherwise).
        """
        model_key = self._routes.get("default")
        if not model_key:
            return ""
        client = self._clients.get(model_key)
        if client is None:
            return ""
        return str(client.config.base_url or "")

    def think_tag_for_stage(self, stage_name: str) -> str:
        """
        Returns a prefix tag to inject into user content for some models (e.g. Qwen /think or /no_think).
        Config keys:
          - options.think_tag_default: string
          - options.think_tag_by_stage: { "<stage_name>": "<tag>" }
          - options.think_tag_by_profile: { "<model_key>": "<tag>" }  # model_key is the routed profile, e.g. think_fast/think_slow/base
        """
        by_stage = self._options.get("think_tag_by_stage")
        if isinstance(by_stage, dict):
            v = by_stage.get((stage_name or "").strip())
            if isinstance(v, str):
                return v

        # If not overridden by stage, infer tag by routed profile (model key).
        by_profile = self._options.get("think_tag_by_profile")
        if isinstance(by_profile, dict):
            key = (stage_name or "").strip()
            model_key = self._routes.get(key) or self._routes.get("default") or ""
            if isinstance(model_key, str) and model_key:
                v2 = by_profile.get(model_key)
                if isinstance(v2, str):
                    return v2
        return self.option_str("think_tag_default", "")

    def answer_keyword_for_stage(self, stage_name: str) -> str:
        """
        Return the configured answer delimiter keyword for solve prompts.
        Source: options.answer_extract_keywords
          - string: "FINAL:"
          - list: ["FINAL:"]
          - dict: { "default": ["FINAL:"], "stage3_solve": ["FINAL:"] }
        We always take the first keyword as the generation delimiter.
        """
        default = ["FINAL:"]
        raw = self.option_any("answer_extract_keywords", default)
        key = (stage_name or "").strip()
        stage_key_lower = key.lower()
        if isinstance(raw, dict):
            v = raw.get(stage_key_lower)
            if v is None:
                v = raw.get(key)
            if v is None:
                v = raw.get("default")
            raw = v if v is not None else default
        if isinstance(raw, str):
            s = raw.strip()
            return s if s else "FINAL:"
        if isinstance(raw, list):
            for x in raw:
                s = str(x).strip()
                if s:
                    return s
        return "FINAL:"

    def prompt_text(self, name: str) -> str:
        """
        Fetch a prompt text from options.prompts.<name> if present, else empty string.
        """
        obj = self.option_any("prompts", {})
        if isinstance(obj, dict):
            v = obj.get(name)
            return v if isinstance(v, str) else ""
        return ""

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
        max_workers: Optional[int] = None,
        debug_print_prompts: Optional[bool] = None,
        debug_print_prompts_max_chars: Optional[int] = None,
        debug_print_outputs: Optional[bool] = None,
        debug_print_outputs_max_chars: Optional[int] = None,
        debug_stream_outputs: Optional[bool] = None,
    ):
        p = self.stage_params(stage_name)
        client = self._pick_client(stage_name)
        answer_keyword = self.answer_keyword_for_stage(stage_name)
        solve_system_tmpl = self.prompt_text("solve_system")
        solve_user_tmpl = self.prompt_text("solve_user")
        raw_prompt_eval_system_tmpl = self.prompt_text("raw_prompt_eval_system")
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
            answer_keyword=answer_keyword,
            solve_system_template=solve_system_tmpl,
            solve_user_template=solve_user_tmpl,
            raw_prompt_eval_system_template=raw_prompt_eval_system_tmpl,
            max_workers=self.option_int("generate_n_max_workers", 8) if max_workers is None else int(max_workers),
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


