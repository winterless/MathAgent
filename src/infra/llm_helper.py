from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from typing import Any

from infra.llm_router import LLMRouter


def _parse_float(s: Any, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return float(default)


def _health_ok(url: str, timeout_s: float = 2.0) -> bool:
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=float(timeout_s)) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            return 200 <= status < 300
    except Exception:
        return False


def _wait_for_health(url: str, wait_s: float) -> bool:
    if not url or wait_s <= 0:
        return False
    deadline = time.time() + float(wait_s)
    sleep_s = 0.5
    while time.time() < deadline:
        if _health_ok(url, timeout_s=2.0):
            return True
        time.sleep(sleep_s)
        sleep_s = min(sleep_s * 1.5, 2.0)
    return False


def maybe_prestart_vllm(llm: LLMRouter) -> None:
    """
    Best-effort cleanup before vLLM start: optional nvidia-smi, gpu reset, and stop_cmd.
    (Policy is driven entirely by llm.options.*)
    """
    if not llm.option_bool("vllm_autostart_force_reset_and_kill", False):
        return
    # Print current GPU status.
    if llm.option_bool("vllm_pre_restart_nvidia_smi", False):
        try:
            res = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            print("[vLLM] prestart nvidia-smi:", file=sys.stderr, flush=True)
            print(res.stdout or "", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[vLLM] prestart nvidia-smi failed: {e}", file=sys.stderr, flush=True)

    # Optional GPU reset.
    if llm.option_bool("vllm_gpu_reset_on_restart", False):
        ids = str(llm.option_str("vllm_gpu_reset_ids", "") or "").strip().lower()
        if ids in ("all", "*"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
                ids = ",".join(lines)
            except Exception as e:
                print(f"[vLLM] gpu reset: failed to list gpu ids: {e}", file=sys.stderr, flush=True)
                ids = ""
        if ids and ids not in ("none", "false", "off"):
            try:
                reset_cmd = ["sudo", "-n", "nvidia-smi", "--gpu-reset", "-i", ids]
                print(f"[vLLM] prestart gpu reset: {' '.join(reset_cmd)}", file=sys.stderr, flush=True)
                res = subprocess.run(
                    reset_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                print(res.stdout or "", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[vLLM] gpu reset failed: {e}", file=sys.stderr, flush=True)

    # Stop vLLM process if configured.
    stop_cmd = llm.option_str("vllm_stop_cmd", "").strip()
    if stop_cmd:
        print(f"[vLLM] prestart stop cmd: {stop_cmd}", file=sys.stderr, flush=True)
        try:
            subprocess.Popen(["bash", "-lc", stop_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[vLLM] prestart stop cmd failed: {e}", file=sys.stderr, flush=True)

    # Optional delay before starting.
    delay_s = float(llm.option_int("vllm_restart_delay_s", 0))
    if delay_s > 0:
        time.sleep(delay_s)


def maybe_autostart_vllm(llm: LLMRouter) -> bool:
    """
    If enabled in config, start vLLM on pipeline startup and wait for health.
    """
    if not llm.option_bool("vllm_autostart", False):
        return False

    start_cmd = llm.vllm_start_cmd_resolved()
    restart_cmd = llm.option_str("vllm_restart_cmd", "").strip()
    use_restart_on_autostart = llm.option_bool("vllm_autostart_use_restart_cmd", True)
    cmd = restart_cmd if (use_restart_on_autostart and restart_cmd) else start_cmd
    if not cmd:
        return False

    health_url = llm.vllm_health_url_resolved()
    wait_s = _parse_float(llm.option_str("vllm_wait_s", "30"), 30.0)

    # If already healthy, skip.
    if health_url and _health_ok(health_url, timeout_s=2.0):
        return False

    log_path = llm.option_str("vllm_log_path", "").strip() or "/tmp/mathagent_vllm.log"
    log_to_stderr = llm.option_bool("vllm_log_to_stderr", True)
    start_with_bash = llm.option_bool("vllm_start_with_bash", False)

    try:
        maybe_prestart_vllm(llm)
        if use_restart_on_autostart and restart_cmd:
            print(f"[vLLM] autostart using restart cmd: {cmd}", file=sys.stderr, flush=True)
        if log_to_stderr:
            # Tee vLLM logs to stderr so terminal shows raw startup/runtime logs.
            import shlex

            log_q = shlex.quote(log_path)
            tee_cmd = f"{cmd} 2>&1 | tee -a {log_q} >&2"
            subprocess.Popen(
                ["bash", "-lc", tee_cmd],
                stdout=None,
                stderr=None,
                start_new_session=True,
            )
        elif start_with_bash:
            with open(log_path, "ab") as log_f:
                subprocess.Popen(
                    ["bash", "-lc", cmd],
                    stdout=log_f,
                    stderr=log_f,
                    start_new_session=True,
                )
        else:
            with open(log_path, "ab") as log_f:
                subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=log_f,
                    stderr=log_f,
                    start_new_session=True,
                )
    except Exception as e:
        print(f"[WARN] vLLM autostart failed to spawn: {e}", file=sys.stderr, flush=True)
        return False

    if health_url and wait_s > 0:
        ok = _wait_for_health(health_url, wait_s)
        if not ok:
            print(
                f"[WARN] vLLM autostart did not become healthy within {wait_s:.1f}s: {health_url}",
                file=sys.stderr,
                flush=True,
            )
    return True


def maybe_shutdown_vllm(llm: LLMRouter) -> None:
    if not llm.option_bool("vllm_shutdown_on_exit", False):
        return
    cmd = llm.option_str("vllm_stop_cmd", "").strip()
    if not cmd:
        return
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return

