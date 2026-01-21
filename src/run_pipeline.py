"""
MathAgent minimal pipeline (JSONL between stages).

This is the **single entrypoint** for the project.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import re
import subprocess
import shlex
import urllib.request
import urllib.error
import atexit
from typing import Any, Dict, List, Tuple

from core.stages import (
    append_choice_map_if_any,
    extract_boxed_answer,
    extract_boxed_counts_from_output,
    extract_choice_map,
    extract_final_answer,
    normalize_for_model,
    standardize_choice_answer,
)
from core.voting import majority_vote
from infra.llm_router import LLMRouter
from dataio.jsonl_io import append_jsonl_line, iter_jsonl, write_jsonl_atomic
from dataio.sample_schema import CANONICAL_KEYS, normalize_output_wrapper, normalize_record


_CONN_ERROR_MARKERS = (
    # Typical urllib/vLLM/OpenAI-compatible error text
    "connection refused",
    "errno 111",
    "llm network error",
    "urlopen error",
    "connection reset",
    "broken pipe",
    "timed out",
    "timeout",
    "temporary failure in name resolution",
)


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


def _maybe_prestart_vllm(llm: LLMRouter) -> None:
    """
    Best-effort cleanup before vLLM start: optional nvidia-smi, gpu reset, and stop_cmd.
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


def _maybe_autostart_vllm(llm: LLMRouter) -> bool:
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
        _maybe_prestart_vllm(llm)
        if use_restart_on_autostart and restart_cmd:
            print(f"[vLLM] autostart using restart cmd: {cmd}", file=sys.stderr, flush=True)
        if log_to_stderr:
            # Tee vLLM logs to stderr so terminal shows raw startup/runtime logs.
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
            print(f"[WARN] vLLM autostart did not become healthy within {wait_s:.1f}s: {health_url}", file=sys.stderr, flush=True)
    return True


def _maybe_shutdown_vllm(llm: LLMRouter) -> None:
    if not llm.option_bool("vllm_shutdown_on_exit", False):
        return
    cmd = llm.option_str("vllm_stop_cmd", "").strip()
    if not cmd:
        return
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return


def _contains_conn_error_marker(obj: Any, *, max_nodes: int = 5000) -> bool:
    """
    Best-effort: recursively search for connection/network error markers inside an object.
    We keep it conservative: only trigger when we find explicit markers or our "[LLM_ERROR...]" wrapper.
    """
    n = 0
    stack: List[Any] = [obj]
    while stack:
        cur = stack.pop()
        n += 1
        if n > int(max_nodes):
            return False
        if cur is None:
            continue
        if isinstance(cur, str):
            s = cur.strip()
            if not s:
                continue
            s_low = s.lower()
            if s_low.startswith("[llm_error"):
                # Only treat as "connection" issue if the message indicates connectivity.
                return any(m in s_low for m in _CONN_ERROR_MARKERS)
            if any(m in s_low for m in _CONN_ERROR_MARKERS):
                return True
            continue
        if isinstance(cur, dict):
            for v in cur.values():
                stack.append(v)
            continue
        if isinstance(cur, (list, tuple)):
            for v in cur:
                stack.append(v)
            continue
    return False


def _iter_out_jsonl_files(out_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(out_dir):
        for name in files:
            if name.endswith(".jsonl"):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def _should_purge_from_file(path: str) -> bool:
    """
    Purge only from derived artifacts that act as "done markers" or archived results.
    Do NOT purge from:
      - stage0 copies (inputs)
      - stage2_input / stage3_input (task lists)
      - stage1_output (upstream input for modular routes)
    """
    base = os.path.basename(path)
    if base == "stage0.jsonl" or base.endswith(".stage0.jsonl"):
        return False
    if base.endswith("stage2_input.stage2.jsonl") or base.endswith("stage3_input.stage3.jsonl"):
        return False
    if base.endswith("stage1_output.stage1.jsonl"):
        return False
    return True


def _scan_bad_uuids_for_conn_errors(out_dir: str) -> set:
    """
    Scan existing produced artifacts under out_dir and collect uuids that contain connection-related LLM errors.
    Returns a set of str(uuid).
    """
    bad: set = set()
    for pth in _iter_out_jsonl_files(out_dir):
        if not _should_purge_from_file(pth):
            continue
        try:
            with open(pth, "r", encoding="utf-8") as f:
                for line in f:
                    s = (line or "").strip()
                    if not s:
                        continue
                    try:
                        row = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    u = row.get("uuid")
                    if u is None:
                        continue
                    if _contains_conn_error_marker(row):
                        bad.add(str(u))
        except Exception:
            # best-effort
            continue
    return bad


def _purge_uuids_from_out_dir(out_dir: str, uuids: set) -> Dict[str, int]:
    """
    Purge lines whose uuid is in `uuids` from selected output artifacts under out_dir.
    Returns per-file removed line counts.
    """
    removed: Dict[str, int] = {}
    if not uuids:
        return removed
    for pth in _iter_out_jsonl_files(out_dir):
        if not _should_purge_from_file(pth):
            continue
        tmp = f"{pth}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
        n_removed = 0
        changed = False
        try:
            with open(pth, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
                for line in fin:
                    s = (line or "").strip()
                    if not s:
                        fout.write(line)
                        continue
                    keep = True
                    try:
                        row = json.loads(s)
                    except Exception:
                        row = None
                    if isinstance(row, dict):
                        u = row.get("uuid")
                        if u is not None and str(u) in uuids:
                            keep = False
                    if keep:
                        fout.write(line)
                    else:
                        n_removed += 1
                        changed = True
            if changed:
                os.replace(tmp, pth)
                removed[pth] = int(n_removed)
            else:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            continue
    return removed


def _default_out_dir() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join("datasets", "out", ts)


def _read_all(path: str) -> List[dict]:
    return list(iter_jsonl(path, tolerate_errors=True))


def _load_status_map(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load a status.jsonl file into a map keyed by str(uuid).
    Each line is a JSON object at least containing 'uuid'.
    """
    if not path or not os.path.exists(path):
        return {}
    m: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path, tolerate_errors=True):
        u = row.get("uuid")
        if u is None:
            continue
        m[str(u)] = row
    return m


def _load_done_uuid_set(path: str) -> set[str]:
    """
    Load a status/result jsonl file into a set of processed UUID strings.
    This is much lighter than _load_status_map() for very large runs.
    """
    if not path or not os.path.exists(path):
        return set()
    s: set[str] = set()
    for row in iter_jsonl(path, tolerate_errors=True):
        u = row.get("uuid")
        if u is None:
            continue
        s.add(str(u))
    return s


def _count_total_and_done_uuids(*, input_path: str, done_uuids: set[str] | None = None) -> tuple[int, int]:
    """
    Count how many rows in input_path have a UUID, and how many are in done_uuids.
    Uses streaming scan; does not materialize the file in memory.
    """
    total = 0
    done = 0
    dset = done_uuids or set()
    for row in iter_jsonl(input_path, tolerate_errors=True):
        u = row.get("uuid")
        if u is None:
            continue
        total += 1
        if dset and str(u) in dset:
            done += 1
    return total, done


def _write_jsonl_stream_atomic(path: str, rows: Any) -> None:
    """
    Write an iterator of dict rows to jsonl atomically (streaming, low memory).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            try:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception:
                # Best-effort: skip bad rows; never crash long runs.
                continue
    os.replace(tmp, path)


def _iter_input_jsonl_paths(input_arg: str) -> List[str]:
    """
    Return a sorted list of input JSONL files.

    - If input_arg is a file: return [input_arg]
    - If input_arg is a directory:
      - If it looks like an output root (contains stage subdirs) and has *.stage0.jsonl, use those.
      - Otherwise, use top-level *.jsonl files (non-recursive, historical behavior).
    """
    if os.path.isdir(input_arg):
        # If the user points --input at an out root (e.g. datasets/out/demo_modular),
        # prefer the original copied inputs (<prefix>.stage0.jsonl) to avoid accidentally
        # treating artifacts as raw inputs.
        has_stage_subdir = any(os.path.isdir(os.path.join(input_arg, d)) for d in ("stage1", "stage2", "stage3"))
        stage0s: List[str] = []
        try:
            for name in os.listdir(input_arg):
                p = os.path.join(input_arg, name)
                if os.path.isfile(p) and name.lower().endswith(".stage0.jsonl"):
                    stage0s.append(p)
        except FileNotFoundError:
            stage0s = []
        if stage0s:
            stage0s.sort()
            return stage0s
        if has_stage_subdir:
            raise ValueError(
                f"--input points to an output root without any '*.stage0.jsonl' inputs: {input_arg}. "
                f"For stage1_infer please pass a raw input JSONL (file/dir) or an out root that contains stage0 copies."
            )
        files: List[str] = []
        for name in os.listdir(input_arg):
            p = os.path.join(input_arg, name)
            if os.path.isfile(p) and name.lower().endswith(".jsonl"):
                files.append(p)
        files.sort()
        return files
    return [input_arg]


def _input_prefix(path: str) -> str:
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    s = (stem or "input").strip()
    # Make prefix filesystem-friendly: keep letters/numbers/._-; replace others with '_'
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "input"



class _UUIDProgress:
    """
    Lightweight per-UUID progress printer (no external deps).

    Shows:
      - percent
      - done/total UUIDs (done includes previously-done + newly processed)
      - avg seconds/uuid (only for newly processed in this run)
    """

    def __init__(self, *, label: str, total: int, already_done: int = 0, prefix: str = "") -> None:
        self.label = str(label or "stage").strip()
        self.prefix = str(prefix or "").strip()
        self.total = int(total or 0)
        self.already_done = int(already_done or 0)
        self.processed = 0
        self.sum_uuid_s = 0.0
        self._last_print = 0.0
        # NOTE:
        # - In a real TTY we can use '\r' to repaint a single line.
        # - In many log collectors, '\r' without '\n' is not visible.
        # - Also, in orchestration (python parent process + subprocess), stderr may not be captured/displayed.
        #   So in non-TTY contexts we default to stdout for progress lines.
        #
        # Override via env:
        #   - MATHAGENT_PROGRESS_STREAM=stdout|stderr|auto
        #   - MATHAGENT_PROGRESS_INTERVAL_S=<float>
        try:
            stream_opt = str(os.environ.get("MATHAGENT_PROGRESS_STREAM", "auto") or "auto").strip().lower()
        except Exception:
            stream_opt = "auto"
        if stream_opt in ("stdout", "out", "1"):
            self._stream = sys.stdout
        elif stream_opt in ("stderr", "err", "2"):
            self._stream = sys.stderr
        else:
            # auto: tty -> stderr (keeps progress separate from normal output); non-tty -> stdout (more visible in logs)
            self._stream = sys.stderr
            try:
                if not bool(sys.stderr.isatty()):
                    self._stream = sys.stdout
            except Exception:
                self._stream = sys.stdout

        try:
            self._isatty = bool(getattr(self._stream, "isatty", lambda: False)())
        except Exception:
            self._isatty = False

        try:
            self._interval_s = float(os.environ.get("MATHAGENT_PROGRESS_INTERVAL_S", "") or 0.0)
        except Exception:
            self._interval_s = 0.0

    def start(self) -> None:
        self._print(force=True)

    def tick(self, uuid_s: float) -> None:
        self.processed += 1
        try:
            self.sum_uuid_s += float(uuid_s)
        except Exception:
            pass
        self._print(force=False)

    def finish(self) -> None:
        self._print(force=True, final=True)

    def _print(self, *, force: bool, final: bool = False) -> None:
        now = time.monotonic()
        min_interval_s = float(self._interval_s) if float(self._interval_s) > 0 else (0.25 if self._isatty else 5.0)
        if not force and (now - self._last_print) < float(min_interval_s):
            return
        self._last_print = now
        done_total = self.already_done + self.processed
        if self.total <= 0:
            pct = 100.0
        else:
            pct = 100.0 * float(done_total) / float(self.total)
            if pct > 100.0:
                pct = 100.0
        avg = (self.sum_uuid_s / float(self.processed)) if self.processed > 0 else 0.0
        tag = f"{self.label}" + (f"/{self.prefix}" if self.prefix else "")
        msg = f"[{tag}] {pct:6.2f}% ({done_total}/{self.total}) avg {avg:.2f}s/uuid"
        try:
            if self._isatty:
                self._stream.write("\r" + msg + "\033[K")
                if final:
                    self._stream.write("\n")
                self._stream.flush()
            else:
                # Log-friendly: always newline-terminated, so collectors show it.
                self._stream.write(msg + "\n")
                self._stream.flush()
        except Exception:
            # Best-effort progress output; never fail pipeline.
            if final:
                try:
                    print(msg, file=self._stream, flush=True)
                except Exception:
                    pass


# ---- Route-A modular modes (infer/eval split for stage2/stage3) ----
MODES = [
    "stage1_infer",
    "stage1_eval",
    "stage2_infer",
    "stage2_eval",
    "stage3_infer",
    "stage3_eval",
    "result_rebuild",
]


def _iter_artifacts(input_arg: str, *, suffix: str) -> List[str]:
    """
    Return artifact file paths ending with suffix (input_arg may be file or directory).

    Supports passing an out root directory (e.g. datasets/out/demo_modular); in that case we
    search recursively under the directory for matching artifacts.
    """
    if os.path.isdir(input_arg):
        outs: List[str] = []
        for root, _dirs, files in os.walk(input_arg):
            for name in files:
                if name.lower().endswith(".jsonl") and name.endswith(suffix):
                    outs.append(os.path.join(root, name))
        outs.sort()
        return outs
    return [input_arg]


def _infer_prefix_from_artifact(path: str, *, suffix: str) -> str:
    """
    Infer prefix from '<prefix>.<suffix>'.
    If unprefixed (exactly suffix), fall back to parent directory name (reduces collisions).
    """
    base = os.path.basename(path)
    if base == suffix:
        parent = os.path.basename(os.path.dirname(path) or "")
        return _input_prefix(parent) if parent else ""
    dot_suf = "." + suffix
    if base.endswith(dot_suf):
        return base[: -len(dot_suf)]
    # final fallback: use filename stem
    stem, _ext = os.path.splitext(base)
    return _input_prefix(stem)


def _ensure_stage_dirs(out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stage1_dir = os.path.join(out_dir, "stage1")
    stage2_dir = os.path.join(out_dir, "stage2")
    stage3_dir = os.path.join(out_dir, "stage3")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(stage3_dir, exist_ok=True)
    return {"stage1": stage1_dir, "stage2": stage2_dir, "stage3": stage3_dir}


def _compact_outputs(llm: LLMRouter) -> bool:
    return llm.option_bool("compact_outputs", False)


def _select_answer(*, gold: str, majority: Dict[str, Any], min_votes_to_accept: int) -> Dict[str, Any]:
    """
    Select the final answer for this stage:
    - If voting is confident (majority_count >= min_votes_to_accept): use voting result
    - Else: no answer (do NOT fall back to gold)

    Note: `gold` is kept only for backward-compatible call sites.
    """
    maj_raw = str((majority or {}).get("majority") or "").strip()
    maj_cnt = int((majority or {}).get("majority_count") or 0)
    if maj_cnt >= int(min_votes_to_accept) and maj_raw:
        return {"final_answer": maj_raw, "final_source": "majority", "final_vote_count": maj_cnt}
    return {"final_answer": "", "final_source": "no_majority", "final_vote_count": maj_cnt}


def _llm_extract_answers_batch(
    *,
    llm: LLMRouter,
    uuid: Any,
    question: str,
    raw_outputs: List[str],
    choice_map: Dict[str, str],
    stage: str,
    sleep_s: float,
    stats: Dict[str, int] | None = None,
) -> List[str]:
    """
    Step-1 (non-LLM): extract answers from raw solver outputs by keyword.
    Contract: the final answer must appear AFTER a keyword; we take the LAST occurrence.
    If keyword not found => "".

    NOTE: keep signature for backward compatibility (llm/stats unused here).
    """
    raws = [str(x) for x in (raw_outputs or [])]
    if not raws:
        return []

    def _get_keywords_for_stage(stage_name: str) -> List[str]:
        default = ["FINAL:"]
        try:
            raw = llm.option_any("answer_extract_keywords", default)
        except Exception:
            raw = default
        stage_key = str(stage_name or "").strip().lower()
        if isinstance(raw, dict):
            v = raw.get(stage_key)
            if v is None:
                v = raw.get("default")
            raw = v if v is not None else default
        if isinstance(raw, str):
            # Allow "FINAL:" or "FINAL:,FINAL：" style strings.
            parts = [p.strip() for p in raw.replace("|", ",").split(",")]
            kws = [p for p in parts if p]
            return kws or default
        if isinstance(raw, list):
            kws = [str(x).strip() for x in raw if str(x).strip()]
            return kws or default
        return default

    def _extract_after_last_keyword(text: str, keywords: List[str]) -> str:
        s = str(text or "")
        best_i = -1
        best_kw = ""
        for kw in keywords:
            i = s.rfind(kw)
            if i > best_i:
                best_i = i
                best_kw = kw
        if best_i < 0:
            return ""
        # Contract: take everything after the LAST keyword occurrence.
        tail = s[best_i + len(best_kw) :]
        return str(tail).strip()

    out: List[str] = []
    for t in raws:
        # If upstream produced explicit error markers, do not accidentally extract numeric codes (e.g. 503).
        if (t or "").strip().startswith("[LLM_ERROR"):
            out.append("")
            continue
        keywords = _get_keywords_for_stage(stage)
        ans = _extract_after_last_keyword(t, keywords)
        # No local normalization: keep as-is (model eval handles any canonicalization).
        out.append(ans)
    return out


def _llm_vote_majority_from_extracted(
    *,
    llm: LLMRouter,
    question: str,
    extracted_answers: List[str],
    choice_map: Dict[str, str],
    stage: str,
    sleep_s: float,
    stats: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """
    Step-2 (LLM): given question + multiple extracted answers, ask LLM to:
    - (optionally) normalize/cluster equivalent answers
    - compute majority vote and counts
    Tie-break rule: if multiple answers share the max vote count, pick the one that appears first.

    Returns (best-effort):
      {
        "normalized": List[str] (aligned with extracted_answers),
        "majority": str,
        "majority_count": int,
        "counts": Dict[str, int]
      }
    """
    answers = [str(x or "") for x in (extracted_answers or [])]
    if not answers:
        return {"normalized": [], "majority": "", "majority_count": 0, "counts": {}}

    choice_lines = []
    for k in ("A", "B", "C", "D"):
        if k in (choice_map or {}):
            choice_lines.append(f"{k} = {choice_map[k]}")
    choice_block = "\n".join(choice_lines)

    # Keep prompt strict: JSON only, with an explicit example.
    items = "\n".join([f"{i+1}. {a}" for i, a in enumerate(answers)])
    default_template = (
        "你是“答案归一化 + 多数投票评估器”，禁止解题。\n"
        "禁止输出 <think> 或任何推理过程。\n"
        "你的输出必须是 **严格 JSON**（不要代码块，不要解释，不要额外文本）。\n\n"
        "任务：\n"
        "1) （可选）把同义/等价的候选答案归一化为相同字符串；不确定可输出空字符串 \"\"。\n"
        "2) 基于归一化后的答案做多数投票（空字符串不参与投票），并给出 counts。\n"
        "   - 平票规则：若最高票数并列，选择在列表中最早出现的那个。\n\n"
        "输出 JSON schema（必须遵守）：\n"
        "{\"normalized\": [\"...\"], \"majority\": \"...\", \"majority_count\": 0, \"counts\": {\"...\": 0}}\n\n"
        "示例（仅供参考，勿照抄内容）：\n"
        "[候选答案]\n"
        "1. B\n"
        "2. B\n"
        "3. D\n\n"
        "输出：\n"
        "{\"normalized\":[\"B\",\"B\",\"D\"],\"majority\":\"B\",\"majority_count\":2,\"counts\":{\"B\":2,\"D\":1}}\n\n"
        "[题目]\n$question\n\n"
        "[选项映射(如有)]\n$choice_block\n\n"
        "[候选答案]\n$candidates\n"
    ).strip()
    prompts_obj = llm.option_any("prompts", {}) if hasattr(llm, "option_any") else {}
    tmpl = default_template
    if isinstance(prompts_obj, dict):
        # Prefer the newer key name, but keep backward compatibility.
        v = prompts_obj.get("majority_vote")
        if not (isinstance(v, str) and v.strip()):
            v = prompts_obj.get("vote_majority")
        if isinstance(v, str) and v.strip():
            tmpl = v
    from string import Template

    # Render user prompt from template, but ALWAYS append the concrete input payload
    # (question + raw candidates) to ensure traceability in vote_model_input.
    rendered = Template(str(tmpl)).safe_substitute(
        question=str(question or "").strip(),
        choice_block=str(choice_block or ""),
        candidates=str(items or ""),
    ).strip()
    q_payload = str(question or "").strip()
    # Use a JSON array for candidates to avoid ambiguity / numbering artifacts.
    cand_payload = json.dumps(answers, ensure_ascii=False)
    prompt = (
        f"{rendered}\n\n"
        f"[题目]\n{q_payload}\n\n"
        f"[候选答案]\n{cand_payload}\n"
    ).strip()

    resp = llm.generate_n(
        stage_name=f"{stage}_vote",
        question=prompt,
        prompt_mode="raw_prompt",
        n=1,
        temperature=0.0,
        # Do NOT hardcode max_tokens here; use stage_params(<stage>_vote) from config
        # so JSON output won't be truncated by an arbitrary small limit.
        sleep_s=sleep_s,
        stats=stats,
    )[0]
    txt = (resp or "").strip()
    # Record the actual user prompt we feed to the model (including think-tag injection).
    vote_stage_name = f"{stage}_vote"
    vote_tag = ""
    try:
        vote_tag = str(llm.think_tag_for_stage(vote_stage_name) or "").strip()
    except Exception:
        vote_tag = ""
    effective_prompt = prompt
    # Injection rule mirrors llm_client.py: inject for all modes except raw_prompt_eval.
    if vote_tag:
        if not vote_tag.startswith("/"):
            vote_tag = "/" + vote_tag
        effective_prompt = f"{vote_tag}\n{prompt}"

    def _safe_load_json(s: str) -> Dict[str, Any]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        # Best-effort: extract the LAST valid JSON object in the text.
        # Rationale: model may output long <think> with many LaTeX braces '{...}',
        # and a JSON blob at the end. A naive s.find("{") will hit LaTeX first.
        try:
            end = s.rfind("}")
            if end < 0:
                return {}
            # Scan backwards for a '{' that yields a valid JSON dict.
            for i in range(end, -1, -1):
                if s[i] != "{":
                    continue
                cand = s[i : end + 1].strip()
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        except Exception:
            return {}
        return {}

    obj = _safe_load_json(txt)
    normalized = obj.get("normalized")
    if not (isinstance(normalized, list) and len(normalized) == len(answers)):
        normalized = answers[:]  # fallback: keep original (no local normalization)
    normalized = [str(x or "") for x in normalized]
    majority = str(obj.get("majority") or "")
    try:
        majority_count = int(obj.get("majority_count") or 0)
    except Exception:
        majority_count = 0
    counts_obj = obj.get("counts")
    counts: Dict[str, int] = {}
    if isinstance(counts_obj, dict):
        for k, v in counts_obj.items():
            try:
                counts[str(k)] = int(v)
            except Exception:
                continue
    return {
        "normalized": normalized,
        "majority": majority,
        "majority_count": int(majority_count),
        "counts": counts,
        # Store the raw LLM output for debugging/auditing in status files.
        "raw_output": txt,
        # Store the actual prompt fed to the model (user content).
        "model_input": effective_prompt,
    }


def _render_solve_prompts(
    *,
    llm: LLMRouter,
    stage_name: str,
    question_for_model: str,
    n: int,
) -> tuple[str, str]:
    """
    Render the (system, base_user) prompts used for solve-like calls.
    Note: actual per-sample requests will append a sample suffix (e.g. '采样编号=i'),
    but for logging we only keep the base user prompt once.
    """
    kw = ""
    try:
        kw = str(llm.answer_keyword_for_stage(stage_name) or "").strip()
    except Exception:
        kw = ""
    if not kw:
        kw = "FINAL:"

    # Templates (optional)
    solve_system_tmpl = ""
    solve_user_tmpl = ""
    try:
        solve_system_tmpl = str(llm.prompt_text("solve_system") or "")
        solve_user_tmpl = str(llm.prompt_text("solve_user") or "")
    except Exception:
        solve_system_tmpl = ""
        solve_user_tmpl = ""

    from string import Template

    default_system = (
        f"你是一个数学解题助手。Stage={stage_name}。\n"
        f"最后一行必须输出最终答案，且格式必须严格为：{kw} <答案>\n"
        "如果题目是选择题：<答案> 只能是单个大写字母 A/B/C/D。\n"
        "如果题目不是选择题：<答案> 为最终数值/表达式。"
    )
    system_t = solve_system_tmpl.strip() if solve_system_tmpl.strip() else default_system
    system = Template(system_t).safe_substitute(stage_name=str(stage_name), answer_keyword=str(kw))

    finish_early = True
    try:
        finish_early = bool(llm.option_bool("finish_early", True))
    except Exception:
        finish_early = True
    if finish_early:
        system += (
            "\n"
            "如果你感觉推理会很长、或可能来不及写完，请立刻停止推理，"
            f"直接在最后一行输出 {kw} <你认为最可能的答案>（选择题在 A/B/C/D 中猜一个）。"
        )

    default_user = f"题目：\n{question_for_model}\n\n要求：你可以写推理过程，但最后一行必须是 {kw} <答案>（严格格式）。"
    user_t = solve_user_tmpl.strip() if solve_user_tmpl.strip() else default_user
    base_user = Template(user_t).safe_substitute(question=str(question_for_model), answer_keyword=str(kw))

    # Think-tag injection mirrors llm_client.py (inject for all modes except raw_prompt_eval).
    tag = ""
    try:
        tag = str(llm.think_tag_for_stage(stage_name) or "").strip()
    except Exception:
        tag = ""
    if tag:
        if not tag.startswith("/"):
            tag = "/" + tag
        base_user = f"{tag}\n{base_user}"

    _ = n  # keep signature stable; sampling count doesn't change the base prompt
    return system, base_user


def _append_stage_infer_row(
    *,
    infer_path: str,
    stage: str,
    row: Dict[str, Any],
    model_input: str,
    model_prompt_system: str,
    model_prompt_user: str,
    raw_solutions: List[str],
    extracted_answers: List[str],
    n: int,
    min_votes_to_accept: int,
    solve_stats: Dict[str, int],
) -> None:
    stage_key = str(stage or "").strip().lower()
    stage_label = f"{stage_key}_infer"
    solve_key = f"{stage_key}_solve"
    append_jsonl_line(
        infer_path,
        {
            "uuid": row.get("uuid"),
            "line_number": row.get("line_number"),
            "stage": stage_label,
            "question": row.get("question"),
            "answer": row.get("answer"),
            "gold": row.get("answer"),
            "model_input": model_input,
            "model_prompt_system": str(model_prompt_system or ""),
            "model_prompt_user": str(model_prompt_user or ""),
            "raw_model_outputs": [str(x) for x in raw_solutions][:n],
            "extracted_answers": [str(x) for x in extracted_answers][:n],
            "min_votes_to_accept": int(min_votes_to_accept),
            "llm_call_counts": {
                solve_key: int(n),
                f"{solve_key}_http_calls": int(solve_stats.get("http_calls", 0)),
                f"{solve_key}_retries": int(solve_stats.get("retries", 0)),
                f"{solve_key}_timeouts": int(solve_stats.get("timeouts", 0)),
                f"{solve_key}_errors": int(solve_stats.get("errors", 0)),
            },
        },
    )




def _result_path_for_prefix(out_dir: str, prefix: str) -> str:
    result_dir = os.path.join(out_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    pfx = f"{prefix}." if prefix else ""
    return os.path.join(result_dir, f"{pfx}result.stage_final.jsonl")


def result_rebuild(*, out_dir: str, min_votes_to_accept: int) -> None:
    """
    Scan output artifacts under out_dir and (re)build result/*.result.stage_final.jsonl.
    Only include rows that:
      - have majority_count >= min_votes_to_accept
      - contain an attempt whose answer matches the majority answer
    Result rows only include: {"uuid", "text"} (text is raw_text of the matching attempt).
    Sources:
      - stage2_infer.stage2.jsonl
      - stage3_infer.stage3.jsonl
    """
    out_dir = str(out_dir)
    stage2_infer_paths = _iter_artifacts(out_dir, suffix="stage2_infer.stage2.jsonl")
    stage3_infer_paths = _iter_artifacts(out_dir, suffix="stage3_infer.stage3.jsonl")
    all_paths = stage2_infer_paths + stage3_infer_paths

    # Cache existing result uuids per prefix
    existing: Dict[str, set[str]] = {}
    stats = {
        "total_rows_scanned": 0,
        "total_candidates": 0,
        "total_written": 0,
        "estimated_tokens": 0,
        "by_stage": {"stage2": 0, "stage3": 0, "unknown": 0},
        "by_source": {
            "stage2_infer": 0,
            "stage3_infer": 0,
        },
        "files_scanned": list(all_paths),
    }

    def _estimate_tokens(text: str) -> int:
        # Conservative heuristic: ~4 chars per token for mixed CJK/ASCII.
        s = str(text or "")
        return max(1, int(len(s) / 4)) if s else 0

    def _get_done(prefix: str) -> set[str]:
        if prefix in existing:
            return existing[prefix]
        result_path = _result_path_for_prefix(out_dir, prefix)
        done = _load_done_uuid_set(result_path)
        existing[prefix] = done
        return done

    def _match_answer(pred: str, gold: str) -> bool:
        p = str(pred or "").strip()
        g = str(gold or "").strip()
        if not p or not g:
            return False
        return p == g

    def _build_result_text(question: str, raw: str, pred: str) -> str:
        q = str(question or "").strip()
        r = str(raw or "").strip()
        p = str(pred or "").strip()
        return f"问题：{q}\n\n思考：{r}\n\n答案：{p}"

    def _infer_status_path(infer_path: str, *, stage: str) -> str:
        base = os.path.basename(infer_path)
        d = os.path.dirname(infer_path)
        if stage == "stage2" and base.endswith("stage2_infer.stage2.jsonl"):
            return os.path.join(d, base.replace("stage2_infer.stage2.jsonl", "status.stage2.jsonl"))
        if stage == "stage3" and base.endswith("stage3_infer.stage3.jsonl"):
            return os.path.join(d, base.replace("stage3_infer.stage3.jsonl", "status.stage3.jsonl"))
        return ""

    def _load_status_majority_map(status_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load status rows keyed by uuid (best-effort). We only need:
        - vote_majority / vote_majority_count
        - min_votes_to_accept
        """
        out: Dict[str, Dict[str, Any]] = {}
        if not status_path or not os.path.exists(status_path):
            return out
        for rr in iter_jsonl(status_path, tolerate_errors=True):
            if not isinstance(rr, dict):
                continue
            u = rr.get("uuid")
            if u is None:
                continue
            out[str(u)] = rr
        return out

    def _iter_majority_infer_attempts(row: Dict[str, Any], *, status_row: Dict[str, Any] | None) -> List[Tuple[int, str]]:
        raws = row.get("raw_model_outputs") if isinstance(row.get("raw_model_outputs"), list) else []
        extracted = row.get("extracted_answers") if isinstance(row.get("extracted_answers"), list) else []
        if not raws or not extracted:
            return []
        question = row.get("question")
        if question is None:
            return []
        # Prefer status-derived majority (authoritative in normal modular flow).
        maj_ans = ""
        maj_cnt = 0
        threshold = int(min_votes_to_accept)
        if isinstance(status_row, dict) and status_row:
            maj_ans = str(status_row.get("vote_majority") or "").strip()
            try:
                maj_cnt = int(status_row.get("vote_majority_count") or 0)
            except Exception:
                maj_cnt = 0
            try:
                threshold = int(status_row.get("min_votes_to_accept") or threshold)
            except Exception:
                threshold = int(threshold)
        else:
            # Fallback: self-contained rebuild from infer only.
            cleaned = [str(x or "").strip() for x in extracted]
            cleaned = [x for x in cleaned if x]
            if not cleaned:
                return []
            v = majority_vote(cleaned)
            maj_ans = str(v.majority or "").strip()
            try:
                threshold = int(row.get("min_votes_to_accept") or min_votes_to_accept)
            except Exception:
                threshold = int(min_votes_to_accept)
            maj_cnt = int(v.majority_count or 0)

        if not maj_ans or int(maj_cnt) < int(threshold):
            return []
        results: List[Tuple[int, str]] = []
        for idx, (raw, pred) in enumerate(zip(raws, extracted)):
            if not _match_answer(pred, maj_ans):
                continue
            text = _build_result_text(question, raw, pred)
            results.append((idx, text))
        return results

    def _rebuild_prefix_for_path(path: str, *, suffix: str) -> str:
        """
        Derive result prefix. If artifacts are under a subdirectory of out_dir,
        prepend that subdirectory name to avoid collisions across multi-run outputs.
        """
        prefix = _infer_prefix_from_artifact(path, suffix=suffix)
        try:
            rel = os.path.relpath(os.path.dirname(path), out_dir)
            parts = [p for p in rel.split(os.sep) if p and p != "."]
        except Exception:
            parts = []
        # If artifact is under a child run dir (not directly under stage*/result),
        # prepend that child dir name to keep outputs unique in the parent result/.
        if parts and parts[0] not in ("stage1", "stage2", "stage3", "result"):
            run_name = parts[0]
            return f"{run_name}.{prefix}" if prefix else run_name
        return prefix

    total_files = len(all_paths)
    print(f"[result_rebuild] start: files={total_files}", file=sys.stderr, flush=True)

    for idx, pth in enumerate(all_paths, start=1):
        base = os.path.basename(pth)
        if base.endswith("stage2_infer.stage2.jsonl"):
            prefix = _rebuild_prefix_for_path(pth, suffix="stage2_infer.stage2.jsonl")
            stage = "stage2"
            source = "stage2_infer"
        elif base.endswith("stage3_infer.stage3.jsonl"):
            prefix = _rebuild_prefix_for_path(pth, suffix="stage3_infer.stage3.jsonl")
            stage = "stage3"
            source = "stage3_infer"
        else:
            continue

        status_path = _infer_status_path(pth, stage=stage)
        status_by_uuid = _load_status_majority_map(status_path)

        result_path = _result_path_for_prefix(out_dir, prefix)
        done = _get_done(prefix)

        processed = 0
        for row in iter_jsonl(pth, tolerate_errors=True):
            stats["total_rows_scanned"] += 1
            processed += 1
            if not isinstance(row, dict):
                continue
            u = row.get("uuid")
            if u is None:
                continue
            u_str = str(u)
            if u_str in done:
                continue
            matches = _iter_majority_infer_attempts(row, status_row=status_by_uuid.get(u_str))
            if not matches:
                continue
            for attempt_idx, raw_text in matches:
                attempt_uuid = f"{u_str}-{attempt_idx + 1}"
                if attempt_uuid in done:
                    continue
                stats["total_candidates"] += 1
                append_jsonl_line(result_path, {"uuid": attempt_uuid, "text": raw_text})
                done.add(attempt_uuid)
                stats["total_written"] += 1
                stats["estimated_tokens"] += _estimate_tokens(raw_text)
                stats["by_source"][source] = int(stats["by_source"].get(source, 0)) + 1
                stats["by_stage"][stage] += 1

        print(
            f"[result_rebuild] {idx}/{total_files} files "
            f"({(idx / total_files * 100.0) if total_files else 100.0:.1f}%) "
            f"rows={processed} written_total={stats['total_written']}",
            file=sys.stderr,
            flush=True,
        )

    # Write summary file under result/
    summary_path = os.path.join(out_dir, "result", "summary.result_rebuild.json")
    try:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        pass


def mode_stage1_infer(*, input_arg: str, out_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> List[str]:
    """
    Route-A mode: stage1_infer
    Input: raw dataset JSONL (file) or a directory containing *.jsonl.
    Output: stage1/<prefix>.stage1_infer.stage1.jsonl (raw solve outputs only, for stage1_eval).
    """
    dirs = _ensure_stage_dirs(out_dir)
    stage1_dir = dirs["stage1"]

    outs: List[str] = []
    input_paths = _iter_input_jsonl_paths(input_arg)
    for input_path in input_paths:
        prefix = _input_prefix(input_path) if os.path.isfile(input_path) else _input_prefix(os.path.basename(input_path))
        pfx = f"{prefix}." if prefix else ""
        stage1_infer_path = os.path.join(stage1_dir, f"{pfx}stage1_infer.stage1.jsonl")

        infer_done: set[str] = set()
        if os.path.exists(stage1_infer_path):
            for rr in iter_jsonl(stage1_infer_path, tolerate_errors=True):
                uu = rr.get("uuid")
                if uu is not None:
                    infer_done.add(str(uu))

        rows = [normalize_record(r) for r in _read_all(input_path)]
        all_uuid_rows = [r for r in rows if r.get("uuid") is not None]
        to_process = [r for r in all_uuid_rows if str(r.get("uuid")) not in infer_done]
        prog = _UUIDProgress(label="stage1_infer", prefix=prefix, total=len(all_uuid_rows), already_done=len(all_uuid_rows) - len(to_process))
        prog.start()

        for r in to_process:
            t0 = time.monotonic()
            uuid = r.get("uuid")
            uuid_key = str(uuid)

            q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
            gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
            # If input rows come from an "output wrapper" file (sample-like),
            # prefer extracting answer from output.content for later comparison.
            try:
                out0 = r.get("output") if isinstance(r.get("output"), dict) else {}
                content0 = out0.get("content") if isinstance(out0.get("content"), dict) else {}
                choices0 = content0.get("choices")
                msg0 = (choices0[0].get("message") if isinstance(choices0, list) and choices0 and isinstance(choices0[0], dict) else {}) or {}
                out_text = msg0.get("content") if isinstance(msg0, dict) else ""
                out_text = out_text if isinstance(out_text, str) else str(out_text)
                ans_from_content = (extract_final_answer(out_text) or "").strip()
                if ans_from_content:
                    gold = ans_from_content
            except Exception:
                pass
            if not q_raw.strip():
                raise ValueError(f"Missing question: uuid={uuid}")

            model_q = append_choice_map_if_any(normalize_for_model(q_raw))
            choice_map = extract_choice_map(model_q)
            s1_solve_stats: Dict[str, int] = {}
            raw_solutions = llm.generate_n(
                stage_name="stage1_solve",
                question=model_q,
                prompt_mode="problem",
                sleep_s=sleep_s,
                stats=s1_solve_stats,
            )
            n1 = int(llm.stage_params("stage1_solve").n)
            sys_p, user_p = _render_solve_prompts(llm=llm, stage_name="stage1_solve", question_for_model=model_q, n=n1)
            extracted = _llm_extract_answers_batch(
                llm=llm,
                uuid=uuid,
                question=q_raw,
                raw_outputs=[str(x) for x in raw_solutions][:n1],
                choice_map=choice_map,
                stage="stage1",
                sleep_s=sleep_s,
                stats=None,
            )

            append_jsonl_line(
                stage1_infer_path,
                {
                    "uuid": uuid,
                    "line_number": r.get("line_number"),
                    "stage": "stage1_infer",
                    "question": q_raw,
                    "answer": gold,
                    "gold": gold,
                    "model_input": model_q,
                    "model_prompt_system": str(sys_p or ""),
                    "model_prompt_user": str(user_p or ""),
                    "raw_model_outputs": [str(x) for x in raw_solutions][:n1],
                    "extracted_answers": [str(x) for x in extracted][:n1],
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "llm_call_counts": {
                        "stage1_solve": llm.stage_params("stage1_solve").n,
                        "stage1_solve_http_calls": int(s1_solve_stats.get("http_calls", 0)),
                        "stage1_solve_retries": int(s1_solve_stats.get("retries", 0)),
                        "stage1_solve_timeouts": int(s1_solve_stats.get("timeouts", 0)),
                        "stage1_solve_errors": int(s1_solve_stats.get("errors", 0)),
                    },
                    "raw_source_path": r.get("raw_source_path"),
                },
            )
            prog.tick(time.monotonic() - t0)

        outs.append(stage1_infer_path)
        prog.finish()

    return outs


def mode_stage1_eval(*, input_arg: str, out_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> List[str]:
    """
    Route-A mode: stage1_eval
    Input (preferred): stage1_output.stage1.jsonl (file) or directory/out-root containing *stage1_output.stage1.jsonl.
        - stage1_eval derives routing purely from majority vote results (no gold-based judging).
        - Backward-compat: if no stage1_output artifacts are found, we fall back to stage1_infer.stage1.jsonl logic.
    Output: stage1/<prefix>.status.stage1.jsonl
            stage2/<prefix>.stage2_input.stage2.jsonl (for stage2_infer, optional convenience)
    """
    dirs = _ensure_stage_dirs(out_dir)
    stage1_dir = dirs["stage1"]
    stage2_dir = dirs["stage2"]

    outs: List[str] = []
    # Prefer stage1_output-based routing (no LLM required). If absent, fall back to stage1_infer-based evaluation.
    artifact_paths: List[str] = []
    if os.path.isdir(input_arg):
        artifact_paths = _iter_artifacts(input_arg, suffix="stage1_output.stage1.jsonl")
        if not artifact_paths:
            artifact_paths = _iter_artifacts(input_arg, suffix="stage1_infer.stage1.jsonl")
    else:
        artifact_paths = [input_arg]

    is_output_mode = True
    if artifact_paths:
        base0 = os.path.basename(artifact_paths[0])
        is_output_mode = base0.endswith("stage1_output.stage1.jsonl")

    multi = os.path.isdir(input_arg) and len(artifact_paths) > 1 and is_output_mode
    overall: _UUIDProgress | None = None
    if multi:
        total_all = 0
        done_all = 0
        for p in artifact_paths:
            prefix = _infer_prefix_from_artifact(p, suffix="stage1_output.stage1.jsonl")
            pfx = f"{prefix}." if prefix else ""
            stage1_status_path = os.path.join(dirs["stage1"], f"{pfx}status.stage1.jsonl")
            done_map = _load_status_map(stage1_status_path)
            uuids: List[str] = []
            for r in iter_jsonl(p, tolerate_errors=True):
                u = r.get("uuid")
                if u is not None:
                    uuids.append(str(u))
            total_all += len(uuids)
            done_all += sum(1 for u in uuids if u in done_map)
        overall = _UUIDProgress(label="stage1_eval", total=total_all, already_done=done_all, prefix="ALL")
        overall.start()

    for in_path in artifact_paths:
        base = os.path.basename(in_path)
        if base.endswith("stage1_output.stage1.jsonl"):
            # ---- Preferred: parse counts from existing stage1_output.output ----
            prefix = _infer_prefix_from_artifact(in_path, suffix="stage1_output.stage1.jsonl")
            pfx = f"{prefix}." if prefix else ""
            stage1_output_path = in_path
            stage1_status_path = os.path.join(stage1_dir, f"{pfx}status.stage1.jsonl")
            stage2_input_path = os.path.join(stage2_dir, f"{pfx}stage2_input.stage2.jsonl")
            stage1_infer_path = os.path.join(stage1_dir, f"{pfx}stage1_infer.stage1.jsonl")
            compact = _compact_outputs(llm)

            stage1_done = _load_status_map(stage1_status_path)
            stage2_inputs: List[Dict[str, Any]] = []

            all_rows: List[Dict[str, Any]] = []
            for r in iter_jsonl(stage1_output_path, tolerate_errors=True):
                uuid = r.get("uuid")
                if uuid is None:
                    continue
                all_rows.append(r)

            to_process = [r for r in all_rows if str(r.get("uuid")) not in stage1_done]
            prog: _UUIDProgress | None = None
            if not multi:
                prog = _UUIDProgress(label="stage1_eval", prefix=prefix, total=len(all_rows), already_done=len(all_rows) - len(to_process))
                prog.start()

            for r in to_process:
                t0 = time.monotonic()
                uuid = r.get("uuid")
                uuid_key = str(uuid)

                q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
                gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""

                # Preferred (output-mode): parse boxed ok/bad counts from the embedded output wrapper.
                # stage1_output.stage1.jsonl is expected to contain a judge-style output:
                #   \boxed{解答正确：x，解答错误：y}
                out0 = r.get("output") if isinstance(r.get("output"), dict) else {}
                counts = None
                try:
                    counts = extract_boxed_counts_from_output(out0)
                except Exception:
                    counts = None

                if counts is not None:
                    ok_i, bad_i = counts
                    try:
                        ok_i = int(ok_i)
                    except Exception:
                        ok_i = 0
                    try:
                        bad_i = int(bad_i)
                    except Exception:
                        bad_i = 0
                    ok_i = int(max(0, ok_i))
                    bad_i = int(max(0, bad_i))

                    # Keep the status schema stable: reuse majority_count slot to carry ok-count for routing.
                    # (There is no meaningful "majority answer" in judge-style stage1_output artifacts.)
                    majority_answer_json = {"majority": "", "majority_count": int(ok_i), "counts": {}}
                    sel1 = _select_answer(gold=gold, majority=majority_answer_json, min_votes_to_accept=min_votes_to_accept)
                    next_stage = "stage2" if int(sel1.get("final_vote_count") or 0) < int(min_votes_to_accept) else "accepted"
                    vote_raw_output = ""
                    vote_model_input = ""
                    vote_candidates = []
                else:
                    # Fallback (legacy): if the artifact contains extracted_answers, derive routing from vote strength.
                    extracted_answers = r.get("extracted_answers") if isinstance(r.get("extracted_answers"), list) else []
                    model_q = r.get("model_input") if isinstance(r.get("model_input"), str) else append_choice_map_if_any(normalize_for_model(q_raw))
                    choice_map = extract_choice_map(model_q)
                    vote1 = _llm_vote_majority_from_extracted(
                        llm=llm,
                        question=q_raw,
                        extracted_answers=[str(x or "") for x in extracted_answers],
                        choice_map=choice_map,
                        stage="stage1",
                        sleep_s=sleep_s,
                        stats=None,
                    )
                    majority_answer_json = {
                        "majority": str(vote1.get("majority") or ""),
                        "majority_count": int(vote1.get("majority_count") or 0),
                        "counts": (vote1.get("counts") if isinstance(vote1.get("counts"), dict) else {}),
                    }
                    vote_raw_output = str(vote1.get("raw_output") or "")
                    vote_model_input = str(vote1.get("model_input") or "")
                    vote_candidates = [str(x or "") for x in extracted_answers]
                    try:
                        n_total = int(llm.stage_params("stage1_solve").n)
                    except Exception:
                        n_total = int(len(extracted_answers) or 0)
                    maj_cnt = int(majority_answer_json.get("majority_count", 0) or 0)
                    ok_i = int(maj_cnt)
                    bad_i = int(max(0, n_total - maj_cnt))
                    sel1 = _select_answer(gold=gold, majority=majority_answer_json, min_votes_to_accept=min_votes_to_accept)
                    next_stage = "stage2" if int(sel1.get("final_vote_count") or 0) < int(min_votes_to_accept) else "accepted"

                paths = {"output": stage1_output_path}
                if os.path.exists(stage1_infer_path):
                    paths["infer"] = stage1_infer_path
                append_jsonl_line(
                    stage1_status_path,
                    {
                        "uuid": uuid,
                        "stage": "stage1",
                        "ok": int(ok_i),
                        "bad": int(bad_i),
                        "min_votes_to_accept": int(min_votes_to_accept),
                        "vote_majority": majority_answer_json.get("majority"),
                        "vote_majority_count": int(majority_answer_json.get("majority_count", 0)),
                        "vote_counts": (majority_answer_json.get("counts") if isinstance(majority_answer_json.get("counts"), dict) else {}),
                        "vote_raw_output": str(vote_raw_output or ""),
                        "vote_model_input": str(vote_model_input or ""),
                        "vote_candidates": [str(x or "") for x in (vote_candidates or [])],
                        **sel1,
                        "next_stage": next_stage,
                        "paths": paths,
                    },
                )
                stage1_done[uuid_key] = {"uuid": uuid, "ok": int(ok_i), "bad": int(bad_i), "next_stage": next_stage}

                if next_stage == "stage2":
                    stage2_inputs.append(
                        {
                            **{k: r.get(k) for k in CANONICAL_KEYS},
                            "_stage1_ok": int(ok_i),
                            "_stage1_bad": int(bad_i),
                            "_difficulty": int(bad_i),
                        }
                    )

                dt = time.monotonic() - t0
                if prog is not None:
                    prog.tick(dt)
                if overall is not None:
                    overall.tick(dt)

            if stage2_inputs and not compact:
                write_jsonl_atomic(stage2_input_path, stage2_inputs)
            outs.append(stage1_status_path)
            if prog is not None:
                prog.finish()
        else:
            # ---- Backward compat: old behavior (stage1_infer-based) ----
            stage1_infer_path = in_path
            prefix = _infer_prefix_from_artifact(stage1_infer_path, suffix="stage1_infer.stage1.jsonl")
            pfx = f"{prefix}." if prefix else ""
            stage1_output_path = os.path.join(stage1_dir, f"{pfx}stage1_output.stage1.jsonl")
            stage1_raw_generations_path = os.path.join(stage1_dir, f"{pfx}stage1_raw_generations.stage1.jsonl")
            stage1_status_path = os.path.join(stage1_dir, f"{pfx}status.stage1.jsonl")
            stage2_input_path = os.path.join(stage2_dir, f"{pfx}stage2_input.stage2.jsonl")
            compact = _compact_outputs(llm)

            stage1_done = _load_status_map(stage1_status_path)
            stage2_inputs: List[Dict[str, Any]] = []

            all_rows: List[Dict[str, Any]] = []
            for r in iter_jsonl(stage1_infer_path, tolerate_errors=True):
                uuid = r.get("uuid")
                if uuid is None:
                    continue
                all_rows.append(r)

            to_process = [r for r in all_rows if str(r.get("uuid")) not in stage1_done]
            prog: _UUIDProgress | None = None
            if not multi:
                prog = _UUIDProgress(label="stage1_eval", prefix=prefix, total=len(all_rows), already_done=len(all_rows) - len(to_process))
                prog.start()

            for r in to_process:
                t0 = time.monotonic()
                uuid = r.get("uuid")
                uuid_key = str(uuid)

                q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
                gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
                extracted_answers = r.get("extracted_answers") if isinstance(r.get("extracted_answers"), list) else []
                # Infer artifacts no longer store majority; derive it here from extracted_answers.
                majority_answer_json = r.get("majority_answer") if isinstance(r.get("majority_answer"), dict) else None
                if majority_answer_json is None:
                    model_q = r.get("model_input") if isinstance(r.get("model_input"), str) else append_choice_map_if_any(normalize_for_model(q_raw))
                    choice_map = extract_choice_map(model_q)
                    vote0 = _llm_vote_majority_from_extracted(
                        llm=llm,
                        question=q_raw,
                        extracted_answers=[str(x or "") for x in extracted_answers],
                        choice_map=choice_map,
                        stage="stage1",
                        sleep_s=sleep_s,
                        stats=None,
                    )
                    majority_answer_json = {
                        "majority": str(vote0.get("majority") or ""),
                        "majority_count": int(vote0.get("majority_count") or 0),
                        "counts": (vote0.get("counts") if isinstance(vote0.get("counts"), dict) else {}),
                    }

                try:
                    n_total = int(llm.stage_params("stage1_solve").n)
                except Exception:
                    n_total = int(len(extracted_answers) or 0)
                maj_cnt = int((majority_answer_json or {}).get("majority_count") or 0)
                ok_i = int(maj_cnt)
                bad_i = int(max(0, n_total - maj_cnt))

                sel1 = _select_answer(gold=gold, majority=majority_answer_json, min_votes_to_accept=min_votes_to_accept)
                next_stage = "stage2" if int(sel1.get("final_vote_count") or 0) < int(min_votes_to_accept) else "accepted"
                paths = {"infer": stage1_infer_path}
                append_jsonl_line(
                    stage1_status_path,
                    {
                        "uuid": uuid,
                        "stage": "stage1",
                        "ok": int(ok_i),
                        "bad": int(bad_i),
                        "min_votes_to_accept": int(min_votes_to_accept),
                        "vote_majority": majority_answer_json.get("majority"),
                        "vote_majority_count": int(majority_answer_json.get("majority_count", 0)),
                        "vote_counts": (majority_answer_json.get("counts") if isinstance(majority_answer_json.get("counts"), dict) else {}),
                        "vote_raw_output": str(vote0.get("raw_output") or ""),
                        "vote_model_input": str(vote0.get("model_input") or ""),
                        "vote_candidates": [str(x or "") for x in extracted_answers],
                        **sel1,
                        "next_stage": next_stage,
                        "paths": paths,
                    },
                )
                stage1_done[uuid_key] = {"uuid": uuid, "ok": int(ok_i), "bad": int(bad_i), "next_stage": next_stage}

                if next_stage == "stage2":
                    stage2_inputs.append(
                        {
                            **{k: r.get(k) for k in CANONICAL_KEYS},
                            "_stage1_ok": int(ok_i),
                            "_stage1_bad": int(bad_i),
                            "_difficulty": int(bad_i),
                        }
                    )
                dt = time.monotonic() - t0
                if prog is not None:
                    prog.tick(dt)
                if overall is not None:
                    overall.tick(dt)

            if stage2_inputs and not compact:
                write_jsonl_atomic(stage2_input_path, stage2_inputs)
            outs.append(stage1_status_path)
            if prog is not None:
                prog.finish()

    if overall is not None:
        overall.finish()
    return outs


def mode_stage2_infer(*, input_arg: str, out_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> List[str]:
    """
    Route-A mode: stage2_infer
    Input: stage2_input.stage2.jsonl (file) or a directory/out-root containing *stage2_input.stage2.jsonl files.
           (backward-compatible fallback: stage1_output.stage1.jsonl)
    Output: stage2/<prefix>.stage2_infer.stage2.jsonl (one per input artifact)
            stage2/<prefix>.stage2_input.stage2.jsonl (the derived next-step inputs)
    """
    dirs = _ensure_stage_dirs(out_dir)
    stage2_dir = dirs["stage2"]

    outs: List[str] = []
    # Prefer task-list input (stage2_input). If the user passes an out-root dir, this aligns with stage3_infer.
    # Fallbacks: stage1_output, then stage1_infer (compact mode).
    artifact_paths: List[str] = []
    if os.path.isdir(input_arg):
        artifact_paths = _iter_artifacts(input_arg, suffix="stage2_input.stage2.jsonl")
        if not artifact_paths:
            artifact_paths = _iter_artifacts(input_arg, suffix="stage1_output.stage1.jsonl")
        if not artifact_paths:
            artifact_paths = _iter_artifacts(input_arg, suffix="stage1_infer.stage1.jsonl")
    else:
        artifact_paths = [input_arg]

    multi = os.path.isdir(input_arg) and len(artifact_paths) > 1 and all(
        os.path.basename(p).endswith("stage2_input.stage2.jsonl") for p in artifact_paths
    )
    overall: _UUIDProgress | None = None
    if multi:
        total_all = 0
        done_all = 0
        for in_path in artifact_paths:
            prefix = _infer_prefix_from_artifact(in_path, suffix="stage2_input.stage2.jsonl")
            pfx = f"{prefix}." if prefix else ""
            stage2_infer_path = os.path.join(stage2_dir, f"{pfx}stage2_infer.stage2.jsonl")
            infer_done = _load_done_uuid_set(stage2_infer_path)
            t_i, d_i = _count_total_and_done_uuids(input_path=in_path, done_uuids=infer_done)
            total_all += int(t_i)
            done_all += int(d_i)
        overall = _UUIDProgress(label="stage2_infer", total=total_all, already_done=done_all, prefix="ALL")
        overall.start()

    for in_path in artifact_paths:
        base = os.path.basename(in_path)
        is_stage2_input = base.endswith("stage2_input.stage2.jsonl")
        is_stage1_output = base.endswith("stage1_output.stage1.jsonl")
        is_stage1_infer = base.endswith("stage1_infer.stage1.jsonl")

        if is_stage2_input:
            prefix = _infer_prefix_from_artifact(in_path, suffix="stage2_input.stage2.jsonl")
        elif is_stage1_output:
            prefix = _infer_prefix_from_artifact(in_path, suffix="stage1_output.stage1.jsonl")
        elif is_stage1_infer:
            prefix = _infer_prefix_from_artifact(in_path, suffix="stage1_infer.stage1.jsonl")
        else:
            raise ValueError(
                f"stage2_infer expects stage2_input.stage2.jsonl (preferred) or stage1_output.stage1.jsonl, got: {in_path}"
            )

        pfx = f"{prefix}." if prefix else ""
        stage2_infer_path = os.path.join(stage2_dir, f"{pfx}stage2_infer.stage2.jsonl")
        stage2_input_path = os.path.join(stage2_dir, f"{pfx}stage2_input.stage2.jsonl")

        compact = _compact_outputs(llm)

        # Ensure we have a stage2_input task list to consume, without materializing the whole file.
        effective_stage2_input = in_path
        if is_stage2_input:
            # Ensure a local copy exists (idempotent) without loading all rows into memory.
            try:
                need_copy = (not os.path.exists(stage2_input_path)) or (os.path.getsize(stage2_input_path) == 0)
            except Exception:
                need_copy = True
            if need_copy:
                try:
                    tmp = f"{stage2_input_path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
                    shutil.copyfile(in_path, tmp)
                    os.replace(tmp, stage2_input_path)
                except Exception:
                    # Best-effort; the pipeline can still read from in_path directly.
                    pass
            effective_stage2_input = in_path
        else:
            # Backward-compatible: derive stage2_input from stage1_output/stage1_infer + status.stage1.
            if not compact:
                effective_stage2_input = stage2_input_path
                try:
                    need_build = (not os.path.exists(stage2_input_path)) or (os.path.getsize(stage2_input_path) == 0)
                except Exception:
                    need_build = True
                if need_build:
                    stage1_status_path = os.path.join(os.path.dirname(in_path), f"{pfx}status.stage1.jsonl")
                    stage1_status = _load_status_map(stage1_status_path)

                    def _rows() -> Any:
                        for row in iter_jsonl(in_path, tolerate_errors=True):
                            u = row.get("uuid")
                            if u is None:
                                continue
                            u_key = str(u)
                            st = stage1_status.get(u_key, {})
                            fv = st.get("final_vote_count")
                            mv = st.get("min_votes_to_accept")
                            try:
                                fv_i = int(fv) if fv is not None else None
                            except Exception:
                                fv_i = None
                            try:
                                mv_i = int(mv) if mv is not None else int(min_votes_to_accept)
                            except Exception:
                                mv_i = int(min_votes_to_accept)

                            if fv_i is not None:
                                need_stage2 = fv_i < mv_i
                            else:
                                need_stage2 = str(st.get("next_stage") or "") == "stage2"
                            if not need_stage2:
                                continue

                            okc = st.get("ok")
                            badc = st.get("bad")
                            try:
                                okc_i = int(okc) if okc is not None else 0
                            except Exception:
                                okc_i = 0
                            try:
                                badc_i = int(badc) if badc is not None else 0
                            except Exception:
                                badc_i = 0
                            yield {
                                **{k: row.get(k) for k in CANONICAL_KEYS},
                            }

                    _write_jsonl_stream_atomic(stage2_input_path, _rows())
            else:
                effective_stage2_input = in_path

        infer_done = _load_done_uuid_set(stage2_infer_path)
        stage1_status: Dict[str, Any] = {}
        if compact and not is_stage2_input:
            stage1_status_path = os.path.join(os.path.dirname(in_path), f"{pfx}status.stage1.jsonl")
            stage1_status = _load_status_map(stage1_status_path)
            # Count rows needing stage2 directly from stage1_infer/output + status.
            total_rows = 0
            done_rows = 0
            for row in iter_jsonl(in_path, tolerate_errors=True):
                u = row.get("uuid")
                if u is None:
                    continue
                u_key = str(u)
                st = stage1_status.get(u_key, {})
                fv = st.get("final_vote_count")
                mv = st.get("min_votes_to_accept")
                try:
                    fv_i = int(fv) if fv is not None else None
                except Exception:
                    fv_i = None
                try:
                    mv_i = int(mv) if mv is not None else int(min_votes_to_accept)
                except Exception:
                    mv_i = int(min_votes_to_accept)
                if fv_i is not None:
                    need_stage2 = fv_i < mv_i
                elif st:
                    need_stage2 = str(st.get("next_stage") or "") == "stage2"
                else:
                    if is_stage1_output:
                        need_stage2 = True
                    else:
                        maj = row.get("majority_answer") if isinstance(row.get("majority_answer"), dict) else {}
                        try:
                            maj_cnt = int((maj or {}).get("majority_count") or 0)
                        except Exception:
                            maj_cnt = 0
                        need_stage2 = maj_cnt < int(min_votes_to_accept)
                if not need_stage2:
                    continue
                total_rows += 1
                if u_key in infer_done:
                    done_rows += 1
        else:
            total_rows, done_rows = _count_total_and_done_uuids(input_path=effective_stage2_input, done_uuids=infer_done)
        total_rows = int(total_rows)
        done_rows = int(done_rows)

        prog: _UUIDProgress | None = None
        if not multi:
            prog = _UUIDProgress(label="stage2_infer", prefix=prefix, total=total_rows, already_done=done_rows)
            prog.start()

        for r in iter_jsonl(effective_stage2_input, tolerate_errors=True):
            uuid = r.get("uuid")
            if uuid is None:
                continue
            uuid_key = str(uuid)
            if uuid_key in infer_done:
                continue
            if compact and not is_stage2_input:
                # Filter rows using status if present (only keep stage2 candidates).
                st = stage1_status.get(uuid_key, {})
                fv = st.get("final_vote_count")
                mv = st.get("min_votes_to_accept")
                try:
                    fv_i = int(fv) if fv is not None else None
                except Exception:
                    fv_i = None
                try:
                    mv_i = int(mv) if mv is not None else int(min_votes_to_accept)
                except Exception:
                    mv_i = int(min_votes_to_accept)
                if fv_i is not None:
                    need_stage2 = fv_i < mv_i
                elif st:
                    need_stage2 = str(st.get("next_stage") or "") == "stage2"
                else:
                    if is_stage1_output:
                        need_stage2 = True
                    else:
                        maj = r.get("majority_answer") if isinstance(r.get("majority_answer"), dict) else {}
                        try:
                            maj_cnt = int((maj or {}).get("majority_count") or 0)
                        except Exception:
                            maj_cnt = 0
                        need_stage2 = maj_cnt < int(min_votes_to_accept)
                if not need_stage2:
                    continue
                okc = st.get("ok")
                badc = st.get("bad")
                try:
                    okc_i = int(okc) if okc is not None else None
                except Exception:
                    okc_i = None
                try:
                    badc_i = int(badc) if badc is not None else None
                except Exception:
                    badc_i = None
            t0 = time.monotonic()
            q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
            gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
            model_q = append_choice_map_if_any(normalize_for_model(q_raw))
            choice_map = extract_choice_map(model_q)
            s2_solve_stats: Dict[str, int] = {}
            raw_solutions = llm.generate_n(
                stage_name="stage2_solve",
                question=model_q,
                prompt_mode="problem",
                sleep_s=sleep_s,
                stats=s2_solve_stats,
            )
            n2 = int(llm.stage_params("stage2_solve").n)
            s2_extract_stats: Dict[str, int] = {}
            sys_p, user_p = _render_solve_prompts(llm=llm, stage_name="stage2_solve", question_for_model=model_q, n=n2)
            extracted = _llm_extract_answers_batch(
                llm=llm,
                uuid=uuid,
                question=q_raw,
                raw_outputs=[str(x) for x in raw_solutions][:n2],
                choice_map=choice_map,
                stage="stage2",
                sleep_s=sleep_s,
                stats=s2_extract_stats,
            )
            _append_stage_infer_row(
                infer_path=stage2_infer_path,
                stage="stage2",
                row=r,
                model_input=model_q,
                model_prompt_system=sys_p,
                model_prompt_user=user_p,
                raw_solutions=[str(x) for x in raw_solutions],
                extracted_answers=extracted,
                n=n2,
                min_votes_to_accept=min_votes_to_accept,
                solve_stats=s2_solve_stats,
            )
            dt = time.monotonic() - t0
            if prog is not None:
                prog.tick(dt)
            if overall is not None:
                overall.tick(dt)

        outs.append(stage2_infer_path)
        if prog is not None:
            prog.finish()
    if overall is not None:
        overall.finish()
    return outs


def mode_stage2_eval(*, input_arg: str, out_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> List[str]:
    """
    Route-A mode: stage2_eval
    Input: stage2_infer.stage2.jsonl (file) or directory containing *stage2_infer.stage2.jsonl files.
    Output: stage2/<prefix>.stage2_archive.stage2.jsonl + stage2/<prefix>.status.stage2.jsonl
            stage3/<prefix>.stage3_input.stage3.jsonl (for stage3_infer)
    """
    dirs = _ensure_stage_dirs(out_dir)
    stage2_dir = dirs["stage2"]
    stage3_dir = dirs["stage3"]
    outs: List[str] = []

    infer_paths = _iter_artifacts(input_arg, suffix="stage2_infer.stage2.jsonl")
    multi = os.path.isdir(input_arg) and len(infer_paths) > 1
    overall: _UUIDProgress | None = None
    if multi:
        total_all = 0
        done_all = 0
        for p in infer_paths:
            prefix = _infer_prefix_from_artifact(p, suffix="stage2_infer.stage2.jsonl")
            pfx = f"{prefix}." if prefix else ""
            stage2_status_path = os.path.join(stage2_dir, f"{pfx}status.stage2.jsonl")
            done_set = _load_done_uuid_set(stage2_status_path)
            t_i, d_i = _count_total_and_done_uuids(input_path=p, done_uuids=done_set)
            total_all += int(t_i)
            done_all += int(d_i)
        overall = _UUIDProgress(label="stage2_eval", total=total_all, already_done=done_all, prefix="ALL")
        overall.start()

    for stage2_infer_path in infer_paths:
        prefix = _infer_prefix_from_artifact(stage2_infer_path, suffix="stage2_infer.stage2.jsonl")
        pfx = f"{prefix}." if prefix else ""
        stage2_archive_path = os.path.join(stage2_dir, f"{pfx}stage2_archive.stage2.jsonl")
        stage2_raw_generations_path = os.path.join(stage2_dir, f"{pfx}stage2_raw_generations.stage2.jsonl")
        stage2_status_path = os.path.join(stage2_dir, f"{pfx}status.stage2.jsonl")
        stage3_input_path = os.path.join(stage3_dir, f"{pfx}stage3_input.stage3.jsonl")
        compact = _compact_outputs(llm)

        done_set = _load_done_uuid_set(stage2_status_path)
        total_rows, done_rows = _count_total_and_done_uuids(input_path=stage2_infer_path, done_uuids=done_set)
        prog: _UUIDProgress | None = None
        if not multi:
            prog = _UUIDProgress(label="stage2_eval", prefix=prefix, total=int(total_rows), already_done=int(done_rows))
            prog.start()

        for r in iter_jsonl(stage2_infer_path, tolerate_errors=True):
            uuid = r.get("uuid")
            if uuid is None:
                continue
            uuid_key = str(uuid)
            if uuid_key in done_set:
                continue
            t0 = time.monotonic()
            q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
            gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
            model_q = r.get("model_input") if isinstance(r.get("model_input"), str) else append_choice_map_if_any(normalize_for_model(q_raw))
            raw_model_outputs = r.get("raw_model_outputs") if isinstance(r.get("raw_model_outputs"), list) else []
            extracted_answers = r.get("extracted_answers") if isinstance(r.get("extracted_answers"), list) else []
            n2 = int(llm.stage_params("stage2_solve").n)
            choice_map = extract_choice_map(model_q)

            stage2_attempts: List[Dict[str, Any]] = []
            extracted_trim = [str(x or "").strip() for x in extracted_answers[:n2]]
            s2_vote_stats: Dict[str, int] = {}
            vote2 = _llm_vote_majority_from_extracted(
                llm=llm,
                question=q_raw,
                extracted_answers=extracted_trim,
                choice_map=choice_map,
                stage="stage2",
                sleep_s=sleep_s,
                stats=s2_vote_stats,
            )
            normalized2 = vote2.get("normalized")
            if not (isinstance(normalized2, list) and len(normalized2) == len(extracted_trim)):
                normalized2 = extracted_trim[:]
            normalized2 = [str(x or "").strip() for x in normalized2]
            for raw, ex_i, norm_i in zip(raw_model_outputs[:n2], extracted_trim, normalized2):
                stage2_attempts.append(
                    {
                        "raw_text": str(raw),
                        "boxed_answer": "",
                        "extracted_answer": str(ex_i or "").strip(),
                        "normalized_answer": str(norm_i or "").strip(),
                    }
                )

            stage2_majority_answer = {
                "majority": str(vote2.get("majority") or ""),
                "majority_count": int(vote2.get("majority_count") or 0),
                "counts": (vote2.get("counts") if isinstance(vote2.get("counts"), dict) else {}),
            }
            vote_model_input = str(vote2.get("model_input") or "")
            sel2 = _select_answer(gold=gold, majority=stage2_majority_answer, min_votes_to_accept=min_votes_to_accept)
            final_answer2 = str(sel2.get("final_answer") or "").strip()
            maj_cnt = int(stage2_majority_answer.get("majority_count", 0) or 0)

            entry: Dict[str, Any] = {
                "uuid": uuid,
                "line_number": r.get("line_number"),
                "stage": "stage2",
                "question": q_raw,
                "gold": gold,
                "model_input": model_q,
                "raw_model_outputs": [str(x) for x in raw_model_outputs][:n2],
                "extracted_answers": [str(x) for x in extracted_answers][:n2],
                "majority_answer": stage2_majority_answer,
                "attempts": stage2_attempts,
                **sel2,
                "ok": int(maj_cnt),
                "bad": int(max(0, int(n2) - maj_cnt)),
                "llm_call_counts": {
                    **(r.get("llm_call_counts") if isinstance(r.get("llm_call_counts"), dict) else {}),
                    "stage2_vote_http_calls": int(s2_vote_stats.get("http_calls", 0)),
                    "stage2_vote_retries": int(s2_vote_stats.get("retries", 0)),
                    "stage2_vote_timeouts": int(s2_vote_stats.get("timeouts", 0)),
                    "stage2_vote_errors": int(s2_vote_stats.get("errors", 0)),
                },
            }
            # Keep a stage2 raw-generations file aligned with stage1_raw_generations:
            # it captures multi-sample raw outputs + attempt judgments + voting stats (no stage3 routing).
            if not compact:
                raw_entry = dict(entry)
                raw_entry.pop("final_answer", None)
                raw_entry.pop("final_source", None)
                raw_entry.pop("final_vote_count", None)
                append_jsonl_line(stage2_raw_generations_path, raw_entry)
                append_jsonl_line(stage2_archive_path, entry)

            # Routing policy:
            # If final_vote_count < min_votes_to_accept => go to next stage; else accepted.
            try:
                fv_i = int(sel2.get("final_vote_count") or 0)
            except Exception:
                fv_i = 0
            next_stage = "stage3" if fv_i < int(min_votes_to_accept) else "accepted"
        paths = {"infer": stage2_infer_path}
        if not compact:
            paths["archive"] = stage2_archive_path
        append_jsonl_line(
            stage2_status_path,
            {
                "uuid": uuid,
                "stage": "stage2",
                    "ok": int(maj_cnt),
                    "bad": int(max(0, int(n2) - maj_cnt)),
                "min_votes_to_accept": int(min_votes_to_accept),
                "vote_majority": stage2_majority_answer.get("majority"),
                "vote_majority_count": int(stage2_majority_answer.get("majority_count", 0)),
                "vote_counts": (stage2_majority_answer.get("counts") if isinstance(stage2_majority_answer.get("counts"), dict) else {}),
                "vote_raw_output": str(vote2.get("raw_output") or ""),
                "vote_model_input": str(vote_model_input or ""),
                "vote_candidates": [str(x or "") for x in extracted_trim],
                **_select_answer(gold=gold, majority=stage2_majority_answer, min_votes_to_accept=min_votes_to_accept),
                "next_stage": next_stage,
                "paths": paths,
            },
        )

        if next_stage == "stage3" and not compact:
            append_jsonl_line(
                stage3_input_path,
                {
                    **{k: r.get(k) for k in CANONICAL_KEYS},
                },
            )
            dt = time.monotonic() - t0
            if prog is not None:
                prog.tick(dt)
            if overall is not None:
                overall.tick(dt)

        outs.append(stage2_status_path)
        if prog is not None:
            prog.finish()

    if overall is not None:
        overall.finish()
    return outs


def mode_stage3_infer(*, input_arg: str, out_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> List[str]:
    """
    Route-A mode: stage3_infer
    Input: stage3_input.stage3.jsonl (file) or directory containing *stage3_input.stage3.jsonl files.
    Output: stage3/<prefix>.stage3_infer.stage3.jsonl
    """
    dirs = _ensure_stage_dirs(out_dir)
    stage2_dir = dirs["stage2"]
    stage3_dir = dirs["stage3"]
    outs: List[str] = []

    input_paths = _iter_artifacts(input_arg, suffix="stage3_input.stage3.jsonl")
    if not input_paths:
        input_paths = _iter_artifacts(input_arg, suffix="stage2_infer.stage2.jsonl")
    multi = os.path.isdir(input_arg) and len(input_paths) > 1
    overall: _UUIDProgress | None = None
    if multi:
        total_all = 0
        done_all = 0
        for p in input_paths:
            prefix = _infer_prefix_from_artifact(p, suffix="stage3_input.stage3.jsonl")
            pfx = f"{prefix}." if prefix else ""
            stage3_infer_path = os.path.join(stage3_dir, f"{pfx}stage3_infer.stage3.jsonl")
            infer_done = _load_done_uuid_set(stage3_infer_path)
            t_i, d_i = _count_total_and_done_uuids(input_path=p, done_uuids=infer_done)
            total_all += int(t_i)
            done_all += int(d_i)
        overall = _UUIDProgress(label="stage3_infer", total=total_all, already_done=done_all, prefix="ALL")
        overall.start()

    for stage3_input_path in input_paths:
        base = os.path.basename(stage3_input_path)
        is_stage3_input = base.endswith("stage3_input.stage3.jsonl")
        if is_stage3_input:
            prefix = _infer_prefix_from_artifact(stage3_input_path, suffix="stage3_input.stage3.jsonl")
        else:
            prefix = _infer_prefix_from_artifact(stage3_input_path, suffix="stage2_infer.stage2.jsonl")
        pfx = f"{prefix}." if prefix else ""
        stage3_infer_path = os.path.join(stage3_dir, f"{pfx}stage3_infer.stage3.jsonl")

        infer_done = _load_done_uuid_set(stage3_infer_path)
        stage2_status: Dict[str, Any] = {}
        if not is_stage3_input:
            stage2_status_path = os.path.join(stage2_dir, f"{pfx}status.stage2.jsonl")
            stage2_status = _load_status_map(stage2_status_path)
        if is_stage3_input:
            total_rows, done_rows = _count_total_and_done_uuids(input_path=stage3_input_path, done_uuids=infer_done)
        else:
            # Derive stage3 candidates from stage2_infer + status.stage2
            total_rows = 0
            done_rows = 0
            for row in iter_jsonl(stage3_input_path, tolerate_errors=True):
                u = row.get("uuid")
                if u is None:
                    continue
                u_key = str(u)
                st = stage2_status.get(u_key, {})
                fv = st.get("final_vote_count")
                mv = st.get("min_votes_to_accept")
                try:
                    fv_i = int(fv) if fv is not None else None
                except Exception:
                    fv_i = None
                try:
                    mv_i = int(mv) if mv is not None else int(min_votes_to_accept)
                except Exception:
                    mv_i = int(min_votes_to_accept)
                if fv_i is not None:
                    need_stage3 = fv_i < mv_i
                elif st:
                    need_stage3 = str(st.get("next_stage") or "") == "stage3"
                else:
                    # Fallback: vote from extracted_answers if available
                    extracted = row.get("extracted_answers") if isinstance(row.get("extracted_answers"), list) else []
                    cleaned = [str(x or "").strip() for x in extracted if str(x or "").strip()]
                    v = majority_vote(cleaned) if cleaned else None
                    need_stage3 = True
                    if v is not None and int(v.majority_count or 0) >= int(min_votes_to_accept):
                        need_stage3 = False
                if not need_stage3:
                    continue
                total_rows += 1
                if u_key in infer_done:
                    done_rows += 1
        prog: _UUIDProgress | None = None
        if not multi:
            prog = _UUIDProgress(label="stage3_infer", prefix=prefix, total=int(total_rows), already_done=int(done_rows))
            prog.start()

        for r in iter_jsonl(stage3_input_path, tolerate_errors=True):
            uuid = r.get("uuid")
            if uuid is None:
                continue
            uuid_key = str(uuid)
            if uuid_key in infer_done:
                continue
            if not is_stage3_input:
                st = stage2_status.get(uuid_key, {})
                fv = st.get("final_vote_count")
                mv = st.get("min_votes_to_accept")
                try:
                    fv_i = int(fv) if fv is not None else None
                except Exception:
                    fv_i = None
                try:
                    mv_i = int(mv) if mv is not None else int(min_votes_to_accept)
                except Exception:
                    mv_i = int(min_votes_to_accept)
                if fv_i is not None:
                    need_stage3 = fv_i < mv_i
                elif st:
                    need_stage3 = str(st.get("next_stage") or "") == "stage3"
                else:
                    extracted = r.get("extracted_answers") if isinstance(r.get("extracted_answers"), list) else []
                    cleaned = [str(x or "").strip() for x in extracted if str(x or "").strip()]
                    v = majority_vote(cleaned) if cleaned else None
                    need_stage3 = True
                    if v is not None and int(v.majority_count or 0) >= int(min_votes_to_accept):
                        need_stage3 = False
                if not need_stage3:
                    continue
            t0 = time.monotonic()
            q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
            gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
            model_q = append_choice_map_if_any(normalize_for_model(q_raw))
            choice_map = extract_choice_map(model_q)
            s3_solve_stats: Dict[str, int] = {}
            raw_solutions = llm.generate_n(
                stage_name="stage3_solve",
                question=model_q,
                prompt_mode="problem",
                sleep_s=sleep_s,
                stats=s3_solve_stats,
            )
            n3 = int(llm.stage_params("stage3_solve").n)
            s3_extract_stats: Dict[str, int] = {}
            sys_p, user_p = _render_solve_prompts(llm=llm, stage_name="stage3_solve", question_for_model=model_q, n=n3)
            extracted = _llm_extract_answers_batch(
                llm=llm,
                uuid=uuid,
                question=q_raw,
                raw_outputs=[str(x) for x in raw_solutions][:n3],
                choice_map=choice_map,
                stage="stage3",
                sleep_s=sleep_s,
                stats=s3_extract_stats,
            )
            _append_stage_infer_row(
                infer_path=stage3_infer_path,
                stage="stage3",
                row=r,
                model_input=model_q,
                model_prompt_system=sys_p,
                model_prompt_user=user_p,
                raw_solutions=[str(x) for x in raw_solutions],
                extracted_answers=extracted,
                n=n3,
                min_votes_to_accept=min_votes_to_accept,
                solve_stats=s3_solve_stats,
            )
            dt = time.monotonic() - t0
            if prog is not None:
                prog.tick(dt)
            if overall is not None:
                overall.tick(dt)
        outs.append(stage3_infer_path)
        if prog is not None:
            prog.finish()
    if overall is not None:
        overall.finish()
    return outs


def mode_stage3_eval(*, input_arg: str, out_dir: str, llm: LLMRouter, min_votes_to_accept: int, sleep_s: float) -> List[str]:
    """
    Route-A mode: stage3_eval
    Input: stage3_infer.stage3.jsonl (file) or directory containing *stage3_infer.stage3.jsonl files.
    Output: stage3/<prefix>.stage3_archive.stage3.jsonl + stage3/<prefix>.status.stage3.jsonl
            accepted_bank + result (same conventions as the original pipeline)
    """
    dirs = _ensure_stage_dirs(out_dir)
    stage3_dir = dirs["stage3"]
    outs: List[str] = []

    infer_paths = _iter_artifacts(input_arg, suffix="stage3_infer.stage3.jsonl")
    multi = os.path.isdir(input_arg) and len(infer_paths) > 1
    overall: _UUIDProgress | None = None
    if multi:
        total_all = 0
        done_all = 0
        for p in infer_paths:
            prefix = _infer_prefix_from_artifact(p, suffix="stage3_infer.stage3.jsonl")
            pfx = f"{prefix}." if prefix else ""
            stage3_status_path = os.path.join(stage3_dir, f"{pfx}status.stage3.jsonl")
            done_set = _load_done_uuid_set(stage3_status_path)
            t_i, d_i = _count_total_and_done_uuids(input_path=p, done_uuids=done_set)
            total_all += int(t_i)
            done_all += int(d_i)
        overall = _UUIDProgress(label="stage3_eval", total=total_all, already_done=done_all, prefix="ALL")
        overall.start()

    for stage3_infer_path in infer_paths:
        prefix = _infer_prefix_from_artifact(stage3_infer_path, suffix="stage3_infer.stage3.jsonl")
        pfx = f"{prefix}." if prefix else ""
        stage3_archive_path = os.path.join(stage3_dir, f"{pfx}stage3_archive.stage3.jsonl")
        stage3_raw_generations_path = os.path.join(stage3_dir, f"{pfx}stage3_raw_generations.stage3.jsonl")
        stage3_status_path = os.path.join(stage3_dir, f"{pfx}status.stage3.jsonl")

        accepted_bank_path = os.path.join(out_dir, f"{pfx}accepted_bank.stage_final.jsonl")
        result_path = os.path.join(out_dir, "result", f"{pfx}result.stage_final.jsonl")
        compact = _compact_outputs(llm)

        stage3_done_set = _load_done_uuid_set(stage3_status_path)
        total_rows, done_rows = _count_total_and_done_uuids(input_path=stage3_infer_path, done_uuids=stage3_done_set)
        prog: _UUIDProgress | None = None
        if not multi:
            prog = _UUIDProgress(label="stage3_eval", prefix=prefix, total=int(total_rows), already_done=int(done_rows))
            prog.start()

        for r in iter_jsonl(stage3_infer_path, tolerate_errors=True):
            uuid = r.get("uuid")
            if uuid is None:
                continue
            uuid_key = str(uuid)
            if uuid_key in stage3_done_set:
                continue
            t0 = time.monotonic()
            q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
            gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
            model_q = r.get("model_input") if isinstance(r.get("model_input"), str) else append_choice_map_if_any(normalize_for_model(q_raw))
            raw_model_outputs = r.get("raw_model_outputs") if isinstance(r.get("raw_model_outputs"), list) else []
            extracted_answers = r.get("extracted_answers") if isinstance(r.get("extracted_answers"), list) else []
            n3 = int(llm.stage_params("stage3_solve").n)
            choice_map = extract_choice_map(model_q)

            stage3_attempts: List[Dict[str, Any]] = []
            extracted_trim = [str(x or "").strip() for x in extracted_answers[:n3]]
            s3_vote_stats: Dict[str, int] = {}
            vote3 = _llm_vote_majority_from_extracted(
                llm=llm,
                question=q_raw,
                extracted_answers=extracted_trim,
                choice_map=choice_map,
                stage="stage3",
                sleep_s=sleep_s,
                stats=s3_vote_stats,
            )
            normalized3 = vote3.get("normalized")
            if not (isinstance(normalized3, list) and len(normalized3) == len(extracted_trim)):
                normalized3 = extracted_trim[:]
            normalized3 = [str(x or "").strip() for x in normalized3]
            for raw, ex_i, norm_i in zip(raw_model_outputs[:n3], extracted_trim, normalized3):
                stage3_attempts.append(
                    {
                        "raw_text": str(raw),
                        "boxed_answer": "",
                        "extracted_answer": str(ex_i or "").strip(),
                        "normalized_answer": str(norm_i or "").strip(),
                    }
                )

            stage3_majority_answer = {
                "majority": str(vote3.get("majority") or ""),
                "majority_count": int(vote3.get("majority_count") or 0),
                "counts": (vote3.get("counts") if isinstance(vote3.get("counts"), dict) else {}),
            }
            vote_model_input = str(vote3.get("model_input") or "")
            sel3 = _select_answer(gold=gold, majority=stage3_majority_answer, min_votes_to_accept=min_votes_to_accept)
            final_answer3 = str(sel3.get("final_answer") or "").strip()
            maj_cnt = int(stage3_majority_answer.get("majority_count", 0) or 0)

            entry = {
                "uuid": uuid,
                "line_number": r.get("line_number"),
                "stage": "stage3",
                "question": q_raw,
                "gold": gold,
                "model_input": model_q,
                "raw_model_outputs": [str(x) for x in raw_model_outputs][:n3],
                "extracted_answers": [str(x) for x in extracted_answers][:n3],
                "majority_answer": stage3_majority_answer,
                "attempts": stage3_attempts,
                **sel3,
                "ok": int(maj_cnt),
                "bad": int(max(0, int(n3) - maj_cnt)),
                "llm_call_counts": {
                    **(r.get("llm_call_counts") if isinstance(r.get("llm_call_counts"), dict) else {}),
                    "stage3_vote_http_calls": int(s3_vote_stats.get("http_calls", 0)),
                    "stage3_vote_retries": int(s3_vote_stats.get("retries", 0)),
                    "stage3_vote_timeouts": int(s3_vote_stats.get("timeouts", 0)),
                    "stage3_vote_errors": int(s3_vote_stats.get("errors", 0)),
                },
            }

            if not compact:
                # Keep a stage3 raw-generations file aligned with stage1_raw_generations.
                raw_entry = dict(entry)
                raw_entry.pop("final_answer", None)
                raw_entry.pop("final_source", None)
                raw_entry.pop("final_vote_count", None)
                append_jsonl_line(stage3_raw_generations_path, raw_entry)
                append_jsonl_line(stage3_archive_path, entry)
                if sel3.get("final_source") == "majority":
                    append_jsonl_line(accepted_bank_path, {**entry, **sel3, "accepted_from": "stage3"})
            paths = {"infer": stage3_infer_path}
            if not compact:
                paths.update({"archive": stage3_archive_path, "result": result_path})
            append_jsonl_line(
                stage3_status_path,
                {
                    "uuid": uuid,
                    "stage": "stage3",
                    "ok": int(maj_cnt),
                    "bad": int(max(0, int(n3) - maj_cnt)),
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "vote_majority": stage3_majority_answer.get("majority"),
                    "vote_majority_count": int(stage3_majority_answer.get("majority_count", 0)),
                    "vote_counts": (stage3_majority_answer.get("counts") if isinstance(stage3_majority_answer.get("counts"), dict) else {}),
                    "vote_raw_output": str(vote3.get("raw_output") or ""),
                    "vote_model_input": str(vote_model_input or ""),
                    "vote_candidates": [str(x or "") for x in extracted_trim],
                    **sel3,
                    "next_stage": ("accepted" if sel3.get("final_source") == "majority" else "no_answer"),
                    "paths": paths,
                },
            )
            dt = time.monotonic() - t0
            if prog is not None:
                prog.tick(dt)
            if overall is not None:
                overall.tick(dt)

        outs.append(stage3_status_path)
        if prog is not None:
            prog.finish()

    if overall is not None:
        overall.finish()
    return outs


def main() -> None:
    p = argparse.ArgumentParser(description="MathAgent minimal pipeline (JSONL between stages).")
    p.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help=(
            "Execution mode (modular only):\n"
            "  - stage2_infer: input is stage1_output.stage1.jsonl (or a dir of them) -> emit stage2_infer\n"
            "  - stage2_eval: input is stage2_infer.stage2.jsonl (or a dir of them) -> emit stage2 status + stage3_input\n"
            "  - stage3_infer: input is stage3_input.stage3.jsonl (or a dir of them) -> emit stage3_infer\n"
            "  - stage3_eval: input is stage3_infer.stage3.jsonl (or a dir of them) -> emit stage3 status + result"
        ),
    )
    p.add_argument("--input", default=None, help="Input JSONL file path OR a directory containing JSONL files.")
    p.add_argument("--out", default=_default_out_dir(), help="Output directory (default datasets/out/<timestamp>)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM calls (rate limit)")
    p.add_argument(
        "--llm-config",
        default="config/llm_models.json",
        help="Path to JSON config describing models + stage routing.",
    )
    p.add_argument("--vllm-model", default=None, help="Override vLLM model path for autostart/restart.")
    p.add_argument("--vllm-served-model-name", default=None, help="Override vLLM served model name.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    mode = str(args.mode or "").strip()
    if not args.input:
        raise ValueError("--mode requires --input as the artifact path (file or directory).")

    llm = LLMRouter(config_path=args.llm_config)
    llm.override_options(
        {
            "vllm_model_path": args.vllm_model,
            "vllm_model_name": args.vllm_served_model_name,
        }
    )
    min_votes_to_accept = llm.threshold_int("min_votes_to_accept", 5)

    # ---- Startup cleanup: purge connection-error rows so reruns can reprocess those uuids ----
    # Default: enabled; toggle via config option `purge_conn_errors_on_start`.
    if llm.option_bool("purge_conn_errors_on_start", True):
        bad = _scan_bad_uuids_for_conn_errors(str(args.out))
        if bad:
            removed = _purge_uuids_from_out_dir(str(args.out), bad)
            # Log a compact summary to stderr (so stdout stays clean for pipelines).
            total_removed = sum(int(v) for v in removed.values())
            print(
                f"[CLEAN] Purged {len(bad)} uuid(s) with connection errors from {len(removed)} file(s); "
                f"removed_lines={total_removed}. Disable via options.purge_conn_errors_on_start=false.",
                file=sys.stderr,
                flush=True,
            )
    # ---- Route-A modular modes ----
    assert args.input
    if mode == "result_rebuild":
        # Rebuild results from existing artifacts; uses --input as out_dir.
        result_rebuild(
            out_dir=str(args.input),
            min_votes_to_accept=min_votes_to_accept,
        )
        print("Done.")
        print(f"- mode: {mode}")
        print(f"- out_dir: {args.input}")
        return

    stage1_dir = os.path.join(args.out, "stage1")
    stage2_dir = os.path.join(args.out, "stage2")
    stage3_dir = os.path.join(args.out, "stage3")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(stage3_dir, exist_ok=True)

    _maybe_autostart_vllm(llm)
    if llm.option_bool("vllm_shutdown_on_exit", False):
        atexit.register(_maybe_shutdown_vllm, llm)

    if mode == "stage1_infer":
        outs = mode_stage1_infer(
            input_arg=args.input,
            out_dir=args.out,
            llm=llm,
            min_votes_to_accept=min_votes_to_accept,
            sleep_s=float(args.sleep),
        )
    elif mode == "stage1_eval":
        outs = mode_stage1_eval(
            input_arg=args.input,
            out_dir=args.out,
            llm=llm,
            min_votes_to_accept=min_votes_to_accept,
            sleep_s=float(args.sleep),
        )
    elif mode == "stage2_infer":
        outs = mode_stage2_infer(
            input_arg=args.input,
            out_dir=args.out,
            llm=llm,
            min_votes_to_accept=min_votes_to_accept,
            sleep_s=float(args.sleep),
        )
    elif mode == "stage2_eval":
        outs = mode_stage2_eval(
            input_arg=args.input,
            out_dir=args.out,
            llm=llm,
            min_votes_to_accept=min_votes_to_accept,
            sleep_s=float(args.sleep),
        )
    elif mode == "stage3_infer":
        outs = mode_stage3_infer(
            input_arg=args.input,
            out_dir=args.out,
            llm=llm,
            min_votes_to_accept=min_votes_to_accept,
            sleep_s=float(args.sleep),
        )
    elif mode == "stage3_eval":
        outs = mode_stage3_eval(
            input_arg=args.input,
            out_dir=args.out,
            llm=llm,
            min_votes_to_accept=min_votes_to_accept,
            sleep_s=float(args.sleep),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("Done.")
    print(f"- mode: {mode}")
    print(f"- input: {args.input}")
    print(f"- out_dir: {args.out}")
    for pth in outs:
        print(f"- output: {pth}")
    return


if __name__ == "__main__":
    main()


