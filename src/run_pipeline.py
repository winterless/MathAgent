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

from core.prompt_assemble import assemble_stored_prompt
from core.stages import (
    append_choice_map_if_any,
    extract_boxed_counts,
    extract_boxed_counts_from_output,
    extract_boxed_answer,
    extract_choice_map,
    extract_final_answer,
    normalize_for_model,
    rule_equivalent,
    strip_think,
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


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on", "enable", "enabled"):
        return True
    if s in ("0", "false", "no", "n", "off", "disable", "disabled"):
        return False
    return bool(default)


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


def _maybe_autostart_vllm(llm: LLMRouter) -> bool:
    """
    If enabled in config, start vLLM on pipeline startup and wait for health.
    """
    if not llm.option_bool("vllm_autostart", False):
        return False

    cmd = llm.vllm_start_cmd_resolved()
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


def _fmt_s(x: float) -> str:
    try:
        return f"{float(x):.2f}s"
    except Exception:
        return "0.00s"


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
    "full",
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


def _as_choice_letter(s: str) -> str:
    """If s looks like 'A' or 'A.xxx', return 'A'; else return original trimmed string."""
    if not isinstance(s, str):
        return ""
    ss = s.strip()
    if not ss:
        return ""
    c0 = ss[0].upper()
    if c0 in ("A", "B", "C", "D"):
        return c0
    return ss


def _compact_outputs(llm: LLMRouter) -> bool:
    return llm.option_bool("compact_outputs", False)


def _select_answer(*, gold: str, majority: Dict[str, Any], min_votes_to_accept: int) -> Dict[str, Any]:
    """
    Select the final answer for this stage:
    - If voting is confident (majority_count >= min_votes_to_accept): use voting result
    - Else: fall back to provided gold answer
    """
    maj_raw = str((majority or {}).get("majority") or "").strip()
    maj_cnt = int((majority or {}).get("majority_count") or 0)
    gold_letter = _as_choice_letter(gold)
    maj_letter = _as_choice_letter(maj_raw)
    if maj_cnt >= int(min_votes_to_accept) and maj_letter:
        return {"final_answer": maj_letter, "final_source": "majority", "final_vote_count": maj_cnt}
    return {"final_answer": gold_letter or gold.strip(), "final_source": "answer_fallback", "final_vote_count": maj_cnt}


def _iter_stage1_output_paths(stage1_dir: str) -> List[str]:
    """
    Find stage1 output files under a stage1 directory.
    Supports:
    - stage1_output.stage1.jsonl
    - <prefix>.stage1_output.stage1.jsonl
    """
    if not stage1_dir or not os.path.isdir(stage1_dir):
        return []
    outs: List[str] = []
    for name in os.listdir(stage1_dir):
        if not name.endswith(".jsonl"):
            continue
        if name.endswith("stage1_output.stage1.jsonl"):
            outs.append(os.path.join(stage1_dir, name))
    outs.sort()
    return outs


def _stage1_output_prefix(path: str) -> str:
    """Infer prefix from '<prefix>.stage1_output.stage1.jsonl' (or '' for unprefixed)."""
    base = os.path.basename(path)
    suf = ".stage1_output.stage1.jsonl"
    if base == "stage1_output.stage1.jsonl":
        return ""
    if base.endswith(suf):
        return base[: -len(suf)]
    return ""


def _llm_judge_equivalence(
    *,
    llm: LLMRouter,
    uuid: Any,
    question: str,
    gold: str,
    pred: str,
    choice_map: Dict[str, str],
    stage: str,
    sleep_s: float,
    stats: Dict[str, int] | None = None,
) -> bool | None:
    """
    Ask LLM to judge equivalence only when rules cannot decide.
    Return True/False/None.
    """
    choice_lines = []
    for k in ("A", "B", "C", "D"):
        if k in choice_map:
            choice_lines.append(f"{k} = {choice_map[k]}")
    choice_block = "\n".join(choice_lines)
    user = (
        "你是“答案一致性判定器”，只判断 pred 是否与 gold 含义一致，禁止解题。\n"
        "输出必须是且只能是三者之一：一致 / 不一致 / 不确定\n\n"
        f"[题目]\n{question}\n\n"
        f"[选项映射]\n{choice_block}\n\n"
        f"[gold]\n{gold}\n\n"
        f"[pred]\n{pred}\n"
    ).strip()
    resp = llm.generate_n(
        stage_name=f"{stage}_judge",
        question=user,
        prompt_mode="raw_prompt",
        sleep_s=sleep_s,
        stats=stats,
    )[0]
    t = (resp or "").strip()
    first = (t.splitlines()[0] if t else "").strip().strip("。.!！?？")
    if first == "不确定":
        return None
    if first == "不一致":
        return False
    if first == "一致":
        return True
    return None


def _parse_json_array_loose(text: str) -> List[Any] | None:
    """
    Best-effort parse of a JSON array from model output.
    Accepts either pure JSON or text containing a JSON array substring.
    """
    s = (text or "").strip()
    if not s:
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else None
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return None
    try:
        v2 = json.loads(m.group(0))
        return v2 if isinstance(v2, list) else None
    except Exception:
        return None


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
    Use LLM to extract canonical final answers from a batch of raw solver outputs.
    Returns a list of strings aligned with raw_outputs.
    """
    raws = [str(x) for x in (raw_outputs or [])]
    if not raws:
        return []

    # If upstream produced explicit error markers, do not accidentally extract numeric codes (e.g. 503).
    # Keep the "extract" step LLM-based, but we can safely short-circuit obvious error markers to "".
    short: List[str] = []
    todo: List[tuple[int, str]] = []
    for i, t in enumerate(raws):
        if (t or "").strip().startswith("[LLM_ERROR"):
            short.append("")
        else:
            short.append("__TO_FILL__")
            todo.append((i, t))
    if not todo:
        return short

    choice_lines = []
    for k in ("A", "B", "C", "D"):
        if k in choice_map:
            choice_lines.append(f"{k} = {choice_map[k]}")
    choice_block = "\n".join(choice_lines)

    attempts_block = "\n\n".join([f"[输出{i+1}]\n{t}" for i, t in enumerate(raws)])
    user = (
        "你是“最终答案抽取器”，只负责从模型输出中抽取最终答案，禁止解题。\n"
        "请对下面 N 条输出逐条抽取最终答案，并按顺序返回 JSON 数组（长度必须等于 N）。\n"
        "要求：\n"
        "- 每个元素是字符串。\n"
        "- 若是选择题，必须只输出 A/B/C/D（大写），不要输出选项内容。\n"
        "- 若无法确定或输出里没有明确最终答案，输出空字符串 \"\"。\n"
        "- 只输出 JSON 数组，不要输出任何额外文字。\n\n"
        f"[题目]\n{question}\n\n"
        f"[选项映射]\n{choice_block}\n\n"
        f"[N]={len(raws)}\n\n"
        f"{attempts_block}\n"
    ).strip()

    resp = llm.generate_n(
        stage_name=f"{stage}_extract",
        question=user,
        prompt_mode="raw_prompt",
        n=1,
        temperature=0.0,
        max_tokens=512,
        sleep_s=sleep_s,
        stats=stats,
    )[0]
    arr = _parse_json_array_loose(resp)
    if not isinstance(arr, list) or len(arr) != len(raws):
        # Retry once with a stricter instruction.
        user2 = (
            "只输出 JSON 数组（例如 [\"A\",\"\",... ]），长度必须等于 N。\n"
            f"N={len(raws)}\n"
            f"{attempts_block}\n"
        ).strip()
        resp2 = llm.generate_n(
            stage_name=f"{stage}_extract",
            question=user2,
            prompt_mode="raw_prompt",
            n=1,
            temperature=0.0,
            max_tokens=256,
            sleep_s=sleep_s,
            stats=stats,
        )[0]
        arr = _parse_json_array_loose(resp2)
    if not isinstance(arr, list) or len(arr) != len(raws):
        # Hard fallback: empty answers (still avoids false consensus).
        out = ["" for _ in raws]
    else:
        out = [str(x or "").strip() for x in arr]
        out = [(x.upper() if len(x) == 1 and x.lower() in ("a", "b", "c", "d") else x) for x in out]

    # Fill short-circuited error slots.
    for i, _t in todo:
        short[i] = out[i]
    return short


def _to_result_entry(*, stage: str, final_answer: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal result entry (stage2/stage3 only)."""
    answer = entry.get("answer")
    answer = (answer or "").strip() if isinstance(answer, str) else ""
    attempts = entry.get("attempts") if isinstance(entry.get("attempts"), list) else []
    attempts_slim: List[Dict[str, Any]] = []
    for a in attempts:
        if not isinstance(a, dict):
            continue
        attempts_slim.append(
            {
                "raw_text": a.get("raw_text"),
                "boxed_answer": a.get("boxed_answer"),
                "verdict": a.get("verdict"),
                "final_answer": final_answer,
            }
        )
    return {
        "uuid": entry.get("uuid"),
        "question": entry.get("question"),
        "answer": answer,
        "stage": stage,
        "final_answer": final_answer,
        "majority_answer": entry.get("majority_answer"),
        "attempts": attempts_slim,
    }


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
        return _as_choice_letter(p) == _as_choice_letter(g) or p == g

    def _build_result_text(question: str, raw: str, pred: str) -> str:
        q = str(question or "").strip()
        r = str(raw or "").strip()
        p = str(pred or "").strip()
        return f"问题：{q}\n\n思考：{r}\n\n答案：{p}"

    def _iter_majority_infer_attempts(row: Dict[str, Any]) -> List[Tuple[int, str]]:
        raws = row.get("raw_model_outputs") if isinstance(row.get("raw_model_outputs"), list) else []
        extracted = row.get("extracted_answers") if isinstance(row.get("extracted_answers"), list) else []
        if not raws or not extracted:
            return []
        question = row.get("question")
        if question is None:
            return []
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
        if not maj_ans or maj_cnt < threshold:
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
            matches = _iter_majority_infer_attempts(row)
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
            if not q_raw.strip():
                raise ValueError(f"Missing question: uuid={uuid}")
            if not gold:
                raise ValueError(f"Missing gold in field 'answer': uuid={uuid}")

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
            extracted = [extract_final_answer(x) for x in raw_solutions]
            standardized = [standardize_choice_answer(a, choice_map=choice_map) for a in extracted]

            # Majority vote: filter empty answers to avoid false consensus on failures.
            vote_inputs1 = [str(x or "").strip() for x in standardized][:n1]
            vote_inputs1 = [x for x in vote_inputs1 if x]
            v1 = majority_vote(vote_inputs1)
            majority_answer_json = {"majority": v1.majority, "majority_count": int(v1.majority_count), "counts": dict(v1.counts)}

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
                    "raw_model_outputs": [str(x) for x in raw_solutions][:n1],
                    "extracted_answers": [str(x) for x in standardized][:n1],
                    "majority_answer": majority_answer_json,
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
        - stage1_eval will first try to parse boxed counts from existing output content:
          \\boxed{解答正确：x，解答错误：y}
        - Only when parsing fails will it fall back to an LLM eval call (best-effort), using the stored `prompt`
          plus [GOLD_STANDARD_ANSWER]=... as input.
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
            stage1_raw_generations_path = os.path.join(stage1_dir, f"{pfx}stage1_raw_generations.stage1.jsonl")
            stage1_infer_path = os.path.join(stage1_dir, f"{pfx}stage1_infer.stage1.jsonl")
            compact = _compact_outputs(llm)

            stage1_done = _load_status_map(stage1_status_path)
            stage2_inputs: List[Dict[str, Any]] = []

            # Optional: if stage1_raw_generations exists, use it to populate final_answer/final_vote_count.
            raw_stage1_by_uuid: Dict[str, Dict[str, Any]] = {}
            if os.path.exists(stage1_raw_generations_path):
                for rr in iter_jsonl(stage1_raw_generations_path, tolerate_errors=True):
                    u0 = rr.get("uuid")
                    if u0 is not None and isinstance(rr, dict):
                        raw_stage1_by_uuid[str(u0)] = rr

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

                # 1) Prefer parsing boxed counts from existing output (no LLM).
                counts = extract_boxed_counts_from_output(r.get("output"))
                eval_calls = 0
                eval_text = ""
                s1_eval_stats: Dict[str, int] = {}

                # 2) Only if parsing failed, do best-effort LLM fallback using stored prompt + gold.
                if counts is None:
                    stored_prompt = r.get("prompt") if isinstance(r.get("prompt"), str) else ""
                    if stored_prompt and gold:
                        eval_user = f"{stored_prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
                        for _ in range(2):
                            eval_calls += 1
                            eval_text = llm.generate_n(
                                stage_name="stage1_eval",
                                question=eval_user,
                                prompt_mode="raw_prompt_eval",
                                sleep_s=sleep_s,
                                stats=s1_eval_stats,
                            )[0]
                            eval_text = strip_think(eval_text)
                            if extract_boxed_counts(eval_text) is not None:
                                break
                        counts = extract_boxed_counts(eval_text)

                # Routing decision derived from parsed boxed counts:
                # - accept iff ok >= min_votes_to_accept
                # - otherwise go to stage2
                if counts is not None:
                    ok_i, bad_i = int(counts[0]), int(counts[1])
                else:
                    ok_i, bad_i = 0, 0
                next_stage = "accepted" if ok_i >= int(min_votes_to_accept) else "stage2"

                # Populate final_answer/final_vote_count if raw generations exist; otherwise conservative fallback.
                raw_stage1 = raw_stage1_by_uuid.get(uuid_key, {})
                maj = raw_stage1.get("majority_answer") if isinstance(raw_stage1.get("majority_answer"), dict) else None
                if isinstance(maj, dict) and maj.get("majority_count") is not None:
                    sel1 = _select_answer(gold=gold, majority=maj, min_votes_to_accept=min_votes_to_accept)
                    vote_majority = maj.get("majority")
                    vote_majority_count = int(maj.get("majority_count", 0) or 0)
                else:
                    sel1 = {"final_answer": gold, "final_source": "answer_fallback", "final_vote_count": int(ok_i)}
                    vote_majority = None
                    vote_majority_count = 0

                paths = {"infer": stage1_infer_path} if compact else {"raw_generations": stage1_raw_generations_path, "output": stage1_output_path}
                append_jsonl_line(
                    stage1_status_path,
                    {
                        "uuid": uuid,
                        "stage": "stage1",
                        "ok": int(ok_i),
                        "bad": int(bad_i),
                        "eval_ok": int(ok_i) if counts is not None else None,
                        "eval_bad": int(bad_i) if counts is not None else None,
                        "judge_ok": raw_stage1.get("ok") if isinstance(raw_stage1, dict) else None,
                        "judge_bad": raw_stage1.get("bad") if isinstance(raw_stage1, dict) else None,
                        "min_votes_to_accept": int(min_votes_to_accept),
                        "vote_majority": vote_majority,
                        "vote_majority_count": int(vote_majority_count),
                        **sel1,
                        "next_stage": next_stage,
                        "paths": paths,
                        "llm_call_counts": {
                            "stage1_eval": int(eval_calls),
                            "stage1_eval_http_calls": int(s1_eval_stats.get("http_calls", 0)),
                            "stage1_eval_retries": int(s1_eval_stats.get("retries", 0)),
                            "stage1_eval_timeouts": int(s1_eval_stats.get("timeouts", 0)),
                            "stage1_eval_errors": int(s1_eval_stats.get("errors", 0)),
                        }
                        if eval_calls > 0
                        else {"stage1_eval": 0},
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
                model_q = r.get("model_input") if isinstance(r.get("model_input"), str) else append_choice_map_if_any(normalize_for_model(q_raw))
                choice_map = extract_choice_map(model_q)
                raw_model_outputs = r.get("raw_model_outputs") if isinstance(r.get("raw_model_outputs"), list) else []
                extracted_answers = r.get("extracted_answers") if isinstance(r.get("extracted_answers"), list) else []
                majority_answer_json = (
                    r.get("majority_answer") if isinstance(r.get("majority_answer"), dict) else {"majority": "", "majority_count": 0, "counts": {}}
                )
                n1 = int(llm.stage_params("stage1_solve").n)

                # Judge attempts (rule first, optional LLM judge).
                stage1_attempts: List[Dict[str, Any]] = []
                ok1 = 0
                s1_judge_stats: Dict[str, int] = {}
                for raw, pred_i in zip(raw_model_outputs[:n1], extracted_answers[:n1]):
                    boxed_i = extract_boxed_answer(str(raw))
                    extracted_final = boxed_i or str(pred_i)
                    pred_final = standardize_choice_answer(extracted_final, choice_map=choice_map)
                    eq = rule_equivalent(pred_final, gold, choice_map=choice_map)
                    judge_src = "rules"
                    if eq is None:
                        eq = _llm_judge_equivalence(
                            llm=llm,
                            uuid=uuid,
                            question=q_raw,
                            gold=gold,
                            pred=pred_final,
                            choice_map=choice_map,
                            stage="stage1",
                            sleep_s=sleep_s,
                            stats=s1_judge_stats,
                        )
                        judge_src = "llm" if eq is not None else "unknown"
                    verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
                    if eq is True:
                        ok1 += 1
                    stage1_attempts.append(
                        {
                            "raw_text": str(raw),
                            "boxed_answer": boxed_i,
                            "extracted_answer": extracted_final,
                            "normalized_answer": pred_final,
                            "verdict": verdict,
                            "judge_source": judge_src,
                        }
                    )

                stored_prompt = assemble_stored_prompt(
                    question=q_raw,
                    standard_answer="",
                    stage1_answers=[str(x) for x in extracted_answers][:n1],
                )

                # Eval prompt (must end with boxed counts ideally).
                eval_user = f"{stored_prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
                eval_calls = 0
                eval_text = ""
                s1_eval_stats: Dict[str, int] = {}
                for _ in range(2):
                    eval_calls += 1
                    eval_text = llm.generate_n(
                        stage_name="stage1_eval",
                        question=eval_user,
                        prompt_mode="raw_prompt_eval",
                        sleep_s=sleep_s,
                        stats=s1_eval_stats,
                    )[0]
                    eval_text = strip_think(eval_text)
                    if extract_boxed_counts(eval_text) is not None:
                        break

                stage1_raw_entry: Dict[str, Any] = {
                    "uuid": uuid,
                    "line_number": r.get("line_number"),
                    "stage": "stage1",
                    "question": q_raw,
                    "gold": gold,
                    "model_input": model_q,
                    "raw_model_outputs": [str(x) for x in raw_model_outputs][:n1],
                    "extracted_answers": [str(x) for x in extracted_answers][:n1],
                    "majority_answer": majority_answer_json,
                    "attempts": stage1_attempts,
                    "ok": ok1,
                    "bad": n1 - ok1,
                    "llm_call_counts": {
                        **(r.get("llm_call_counts") if isinstance(r.get("llm_call_counts"), dict) else {}),
                        "stage1_eval": eval_calls,
                        "stage1_eval_http_calls": int(s1_eval_stats.get("http_calls", 0)),
                        "stage1_eval_retries": int(s1_eval_stats.get("retries", 0)),
                        "stage1_eval_timeouts": int(s1_eval_stats.get("timeouts", 0)),
                        "stage1_eval_errors": int(s1_eval_stats.get("errors", 0)),
                        "stage1_judge_llm_fallback": sum(1 for a in stage1_attempts if a.get("judge_source") == "llm"),
                        "stage1_judge_unknown": sum(1 for a in stage1_attempts if a.get("judge_source") == "unknown"),
                        "stage1_judge_http_calls": int(s1_judge_stats.get("http_calls", 0)),
                        "stage1_judge_retries": int(s1_judge_stats.get("retries", 0)),
                        "stage1_judge_timeouts": int(s1_judge_stats.get("timeouts", 0)),
                        "stage1_judge_errors": int(s1_judge_stats.get("errors", 0)),
                    },
                }

                out = normalize_output_wrapper(
                    {"status": "SUCCESS", "content": {"choices": [{"indext": 0, "message": {"role": "assistant", "content": eval_text}}]}},
                    uuid=uuid,
                    stage="stage1",
                )
                clean = {k: r.get(k) for k in CANONICAL_KEYS}
                clean["prompt"] = stored_prompt
                clean["output"] = out

                if not compact:
                    append_jsonl_line(stage1_raw_generations_path, stage1_raw_entry)
                    append_jsonl_line(stage1_output_path, clean)

                counts = extract_boxed_counts(eval_text)
                route_ok, route_bad = (counts if counts is not None else (ok1, n1 - ok1))
                sel1 = _select_answer(gold=gold, majority=majority_answer_json, min_votes_to_accept=min_votes_to_accept)
                next_stage = "stage2" if int(majority_answer_json.get("majority_count", 0)) < int(min_votes_to_accept) else "accepted"
                paths = {"infer": stage1_infer_path}
                if not compact:
                    paths.update({"raw_generations": stage1_raw_generations_path, "output": stage1_output_path})
                append_jsonl_line(
                    stage1_status_path,
                    {
                        "uuid": uuid,
                        "stage": "stage1",
                        "ok": int(route_ok),
                        "bad": int(route_bad),
                        "eval_ok": int(counts[0]) if counts is not None else None,
                        "eval_bad": int(counts[1]) if counts is not None else None,
                        "judge_ok": int(ok1),
                        "judge_bad": int(n1 - ok1),
                        "min_votes_to_accept": int(min_votes_to_accept),
                        "vote_majority": majority_answer_json.get("majority"),
                        "vote_majority_count": int(majority_answer_json.get("majority_count", 0)),
                        **sel1,
                        "next_stage": next_stage,
                        "paths": paths,
                    },
                )
                stage1_done[uuid_key] = {"uuid": uuid, "ok": int(route_ok), "bad": int(route_bad), "next_stage": next_stage}

                if next_stage == "stage2":
                    stage2_inputs.append(
                        {
                            **{k: r.get(k) for k in CANONICAL_KEYS},
                            "_stage1_ok": int(route_ok),
                            "_stage1_bad": int(route_bad),
                            "_difficulty": int(route_bad),
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
                            if not stage1_status and is_stage1_output:
                                counts = extract_boxed_counts_from_output(row.get("output"))
                                if counts is not None:
                                    okc_i, badc_i = int(counts[0]), int(counts[1])
                            diff = badc_i
                            yield {
                                **{k: row.get(k) for k in CANONICAL_KEYS},
                                "_stage1_ok": int(okc_i),
                                "_stage1_bad": int(badc_i),
                                "_difficulty": int(diff),
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
                        counts = extract_boxed_counts_from_output(row.get("output"))
                        if counts is not None:
                            need_stage2 = int(counts[0]) < int(min_votes_to_accept)
                        else:
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
            stage1_ok_val = r.get("_stage1_ok") if "_stage1_ok" in r else r.get("stage1_ok")
            stage1_bad_val = r.get("_stage1_bad") if "_stage1_bad" in r else r.get("stage1_bad")
            difficulty_val = r.get("_difficulty") if "_difficulty" in r else r.get("difficulty")
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
                        counts = extract_boxed_counts_from_output(r.get("output"))
                        if counts is not None:
                            need_stage2 = int(counts[0]) < int(min_votes_to_accept)
                        else:
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
                if okc_i is None or badc_i is None:
                    if is_stage1_output:
                        counts = extract_boxed_counts_from_output(r.get("output"))
                        if counts is not None:
                            okc_i, badc_i = int(counts[0]), int(counts[1])
                if okc_i is not None:
                    stage1_ok_val = okc_i
                if badc_i is not None:
                    stage1_bad_val = badc_i
                    difficulty_val = badc_i
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
            append_jsonl_line(
                stage2_infer_path,
                {
                    "uuid": uuid,
                    "line_number": r.get("line_number"),
                    "stage": "stage2_infer",
                    "difficulty": difficulty_val,
                    "question": q_raw,
                    "answer": gold,
                    "gold": gold,
                    "model_input": model_q,
                    "raw_model_outputs": [str(x) for x in raw_solutions][:n2],
                    "extracted_answers": [str(x) for x in extracted][:n2],
                    "stage1_ok": stage1_ok_val,
                    "stage1_bad": stage1_bad_val,
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "llm_call_counts": {
                        "stage2_solve": llm.stage_params("stage2_solve").n,
                        "stage2_solve_http_calls": int(s2_solve_stats.get("http_calls", 0)),
                        "stage2_solve_retries": int(s2_solve_stats.get("retries", 0)),
                        "stage2_solve_timeouts": int(s2_solve_stats.get("timeouts", 0)),
                        "stage2_solve_errors": int(s2_solve_stats.get("errors", 0)),
                        "stage2_extract_http_calls": int(s2_extract_stats.get("http_calls", 0)),
                        "stage2_extract_retries": int(s2_extract_stats.get("retries", 0)),
                        "stage2_extract_timeouts": int(s2_extract_stats.get("timeouts", 0)),
                        "stage2_extract_errors": int(s2_extract_stats.get("errors", 0)),
                    },
                },
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
            choice_map = extract_choice_map(model_q)
            raw_model_outputs = r.get("raw_model_outputs") if isinstance(r.get("raw_model_outputs"), list) else []
            extracted_answers = r.get("extracted_answers") if isinstance(r.get("extracted_answers"), list) else []
            n2 = int(llm.stage_params("stage2_solve").n)

            stage2_attempts: List[Dict[str, Any]] = []
            ok2 = 0
            s2_judge_stats: Dict[str, int] = {}
            for raw, pred_i in zip(raw_model_outputs[:n2], extracted_answers[:n2]):
                extracted_final = str(pred_i or "").strip()
                eq = _llm_judge_equivalence(
                    llm=llm,
                    uuid=uuid,
                    question=q_raw,
                    gold=gold,
                    pred=extracted_final,
                    choice_map=choice_map,
                    stage="stage2",
                    sleep_s=sleep_s,
                    stats=s2_judge_stats,
                )
                verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
                if eq is True:
                    ok2 += 1
                stage2_attempts.append(
                    {
                        "raw_text": str(raw),
                        "boxed_answer": "",
                        "extracted_answer": extracted_final,
                        "normalized_answer": extracted_final,
                        "verdict": verdict,
                        "judge_source": "llm" if eq is not None else "unknown",
                    }
                )

            # Filter out empty answers (e.g. upstream LLM errors) to avoid false consensus.
            vote_inputs2 = [str(a.get("normalized_answer") or "").strip() for a in stage2_attempts][:n2]
            vote_inputs2 = [x for x in vote_inputs2 if x]
            s2_vote = majority_vote(vote_inputs2)
            stage2_majority_answer = {
                "majority": s2_vote.majority,
                "majority_count": int(s2_vote.majority_count),
                "counts": dict(s2_vote.counts),
            }
            sel2 = _select_answer(gold=gold, majority=stage2_majority_answer, min_votes_to_accept=min_votes_to_accept)
            final_answer2 = str(sel2.get("final_answer") or "").strip()
            # ok/bad are based on LLM judge against gold (not by comparing to voted final answer).

            entry: Dict[str, Any] = {
                "uuid": uuid,
                "line_number": r.get("line_number"),
                "stage": "stage2",
                "difficulty": r.get("difficulty"),
                "question": q_raw,
                "gold": gold,
                "model_input": model_q,
                "raw_model_outputs": [str(x) for x in raw_model_outputs][:n2],
                "extracted_answers": [str(x) for x in extracted_answers][:n2],
                "majority_answer": stage2_majority_answer,
                "attempts": stage2_attempts,
                **sel2,
                "ok": int(ok2),
                "bad": int(n2 - ok2),
                "llm_call_counts": {
                    **(r.get("llm_call_counts") if isinstance(r.get("llm_call_counts"), dict) else {}),
                    "stage2_judge_llm_calls": int(n2),
                    "stage2_judge_unknown": sum(1 for a in stage2_attempts if a.get("judge_source") == "unknown"),
                    "stage2_judge_http_calls": int(s2_judge_stats.get("http_calls", 0)),
                    "stage2_judge_retries": int(s2_judge_stats.get("retries", 0)),
                    "stage2_judge_timeouts": int(s2_judge_stats.get("timeouts", 0)),
                    "stage2_judge_errors": int(s2_judge_stats.get("errors", 0)),
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
                "ok": int(ok2),
                "bad": int(n2 - ok2),
                "min_votes_to_accept": int(min_votes_to_accept),
                "vote_majority": stage2_majority_answer.get("majority"),
                "vote_majority_count": int(stage2_majority_answer.get("majority_count", 0)),
                **_select_answer(gold=gold, majority=stage2_majority_answer, min_votes_to_accept=min_votes_to_accept),
                "next_stage": next_stage,
                "difficulty": r.get("difficulty"),
                "stage1_ok": r.get("stage1_ok"),
                "stage1_bad": r.get("stage1_bad"),
                "paths": paths,
            },
        )

        if next_stage == "stage3" and not compact:
            append_jsonl_line(
                stage3_input_path,
                {
                    **{k: r.get(k) for k in CANONICAL_KEYS},
                    "_difficulty": r.get("difficulty"),
                    "_stage1_ok": r.get("stage1_ok"),
                    "_stage1_bad": r.get("stage1_bad"),
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
            append_jsonl_line(
                stage3_infer_path,
                {
                    "uuid": uuid,
                    "line_number": r.get("line_number"),
                    "stage": "stage3_infer",
                    "difficulty": r.get("_difficulty"),
                    "question": q_raw,
                    "answer": gold,
                    "gold": gold,
                    "model_input": model_q,
                    "raw_model_outputs": [str(x) for x in raw_solutions][:n3],
                    "extracted_answers": [str(x) for x in extracted][:n3],
                    "stage1_ok": r.get("_stage1_ok"),
                    "stage1_bad": r.get("_stage1_bad"),
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "llm_call_counts": {
                        "stage3_solve": llm.stage_params("stage3_solve").n,
                        "stage3_solve_http_calls": int(s3_solve_stats.get("http_calls", 0)),
                        "stage3_solve_retries": int(s3_solve_stats.get("retries", 0)),
                        "stage3_solve_timeouts": int(s3_solve_stats.get("timeouts", 0)),
                        "stage3_solve_errors": int(s3_solve_stats.get("errors", 0)),
                        "stage3_extract_http_calls": int(s3_extract_stats.get("http_calls", 0)),
                        "stage3_extract_retries": int(s3_extract_stats.get("retries", 0)),
                        "stage3_extract_timeouts": int(s3_extract_stats.get("timeouts", 0)),
                        "stage3_extract_errors": int(s3_extract_stats.get("errors", 0)),
                    },
                },
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
            choice_map = extract_choice_map(model_q)
            raw_model_outputs = r.get("raw_model_outputs") if isinstance(r.get("raw_model_outputs"), list) else []
            extracted_answers = r.get("extracted_answers") if isinstance(r.get("extracted_answers"), list) else []
            n3 = int(llm.stage_params("stage3_solve").n)

            stage3_attempts: List[Dict[str, Any]] = []
            ok3 = 0
            s3_judge_stats: Dict[str, int] = {}
            for raw, pred_i in zip(raw_model_outputs[:n3], extracted_answers[:n3]):
                extracted_final = str(pred_i or "").strip()
                eq = _llm_judge_equivalence(
                    llm=llm,
                    uuid=uuid,
                    question=q_raw,
                    gold=gold,
                    pred=extracted_final,
                    choice_map=choice_map,
                    stage="stage3",
                    sleep_s=sleep_s,
                    stats=s3_judge_stats,
                )
                verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
                if eq is True:
                    ok3 += 1
                stage3_attempts.append(
                    {
                        "raw_text": str(raw),
                        "boxed_answer": "",
                        "extracted_answer": extracted_final,
                        "normalized_answer": extracted_final,
                        "verdict": verdict,
                        "judge_source": "llm" if eq is not None else "unknown",
                    }
                )

            # Filter out empty answers (e.g. upstream LLM errors) to avoid false consensus.
            vote_inputs3 = [str(a.get("normalized_answer") or "").strip() for a in stage3_attempts][:n3]
            vote_inputs3 = [x for x in vote_inputs3 if x]
            s3_vote = majority_vote(vote_inputs3)
            stage3_majority_answer = {
                "majority": s3_vote.majority,
                "majority_count": int(s3_vote.majority_count),
                "counts": dict(s3_vote.counts),
            }
            sel3 = _select_answer(gold=gold, majority=stage3_majority_answer, min_votes_to_accept=min_votes_to_accept)
            final_answer3 = str(sel3.get("final_answer") or "").strip()
            # ok/bad are based on LLM judge against gold (not by comparing to voted final answer).

            entry = {
                "uuid": uuid,
                "line_number": r.get("line_number"),
                "stage": "stage3",
                "difficulty": r.get("difficulty"),
                "question": q_raw,
                "gold": gold,
                "model_input": model_q,
                "raw_model_outputs": [str(x) for x in raw_model_outputs][:n3],
                "extracted_answers": [str(x) for x in extracted_answers][:n3],
                "majority_answer": stage3_majority_answer,
                "attempts": stage3_attempts,
                **sel3,
                "ok": int(ok3),
                "bad": int(n3 - ok3),
                "llm_call_counts": {
                    **(r.get("llm_call_counts") if isinstance(r.get("llm_call_counts"), dict) else {}),
                    "stage3_judge_llm_calls": int(n3),
                    "stage3_judge_unknown": sum(1 for a in stage3_attempts if a.get("judge_source") == "unknown"),
                    "stage3_judge_http_calls": int(s3_judge_stats.get("http_calls", 0)),
                    "stage3_judge_retries": int(s3_judge_stats.get("retries", 0)),
                    "stage3_judge_timeouts": int(s3_judge_stats.get("timeouts", 0)),
                    "stage3_judge_errors": int(s3_judge_stats.get("errors", 0)),
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
                accepted_from = "stage3" if sel3.get("final_source") == "majority" else "stage3_gold_fallback"
                append_jsonl_line(accepted_bank_path, {**entry, **sel3, "accepted_from": accepted_from})
            paths = {"infer": stage3_infer_path}
            if not compact:
                paths.update({"archive": stage3_archive_path, "result": result_path})
            append_jsonl_line(
                stage3_status_path,
                {
                    "uuid": uuid,
                    "stage": "stage3",
                    "ok": int(ok3),
                    "bad": int(n3 - ok3),
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "vote_majority": stage3_majority_answer.get("majority"),
                    "vote_majority_count": int(stage3_majority_answer.get("majority_count", 0)),
                    **sel3,
                    "next_stage": "accepted",
                    "difficulty": r.get("difficulty"),
                    "stage1_ok": r.get("stage1_ok"),
                    "stage1_bad": r.get("stage1_bad"),
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


def _run_one_input(
    *,
    input_path: str,
    prefix: str,
    out_dir: str,
    stage1_dir: str,
    stage2_dir: str,
    stage3_dir: str,
    llm: LLMRouter,
    min_votes_to_accept: int,
    sleep_s: float,
    start_stage: str = "stage1",
) -> None:
    """
    Run the whole pipeline for a single input JSONL file, writing outputs with a prefix
    (derived from filename) to avoid collisions when --input is a directory.
    """
    if start_stage not in ("stage1", "stage2"):
        raise ValueError(f"start_stage must be 'stage1' or 'stage2', got: {start_stage}")
    pfx = f"{prefix}." if prefix else ""
    raw_input_copy_path = os.path.join(out_dir, f"{pfx}stage0.jsonl")

    stage1_output_path = os.path.join(stage1_dir, f"{pfx}stage1_output.stage1.jsonl")
    stage1_raw_generations_path = os.path.join(stage1_dir, f"{pfx}stage1_raw_generations.stage1.jsonl")
    stage1_status_path = os.path.join(stage1_dir, f"{pfx}status.stage1.jsonl")
    stage1_infer_path = os.path.join(stage1_dir, f"{pfx}stage1_infer.stage1.jsonl")

    stage2_archive_path = os.path.join(stage2_dir, f"{pfx}stage2_archive.stage2.jsonl")
    stage2_raw_generations_path = os.path.join(stage2_dir, f"{pfx}stage2_raw_generations.stage2.jsonl")
    stage2_status_path = os.path.join(stage2_dir, f"{pfx}status.stage2.jsonl")
    stage2_infer_path = os.path.join(stage2_dir, f"{pfx}stage2_infer.stage2.jsonl")

    stage3_archive_path = os.path.join(stage3_dir, f"{pfx}stage3_archive.stage3.jsonl")
    stage3_raw_generations_path = os.path.join(stage3_dir, f"{pfx}stage3_raw_generations.stage3.jsonl")
    stage3_status_path = os.path.join(stage3_dir, f"{pfx}status.stage3.jsonl")
    stage3_infer_path = os.path.join(stage3_dir, f"{pfx}stage3_infer.stage3.jsonl")

    accepted_bank_path = os.path.join(out_dir, f"{pfx}accepted_bank.stage_final.jsonl")
    result_path = os.path.join(out_dir, "result", f"{pfx}result.stage_final.jsonl")
    compact = _compact_outputs(llm)

    # ---- Resume bookkeeping ----
    stage1_done = _load_status_map(stage1_status_path)
    stage2_done = _load_status_map(stage2_status_path)
    stage3_done = _load_status_map(stage3_status_path)

    def _to_result_entry(*, stage: str, final_answer: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Canonical result schema (one line per uuid per input file):
        - uuid, question
        - each attempt: raw_text, boxed_answer, verdict, final_answer
        - final_answer is selected by vote strength: majority if confident else input.answer fallback
        """
        # Keep the original input answer (a.k.a. gold) for analysis.
        answer = entry.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = entry.get("gold_answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = entry.get("gold")
        answer = (answer or "").strip() if isinstance(answer, str) else ""

        attempts = entry.get("attempts") if isinstance(entry.get("attempts"), list) else []
        attempts_slim: List[Dict[str, Any]] = []
        for a in attempts:
            if not isinstance(a, dict):
                continue
            boxed = (a.get("boxed_answer") or "")
            boxed = boxed.strip() if isinstance(boxed, str) else str(boxed).strip()
            fa = (final_answer or "").strip()
            # Verdict in result is defined as: whether the current boxed_answer equals final_answer.
            # For MCQ-like outputs, compare by choice letter; otherwise compare by trimmed string.
            if not boxed or not fa:
                verdict = "不确定"
            else:
                verdict = "正确" if (_as_choice_letter(boxed) == _as_choice_letter(fa) or boxed == fa) else "错误"
            attempts_slim.append(
                {
                    "raw_text": a.get("raw_text"),
                    "boxed_answer": a.get("boxed_answer"),
                    "verdict": verdict,
                    "final_answer": final_answer,
                }
            )
        return {
            "uuid": entry.get("uuid"),
            "question": entry.get("question"),
            "answer": answer,
            "stage": stage,
            "final_answer": final_answer,
            "majority_answer": entry.get("majority_answer"),
            "attempts": attempts_slim,
        }

    try:
        shutil.copyfile(input_path, raw_input_copy_path)
    except Exception:
        pass

    input_rows = _read_all(input_path)
    normalized = [normalize_record(r) for r in input_rows]

    # Stage1-dir input mode: when starting from stage2 and the "input" is a stage1_output file,
    # always regenerate Stage1 status from the provided stage1_output content (eval boxed counts)
    # and stage1_raw_generations (vote majority) if present.
    #
    # This is important because users may provide only `stage1_output.stage1.jsonl`, and we still
    # need correct `status.stage1.jsonl` + routing decisions for Stage2.
    if start_stage == "stage2" and str(os.path.basename(input_path)).endswith("stage1_output.stage1.jsonl"):
        # Regenerate Stage1 status from existing Stage1 artifacts.
        #
        # In `--stage1` mode we pass `stage1_output.stage1.jsonl` as input and start from Stage2.
        # For routing (accepted vs stage2), and for producing `status.stage1.jsonl`, we should parse:
        # - eval boxed counts from stage1_output.output (\\boxed{解答正确：x，解答错误：y})
        # - vote_majority + vote_majority_count from stage1_raw_generations if present
        raw_stage1_by_uuid: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(stage1_raw_generations_path):
            for row in iter_jsonl(stage1_raw_generations_path, tolerate_errors=True):
                u = row.get("uuid")
                if u is not None and isinstance(row, dict):
                    raw_stage1_by_uuid[str(u)] = row

        regenerated_status_rows: List[Dict[str, Any]] = []
        for r in normalized:
            u = r.get("uuid")
            if u is None:
                continue
            u_str = str(u)

            gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
            counts = extract_boxed_counts_from_output(r.get("output"))
            eval_ok = int(counts[0]) if counts is not None else None
            eval_bad = int(counts[1]) if counts is not None else None

            raw_stage1 = raw_stage1_by_uuid.get(u_str, {})
            maj = raw_stage1.get("majority_answer") if isinstance(raw_stage1.get("majority_answer"), dict) else {}
            vote_majority = maj.get("majority")
            vote_majority_count = int(maj.get("majority_count", 0) or 0) if isinstance(maj, dict) else 0

            # Derive judge_ok/bad from stage1_raw_generations if present.
            judge_ok = raw_stage1.get("ok")
            judge_bad = raw_stage1.get("bad")
            judge_ok_i = int(judge_ok) if isinstance(judge_ok, int) else (int(judge_ok) if str(judge_ok).isdigit() else 0)
            judge_bad_i = int(judge_bad) if isinstance(judge_bad, int) else (int(judge_bad) if str(judge_bad).isdigit() else 0)

            # Routing policy:
            # - Prefer vote consensus when available (same as normal Stage1).
            # - Otherwise fall back to eval boxed counts: accept only if eval_bad == 0 and eval_ok > 0.
            if vote_majority_count > 0:
                next_stage = "accepted" if vote_majority_count >= int(min_votes_to_accept) else "stage2"
            elif counts is not None:
                next_stage = "accepted" if (int(counts[1]) == 0 and int(counts[0]) > 0) else "stage2"
            else:
                next_stage = "stage2"

            # For ok/bad (difficulty), prefer eval boxed counts; else fall back to judge counts; else zeros.
            ok_for_route = int(eval_ok) if eval_ok is not None else int(judge_ok_i)
            bad_for_route = int(eval_bad) if eval_bad is not None else int(judge_bad_i)

            sel1 = _select_answer(
                gold=gold,
                majority={"majority": vote_majority, "majority_count": vote_majority_count, "counts": (maj.get("counts") if isinstance(maj, dict) else {})},
                min_votes_to_accept=min_votes_to_accept,
            )

            row = {
                "uuid": u,
                "stage": "stage1",
                "ok": int(ok_for_route),
                "bad": int(bad_for_route),
                "eval_ok": eval_ok,
                "eval_bad": eval_bad,
                "judge_ok": int(judge_ok_i),
                "judge_bad": int(judge_bad_i),
                "min_votes_to_accept": int(min_votes_to_accept),
                "vote_majority": vote_majority,
                "vote_majority_count": int(vote_majority_count),
                **sel1,
                "next_stage": next_stage,
                "paths": {"raw_generations": stage1_raw_generations_path, "output": input_path},
            }
            stage1_done[u_str] = row
            regenerated_status_rows.append(row)

        # Best-effort: always write/overwrite stage1 status file in this mode so routing is correct.
        try:
            if regenerated_status_rows:
                write_jsonl_atomic(stage1_status_path, regenerated_status_rows)
        except Exception:
            pass

    # ---- Stage 1: per-uuid checkpointing (append) ----
    if start_stage == "stage1":
        all_stage1 = [r for r in normalized if r.get("uuid") is not None]
        to_process_stage1 = [r for r in all_stage1 if str(r.get("uuid")) not in stage1_done]
        prog1 = _UUIDProgress(label="stage1", prefix=prefix, total=len(all_stage1), already_done=len(all_stage1) - len(to_process_stage1))
        prog1.start()
        for r in to_process_stage1:
            t0 = time.monotonic()
            uuid = r.get("uuid")
            uuid_key = str(uuid)

            q_raw = r.get("question") if isinstance(r.get("question"), str) else ""
            gold = (r.get("answer") or "").strip() if isinstance(r.get("answer"), str) else ""
            if not q_raw.strip():
                raise ValueError(f"Missing question: uuid={uuid}")
            if not gold:
                raise ValueError(f"Missing gold in field 'answer': uuid={uuid}")

            model_q = append_choice_map_if_any(normalize_for_model(q_raw))
            s1_solve_stats: Dict[str, int] = {}
            raw_solutions = llm.generate_n(
                stage_name="stage1_solve",
                question=model_q,
                prompt_mode="problem",
                sleep_s=sleep_s,
                stats=s1_solve_stats,
            )

            extracted = [extract_final_answer(x) for x in raw_solutions]
            choice_map = extract_choice_map(model_q)
            standardized = [standardize_choice_answer(a, choice_map=choice_map) for a in extracted]
            n1 = int(llm.stage_params("stage1_solve").n)
            majority_answer = majority_vote(standardized[:n1])
            majority_answer_json = {
                "majority": majority_answer.majority,
                "majority_count": int(majority_answer.majority_count),
                "counts": dict(majority_answer.counts),
            }

            # In compact mode, persist stage1_infer as the primary output artifact.
            if compact:
                vote_inputs1 = [str(x or "").strip() for x in standardized][:n1]
                vote_inputs1 = [x for x in vote_inputs1 if x]
                v1 = majority_vote(vote_inputs1)
                majority_answer_infer = {
                    "majority": v1.majority,
                    "majority_count": int(v1.majority_count),
                    "counts": dict(v1.counts),
                }
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
                        "raw_model_outputs": [str(x) for x in raw_solutions][:n1],
                        "extracted_answers": [str(x) for x in standardized][:n1],
                        "majority_answer": majority_answer_infer,
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

            stage1_attempts: List[Dict[str, Any]] = []
            ok1 = 0
            s1_judge_stats: Dict[str, int] = {}
            for raw, extracted_i, pred_i in zip(raw_solutions[:n1], extracted[:n1], standardized[:n1]):
                boxed_i = extract_boxed_answer(raw)
                extracted_final = boxed_i or extracted_i
                pred_final = standardize_choice_answer(extracted_final, choice_map=choice_map)
                eq = rule_equivalent(pred_final, gold, choice_map=choice_map)
                judge_src = "rules"
                if eq is None:
                    eq = _llm_judge_equivalence(
                        llm=llm,
                        uuid=uuid,
                        question=q_raw,
                        gold=gold,
                        pred=pred_final,
                        choice_map=choice_map,
                        stage="stage1",
                        sleep_s=sleep_s,
                        stats=s1_judge_stats,
                    )
                    judge_src = "llm" if eq is not None else "unknown"
                verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
                if eq is True:
                    ok1 += 1
                stage1_attempts.append(
                    {
                        "raw_text": str(raw),
                        "boxed_answer": boxed_i,
                        "extracted_answer": extracted_final,
                        "normalized_answer": pred_final,
                        "verdict": verdict,
                        "judge_source": judge_src,
                    }
                )

            stored_prompt = assemble_stored_prompt(question=q_raw, standard_answer="", stage1_answers=standardized[:n1])

            stage1_raw_entry: Dict[str, Any] = {
                "uuid": uuid,
                "line_number": r.get("line_number"),
                "stage": "stage1",
                "question": q_raw,
                "gold": gold,
                "model_input": model_q,
                "raw_model_outputs": [str(x) for x in raw_solutions][:n1],
                "extracted_answers": [str(x) for x in standardized][:n1],
                "majority_answer": majority_answer_json,
                "attempts": stage1_attempts,
                "ok": ok1,
                "bad": n1 - ok1,
                "llm_call_counts": {
                    "stage1_solve": llm.stage_params("stage1_solve").n,
                    "stage1_eval": 0,
                    "stage1_judge_llm_fallback": sum(1 for a in stage1_attempts if a.get("judge_source") == "llm"),
                    "stage1_judge_unknown": sum(1 for a in stage1_attempts if a.get("judge_source") == "unknown"),
                    "stage1_solve_http_calls": int(s1_solve_stats.get("http_calls", 0)),
                    "stage1_solve_retries": int(s1_solve_stats.get("retries", 0)),
                    "stage1_solve_timeouts": int(s1_solve_stats.get("timeouts", 0)),
                    "stage1_solve_errors": int(s1_solve_stats.get("errors", 0)),
                    "stage1_judge_http_calls": int(s1_judge_stats.get("http_calls", 0)),
                    "stage1_judge_retries": int(s1_judge_stats.get("retries", 0)),
                    "stage1_judge_timeouts": int(s1_judge_stats.get("timeouts", 0)),
                    "stage1_judge_errors": int(s1_judge_stats.get("errors", 0)),
                },
            }
            eval_user = f"{stored_prompt}\n\n[GOLD_STANDARD_ANSWER]={gold}\n"
            eval_calls = 0
            eval_text = ""
            s1_eval_stats: Dict[str, int] = {}
            for _ in range(2):
                eval_calls += 1
                eval_text = llm.generate_n(
                    stage_name="stage1_eval",
                    question=eval_user,
                    prompt_mode="raw_prompt_eval",
                    sleep_s=sleep_s,
                    stats=s1_eval_stats,
                )[0]
                eval_text = strip_think(eval_text)
                if extract_boxed_counts(eval_text) is not None:
                    break
            stage1_raw_entry["llm_call_counts"]["stage1_eval"] = eval_calls
            stage1_raw_entry["llm_call_counts"]["stage1_eval_http_calls"] = int(s1_eval_stats.get("http_calls", 0))
            stage1_raw_entry["llm_call_counts"]["stage1_eval_retries"] = int(s1_eval_stats.get("retries", 0))
            stage1_raw_entry["llm_call_counts"]["stage1_eval_timeouts"] = int(s1_eval_stats.get("timeouts", 0))
            stage1_raw_entry["llm_call_counts"]["stage1_eval_errors"] = int(s1_eval_stats.get("errors", 0))

            out = normalize_output_wrapper(
                {
                    "status": "SUCCESS",
                    "content": {"choices": [{"indext": 0, "message": {"role": "assistant", "content": eval_text}}]},
                },
                uuid=uuid,
                stage="stage1",
            )

            clean = {k: r.get(k) for k in CANONICAL_KEYS}
            clean["prompt"] = stored_prompt
            clean["output"] = out

            if not compact:
                append_jsonl_line(stage1_raw_generations_path, stage1_raw_entry)
                append_jsonl_line(stage1_output_path, clean)

            counts = extract_boxed_counts(eval_text)
            route_ok, route_bad = (counts if counts is not None else (ok1, n1 - ok1))
            sel1 = _select_answer(gold=gold, majority=majority_answer_json, min_votes_to_accept=min_votes_to_accept)
            # Routing uses vote strength (consensus). If not enough votes, go to Stage2.
            next_stage = (
                "stage2" if int(majority_answer_json.get("majority_count", 0)) < int(min_votes_to_accept) else "accepted"
            )
            paths = {"infer": stage1_infer_path} if compact else {"raw_generations": stage1_raw_generations_path, "output": stage1_output_path}
            append_jsonl_line(
                stage1_status_path,
                {
                    "uuid": uuid,
                    "stage": "stage1",
                    "ok": int(route_ok),
                    "bad": int(route_bad),
                    "eval_ok": int(counts[0]) if counts is not None else None,
                    "eval_bad": int(counts[1]) if counts is not None else None,
                    "judge_ok": int(ok1),
                    "judge_bad": int(n1 - ok1),
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "vote_majority": majority_answer_json.get("majority"),
                    "vote_majority_count": int(majority_answer_json.get("majority_count", 0)),
                    **sel1,
                    "next_stage": next_stage,
                    "paths": paths,
                },
            )
            # NOTE: By design we do NOT put Stage1-accepted samples into `accepted_bank`.
            # Rationale: Stage1-accepted cases are considered "too easy" and we keep them only in stage1 artifacts.
            stage1_done[uuid_key] = {"uuid": uuid, "ok": int(route_ok), "bad": int(route_bad), "next_stage": next_stage}
            prog1.tick(time.monotonic() - t0)
        prog1.finish()

    # ---- Stage 2/3 ----
    # Stage2/Stage3 do NOT need Stage1 output files. They only need:
    #   - the original input rows (question/answer/uuid)
    #   - Stage1 status routing decisions (status.stage1.jsonl), which may be produced externally.
    input_row_by_uuid: Dict[str, Dict[str, Any]] = {str(r.get("uuid")): r for r in normalized if r.get("uuid") is not None}

    # Track already-accepted UUIDs in accepted_bank (Stage2/Stage3 only; Stage1 accepted is intentionally excluded).
    accepted_done: set[str] = set()
    if os.path.exists(accepted_bank_path):
        for row in iter_jsonl(accepted_bank_path, tolerate_errors=True):
            u = row.get("uuid")
            if u is not None:
                accepted_done.add(str(u))

    # IMPORTANT: Do NOT backfill Stage1-accepted UUIDs into accepted_bank.

    hard_rows: List[Dict[str, Any]] = []
    for u_str, st in stage1_done.items():
        if (st.get("next_stage") or "") != "stage2":
            continue
        r = input_row_by_uuid.get(u_str)
        if not r:
            continue
        ok = int(st.get("ok", 0))
        bad = int(st.get("bad", 0))
        r["_stage1_ok"] = ok
        r["_stage1_bad"] = bad
        r["_difficulty"] = bad
        hard_rows.append(r)

    stage3_candidates: List[Dict[str, Any]] = []
    to_process_stage2 = [r for r in hard_rows if str(r.get("uuid")) not in stage2_done]
    prog2 = _UUIDProgress(label="stage2", prefix=prefix, total=len(hard_rows), already_done=len(hard_rows) - len(to_process_stage2))
    prog2.start()
    for r in to_process_stage2:
        t0 = time.monotonic()
        uuid = r.get("uuid")
        uuid_key = str(uuid)
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

        stage1_ok_val = r.get("_stage1_ok") if "_stage1_ok" in r else r.get("stage1_ok")
        stage1_bad_val = r.get("_stage1_bad") if "_stage1_bad" in r else r.get("stage1_bad")
        difficulty_val = r.get("_difficulty") if "_difficulty" in r else r.get("difficulty")
        if compact:
            append_jsonl_line(
                stage2_infer_path,
                {
                    "uuid": uuid,
                    "line_number": r.get("line_number"),
                    "stage": "stage2_infer",
                    "difficulty": difficulty_val,
                    "question": q_raw,
                    "answer": gold,
                    "gold": gold,
                    "model_input": model_q,
                    "raw_model_outputs": [str(x) for x in raw_solutions][:n2],
                    "extracted_answers": [str(x) for x in extracted][:n2],
                    "stage1_ok": stage1_ok_val,
                    "stage1_bad": stage1_bad_val,
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "llm_call_counts": {
                        "stage2_solve": llm.stage_params("stage2_solve").n,
                        "stage2_solve_http_calls": int(s2_solve_stats.get("http_calls", 0)),
                        "stage2_solve_retries": int(s2_solve_stats.get("retries", 0)),
                        "stage2_solve_timeouts": int(s2_solve_stats.get("timeouts", 0)),
                        "stage2_solve_errors": int(s2_solve_stats.get("errors", 0)),
                        "stage2_extract_http_calls": int(s2_extract_stats.get("http_calls", 0)),
                        "stage2_extract_retries": int(s2_extract_stats.get("retries", 0)),
                        "stage2_extract_timeouts": int(s2_extract_stats.get("timeouts", 0)),
                        "stage2_extract_errors": int(s2_extract_stats.get("errors", 0)),
                    },
                },
            )

        stage2_attempts: List[Dict[str, Any]] = []
        ok2 = 0
        s2_judge_stats: Dict[str, int] = {}
        for raw, pred_i in zip(raw_solutions[:n2], extracted[:n2]):
            extracted_final = str(pred_i or "").strip()
            eq = _llm_judge_equivalence(
                llm=llm,
                uuid=uuid,
                question=q_raw,
                gold=gold,
                pred=extracted_final,
                choice_map=choice_map,
                stage="stage2",
                sleep_s=sleep_s,
                stats=s2_judge_stats,
            )
            judge_src = "llm" if eq is not None else "unknown"
            verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
            if eq is True:
                ok2 += 1
            stage2_attempts.append(
                {
                    "raw_text": str(raw),
                    "boxed_answer": "",
                    "extracted_answer": extracted_final,
                    "normalized_answer": extracted_final,
                    "verdict": verdict,
                    "judge_source": judge_src,
                }
            )

        # Majority vote uses the same "boxed-first, then normalize" answers we judge with.
        # Filter out empty answers (e.g. upstream LLM errors) to avoid false consensus.
        vote_inputs2 = [str(a.get("normalized_answer") or "").strip() for a in stage2_attempts][:n2]
        vote_inputs2 = [x for x in vote_inputs2 if x]
        s2_vote = majority_vote(vote_inputs2)
        stage2_majority_answer = {
            "majority": s2_vote.majority,
            "majority_count": int(s2_vote.majority_count),
            "counts": dict(s2_vote.counts),
        }

        sel2 = _select_answer(gold=gold, majority=stage2_majority_answer, min_votes_to_accept=min_votes_to_accept)
        final_answer2 = str(sel2.get("final_answer") or "").strip()

        entry: Dict[str, Any] = {
            "uuid": uuid,
            "line_number": r.get("line_number"),
            "stage": "stage2",
            "difficulty": r.get("_difficulty"),
            "question": q_raw,
            "gold": gold,
            "model_input": model_q,
            "raw_model_outputs": [str(x) for x in raw_solutions][:n2],
            "extracted_answers": [str(x) for x in extracted][:n2],
            "majority_answer": stage2_majority_answer,
            "attempts": stage2_attempts,
            **sel2,
            "ok": int(ok2),
            "bad": int(n2 - ok2),
            "llm_call_counts": {
                "stage2_solve": llm.stage_params("stage2_solve").n,
                "stage2_judge_llm_calls": int(n2),
                "stage2_judge_unknown": sum(1 for a in stage2_attempts if a.get("judge_source") == "unknown"),
                "stage2_solve_http_calls": int(s2_solve_stats.get("http_calls", 0)),
                "stage2_solve_retries": int(s2_solve_stats.get("retries", 0)),
                "stage2_solve_timeouts": int(s2_solve_stats.get("timeouts", 0)),
                "stage2_solve_errors": int(s2_solve_stats.get("errors", 0)),
                "stage2_extract_http_calls": int(s2_extract_stats.get("http_calls", 0)),
                "stage2_extract_retries": int(s2_extract_stats.get("retries", 0)),
                "stage2_extract_timeouts": int(s2_extract_stats.get("timeouts", 0)),
                "stage2_extract_errors": int(s2_extract_stats.get("errors", 0)),
                "stage2_judge_http_calls": int(s2_judge_stats.get("http_calls", 0)),
                "stage2_judge_retries": int(s2_judge_stats.get("retries", 0)),
                "stage2_judge_timeouts": int(s2_judge_stats.get("timeouts", 0)),
                "stage2_judge_errors": int(s2_judge_stats.get("errors", 0)),
            },
        }

        if not compact:
            raw_entry = dict(entry)
            raw_entry.pop("final_answer", None)
            raw_entry.pop("final_source", None)
            raw_entry.pop("final_vote_count", None)
            append_jsonl_line(stage2_raw_generations_path, raw_entry)
            append_jsonl_line(stage2_archive_path, entry)
        # Routing uses vote strength (consensus), not reference-accuracy ok/bad.
        next_stage = "stage3" if int(stage2_majority_answer["majority_count"]) < int(min_votes_to_accept) else "accepted"
        if next_stage == "accepted" and not compact:
            append_jsonl_line(accepted_bank_path, {**entry, **sel2, "accepted_from": "stage2"})
        paths = {"infer": stage2_infer_path} if compact else {"archive": stage2_archive_path}
        append_jsonl_line(
            stage2_status_path,
            {
                "uuid": uuid,
                "stage": "stage2",
                "ok": int(ok2),
                "bad": int(n2 - ok2),
                "min_votes_to_accept": int(min_votes_to_accept),
                "vote_majority": stage2_majority_answer.get("majority"),
                "vote_majority_count": int(stage2_majority_answer.get("majority_count", 0)),
                **_select_answer(gold=gold, majority=stage2_majority_answer, min_votes_to_accept=min_votes_to_accept),
                "next_stage": next_stage,
                "difficulty": r.get("_difficulty"),
                "stage1_ok": r.get("_stage1_ok"),
                "stage1_bad": r.get("_stage1_bad"),
                "paths": paths,
            },
        )
        stage2_done[uuid_key] = {"uuid": uuid, "ok": int(ok2), "bad": int(n2 - ok2), "next_stage": next_stage}
        if next_stage == "stage3":
            stage3_candidates.append(r)
        prog2.tick(time.monotonic() - t0)
    prog2.finish()

    for u_str, st in stage2_done.items():
        if (st.get("next_stage") or "") != "stage3":
            continue
        r = input_row_by_uuid.get(u_str)
        if r and r not in stage3_candidates:
            r["_difficulty"] = st.get("difficulty")
            r["_stage1_ok"] = st.get("stage1_ok")
            r["_stage1_bad"] = st.get("stage1_bad")
            stage3_candidates.append(r)

    to_process_stage3 = [r for r in stage3_candidates if str(r.get("uuid")) not in stage3_done]
    prog3 = _UUIDProgress(label="stage3", prefix=prefix, total=len(stage3_candidates), already_done=len(stage3_candidates) - len(to_process_stage3))
    prog3.start()
    for r in to_process_stage3:
        t0 = time.monotonic()
        uuid = r.get("uuid")
        uuid_key = str(uuid)
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

        stage1_ok_val = r.get("_stage1_ok") if "_stage1_ok" in r else r.get("stage1_ok")
        stage1_bad_val = r.get("_stage1_bad") if "_stage1_bad" in r else r.get("stage1_bad")
        difficulty_val = r.get("_difficulty") if "_difficulty" in r else r.get("difficulty")
        if compact:
            append_jsonl_line(
                stage3_infer_path,
                {
                    "uuid": uuid,
                    "line_number": r.get("line_number"),
                    "stage": "stage3_infer",
                    "difficulty": difficulty_val,
                    "question": q_raw,
                    "answer": gold,
                    "gold": gold,
                    "model_input": model_q,
                    "raw_model_outputs": [str(x) for x in raw_solutions][:n3],
                    "extracted_answers": [str(x) for x in extracted][:n3],
                    "stage1_ok": stage1_ok_val,
                    "stage1_bad": stage1_bad_val,
                    "min_votes_to_accept": int(min_votes_to_accept),
                    "llm_call_counts": {
                        "stage3_solve": llm.stage_params("stage3_solve").n,
                        "stage3_solve_http_calls": int(s3_solve_stats.get("http_calls", 0)),
                        "stage3_solve_retries": int(s3_solve_stats.get("retries", 0)),
                        "stage3_solve_timeouts": int(s3_solve_stats.get("timeouts", 0)),
                        "stage3_solve_errors": int(s3_solve_stats.get("errors", 0)),
                        "stage3_extract_http_calls": int(s3_extract_stats.get("http_calls", 0)),
                        "stage3_extract_retries": int(s3_extract_stats.get("retries", 0)),
                        "stage3_extract_timeouts": int(s3_extract_stats.get("timeouts", 0)),
                        "stage3_extract_errors": int(s3_extract_stats.get("errors", 0)),
                    },
                },
            )

        stage3_attempts: List[Dict[str, Any]] = []
        ok3 = 0
        s3_judge_stats: Dict[str, int] = {}
        for raw, pred_i in zip(raw_solutions[:n3], extracted[:n3]):
            extracted_final = str(pred_i or "").strip()
            eq = _llm_judge_equivalence(
                llm=llm,
                uuid=uuid,
                question=q_raw,
                gold=gold,
                pred=extracted_final,
                choice_map=choice_map,
                stage="stage3",
                sleep_s=sleep_s,
                stats=s3_judge_stats,
            )
            judge_src = "llm" if eq is not None else "unknown"
            verdict = "正确" if eq is True else ("错误" if eq is False else "不确定")
            if eq is True:
                ok3 += 1
            stage3_attempts.append(
                {
                    "raw_text": str(raw),
                    "boxed_answer": "",
                    "extracted_answer": extracted_final,
                    "normalized_answer": extracted_final,
                    "verdict": verdict,
                    "judge_source": judge_src,
                }
            )

        # Majority vote uses the same "boxed-first, then normalize" answers we judge with.
        # Filter out empty answers (e.g. upstream LLM errors) to avoid false consensus.
        vote_inputs3 = [str(a.get("normalized_answer") or "").strip() for a in stage3_attempts][:n3]
        vote_inputs3 = [x for x in vote_inputs3 if x]
        s3_vote = majority_vote(vote_inputs3)
        stage3_majority_answer = {
            "majority": s3_vote.majority,
            "majority_count": int(s3_vote.majority_count),
            "counts": dict(s3_vote.counts),
        }

        sel3 = _select_answer(gold=gold, majority=stage3_majority_answer, min_votes_to_accept=min_votes_to_accept)
        final_answer3 = str(sel3.get("final_answer") or "").strip()

        entry = {
            "uuid": uuid,
            "line_number": r.get("line_number"),
            "stage": "stage3",
            "difficulty": r.get("_difficulty"),
            "question": q_raw,
            "gold": gold,
            "model_input": model_q,
            "raw_model_outputs": [str(x) for x in raw_solutions][:n3],
            "extracted_answers": [str(x) for x in extracted][:n3],
            "majority_answer": stage3_majority_answer,
            "attempts": stage3_attempts,
            **sel3,
            "ok": int(ok3),
            "bad": int(n3 - ok3),
            "llm_call_counts": {
                "stage3_solve": llm.stage_params("stage3_solve").n,
                "stage3_judge_llm_calls": int(n3),
                "stage3_judge_unknown": sum(1 for a in stage3_attempts if a.get("judge_source") == "unknown"),
                "stage3_solve_http_calls": int(s3_solve_stats.get("http_calls", 0)),
                "stage3_solve_retries": int(s3_solve_stats.get("retries", 0)),
                "stage3_solve_timeouts": int(s3_solve_stats.get("timeouts", 0)),
                "stage3_solve_errors": int(s3_solve_stats.get("errors", 0)),
                "stage3_extract_http_calls": int(s3_extract_stats.get("http_calls", 0)),
                "stage3_extract_retries": int(s3_extract_stats.get("retries", 0)),
                "stage3_extract_timeouts": int(s3_extract_stats.get("timeouts", 0)),
                "stage3_extract_errors": int(s3_extract_stats.get("errors", 0)),
                "stage3_judge_http_calls": int(s3_judge_stats.get("http_calls", 0)),
                "stage3_judge_retries": int(s3_judge_stats.get("retries", 0)),
                "stage3_judge_timeouts": int(s3_judge_stats.get("timeouts", 0)),
                "stage3_judge_errors": int(s3_judge_stats.get("errors", 0)),
            },
        }

        if not compact:
            raw_entry = dict(entry)
            raw_entry.pop("final_answer", None)
            raw_entry.pop("final_source", None)
            raw_entry.pop("final_vote_count", None)
            append_jsonl_line(stage3_raw_generations_path, raw_entry)
            append_jsonl_line(stage3_archive_path, entry)
        # Finalization rule:
        # - If vote has consensus: accept by vote
        # - Else: accept by provided gold fallback (do NOT discard)
        next_stage = "accepted"
        accepted_from = "stage3" if sel3.get("final_source") == "majority" else "stage3_gold_fallback"
        if not compact:
            append_jsonl_line(accepted_bank_path, {**entry, **sel3, "accepted_from": accepted_from})
        paths = {"infer": stage3_infer_path} if compact else {"archive": stage3_archive_path}
        append_jsonl_line(
            stage3_status_path,
            {
                "uuid": uuid,
                "stage": "stage3",
                "ok": int(ok3),
                "bad": int(n3 - ok3),
                "min_votes_to_accept": int(min_votes_to_accept),
                "vote_majority": stage3_majority_answer.get("majority"),
                "vote_majority_count": int(stage3_majority_answer.get("majority_count", 0)),
                **sel3,
                "next_stage": next_stage,
                "difficulty": r.get("_difficulty"),
                "stage1_ok": r.get("_stage1_ok"),
                "stage1_bad": r.get("_stage1_bad"),
                "paths": paths,
            },
        )
        stage3_done[uuid_key] = {"uuid": uuid, "ok": int(ok3), "bad": int(n3 - ok3), "next_stage": next_stage}
        prog3.tick(time.monotonic() - t0)
    prog3.finish()

    print("Done.")
    print(f"- input: {input_path}")
    print(f"- prefix: {prefix}")
    print(f"- out_dir: {out_dir}")
    print(f"- input_copy: {raw_input_copy_path}")
    print(f"- stage1_output: {stage1_output_path}")
    print(f"- stage2_archive: {stage2_archive_path}")
    print(f"- stage3_archive: {stage3_archive_path}")
    print(f"- accepted_bank: {accepted_bank_path}")
    print(f"- result: {result_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="MathAgent minimal pipeline (JSONL between stages).")
    p.add_argument(
        "--mode",
        default="full",
        choices=MODES,
        help=(
            "Execution mode. 'full' runs the original pipeline.\n"
            "Route-A modular modes:\n"
            "  - stage2_infer: input is stage1_output.stage1.jsonl (or a dir of them) -> emit stage2_infer\n"
            "  - stage2_eval: input is stage2_infer.stage2.jsonl (or a dir of them) -> emit stage2 status + stage3_input\n"
            "  - stage3_infer: input is stage3_input.stage3.jsonl (or a dir of them) -> emit stage3_infer\n"
            "  - stage3_eval: input is stage3_infer.stage3.jsonl (or a dir of them) -> emit stage3 status + result"
        ),
    )
    p.add_argument("--input", default=None, help="Input JSONL file path OR a directory containing JSONL files.")
    p.add_argument(
        "--stage1",
        default=None,
        help="Run Stage2/Stage3 using an existing stage1 directory as input (reads *stage1_output.stage1.jsonl there).",
    )
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

    mode = str(args.mode or "full").strip()
    if mode != "full":
        if not args.input:
            raise ValueError("--mode is not 'full': must provide --input as the artifact path (file or directory).")
        if args.stage1:
            raise ValueError("--mode is not 'full': do not use --stage1; provide --input pointing to artifacts instead.")
    else:
        if not args.input and not args.stage1:
            raise ValueError("Must provide either --input or --stage1")
        if args.input and args.stage1:
            raise ValueError("Provide only one of --input or --stage1 (not both)")

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
    if mode != "full":
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
        stage1_dir = os.path.join(args.out, "stage1") if not args.stage1 else str(args.stage1)
        stage2_dir = os.path.join(args.out, "stage2")
        stage3_dir = os.path.join(args.out, "stage3")
        # In multi-worker setups, many environments may share the same `--out` directory.
        # Directory creation must be idempotent and must not fail if the directories already exist.
        os.makedirs(stage1_dir, exist_ok=True)
        os.makedirs(stage2_dir, exist_ok=True)
        os.makedirs(stage3_dir, exist_ok=True)

        started_by_us = _maybe_autostart_vllm(llm)
        if llm.option_bool("vllm_shutdown_on_exit", False):
            # Always honor shutdown_on_exit; if you only want to stop when we started it, set vllm_stop_cmd accordingly.
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

    stage1_dir = os.path.join(args.out, "stage1") if not args.stage1 else str(args.stage1)
    stage2_dir = os.path.join(args.out, "stage2")
    stage3_dir = os.path.join(args.out, "stage3")
    # In multi-worker setups, many environments may share the same `--out` directory.
    # Directory creation must be idempotent and must not fail if the directories already exist.
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(stage3_dir, exist_ok=True)

    started_by_us = _maybe_autostart_vllm(llm)
    if llm.option_bool("vllm_shutdown_on_exit", False):
        # Always honor shutdown_on_exit; if you only want to stop when we started it, set vllm_stop_cmd accordingly.
        atexit.register(_maybe_shutdown_vllm, llm)

    # Stage1-dir input mode: run Stage2/Stage3 using existing Stage1 artifacts.
    if args.stage1:
        stage1_outs = _iter_stage1_output_paths(stage1_dir)
        if not stage1_outs:
            raise ValueError(f"--stage1 has no *stage1_output.stage1.jsonl files: {stage1_dir}")
        for stage1_output_path in stage1_outs:
            prefix = _stage1_output_prefix(stage1_output_path)
            _run_one_input(
                input_path=stage1_output_path,
                prefix=prefix,
                out_dir=args.out,
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                stage3_dir=stage3_dir,
                llm=llm,
                min_votes_to_accept=min_votes_to_accept,
                sleep_s=float(args.sleep),
                start_stage="stage2",
            )
        return

    assert args.input
    # Directory input mode: run once per *.jsonl file, prefixing outputs by filename stem.
    if os.path.isdir(args.input):
        input_paths = _iter_input_jsonl_paths(args.input)
        if not input_paths:
            raise ValueError(f"--input is a directory but has no *.jsonl files: {args.input}")
        for input_path in input_paths:
            prefix = _input_prefix(input_path)
            _run_one_input(
                input_path=input_path,
                prefix=prefix,
                out_dir=args.out,
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                stage3_dir=stage3_dir,
                llm=llm,
                min_votes_to_accept=min_votes_to_accept,
                sleep_s=float(args.sleep),
            )
        return

    # Single-file mode: run through the unified implementation.
    # Use input filename stem as output prefix (same convention as directory mode).
    _run_one_input(
        input_path=args.input,
        prefix=_input_prefix(args.input),
        out_dir=args.out,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        stage3_dir=stage3_dir,
        llm=llm,
        min_votes_to_accept=min_votes_to_accept,
        sleep_s=float(args.sleep),
    )
    return


if __name__ == "__main__":
    main()


