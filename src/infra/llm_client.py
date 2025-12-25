from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class LLMConfig:
    """
    Minimal OpenAI-compatible config (HTTP only).
    """

    base_url: str
    api_key: str
    model: str
    timeout_s: int = 60
    retry_max: int = 0
    retry_backoff_s: float = 1.0
    retry_backoff_mult: float = 2.0
    retry_jitter_s: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LLMConfig":
        base_url = str(d.get("base_url") or "")
        api_key = str(d.get("api_key") or "")
        model = str(d.get("model") or "")
        timeout_s = int(d.get("timeout_s") or 60)
        retry_max = int(d.get("retry_max") or 0)
        retry_backoff_s = float(d.get("retry_backoff_s") or 1.0)
        retry_backoff_mult = float(d.get("retry_backoff_mult") or 2.0)
        retry_jitter_s = float(d.get("retry_jitter_s") or 0.0)
        return LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            retry_max=retry_max,
            retry_backoff_s=retry_backoff_s,
            retry_backoff_mult=retry_backoff_mult,
            retry_jitter_s=retry_jitter_s,
        )


class LLMClient:
    def __init__(self, *, config: LLMConfig) -> None:
        self.config = config

    def _ensure_http_config(self) -> None:
        missing = []
        if not self.config.base_url:
            missing.append("base_url")
        if not self.config.model:
            missing.append("model")
        if missing:
            raise RuntimeError("Missing LLM config fields: " + ", ".join(missing) + ".")

    def chat_once(
        self,
        *,
        stage_name: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        stats: Optional[Dict[str, int]] = None,
        stream: bool = False,
        stream_printer: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Call one completion via http (OpenAI-compatible /v1/chat/completions).
        Returns assistant content as string.
        """
        self._ensure_http_config()

        url = self.config.base_url.rstrip("/") + "/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            body["stream"] = True
        data = json.dumps(body).encode("utf-8")
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(url, method="POST", data=data, headers=headers)

        def _bump(key: str, inc: int = 1) -> None:
            if stats is None:
                return
            stats[key] = int(stats.get(key, 0)) + inc

        backoff = max(float(self.config.retry_backoff_s), 0.0)
        attempts = max(int(self.config.retry_max), 0) + 1
        last_err: Exception | None = None
        for attempt in range(attempts):
            _bump("http_calls", 1)
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    if not stream:
                        raw = resp.read().decode("utf-8")
                        obj = json.loads(raw)
                        choices = obj.get("choices") or []
                        if not choices:
                            raise RuntimeError(f"LLM returned no choices: {raw}")
                        msg = (choices[0].get("message") or {}).get("content")
                        if not isinstance(msg, str):
                            raise RuntimeError(f"LLM returned invalid message: {raw}")
                        last_err = None
                        return msg.strip()

                    # Streaming (OpenAI SSE): accumulate deltas and optionally print them.
                    chunks: List[str] = []
                    while True:
                        line_b = resp.readline()
                        if not line_b:
                            break
                        line = line_b.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            evt = json.loads(payload)
                        except Exception:
                            continue

                        choices = evt.get("choices") or []
                        if not choices:
                            continue
                        c0 = choices[0] or {}
                        delta = c0.get("delta") or {}
                        piece = delta.get("content")
                        if not isinstance(piece, str):
                            # Some servers may stream full message; try fallbacks
                            msg = (c0.get("message") or {}).get("content")
                            piece = msg if isinstance(msg, str) else None
                        if isinstance(piece, str) and piece:
                            chunks.append(piece)
                            if stream_printer is not None:
                                try:
                                    stream_printer(piece)
                                except BrokenPipeError:
                                    # stdout closed (e.g. piping to `head`); avoid crashing the pipeline.
                                    stream_printer = None
                    last_err = None
                    return "".join(chunks).strip()
            except urllib.error.HTTPError as e:
                # Retry only for transient server errors / throttling.
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass
                last_err = RuntimeError(f"LLM HTTPError {e.code}: {err_body}")
                retriable = e.code in (429, 500, 502, 503, 504)
                if not retriable or attempt >= attempts - 1:
                    raise last_err from e
            except (TimeoutError, urllib.error.URLError) as e:
                # URLError is a wrapper for many network issues, incl timeouts.
                last_err = RuntimeError(f"LLM network error: {e}")
                _bump("timeouts", 1)
                if attempt >= attempts - 1:
                    raise last_err from e

            # retry path
            _bump("retries", 1)
            if backoff > 0:
                # deterministic "jitter" without random import
                sleep_s = backoff + (float(self.config.retry_jitter_s) if attempt % 2 == 0 else 0.0)
                time.sleep(max(sleep_s, 0.0))
            backoff *= max(float(self.config.retry_backoff_mult), 1.0)

        if last_err is not None:
            raise RuntimeError(f"LLM failed after retries: stage={stage_name!r}") from last_err
        raise RuntimeError(f"LLM failed unexpectedly: stage={stage_name!r}")

    def generate_n(
        self,
        *,
        stage_name: str,
        question: str,
        prompt_mode: str = "problem",
        n: int,
        temperature: float,
        max_tokens: int = 512,
        sleep_s: float = 0.0,
        stats: Optional[Dict[str, int]] = None,
        finish_early: bool = True,
        think_tag: str = "",
        debug_print_prompts: bool = False,
        debug_print_prompts_max_chars: int = 4000,
        debug_print_outputs: bool = False,
        debug_print_outputs_max_chars: int = 2000,
        debug_stream_outputs: bool = False,
    ) -> List[str]:
        """
        Generate n independent answers by calling chat_once n times.
        """
        answers: List[str] = []

        def _safe_print(*args: Any, **kwargs: Any) -> None:
            try:
                print(*args, **kwargs)
            except BrokenPipeError:
                # stdout closed (e.g. piping to `head`); avoid crashing the pipeline.
                return

        mode = (prompt_mode or "problem").strip().lower()
        if mode in ("raw_prompt", "raw_prompt_eval"):
            system = ""
            if mode == "raw_prompt_eval":
                system = (
                    "你是一个**判断答案与标准答案一致**的专家。\n"
                    "你只做一致性判断，严禁解题、严禁复算、严禁推导题目。\n"
                    "注意：你拿到的解答数量以用户输入中的 [N]=... 为准；这些解答已经是“抽取后的答案文本”，你只能用它们来比对。\n"
                    "禁止输出 <think> 或任何推理过程。\n\n"
                    "注意：用户输入末尾可能包含一行 [GOLD_STANDARD_ANSWER]=...，这是唯一可信的标准答案来源。\n"
                    "你必须用该 gold 与 N 个解答做对比（而不是去解题）。\n"
                    "对于选择题：只允许按“选项字母是否一致”判定；\n"
                    "若某个解答给的是选项内容（如 -5、\\frac{1}{5}），必须通过题目选项映射反推它对应的字母后再比较。\n"
                    "禁止把不同选项的数值用小数误差当作“相同”（例如 A=1675/16393 与 B=1675/16390 不得因接近而判一致）。\n\n"
                    "判定规则（必须遵守）：\n"
                    "1) 数值相同但表述不同视为一致（如 A vs -5；必要时用题目选项映射）。\n"
                    "2) （非选择题才可用）允许小误差：把数值/表达式换算为四位小数比较，|误差|<=1e-4 视为一致。\n"
                    "3) （非选择题才可用）表达式需换算为四位小数（例：2π=6.2832，3e=8.1548，√6=2.4495）。\n"
                    "4) 忽略格式。\n"
                    "5) 解答为空或缺少最终答案 -> 错误。\n\n"
                    "输出格式（必须严格输出，参照 sample.jsonl 的 content 样式；禁止输出 JSON/代码块/额外说明）：\n"
                    "你必须只输出一段 Markdown 文本（不是 JSON），结构必须包含且仅包含以下内容（顺序固定）。其中“解答1..解答N”按 [N]=... 的数量生成：\n"
                    "我们按照题目要求，逐条对比每个解答与**标准答案**是否一致。\n\n"
                    "---\n\n"
                    "### 题目回顾：\n"
                    "题目是：\n"
                    "“<把[题目]原文放这里>” \n"
                    "标准答案是：**<只用 [GOLD_STANDARD_ANSWER] + 题目选项映射，写成 A.-5 这种形式；禁止重新解题>** \n\n"
                    "---\n\n"
                    "###解答分析（逐个对比）：\n"
                    "- **解答1**：<解答1原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n"
                    "- ...（一直到解答N，必须逐条列出，不可省略）...\n\n"
                    "---\n\n"
                    "### 判断汇总：\n\n"
                    "| 编号 | 是否正确 | 理由 |\n"
                    "|------|---------|------|\n"
                    "| 解答1 | 正确 ✅/错误 ❌ | <一句理由；必要时写“B（对应 5）”这种映射> |\n"
                    "| ... | ... | ... |\n"
                    "| 解答N | 正确 ✅/错误 ❌ | <一句理由> |\n\n"
                    "### 统计：\n"
                    "- 解题正确数量：<列出正确编号> -> 共**x个** \n"
                    "- 解题错误数量：<列出错误编号> -> 共 **y**个\n\n"
                    "---\n\n"
                    "最后一行必须且只能是：\\boxed{解答正确：x，解答错误：y}\n"
                )
            base_user = question.strip()
        elif mode == "boxed_solve":
            # Stage2/Stage3 solving: enforce \boxed{...} for reliable extraction.
            system = (
                f"你是一个数学解题助手。Stage={stage_name}。\n"
                "你可以输出推理过程（尽量简短，避免复述题目）。\n"
                "为防止长输出被截断：第一行必须先输出最终答案，格式为 \\boxed{...}（必须出现且只需出现一次）。\n"
                "如果题目是选择题：\\boxed{<答案>} 中 <答案> 只能是单个大写字母 A/B/C/D。\n"
                "如果题目不是选择题：\\boxed{<答案>} 中 <答案> 为最终数值/表达式。\n"
                "不要输出 FINAL: 格式。"
            )
            if finish_early:
                system += (
                    "\n\n"
                    "如果你感觉推理会很长、或可能来不及写完，请立刻停止推理，直接给出你认为最可能的最终答案。\n"
                    "选择题必须在 A/B/C/D 中猜测一个字母；不要输出多余内容。"
                )
            # User side only asks the question (per Architecture.md).
            base_user = question.strip()
        else:
            system = (
                f"你是一个数学解题助手。Stage={stage_name}。\n"
                "最后一行必须输出最终答案，且格式必须严格为：FINAL: <答案>\n"
                "如果题目是选择题：<答案> 只能是单个大写字母 A/B/C/D。\n"
                "如果题目不是选择题：<答案> 为最终数值/表达式。"
            )
            if finish_early:
                system += (
                    "\n"
                    "如果你感觉推理会很长、或可能来不及写完，请立刻停止推理，"
                    "直接在最后一行输出 FINAL: <你认为最可能的答案>（选择题在 A/B/C/D 中猜一个）。"
                )
            base_user = (
                f"题目：\n{question}\n\n"
                "要求：你可以写推理过程，但最后一行必须是 FINAL: <答案>（严格格式）。"
            )

        # Optional Qwen-style mode tag injection (/think, /no_think, etc.)
        # By default we only inject for solve-like modes, not for eval/judge prompts.
        tag = (think_tag or "").strip()
        if tag and mode not in ("raw_prompt", "raw_prompt_eval"):
            if not tag.startswith("/"):
                tag = "/" + tag
            base_user = f"{tag}\n{base_user}"

        def _truncate(s: str) -> str:
            m = int(debug_print_prompts_max_chars or 0)
            if m <= 0:
                return s
            if len(s) <= m:
                return s
            return s[:m] + "\n...<truncated>..."

        def _truncate_out(s: str) -> str:
            m = int(debug_print_outputs_max_chars or 0)
            if m <= 0:
                return s
            if len(s) <= m:
                return s
            return s[:m] + "\n...<truncated>..."

        for i in range(n):
            user = base_user if mode in ("raw_prompt", "raw_prompt_eval") else f"{base_user}\n采样编号={i}"
            if debug_print_prompts:
                header = (
                    f"\n========== [LLM_PROMPT] stage={stage_name} mode={mode} "
                    f"sample={i+1}/{n} model={self.config.model} base_url={self.config.base_url} "
                    f"temperature={temperature} max_tokens={max_tokens} timeout_s={self.config.timeout_s} ==========\n"
                )
                _safe_print(header, flush=True)
                _safe_print("---- SYSTEM ----", flush=True)
                _safe_print(_truncate(system), flush=True)
                _safe_print("---- USER ----", flush=True)
                _safe_print(_truncate(user), flush=True)
                _safe_print("========== [LLM_PROMPT END] ==========\n", flush=True)
            try:
                stream_this = bool(debug_stream_outputs and debug_print_outputs)
                if stream_this:
                    _safe_print(
                        f"========== [LLM_OUTPUT_STREAM] stage={stage_name} mode={mode} sample={i+1}/{n} ==========",
                        flush=True,
                    )

                    def _printer(piece: str) -> None:
                        _safe_print(piece, end="", flush=True)

                ans = self.chat_once(
                    stage_name=stage_name,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stats=stats,
                    stream=stream_this,
                    stream_printer=_printer if stream_this else None,
                )
                if stream_this:
                    _safe_print("\n========== [LLM_OUTPUT_STREAM END] ==========\n", flush=True)
                answers.append(ans)
                if debug_print_outputs and not stream_this:
                    _safe_print(
                        f"========== [LLM_OUTPUT] stage={stage_name} mode={mode} sample={i+1}/{n} ==========",
                        flush=True,
                    )
                    _safe_print(_truncate_out(ans), flush=True)
                    _safe_print("========== [LLM_OUTPUT END] ==========\n", flush=True)
            except Exception as e:
                if stats is not None:
                    stats["errors"] = int(stats.get("errors", 0)) + 1
                # Keep pipeline moving; caller can treat this as an incorrect attempt.
                err_text = f"[LLM_ERROR stage={stage_name}]: {type(e).__name__}: {e}"
                answers.append(err_text)
                if debug_print_outputs:
                    _safe_print(
                        f"========== [LLM_OUTPUT] stage={stage_name} mode={mode} sample={i+1}/{n} (ERROR) ==========",
                        flush=True,
                    )
                    _safe_print(_truncate_out(err_text), flush=True)
                    _safe_print("========== [LLM_OUTPUT END] ==========\n", flush=True)
            if sleep_s > 0:
                time.sleep(sleep_s)
        return answers



