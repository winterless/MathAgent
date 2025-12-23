from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


@dataclass(frozen=True)
class LLMConfig:
    """
    Minimal OpenAI-compatible config.

    Expected env vars (you can fill later):
    - LLM_BASE_URL: e.g. https://api.openai.com
    - LLM_API_KEY:  your key
    - LLM_MODEL:    e.g. gpt-4o-mini / your model name
    Optional:
    - LLM_TIMEOUT_S: request timeout (default 60)
    - LLM_MOCK: set to "1" to enable mock answers without real API
    - LLM_BACKEND: "auto"(default) | "http" | "script" | "mock"
    - LLM_SCRIPT_PATH: path to a Python script that prints completion to stdout
    - LLM_SCRIPT_PYTHON: python executable to use (default: current interpreter)
    """

    # http backend (OpenAI-compatible)
    base_url: str
    api_key: str
    model: str
    timeout_s: int = 60
    mock: bool = False

    # backend selection
    backend: str = "auto"

    # script backend
    script_path: str = "/home/unlimitediw/workspace/TYDeepResearch/AgenticRLModelTraining/model/scripts/call_model.py"
    script_python: str = sys.executable

    @staticmethod
    def from_env() -> "LLMConfig":
        base_url = _env("LLM_BASE_URL", "") or ""
        api_key = _env("LLM_API_KEY", "") or ""
        model = _env("LLM_MODEL", "") or ""
        timeout_s = int(_env("LLM_TIMEOUT_S", "60") or "60")
        mock = (_env("LLM_MOCK", "0") or "0") == "1"
        backend = (_env("LLM_BACKEND", "auto") or "auto").strip().lower()
        script_path = _env(
            "LLM_SCRIPT_PATH",
            "/home/unlimitediw/workspace/TYDeepResearch/AgenticRLModelTraining/model/scripts/call_model.py",
        )
        script_python = _env("LLM_SCRIPT_PYTHON", sys.executable) or sys.executable
        return LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            mock=mock,
            backend=backend,
            script_path=script_path or "",
            script_python=script_python,
        )


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()

    def _resolved_backend(self) -> str:
        """
        Decide which backend to use.
        - If LLM_MOCK=1, always mock.
        - If LLM_BACKEND explicitly set, honor it.
        - Otherwise (auto): prefer http if base_url+model are set (api_key optional for local servers);
          else use script.
        """
        if self.config.mock:
            return "mock"
        b = (self.config.backend or "auto").strip().lower()
        if b in ("http", "script", "mock"):
            return b
        # auto
        if self.config.base_url and self.config.model:
            return "http"
        return "script"

    def _ensure_real_config(self) -> None:
        backend = self._resolved_backend()
        if backend == "mock":
            return
        if backend == "script":
            if not self.config.script_path:
                raise RuntimeError(
                    "LLM script backend selected but no script path configured. "
                    "Set LLM_SCRIPT_PATH to your call_model.py."
                )
            if not os.path.exists(self.config.script_path):
                raise RuntimeError(
                    f"LLM script backend selected but script not found: {self.config.script_path}. "
                    "Set LLM_SCRIPT_PATH to a valid path."
                )
            return
        missing = []
        if not self.config.base_url:
            missing.append("LLM_BASE_URL")
        if not self.config.model:
            missing.append("LLM_MODEL")
        if missing:
            raise RuntimeError(
                "Missing LLM config env vars: "
                + ", ".join(missing)
                + ". Set them or export LLM_MOCK=1 to run without a real API."
            )
        # Many local OpenAI-compatible servers (e.g. vLLM) do not require an API key.

    def chat_once(self, *, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """
        Call one completion via selected backend.
        - http: OpenAI-compatible /v1/chat/completions
        - script: run a local Python script (prints response to stdout)
        Returns assistant content as string.
        """
        backend = self._resolved_backend()
        if backend == "mock":
            return self._mock_answer(system=system, user=user)

        self._ensure_real_config()

        if backend == "script":
            return self._script_chat_once(system=system, user=user, temperature=temperature, max_tokens=max_tokens)

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
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            method="POST",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"LLM HTTPError {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM URLError: {e}") from e

        obj = json.loads(raw)
        choices = obj.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned no choices: {raw}")
        msg = (choices[0].get("message") or {}).get("content")
        if not isinstance(msg, str):
            raise RuntimeError(f"LLM returned invalid message: {raw}")
        return msg.strip()

    def _script_chat_once(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:
        """
        Invoke a local script that behaves like TYDeepResearch call_model.py:
        - accepts --prompt/--temperature/--max-tokens (and optionally --base-url/--api-key/--model)
        - prints completion to stdout
        """
        prompt = f"{system}\n\n{user}".strip()
        cmd: List[str] = [
            self.config.script_python,
            self.config.script_path,
            "--prompt",
            prompt,
            "--temperature",
            str(temperature),
            "--max-tokens",
            str(max_tokens),
            "--output",
            "",  # disable JSONL logging by default
        ]

        # Reuse LLM_* vars as defaults for script args if provided.
        if self.config.base_url:
            cmd += ["--base-url", self.config.base_url]
        if self.config.api_key:
            cmd += ["--api-key", self.config.api_key]
        if self.config.model:
            cmd += ["--model", self.config.model]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
                check=False,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Failed to run LLM script backend. Missing executable or script: {e}. "
                "Check LLM_SCRIPT_PYTHON and LLM_SCRIPT_PATH."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"LLM script backend timed out after {self.config.timeout_s}s.") from e

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            details = "\n".join([s for s in [stdout, stderr] if s])
            raise RuntimeError(f"LLM script backend failed (exit={proc.returncode}).\n{details}")

        out = (proc.stdout or "").strip()
        if not out:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"LLM script backend returned empty output.\n{stderr}")
        return out

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
    ) -> List[str]:
        """
        Generate n independent answers by calling chat_once n times.
        In mock mode, answers are deterministic but vary by sample index.
        """
        answers: List[str] = []
        # If caller provides a full "standard prompt" in row["prompt"], do NOT wrap it or
        # constrain with a conflicting system prompt. Send it as-is as user content.
        mode = (prompt_mode or "problem").strip().lower()
        if mode in ("raw_prompt", "raw_prompt_eval"):
            # For eval we still provide a strict system to suppress chain-of-thought and enforce format.
            system = ""
            if mode == "raw_prompt_eval":
                system = (
                    "你是一个**判断答案与标准答案一致**的专家。\n"
                    "你只做一致性判断，不要解题、不要推理题目本身。\n"
                    "禁止输出<think>或任何推理过程。\n\n"
                    "注意：用户输入末尾可能包含一行 [GOLD_STANDARD_ANSWER]=...，这是唯一可信的标准答案来源。\n"
                    "你必须用该 gold 与 8 个解答做对比（而不是去解题）。\n\n"
                    "判定规则（必须遵守）：\n"
                    "1) 数值相同但表述不同视为一致（如 A vs -5；必要时用题目选项映射）。\n"
                    "2) 允许小误差：把数值/表达式换算为四位小数比较，|误差|<=1e-4 视为一致。\n"
                    "3) 表达式需换算为四位小数（例：2π=6.2832，3e=8.1548，√6=2.4495）。\n"
                    "4) 忽略格式。\n"
                    "5) 解答为空或缺少最终答案 -> 错误。\n\n"
                    "输出格式（必须严格输出，参照 sample.jsonl 的 content 样式；禁止输出 JSON/代码块/额外说明）：\n"
                    "你必须只输出一段 Markdown 文本（不是 JSON），结构必须包含且仅包含以下内容（顺序固定）：\n"
                    "我们按照题目要求，逐条对比么个解答与**标准答案**是否一致。\n\n"
                    "---\n\n"
                    "### 题目回顾：\n"
                    "题目是：\n"
                    "“<把[题目]原文放这里>” \n"
                    "标准答案是：**<使用 [GOLD_STANDARD_ANSWER] 的值；若 gold 为 A 且题目有选项A.-5，则写成 A.-5>** \n\n"
                    "---\n\n"
                    "###解答分析（逐个对比）：\n"
                    "- **解答1**：<解答1原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n"
                    "- **解答2**：<解答2原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n"
                    "- **解答3**：<解答3原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n"
                    "- **解答4**：<解答4原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n"
                    "- **解答5**：<解答5原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n"
                    "- **解答6**：<解答6原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n"
                    "- **解答7**：<解答7原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n"
                    "- **解答8**：<解答8原文> -> 与标准答案一致 ✅ 或 与标准答案不一致 ❌\n\n"
                    "---\n\n"
                    "### 判断汇总：\n\n"
                    "| 编号 | 是否正确 | 理由 |\n"
                    "|------|---------|------|\n"
                    "| 解答1 | 正确 ✅/错误 ❌ | <一句理由；必要时写“B（对应 5）”这种映射> |\n"
                    "| 解答2 | 正确 ✅/错误 ❌ | <一句理由> |\n"
                    "| 解答3 | 正确 ✅/错误 ❌ | <一句理由> |\n"
                    "| 解答4 | 正确 ✅/错误 ❌ | <一句理由> |\n"
                    "| 解答5 | 正确 ✅/错误 ❌ | <一句理由> |\n"
                    "| 解答6 | 正确 ✅/错误 ❌ | <一句理由> |\n"
                    "| 解答7 | 正确 ✅/错误 ❌ | <一句理由> |\n"
                    "| 解答8 | 正确 ✅/错误 ❌ | <一句理由> |\n\n"
                    "### 统计：\n"
                    "- 解题正确数量：<列出正确编号> -> 共**x个** \n"
                    "- 解题错误数量：<列出错误编号> -> 共 **y**个\n\n"
                    "---\n\n"
                    "最后一行必须且只能是：\\boxed{解答正确：x，解答错误：y}\n"
                )
            base_user = question.strip()
        else:
            # All math problems are Chinese in this project: keep prompts in Chinese to avoid
            # steering the model into English or changing the task framing.
            system = (
                f"你是一个数学解题助手。Stage={stage_name}。\n"
                "你必须只输出最终答案，不要输出推理过程，不要输出<think>。\n"
                "如果题目是选择题：只输出单个大写字母 A/B/C/D（不要带标点、不要带解释）。\n"
                "如果题目不是选择题：只输出最终数值/表达式一行。"
            )
            base_user = (
                f"题目：\n{question}\n\n"
                "要求：只输出最终答案一行，不要输出推理过程、不要解释、不要<think>。"
            )
        for i in range(n):
            if mode == "raw_prompt":
                user = base_user
            else:
                user = f"{base_user}\n采样编号={i}"
            ans = self.chat_once(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
            answers.append(ans)
            if sleep_s > 0:
                time.sleep(sleep_s)
        return answers

    def _mock_answer(self, *, system: str, user: str) -> str:
        """
        Deterministic mock answer generator.
        Produces task-shaped outputs:
        - If the prompt looks like an evaluator prompt (mentions boxed counts), return a boxed summary.
        - Otherwise return an A/B/C/D style answer (or small numeric) to simulate solvers.
        """
        text = (system + "\n" + user).strip()
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Evaluator-like prompt: must contain both "解答正确" and "解答错误" and a boxed example.
        # In mock mode, still output a *sample.jsonl-like long markdown* (not only boxed),
        # so users can validate the pipeline end-to-end without a real model.
        if "解答正确" in text and "解答错误" in text and "\\boxed" in text:
            gold = ""
            m_gold = re.search(r"(?m)^\[GOLD_STANDARD_ANSWER\]=(.*)$", user)
            if m_gold:
                gold = (m_gold.group(1) or "").strip()

            # Extract question block from stored prompt (between "[题目]" and the next "###" delimiter)
            q_block = ""
            m_q = re.search(r"\[题目\]\s*\n(?P<q>[\s\S]*?)\n###", user)
            if m_q:
                q_block = (m_q.group("q") or "").strip()

            # Extract options (A./B./C./D.) from question block
            opt_map: Dict[str, str] = {}
            for m in re.finditer(r"(?m)^\s*([ABCD])[\.．、]\s*(.+?)\s*$", q_block):
                k = m.group(1).upper()
                v = (m.group(2) or "").strip()
                if k not in opt_map and v:
                    opt_map[k] = v

            def _norm(s: str) -> str:
                return re.sub(r"\s+", "", (s or "")).strip()

            def _std_from_letter(letter: str) -> str:
                letter = (letter or "").strip().upper()
                if len(letter) == 1 and letter in opt_map:
                    return f"{letter}.{opt_map[letter]}"
                return letter

            std_gold = _std_from_letter(gold) if (gold and len(gold.strip()) == 1) else (gold or "").strip()

            answers: List[str] = []
            for i in range(1, 9):
                m_a = re.search(rf"\[解答{i}\]\s*\n(?P<a>[\s\S]*?)\n\[解答{i}\]", user)
                a = (m_a.group("a") if m_a else "") or ""
                answers.append(a.strip())

            correct_ids: List[int] = []
            wrong_ids: List[int] = []
            lines_compare: List[str] = []
            table_rows: List[str] = []

            for idx, a in enumerate(answers, start=1):
                a0 = (a or "").strip()
                a_cmp = a0
                if len(a0) == 1 and a0.upper() in opt_map:
                    a_cmp = _std_from_letter(a0)

                ok = bool(a_cmp) and (_norm(a_cmp) == _norm(std_gold) or _norm(a_cmp) == _norm(gold))
                if ok:
                    correct_ids.append(idx)
                    lines_compare.append(f"- **解答{idx}**：{a0} -> 与标准答案一致 ✅")
                    table_rows.append(f"| 解答{idx} | 正确 ✅ | {a_cmp}（与标准答案一致） |")
                else:
                    wrong_ids.append(idx)
                    lines_compare.append(f"- **解答{idx}**：{a0} -> 与标准答案不一致 ❌")
                    table_rows.append(f"| 解答{idx} | 错误 ❌ | {a_cmp or '（空）'}（与标准答案不一致） |")

            ok_n = len(correct_ids)
            bad_n = len(wrong_ids)
            q_show = q_block or "<缺失题目>"
            gold_show = std_gold or gold or "<缺失标准答案>"
            ok_list = ",".join([f"解答{i}" for i in correct_ids]) if correct_ids else ""
            bad_list = ",".join([f"解答{i}" for i in wrong_ids]) if wrong_ids else ""

            return (
                "我们按照题目要求，逐条对比么个解答与**标准答案**是否一致。\n\n"
                "---\n\n"
                "### 题目回顾：\n"
                "题目是：\n"
                f"“{q_show}” \n"
                f"标准答案是：**{gold_show}** \n\n"
                "---\n\n"
                "###解答分析（逐个对比）：\n\n"
                + "\n".join(lines_compare)
                + "\n\n---\n\n"
                "### 判断汇总：\n\n"
                "| 编号 | 是否正确 | 理由 |\n"
                "|------|---------|------|\n"
                + "\n".join(table_rows)
                + "\n\n"
                "### 统计：\n"
                f"- 解题正确数量：{ok_list} -> 共**{ok_n}个** \n"
                f"- 解题错误数量：{bad_list} -> 共 **{bad_n}**个\n\n"
                "---\n\n"
                f"\\boxed{{解答正确：{ok_n}，解答错误：{bad_n}}}"
            )

        # Solver-like output: choose among A/B/C/D to match multiple-choice tasks
        if any(opt in text for opt in ["A.", "B.", "C.", "D.", "A-", "B-", "C-", "D-"]):
            bucket = int(h[:8], 16) % 4
            letter = ["A", "B", "C", "D"][bucket]
            # Keep a verbose raw output for debugging (archived in stage1_raw_generations.jsonl),
            # while still allowing the pipeline to extract the final answer reliably.
            return (
                "<think>\n"
                "（mock）这里是用于调试的原始长输出，模拟真实模型可能产生的推理/冗余内容。\n"
                f"hash={h}\n"
                "</think>\n\n"
                f"{letter}"
            )

        # Fallback: small integer string
        n = str(int(h[:8], 16) % 5)
        return (
            "<think>\n"
            "（mock）这里是用于调试的原始长输出，模拟真实模型可能产生的推理/冗余内容。\n"
            f"hash={h}\n"
            "</think>\n\n"
            f"{n}"
        )



