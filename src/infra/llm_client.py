from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class LLMConfig:
    """
    Minimal OpenAI-compatible config (HTTP only).
    """

    base_url: str
    api_key: str
    model: str
    timeout_s: int = 60

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LLMConfig":
        base_url = str(d.get("base_url") or "")
        api_key = str(d.get("api_key") or "")
        model = str(d.get("model") or "")
        timeout_s = int(d.get("timeout_s") or 60)
        return LLMConfig(base_url=base_url, api_key=api_key, model=model, timeout_s=timeout_s)


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

    def chat_once(self, *, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
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
        data = json.dumps(body).encode("utf-8")
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(url, method="POST", data=data, headers=headers)

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
        """
        answers: List[str] = []

        mode = (prompt_mode or "problem").strip().lower()
        if mode in ("raw_prompt", "raw_prompt_eval"):
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
            system = (
                f"你是一个数学解题助手。Stage={stage_name}。\n"
                "你可以输出推理过程。\n"
                "最后一行必须输出最终答案，且格式必须严格为：FINAL: <答案>\n"
                "如果题目是选择题：<答案> 只能是单个大写字母 A/B/C/D。\n"
                "如果题目不是选择题：<答案> 为最终数值/表达式。"
            )
            base_user = (
                f"题目：\n{question}\n\n"
                "要求：你可以写推理过程，但最后一行必须是 FINAL: <答案>（严格格式）。"
            )

        for i in range(n):
            user = base_user if mode in ("raw_prompt", "raw_prompt_eval") else f"{base_user}\n采样编号={i}"
            ans = self.chat_once(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
            answers.append(ans)
            if sleep_s > 0:
                time.sleep(sleep_s)
        return answers



