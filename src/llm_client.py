from __future__ import annotations

import hashlib
import json
import os
import time
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
    """

    base_url: str
    api_key: str
    model: str
    timeout_s: int = 60
    mock: bool = False

    @staticmethod
    def from_env() -> "LLMConfig":
        base_url = _env("LLM_BASE_URL", "") or ""
        api_key = _env("LLM_API_KEY", "") or ""
        model = _env("LLM_MODEL", "") or ""
        timeout_s = int(_env("LLM_TIMEOUT_S", "60") or "60")
        mock = (_env("LLM_MOCK", "0") or "0") == "1"
        return LLMConfig(base_url=base_url, api_key=api_key, model=model, timeout_s=timeout_s, mock=mock)


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()

    def _ensure_real_config(self) -> None:
        if self.config.mock:
            return
        missing = []
        if not self.config.base_url:
            missing.append("LLM_BASE_URL")
        if not self.config.api_key:
            missing.append("LLM_API_KEY")
        if not self.config.model:
            missing.append("LLM_MODEL")
        if missing:
            raise RuntimeError(
                "Missing LLM config env vars: "
                + ", ".join(missing)
                + ". Set them or export LLM_MOCK=1 to run without a real API."
            )

    def chat_once(self, *, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """
        Call OpenAI-compatible /v1/chat/completions (single completion).
        Returns assistant content as string.
        """
        if self.config.mock:
            return self._mock_answer(system=system, user=user)

        self._ensure_real_config()

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

    def generate_n(
        self,
        *,
        stage_name: str,
        question: str,
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
        system = f"You are a math problem solver. Return ONLY the final answer. Stage={stage_name}."
        for i in range(n):
            user = f"Problem:\n{question}\n\nSampleIndex={i}\nReturn only the final answer."
            ans = self.chat_once(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
            answers.append(ans)
            if sleep_s > 0:
                time.sleep(sleep_s)
        return answers

    def _mock_answer(self, *, system: str, user: str) -> str:
        """
        Deterministic mock answer generator.
        Produces a small set of candidate answers to simulate agreement/disagreement.
        """
        h = hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()
        # map to a few buckets so voting can sometimes converge and sometimes not
        bucket = int(h[:8], 16) % 5
        candidates = ["0", "1", "2", "3", "4"]
        return candidates[bucket]


