import logging
import time

import httpx

logger = logging.getLogger(__name__)


class SiliconFlowLLM:
    """SiliconFlow LLM（OpenAI 兼容接口），用于 RAG 生成和 Query Rewriting。

    实现 ZhipuLLM 的 chat() 接口签名，可无缝替换。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/Qwen3-8B",
        base_url: str = "https://api.siliconflow.cn/v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        all_msgs: list[dict[str, str]] = []
        if system_prompt:
            all_msgs.append({"role": "system", "content": system_prompt})
        all_msgs.extend(messages)

        for attempt in range(3):
            try:
                resp = httpx.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": all_msgs,
                        "temperature": temperature if temperature is not None else self._temperature,
                        "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                body = resp.json()
                return body["choices"][0]["message"]["content"] or ""
            except Exception as e:
                logger.warning("SiliconFlow API 调用失败 (%d/3): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"SiliconFlow API 连续 3 次调用失败: {self._model}")

    def stream_chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        all_msgs: list[dict[str, str]] = []
        if system_prompt:
            all_msgs.append({"role": "system", "content": system_prompt})
        all_msgs.extend(messages)

        with httpx.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": all_msgs,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
                "stream": True,
            },
            timeout=60.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                import json
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    yield delta["content"]
