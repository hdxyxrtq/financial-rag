import logging
from collections.abc import AsyncGenerator, Generator

from zhipuai import ZhipuAI

from src.utils import (
    _AUTH_KEYWORDS,
    _QUOTA_KEYWORDS,
    _RATE_LIMIT_KEYWORDS,
    _TIMEOUT_KEYWORDS,
    async_call_with_retry,
    call_with_retry,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMAuthError(LLMError):
    pass


class LLMQuotaError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


def _classify_error(error: Exception) -> LLMError:
    msg = str(error).lower()
    if any(kw in msg for kw in _AUTH_KEYWORDS):
        return LLMAuthError("API Key 无效，请在侧边栏检查您的 API Key")
    if any(kw in msg for kw in _QUOTA_KEYWORDS):
        return LLMQuotaError("API 额度已用完，请到智谱开放平台充值或更换 Key")
    if any(kw in msg for kw in _TIMEOUT_KEYWORDS):
        return LLMTimeoutError("API 请求超时，请检查网络连接后重试")
    if any(kw in msg for kw in _RATE_LIMIT_KEYWORDS):
        return LLMRateLimitError("请求过于频繁，请稍后再试")
    return LLMError(f"GLM API 调用失败: {error}")


class ZhipuLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "glm-4-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
    ) -> None:
        self._client = ZhipuAI(api_key=api_key)
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    async def achat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        def do_call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=all_messages,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                top_p=self._top_p,
            )

        response = await async_call_with_retry(
            do_call, _classify_error,
            non_retriable_types=(LLMAuthError, LLMQuotaError),
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    async def astream_chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        async def do_call():
            return await self._get_async_client().chat.completions.create(
                model=self._model,
                messages=all_messages,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                top_p=self._top_p,
                stream=True,
            )

        response = await async_call_with_retry(
            do_call, _classify_error,
            non_retriable_types=(LLMAuthError, LLMQuotaError),
        )
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.warning("异步流式生成中断: %s", e)

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        # Only prepend a system prompt if it's provided
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        def do_call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=all_messages,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                top_p=self._top_p,
            )

        response = call_with_retry(do_call, _classify_error, non_retriable_types=(LLMAuthError, LLMQuotaError))
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    def stream_chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        # Only prepend a system prompt if it's provided
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        def do_call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=all_messages,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                top_p=self._top_p,
                stream=True,
            )

        response = call_with_retry(do_call, _classify_error, non_retriable_types=(LLMAuthError, LLMQuotaError))
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.warning("流式生成中断: %s", e)
