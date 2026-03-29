import logging
import time
import random
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_AUTH_KEYWORDS = ("invalid api key", "authorization", "unauthorized", "authentication")
_QUOTA_KEYWORDS = ("quota", "limit exceeded", "insufficient", "resource exhausted")
_TIMEOUT_KEYWORDS = ("timeout", "timed out", "connection timed out")
_RATE_LIMIT_KEYWORDS = ("rate limit", "too many requests", "429")


def call_with_retry(
    fn: Callable[[], Any],
    classify_fn: Callable[[Exception], Exception],
    max_retries: int = 3,
    non_retriable_types: tuple[type, ...] = (),
) -> Any:
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            classified = classify_fn(e)
            if isinstance(classified, non_retriable_types):
                raise classified from e
            if attempt == max_retries - 1:
                logger.error("API 调用失败（已重试 %d 次）: %s", max_retries, e)
                raise classified from e
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning(
                "API 调用失败，%d 秒后重试 (%d/%d): %s",
                wait, attempt + 1, max_retries, e,
            )
            time.sleep(wait)
