import math
import threading
from dataclasses import dataclass


@dataclass
class _CacheEntry:
    query: str
    embedding: list[float]
    result: dict


class QueryCache:
    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.95,
        max_size: int = 100,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._lock = threading.Lock()
        self._entries: list[_CacheEntry] = []
        self._hit = 0
        self._miss = 0

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    async def get(self, query: str) -> dict | None:
        if not query:
            return None
        try:
            embedding = await self._embedder.aembed_query(query)
        except Exception:
            return None

        with self._lock:
            best_idx = -1
            best_sim = self._threshold
            for idx, entry in enumerate(self._entries):
                sim = self._cosine_similarity(embedding, entry.embedding)
                if sim >= best_sim:
                    best_sim = sim
                    best_idx = idx
            if best_idx >= 0:
                entry = self._entries.pop(best_idx)
                self._entries.append(entry)
                self._hit += 1
                return entry.result
        self._miss += 1
        return None

    async def put(self, query: str, result: dict) -> None:
        if query is None:
            return
        if self._max_size <= 0:
            return
        try:
            embedding = await self._embedder.aembed_query(query)
        except Exception:
            return
        with self._lock:
            if len(self._entries) >= self._max_size:
                self._entries.pop(0)
            self._entries.append(_CacheEntry(query=query, embedding=embedding, result=result))

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._hit = 0
            self._miss = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._entries),
                "hit": self._hit,
                "miss": self._miss,
            }
