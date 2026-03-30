import pytest

from src.cache.lru_cache import QueryCache


class _FakeEmbedder:
    def __init__(self, dim: int = 64):
        self._dim = dim
        self._counter = 0

    async def aembed_query(self, text: str) -> list[float]:
        self._counter += 1
        vec = [float(ord(ch)) for ch in text]
        if len(vec) < self._dim:
            vec += [0.0] * (self._dim - len(vec))
        return vec[: self._dim]


class _ConstantEmbedder:
    async def aembed_query(self, text: str) -> list[float]:
        return [1.0] * 64


class _OrthogonalEmbedder:
    def __init__(self):
        self._vectors: dict[str, list[float]] = {}

    async def aembed_query(self, text: str) -> list[float]:
        if text not in self._vectors:
            idx = len(self._vectors)
            vec = [0.0] * 64
            vec[idx % 64] = 1.0
            self._vectors[text] = vec
        return self._vectors[text]


@pytest.mark.asyncio
async def test_cache_miss_then_hit():
    embedder = _FakeEmbedder()
    cache = QueryCache(embedder, similarity_threshold=0.0, max_size=10)

    result = await cache.get("hello")
    assert result is None

    await cache.put("hello", {"answer": "world", "sources": []})
    result = await cache.get("hello")
    assert result == {"answer": "world", "sources": []}


@pytest.mark.asyncio
async def test_cache_semantic_hit():
    embedder = _ConstantEmbedder()
    cache = QueryCache(embedder, similarity_threshold=0.9, max_size=10)

    await cache.put("什么是GDP", {"answer": "GDP是...", "sources": []})
    result = await cache.get("GDP是什么意思")
    assert result is not None
    assert result["answer"] == "GDP是..."


@pytest.mark.asyncio
async def test_cache_lru_eviction_order():
    embedder = _OrthogonalEmbedder()
    cache = QueryCache(embedder, similarity_threshold=0.5, max_size=2)

    await cache.put("first", {"answer": "1"})
    await cache.put("second", {"answer": "2"})
    await cache.put("third", {"answer": "3"})

    assert cache.stats()["size"] == 2

    assert await cache.get("first") is None
    assert await cache.get("second") is not None
    assert await cache.get("third") is not None


@pytest.mark.asyncio
async def test_cache_stats_counters():
    embedder = _OrthogonalEmbedder()
    cache = QueryCache(embedder, similarity_threshold=0.5, max_size=10)

    assert cache.stats() == {"size": 0, "hit": 0, "miss": 0}

    await cache.put("q1", {"answer": "a1"})
    await cache.put("q2", {"answer": "a2"})

    assert await cache.get("q1") is not None
    assert await cache.get("q1") is not None
    assert await cache.get("not_cached") is None

    stats = cache.stats()
    assert stats["hit"] == 2
    assert stats["miss"] == 1


@pytest.mark.asyncio
async def test_cache_clear():
    embedder = _FakeEmbedder()
    cache = QueryCache(embedder, similarity_threshold=0.0, max_size=10)

    await cache.put("x", {"answer": "9"})
    await cache.get("x")

    cache.clear()
    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["hit"] == 0
    assert stats["miss"] == 0


@pytest.mark.asyncio
async def test_cache_disabled_behaviour():
    embedder = _FakeEmbedder()
    cache = QueryCache(embedder, similarity_threshold=0.0, max_size=0)

    await cache.put("anything", {"answer": "nope"})
    assert cache.stats()["size"] == 0


@pytest.mark.asyncio
async def test_cache_empty_query():
    embedder = _FakeEmbedder()
    cache = QueryCache(embedder, similarity_threshold=0.0, max_size=10)

    result = await cache.get("")
    assert result is None
