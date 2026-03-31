import logging
import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class LLMConfig:
    model: str = "glm-4-flash"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9

    def __post_init__(self):
        if not (0 <= self.temperature <= 2):
            raise ValueError(f"temperature must be 0-2, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not (0 <= self.top_p <= 1):
            raise ValueError(f"top_p must be 0-1, got {self.top_p}")


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "embedding-3"
    batch_size: int = 20

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass(frozen=True)
class ChunkerConfig:
    chunk_size: int = 512
    chunk_overlap: int = 100
    separator: str = "\n\n"
    strategy: str = "paragraph"
    title_chunk_size: int = 1024
    title_patterns: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")


@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int = 5
    score_threshold: float = 0.5

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError(f"score_threshold must be 0-1, got {self.score_threshold}")


@dataclass(frozen=True)
class VectorStoreConfig:
    persist_directory: str = "data/chroma_db"
    collection_name: str = "financial_docs"


@dataclass(frozen=True)
class RAGConfig:
    max_context_tokens: int = 4000
    query_rewrite: bool = False


@dataclass(frozen=True)
class HybridConfig:
    enabled: bool = True
    strategy: str = "hybrid"  # "vector" / "bm25" / "hybrid"
    rrf_k: int = 60
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    bm25_fetch_k: int = 30
    vector_fetch_k: int = 30


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    retrieve_n: int = 20
    top_n: int = 5
    min_score: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    title: str = "Financial RAG - 金融知识库问答"
    page_title: str = "金融 RAG 问答系统"


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = False
    similarity_threshold: float = 0.95
    max_size: int = 100


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_yaml(path: Path) -> dict[str, object]:
    with open(path, encoding="utf-8") as f:
        result = yaml.safe_load(f)
    assert isinstance(result, dict)
    return result


class Config:
    _llm: LLMConfig
    _embedding: EmbeddingConfig
    _chunker: ChunkerConfig
    _retriever: RetrieverConfig
    _vectorstore: VectorStoreConfig
    _rag: RAGConfig
    _hybrid: HybridConfig
    _reranker: RerankerConfig
    _cache: CacheConfig
    _app: AppConfig

    def __init__(self) -> None:
        load_dotenv(_PROJECT_ROOT / ".env")

        config_path = _PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")

        raw = _load_yaml(config_path)
        self._llm = self._make(LLMConfig, cast(dict[str, Any], raw.get("llm") or {}))
        self._embedding = self._make(EmbeddingConfig, cast(dict[str, Any], raw.get("embedding") or {}))
        self._chunker = self._make(ChunkerConfig, cast(dict[str, Any], raw.get("chunker") or {}))
        self._retriever = self._make(RetrieverConfig, cast(dict[str, Any], raw.get("retriever") or {}))
        self._vectorstore = self._make(VectorStoreConfig, cast(dict[str, Any], raw.get("vectorstore") or {}))
        self._rag = self._make(RAGConfig, cast(dict[str, Any], raw.get("rag") or {}))
        self._hybrid = self._make(HybridConfig, cast(dict[str, Any], raw.get("hybrid") or {}))
        self._reranker = self._make(RerankerConfig, cast(dict[str, Any], raw.get("reranker") or {}))
        self._app = self._make(AppConfig, cast(dict[str, Any], raw.get("app") or {}))
        cache_raw: dict[str, Any] = {}
        if isinstance(raw, dict):
            tmp = raw.get("cache")
            if isinstance(tmp, dict):
                cache_raw = tmp
        self._cache = self._make(CacheConfig, cache_raw)
        # Safe retrieval of logging level from YAML
        log_level = "INFO"
        if isinstance(raw, dict):
            tmp = raw.get("logging")
            if isinstance(tmp, dict) and "level" in tmp:
                log_level = str(tmp["level"])
        setup_logging(log_level)

    @staticmethod
    def _make(target_cls: type, raw: dict[str, Any]):
        field_names = {f.name for f in dataclass_fields(target_cls)}
        return target_cls(**{k: v for k, v in raw.items() if k in field_names})

    @property
    def llm(self) -> LLMConfig:
        return self._llm

    @property
    def embedding(self) -> EmbeddingConfig:
        return self._embedding

    @property
    def chunker(self) -> ChunkerConfig:
        return self._chunker

    @property
    def retriever(self) -> RetrieverConfig:
        return self._retriever

    @property
    def vectorstore(self) -> VectorStoreConfig:
        return self._vectorstore

    @property
    def rag(self) -> RAGConfig:
        return self._rag

    @property
    def hybrid(self) -> HybridConfig:
        return self._hybrid

    @property
    def reranker(self) -> RerankerConfig:
        return self._reranker

    @property
    def app(self) -> AppConfig:
        return self._app

    @property
    def api_key(self) -> str | None:
        return os.environ.get("ZHIPU_API_KEY")

    @property
    def cache(self) -> CacheConfig:
        return self._cache
