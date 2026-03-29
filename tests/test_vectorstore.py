import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from src.vectorstore.chroma_store import ChromaStore, VectorStoreError


def _random_embedding(dim: int = 128) -> list[float]:
    """生成随机归一化向量用于测试。"""
    vec = np.random.randn(dim).tolist()
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


class TestChromaStore:
    """ChromaStore 单元测试。"""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.store = ChromaStore(
            persist_directory=self.temp_dir,
            collection_name="test_collection",
        )

    def teardown_method(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_search(self) -> None:
        """添加文档后应能检索到。"""
        embeddings = [_random_embedding(), _random_embedding()]

        self.store.add_documents(
            ids=["doc1", "doc2"],
            documents=["人工智能技术发展迅速", "机器学习是人工智能的分支"],
            embeddings=embeddings,
        )

        results = self.store.search(embeddings[0], top_k=2)

        assert len(results) >= 1
        assert results[0]["content"] == "人工智能技术发展迅速"

    def test_add_documents_validation_mismatch(self) -> None:
        """ids/documents/embeddings 长度不一致应报错。"""
        with pytest.raises(VectorStoreError, match="长度必须一致"):
            self.store.add_documents(
                ids=["a", "b"],
                documents=["doc1"],
                embeddings=[_random_embedding(), _random_embedding()],
            )

    def test_add_documents_metadata_mismatch(self) -> None:
        """metadatas 长度不一致应报错。"""
        with pytest.raises(VectorStoreError, match="长度必须一致"):
            self.store.add_documents(
                ids=["a"],
                documents=["doc1"],
                embeddings=[_random_embedding()],
                metadatas=[{"key": "val"}, {"extra": "val2"}],
            )

    def test_search_with_metadata_filter(self) -> None:
        """元数据过滤应只返回匹配的文档。"""
        embeddings = [_random_embedding(), _random_embedding()]

        self.store.add_documents(
            ids=["doc1", "doc2"],
            documents=["股票市场分析", "债券市场分析"],
            embeddings=embeddings,
            metadatas=[{"doc_type": "news"}, {"doc_type": "report"}],
        )

        results = self.store.search(embeddings[0], top_k=10, where={"doc_type": "report"})

        assert len(results) == 1
        assert results[0]["metadata"]["doc_type"] == "report"

    def test_delete_collection(self) -> None:
        """删除集合后文档数量应为 0。"""
        self.store.add_documents(
            ids=["doc1"],
            documents=["test content"],
            embeddings=[_random_embedding()],
        )
        assert self.store.get_stats()["document_count"] == 1

        self.store.delete_collection()
        assert self.store.get_stats()["document_count"] == 0

    def test_get_stats(self) -> None:
        """get_stats 应返回正确的统计信息。"""
        stats = self.store.get_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["document_count"] == 0

        self.store.add_documents(
            ids=["a", "b"],
            documents=["doc1", "doc2"],
            embeddings=[_random_embedding(), _random_embedding()],
        )

        stats = self.store.get_stats()
        assert stats["document_count"] == 2

    def test_persistence(self) -> None:
        """持久化：重新创建 ChromaStore 应保留数据。"""
        self.store.add_documents(
            ids=["doc1"],
            documents=["持久化测试"],
            embeddings=[_random_embedding()],
        )

        # 使用相同目录创建新实例
        new_store = ChromaStore(
            persist_directory=self.temp_dir,
            collection_name="test_collection",
        )

        assert new_store.get_stats()["document_count"] == 1

    def test_search_empty_collection(self) -> None:
        """空集合搜索应返回空列表。"""
        results = self.store.search(_random_embedding(), top_k=5)

        assert results == []

    def test_score_range(self) -> None:
        """搜索结果的 score 应在 0~1 范围内。"""
        embeddings = [_random_embedding() for _ in range(5)]
        docs = [f"文档内容 {i}" for i in range(5)]
        ids = [f"doc{i}" for i in range(5)]

        self.store.add_documents(ids=ids, documents=docs, embeddings=embeddings)

        results = self.store.search(embeddings[0], top_k=5)

        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_get_all_ids(self) -> None:
        """get_all_ids 应返回所有已存储的 ID。"""
        self.store.add_documents(
            ids=["id1", "id2", "id3"],
            documents=["a", "b", "c"],
            embeddings=[_random_embedding() for _ in range(3)],
        )

        ids = self.store.get_all_ids()

        assert set(ids) == {"id1", "id2", "id3"}
