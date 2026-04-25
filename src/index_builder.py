"""文档索引构建脚本。

将 data/raw/ 下的文档经加载 → 清洗 → 分块 → Embedding → 存储到 ChromaDB。
支持增量索引（已有文档不重复处理）和进度条显示。
"""

import argparse
import hashlib
import logging
import os
from pathlib import Path

from src.config import Config
from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
from src.loaders.base_loader import Document
from src.loaders.pdf_loader import PDFLoader
from src.loaders.qa_loader import QALoader
from src.loaders.text_loader import TextLoader
from src.processor.chunker import Chunk, TextChunker, TitleBasedChunker
from src.processor.cleaner import TextCleaner
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ChunkerType = TextChunker | TitleBasedChunker


class IndexBuilder:

    def __init__(self, api_key: str | None = None) -> None:
        self._config = Config()

        key = api_key or os.environ.get("SILICONFLOW_API_KEY", "") or self._config.api_key
        if not key:
            raise ValueError("未提供 API Key，请设置 SILICONFLOW_API_KEY 环境变量")
        self._embedder = SiliconFlowEmbedder(
            api_key=key,
            model=self._config.embedding.model,
        )
        logger.info("使用 SiliconFlow %s 构建索引", self._config.embedding.model)
        self._store = ChromaStore(
            persist_directory=self._config.vectorstore.persist_directory,
            collection_name=self._config.vectorstore.collection_name,
        )
        # 根据 config 策略选择分块器
        self._chunker: ChunkerType
        if self._config.chunker.strategy == "title":
            self._chunker = TitleBasedChunker(
                chunk_size=self._config.chunker.title_chunk_size,
                title_patterns=self._config.chunker.title_patterns,
            )
        else:
            self._chunker = TextChunker(
                chunk_size=self._config.chunker.chunk_size,
                chunk_overlap=self._config.chunker.chunk_overlap,
                separator=self._config.chunker.separator,
            )
        self._cleaner = TextCleaner()

        # 确保数据目录存在
        self._raw_dir = _PROJECT_ROOT / "data" / "raw"
        self._news_dir = self._raw_dir / "news"
        self._reports_dir = self._raw_dir / "reports"
        self._qa_dir = self._raw_dir / "qa"

    def build_index(self, show_progress: bool = True) -> dict:
        """执行完整的索引构建流程。

        Args:
            show_progress: 是否显示进度条。

        Returns:
            统计信息字典。
        """
        # 1. 加载文档
        logger.info("开始加载文档...")
        documents = self._load_all_documents()
        logger.info("共加载 %d 个文档", len(documents))

        if not documents:
            return {"documents_loaded": 0, "chunks_created": 0, "chunks_indexed": 0, "chunks_skipped": 0}

        # 2. 清洗 + 分块
        logger.info("开始处理文档...")
        chunks = self._process_documents(documents)
        logger.info("共生成 %d 个文本块", len(chunks))

        if not chunks:
            return {"documents_loaded": len(documents), "chunks_created": 0, "chunks_indexed": 0, "chunks_skipped": 0}

        # 3. 获取已有 ID（增量索引）
        existing_ids = set(self._store.get_all_ids())
        logger.info("向量库中已有 %d 条记录", len(existing_ids))

        # 4. Embedding + 存储
        logger.info("开始向量化并存储...")
        stats = self._index_chunks(chunks, existing_ids, show_progress)
        stats["documents_loaded"] = len(documents)
        stats["chunks_created"] = len(chunks)

        logger.info("索引构建完成: %s", stats)
        return stats

    def _load_all_documents(self) -> list[Document]:
        """从 data/raw/ 下所有子目录加载文档。"""
        all_documents: list[Document] = []

        loaders = [
            (self._news_dir, TextLoader()),
            (self._reports_dir, PDFLoader()),
            (self._qa_dir, QALoader()),
        ]

        for directory, loader in loaders:
            if not directory.exists():
                logger.info("目录不存在，跳过: %s", directory)
                continue

            try:
                docs = loader.load(directory)
                all_documents.extend(docs)
                logger.info("从 %s 加载了 %d 个文档", directory.name, len(docs))
            except Exception as e:
                logger.error("加载 %s 失败: %s", directory, e)

        return all_documents

    def _process_documents(self, documents: list[Document]) -> list[Chunk]:
        """清洗文档内容并分块。"""
        # 先清洗文档内容
        for doc in documents:
            doc.content = self._cleaner.clean(doc.content)

        # 分块
        return self._chunker.chunk_documents(documents)

    def _index_chunks(
        self,
        chunks: list[Chunk],
        existing_ids: set[str],
        show_progress: bool = True,
    ) -> dict:
        """向量化并存储文本块，跳过已有 ID。"""
        # 单次遍历即可筛选新块并准备嵌入数据，避免重复遍历
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []
        skipped = 0
        new_chunks_count = 0

        for chunk in chunks:
            chunk_hash = hashlib.md5(chunk.content.encode("utf-8")).hexdigest()
            chunk_uid = f"{chunk.metadata.get('source', 'unknown')}_{chunk_hash}"
            if chunk_uid in existing_ids:
                skipped += 1
                continue
            # 新块，准备数据用于向量化与存储
            new_chunks_count += 1
            ids.append(chunk_uid)
            texts.append(chunk.content)
            meta = dict(chunk.metadata)
            meta["chunk_id"] = str(chunk.chunk_id)
            metadatas.append(meta)

        if not ids:
            logger.info("所有块已存在，无需更新")
            return {"chunks_indexed": 0, "chunks_skipped": skipped}

        # 批量 Embedding（带进度）
        all_embeddings: list[list[float]] = []
        batch_size = self._config.embedding.batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            if show_progress:
                logger.info(
                    "Embedding 进度: %d/%d (%d/%d texts)",
                    batch_num,
                    total_batches,
                    len(all_embeddings),
                    len(texts),
                )
            embeddings = self._embedder.embed_texts(batch)
            all_embeddings.extend(embeddings)

        # 存储
        self._store.add_documents(
            ids=ids,
            documents=texts,
            embeddings=all_embeddings,
            metadatas=metadatas,
        )

        return {
            "chunks_indexed": len(ids),
            "chunks_skipped": skipped,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 Financial RAG 文档索引")
    parser.add_argument("--no-progress", action="store_true", help="禁用进度显示")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    builder = IndexBuilder()
    stats = builder.build_index(show_progress=not args.no_progress)
    print(f"\n索引构建完成: {stats}")
