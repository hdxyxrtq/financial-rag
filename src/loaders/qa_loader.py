from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

from src.loaders.base_loader import BaseLoader, Document

logger = logging.getLogger(__name__)

# 多编码读取支持（JSON/CSV 统一解析容错）
_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030", "latin-1"]


class QALoader(BaseLoader):
    """结构化 Q&A 数据加载器。

    支持 JSON 和 CSV 格式，将 Q&A 拼接为「问题：...\n答案：...」格式。
    """

    SUPPORTED_EXTENSIONS = {".json", ".csv"}

    def load(self, source: str | Path) -> list[Document]:
        """加载 Q&A 数据文件。

        Args:
            source: JSON 或 CSV 文件路径。

        Returns:
            Q&A 文档列表，每条 Q&A 对应一个 Document。
        """
        path = Path(source)

        if not path.exists():
            logger.error("文件不存在: %s", path)
            return []

        if path.is_dir():
            return self._load_directory(path)

        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._load_json(path)
        elif suffix == ".csv":
            return self._load_csv(path)
        else:
            logger.warning("不支持的文件格式: %s", suffix)
            return []

    # load_batch moved to BaseLoader for DRY behavior

    def _load_json(self, path: Path) -> list[Document]:
        """加载 JSON 格式的 Q&A 数据。

        支持两种结构：
        1. 列表格式: [{"question": ..., "answer": ..., ...}, ...]
        2. 对象格式: {"qa_list": [...], ...}  — 自动找到第一个列表字段
        """
        try:
            raw = None
            for enc in _ENCODINGS:
                try:
                    raw = path.read_text(encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if raw is None:
                logger.error("无法解码文件（已尝试所有编码）: %s", path)
                return []
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("JSON 解析失败 %s: %s", path, e)
            return []

        items: list[dict] = []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # 找到第一个值类型为列表的字段
            for key, value in data.items():
                if isinstance(value, list):
                    items = value
                    logger.debug("从 JSON 对象的 '%s' 字段加载 Q&A 数据", key)
                    break
        else:
            logger.error("JSON 格式不受支持，需要列表或对象: %s", path)
            return []

        return self._items_to_documents(items, path)

    def _load_csv(self, path: Path) -> list[Document]:
        """加载 CSV 格式的 Q&A 数据。

        必需列: question
        可选列: answer, source, category
        """
        try:
            raw = path.read_text(encoding="utf-8-sig")
            reader = csv.DictReader(io.StringIO(raw))
            items = list(reader)
        except Exception as e:
            logger.error("CSV 解析失败 %s: %s", path, e)
            return []

        return self._items_to_documents(items, path)

    def _items_to_documents(
        self, items: list[dict], path: Path
    ) -> list[Document]:
        """将 Q&A 字典列表转为 Document 列表。"""
        documents: list[Document] = []

        for item in items:
            question = item.get("question", "").strip()
            if not question:
                logger.warning("跳过缺少 question 字段的记录: %s", item)
                continue

            answer = item.get("answer", "").strip()
            if not answer:
                logger.warning("跳过缺少 answer 字段的记录")
                continue
            source = item.get("source", "").strip()
            category = item.get("category", "").strip()

            content = f"问题：{question}\n答案：{answer}"
            title = question[:30] + ("..." if len(question) > 30 else "")

            metadata: dict[str, str] = {
                "source": source,
                "title": title,
                "doc_type": "qa",
                "category": category,
                "file_path": str(path.resolve()),
            }

            doc = Document(content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def _load_directory(self, dir_path: Path) -> list[Document]:
        """递归加载目录下的 Q&A 文件。"""
        files = sorted(
            p
            for p in dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        results: list[Document] = []
        for f in files:
            results.extend(self.load(f))
        return results
