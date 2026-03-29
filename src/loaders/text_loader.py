from __future__ import annotations

import logging
import re
from pathlib import Path

from src.loaders.base_loader import BaseLoader, Document

logger = logging.getLogger(__name__)

# 编码检测顺序
_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030", "latin-1"]


class TextLoader(BaseLoader):
    """TXT/MD 文本文件加载器。

    支持批量加载目录下所有 .txt 和 .md 文件，
    自动检测编码并提取标题。
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def load(self, source: str | Path) -> list[Document]:
        """加载单个文本文件。

        Args:
            source: 文本文件路径。

        Returns:
            包含一个 Document 的列表。
        """
        path = Path(source)

        if not path.exists():
            logger.error("文件不存在: %s", path)
            return []

        if path.is_dir():
            return self._load_directory(path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.warning("不支持的文件格式: %s", path.suffix)
            return []

        content = self._read_file(path)
        if not content or not content.strip():
            logger.warning("文件为空或仅包含空白: %s", path)
            return []

        title = self._extract_title(content)
        doc = Document(
            content=content,
            metadata={
                "source": path.name,
                "title": title,
                "doc_type": "text",
                "file_path": str(path.resolve()),
            },
        )
        return [doc]

    # load_batch moved to BaseLoader for DRY behavior

    def _read_file(self, path: Path) -> str:
        """尝试多种编码读取文件。"""
        for encoding in _ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue

        logger.error("无法解码文件（已尝试所有编码）: %s", path)
        return ""

    @staticmethod
    def _extract_title(content: str) -> str:
        """从文本内容提取标题（第一行非空文本，去除 Markdown 符号）。"""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped:
                # 去除 Markdown 标题符号
                return re.sub(r"^#+\s*", "", stripped).strip()
        return "Untitled"

    def _load_directory(self, dir_path: Path) -> list[Document]:
        """递归加载目录下的文本文件。"""
        files = sorted(
            p
            for p in dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        results: list[Document] = []
        for f in files:
            results.extend(self.load(f))
        return results
