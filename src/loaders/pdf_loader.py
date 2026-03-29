from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from src.loaders.base_loader import BaseLoader, Document

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """PDF 文档加载器。

    使用 PyMuPDF 提取文本，保留页面结构信息和基础表格。
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, extract_tables: bool = True) -> None:
        self._extract_tables = extract_tables

    def load(self, source: str | Path) -> list[Document]:
        """加载单个 PDF 文件。

        Args:
            source: PDF 文件路径。

        Returns:
            包含一个 Document 的列表（所有页面合并）。
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

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            logger.error("无法打开 PDF 文件 %s: %s", path, e)
            return []

        try:
            total_pages = len(doc)
            page_texts: list[str] = []

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text").strip()

                table_text = self._extract_tables_from_page(page)
                if table_text:
                    text = text + "\n\n" + table_text if text else table_text

                page_texts.append(f"--- 第 {page_num + 1} 页 ---\n{text}")

            full_text = "\n\n".join(page_texts)

            if not full_text.strip():
                logger.warning("PDF 文件内容为空: %s", path)
                return []

            document = Document(
                content=full_text,
                metadata={
                    "source": path.name,
                    "title": path.stem,
                    "doc_type": "pdf",
                    "file_path": str(path.resolve()),
                    "total_pages": str(total_pages),
                },
            )
            return [document]

        finally:
            doc.close()

    def load_batch(self, sources: list[str | Path]) -> list[Document]:
        """批量加载，支持目录自动扫描。

        Args:
            sources: 文件或目录路径列表。

        Returns:
            所有加载到的文档列表。
        """
        results: list[Document] = []
        seen: set[str] = set()

        for src in sources:
            path = Path(src)

            if path.is_dir():
                files = sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
                )
                for f in files:
                    key = str(f.resolve())
                    if key not in seen:
                        seen.add(key)
                        results.extend(self.load(f))
            elif path.is_file():
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    results.extend(self.load(path))

        return results

    def _extract_tables_from_page(self, page: fitz.Page) -> str:
        try:
            tables = page.find_tables()
            if not tables:
                return ""
            table_texts = []
            for table in tables:
                rows = table.extract()
                if not rows:
                    continue
                for row in rows:
                    table_texts.append(" | ".join(str(cell) if cell else "" for cell in row))
                table_texts.append("")
            return "\n".join(table_texts)
        except AttributeError:
            return ""
        except Exception as e:
            logger.debug("表格提取失败: %s", e)
            return ""

    def _load_directory(self, dir_path: Path) -> list[Document]:
        """递归加载目录下的 PDF 文件。"""
        files = sorted(
            p
            for p in dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        results: list[Document] = []
        for f in files:
            results.extend(self.load(f))
        return results
