from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


@dataclass
class Document:
    """文档数据结构，包含正文和元数据。"""

    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: uuid4().hex)

    def __str__(self) -> str:
        title = self.metadata.get("title", "Untitled")
        length = len(self.content)
        return f"Document(title={title!r}, length={length})"


class BaseLoader(ABC):
    """文档加载器抽象基类，定义统一的加载接口。"""

    @abstractmethod
    def load(self, source: str | Path) -> list[Document]:
        """加载单个来源的文档。

        Args:
            source: 文件路径或目录路径。

        Returns:
            加载到的文档列表。
        """
        ...

    def load_batch(self, sources: list[str | Path]) -> list[Document]:
        """批量加载多个来源的文档。直接委托给单源加载以保持行为一致性。"""
        results: list[Document] = []
        for src in sources:
            results.extend(self.load(src))
        return results
