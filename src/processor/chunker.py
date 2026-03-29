from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.loaders.base_loader import Document

if TYPE_CHECKING:
    import tiktoken

# ---------------------------------------------------------------------------
# 模块级工具函数（供 TextChunker 和 TitleBasedChunker 共用）
# ---------------------------------------------------------------------------


def _init_tokenizer() -> tuple[tiktoken.Encoding | None, bool]:
    """初始化 tiktoken tokenizer，返回 (encoder, available)。"""
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return encoder, True
    except ImportError:
        return None, False


_token_encoder: tiktoken.Encoding | None = None
_tiktoken_available: bool = False
_token_encoder, _tiktoken_available = _init_tokenizer()


def count_tokens(text: str) -> int:
    """计算文本的 token 数量（模块级，供所有分块器共用）。

    优先使用 tiktoken cl100k_base，不可用时按字符估算。
    """
    if _token_encoder is not None:
        return len(_token_encoder.encode(text))
    # fallback: 中文约 1 字符 = 1.5 token，英文约 1 字符 = 0.25 token
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - chinese_count
    return int(chinese_count * 1.5 + other_count * 0.25)


# 默认标题匹配模式
_DEFAULT_TITLE_PATTERNS: list[str] = [
    r"^#{1,3}\s+.+",                       # Markdown 标题
    r"^[一二三四五六七八九十]+、.+",          # 中文 一、二、...
    r"^第[一二三四五六七八九十百]+[章节篇].+",  # 第X章/节/篇
    r"^\d+[.、]\s*.+",                      # 1. / 1、
]


@dataclass
class Chunk:
    """文本分块结果。"""

    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    chunk_id: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0


class TextChunker:
    """金融领域智能文本分块器。

    按段落优先切分，支持 token 级块大小控制和重叠。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        separator: str = "\n\n",
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separator = separator

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数量。"""
        return count_tokens(text)

    def chunk(
        self, text: str, metadata: dict[str, str] | None = None
    ) -> list[Chunk]:
        """将文本按段落优先策略分块。

        Args:
            text: 待分块的文本。
            metadata: 原始文档的元数据，会复制到每个 chunk。

        Returns:
            分块结果列表。
        """
        if not text or not text.strip():
            return []

        meta = dict(metadata) if metadata else {}

        # 先按分隔符分割为段落
        paragraphs = text.split(self._separator)

        # 过滤空段落但保留其位置信息
        non_empty = [(i, p.strip()) for i, p in enumerate(paragraphs) if p.strip()]

        if not non_empty:
            return []

        chunks: list[Chunk] = []
        current_text = ""
        current_start = 0
        chunk_id = 0
        # 预先计算段落的累计字符偏移，避免 O(n^2) 的逐段落求和
        _para_offsets: list[int] = [0]
        for i, p in enumerate(paragraphs):
            _para_offsets.append(_para_offsets[-1] + len(p) + len(self._separator))

        for para_idx, (orig_idx, paragraph) in enumerate(non_empty):
            # 计算当前段落在整个原文中的字符位置（O(1)）
            para_start = _para_offsets[orig_idx]

            if not current_text:
                # 第一个段落
                current_text = paragraph
                current_start = para_start
            else:
                # 检查加入后是否超限
                candidate = current_text + self._separator + paragraph
                if self._count_tokens(candidate) <= self._chunk_size:
                    current_text = candidate
                else:
                    # 切出当前块
                    if current_text.strip():
                        chunks.append(self._make_chunk(
                            chunk_id, current_text, meta,
                            current_start, current_start + len(current_text),
                        ))
                        chunk_id += 1

                    # 重叠：从当前块末尾取 overlap tokens
                    overlap_text = self._get_overlap_text(current_text)
                    current_text = (
                        overlap_text + self._separator + paragraph
                        if overlap_text
                        else paragraph
                    )
                    current_start = para_start - len(overlap_text) if overlap_text else para_start

        # 处理最后一个块
        if current_text.strip():
            # 超长段落强制切分
            if self._count_tokens(current_text) > self._chunk_size:
                sub_chunks = self._force_split(current_text, meta, current_start, chunk_id)
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._make_chunk(
                    chunk_id, current_text, meta,
                    current_start, current_start + len(current_text),
                ))

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """批量分块多个文档。

        Args:
            documents: Document 列表。

        Returns:
            所有文档的分块结果。
        """
        all_chunks: list[Chunk] = []
        global_chunk_id = 0

        for doc in documents:
            chunks = self.chunk(doc.content, doc.metadata)
            # 重新编号全局 chunk_id
            for c in chunks:
                c.chunk_id = global_chunk_id
                global_chunk_id += 1
            all_chunks.extend(chunks)

        return all_chunks

    def _make_chunk(
        self,
        chunk_id: int,
        content: str,
        metadata: dict[str, str],
        start_char: int,
        end_char: int,
    ) -> Chunk:
        """创建一个 Chunk 实例。"""
        return Chunk(
            content=content.strip(),
            metadata=dict(metadata),
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char,
            token_count=self._count_tokens(content),
        )

    def _get_overlap_text(self, text: str) -> str:
        """从文本末尾提取 overlap tokens 对应的文本。"""
        if self._chunk_overlap <= 0:
            return ""

        lines = text.split("\n")
        overlap_lines: list[str] = []
        overlap_tokens = 0

        for line in reversed(lines):
            line_tokens = self._count_tokens(line)
            if overlap_tokens + line_tokens > self._chunk_overlap:
                break
            overlap_lines.insert(0, line)
            overlap_tokens += line_tokens

        result = "\n".join(overlap_lines).strip()
        return result

    def _force_split(
        self,
        text: str,
        metadata: dict[str, str],
        base_start: int,
        base_chunk_id: int,
    ) -> list[Chunk]:
        """对超长文本强制切分（按句子边界）。"""
        chunks: list[Chunk] = []

        # 按中文句号、换行、英文句号切分
        sentences = []
        current = ""

        for i, char in enumerate(text):
            current += char
            # Split on Chinese/Japanese/English sentence endings
            if char in "。\n!?":
                if current.strip():
                    sentences.append(current.strip())
                    current = ""
            elif char == ".":
                # Only split on '.' when it's the end of a sentence, i.e. next char is space/newline/end
                if i + 1 >= len(text) or text[i + 1].isspace():
                    if current.strip():
                        sentences.append(current.strip())
                        current = ""

        if current.strip():
            sentences.append(current.strip())

        # 合并为块
        current_text = ""
        char_offset = base_start
        chunk_id = base_chunk_id

        for sentence in sentences:
            candidate = current_text + sentence if not current_text else current_text + "\n" + sentence

            if self._count_tokens(candidate) > self._chunk_size and current_text:
                chunks.append(self._make_chunk(
                    chunk_id, current_text, metadata,
                    char_offset, char_offset + len(current_text),
                ))
                chunk_id += 1
                char_offset += len(current_text) + 1
                current_text = sentence
            else:
                current_text = candidate

        if current_text.strip():
            chunks.append(self._make_chunk(
                chunk_id, current_text, metadata,
                char_offset, char_offset + len(current_text),
            ))

        return chunks


# ---------------------------------------------------------------------------
# 按标题分块策略
# ---------------------------------------------------------------------------


class TitleBasedChunker:
    """按标题/章节分块的文本分块器。

    适用于金融研报等结构化文档，按照 Markdown 标题、中文编号、
    章节号等标题模式将文档按章节切分。超长章节自动回退到句子级切分。
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        title_patterns: list[str] | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._patterns: list[re.Pattern[str]] = [
            re.compile(p, re.MULTILINE)
            for p in (title_patterns or _DEFAULT_TITLE_PATTERNS)
        ]

    def _is_title_line(self, line: str) -> bool:
        """判断一行是否匹配任何标题模式。"""
        stripped = line.strip()
        if not stripped:
            return False
        return any(p.match(stripped) for p in self._patterns)

    def _split_by_titles(self, text: str) -> list[tuple[str, str]]:
        """将文本按标题分割为 (标题, 正文) 列表。

        标题前没有匹配标题的内容归入第一个块（以「引言」为虚拟标题）。
        """
        lines = text.split("\n")
        sections: list[tuple[str, str]] = []
        current_title = "引言"
        current_lines: list[str] = []

        for line in lines:
            if self._is_title_line(line):
                # 保存上一个 section
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        # 最后一个 section
        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        return sections

    def _split_long_section(
        self,
        title: str,
        content: str,
        metadata: dict[str, str],
        base_start: int,
        chunk_id: int,
    ) -> list[Chunk]:
        """对超长章节按句子边界切分。"""
        sentences: list[str] = []
        current = ""
        for char in content:
            current += char
            if char in "。\n！？!?;；":
                if current.strip():
                    sentences.append(current.strip())
                    current = ""
        if current.strip():
            sentences.append(current.strip())

        chunks: list[Chunk] = []
        section_text = ""
        char_offset = base_start

        for sentence in sentences:
            candidate = section_text + ("\n" if section_text else "") + sentence
            if count_tokens(candidate) > self._chunk_size and section_text:
                chunk_meta = dict(metadata)
                chunk_meta["section_title"] = title
                chunk_content = section_text.strip()
                # First sub-chunk should include the section title for context
                if not chunks:
                    chunk_content = title + "\n" + chunk_content
                chunks.append(Chunk(
                    content=chunk_content,
                    metadata=chunk_meta,
                    chunk_id=chunk_id,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_content),
                    token_count=count_tokens(chunk_content),
                ))
                chunk_id += 1
                char_offset += len(chunk_content) + 1
                section_text = sentence
            else:
                section_text = candidate

        if section_text.strip():
            chunk_meta = dict(metadata)
            chunk_meta["section_title"] = title
            chunks.append(Chunk(
                content=section_text.strip(),
                metadata=chunk_meta,
                chunk_id=chunk_id,
                start_char=char_offset,
                end_char=char_offset + len(section_text),
                token_count=count_tokens(section_text),
            ))

        return chunks

    def chunk(
        self, text: str, metadata: dict[str, str] | None = None
    ) -> list[Chunk]:
        """按标题/章节将文本分块。

        Args:
            text: 待分块的文本。
            metadata: 原始文档的元数据，会复制到每个 chunk。

        Returns:
            分块结果列表，每个 chunk 包含 section_title 标注。
        """
        if not text or not text.strip():
            return []

        meta = dict(metadata) if metadata else {}
        sections = self._split_by_titles(text)

        if not sections:
            return []

        chunks: list[Chunk] = []
        chunk_id = 0
        char_offset = 0

        for title, content in sections:
            if not content:
                char_offset += len(title) + 1
                continue

            full_text = title + "\n" + content
            token_count = count_tokens(full_text)

            chunk_meta = dict(meta)
            chunk_meta["section_title"] = title

            if token_count <= self._chunk_size:
                chunks.append(Chunk(
                    content=full_text.strip(),
                    metadata=chunk_meta,
                    chunk_id=chunk_id,
                    start_char=char_offset,
                    end_char=char_offset + len(full_text),
                    token_count=token_count,
                ))
                chunk_id += 1
                char_offset += len(full_text) + 1
            else:
                sub_chunks = self._split_long_section(
                    title, content, meta, char_offset, chunk_id,
                )
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
                char_offset += len(full_text) + 1

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """批量分块多个文档。"""
        all_chunks: list[Chunk] = []
        global_chunk_id = 0

        for doc in documents:
            chunks = self.chunk(doc.content, doc.metadata)
            for c in chunks:
                c.chunk_id = global_chunk_id
                global_chunk_id += 1
            all_chunks.extend(chunks)

        return all_chunks
