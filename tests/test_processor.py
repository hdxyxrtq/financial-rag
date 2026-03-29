"""文档处理器测试。"""

import pytest

from src.loaders.base_loader import Document
from src.processor.chunker import Chunk, TextChunker
from src.processor.cleaner import TextCleaner


# ============================================================
# TextCleaner 测试
# ============================================================


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_basic_clean(self):
        text = "  这是测试  \n\n  文本  "
        result = self.cleaner.clean(text)
        assert "  " not in result
        assert result == "这是测试\n\n文本"

    def test_normalize_whitespace(self):
        text = "多个   空格\t\t合并"
        result = self.cleaner.clean(text)
        assert "多个 空格 合并" == result

    def test_merge_broken_lines_chinese(self):
        text = "这是一段被换行打断的中文文本\n内容在第二行继续"
        result = self.cleaner.clean(text)
        # 中文无标点结尾应合并
        assert "打断的中文文本内容" in result

    def test_not_merge_with_punctuation(self):
        text = "这是第一句话。\n这是第二段开头。"
        result = self.cleaner.clean(text)
        assert "第一句话。\n这是第二段开头。" in result

    def test_merge_broken_english(self):
        text = "This is a broken line of English text\ncontinued on the next line."
        result = self.cleaner.clean(text)
        assert "English textcontinued" in result

    def test_preserve_financial_symbols(self):
        text = "GDP增长5.3%，CPI上涨2.1%，汇率变化±0.5%"
        result = self.cleaner.clean(text)
        assert "5.3%" in result
        assert "2.1%" in result
        assert "±" in result

    def test_preserve_currency_symbols(self):
        text = "金额为￥100万元，相当于$14万美元"
        result = self.cleaner.clean(text)
        assert "￥100" in result
        assert "$14" in result

    def test_remove_control_characters(self):
        text = "正常文本\x00\x01\x02\x03乱码文本"
        result = self.cleaner.clean(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "正常文本" in result

    def test_normalize_punctuation(self):
        text = "百分比100％"
        result = self.cleaner.clean(text)
        assert "100%" in result

    def test_not_merge_numbered_list(self):
        text = "第一条内容\n1. 第二条内容\n2. 第三条内容"
        result = self.cleaner.clean(text)
        assert "1. 第二条内容" in result

    def test_not_merge_title(self):
        text = "正文内容\n## 标题\n更多内容"
        result = self.cleaner.clean(text)
        assert "## 标题" in result

    def test_clean_batch(self):
        texts = ["  文本A  ", "  文本B  "]
        results = self.cleaner.clean_batch(texts)
        assert results == ["文本A", "文本B"]

    def test_preserve_paragraphs(self):
        text = "第一段\n\n第二段\n\n第三段"
        result = self.cleaner.clean(text)
        assert result.count("\n\n") == 2

    def test_empty_text(self):
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean("   \n\t  ") == ""

    def test_preserve_mixed_content(self):
        text = "2024年GDP增长5.3%，其中第一产业增长\n3.3%，第二产业增长6.0%。"
        result = self.cleaner.clean(text)
        assert "5.3%" in result
        assert "3.3%" in result


# ============================================================
# TextChunker 测试
# ============================================================


class TestTextChunker:
    def setup_method(self):
        self.chunker = TextChunker(chunk_size=128, chunk_overlap=20)

    def test_simple_chunk(self):
        text = "短文本。"
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 1
        assert "短文本" in chunks[0].content

    def test_empty_text(self):
        assert self.chunker.chunk("") == []
        assert self.chunker.chunk("   \n  ") == []

    def test_chunk_preserves_metadata(self):
        meta = {"source": "test.txt", "title": "测试"}
        chunks = self.chunker.chunk("测试内容", metadata=meta)
        assert all(c.metadata == meta for c in chunks)

    def test_chunk_within_size(self):
        text = "这是一段不太长的文本。"
        chunks = self.chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].token_count <= 128

    def test_chunk_multiple_paragraphs(self):
        paragraphs = [f"这是第{i}段内容，包含一些文字。" * 10 for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 2
        # 每个 chunk 应该在合理范围内
        for c in chunks:
            assert c.token_count > 0

    def test_chunk_ids_sequential(self):
        text = "第一段。" * 100 + "\n\n" + "第二段。" * 100
        chunks = self.chunker.chunk(text)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_chunk_positions(self):
        text = "短文本。"
        chunks = self.chunker.chunk(text)
        assert chunks[0].start_char >= 0
        assert chunks[0].end_char > chunks[0].start_char

    def test_chunk_documents(self):
        docs = [
            Document(content="文档一的内容。" * 20, metadata={"source": "doc1"}),
            Document(content="文档二的内容。" * 20, metadata={"source": "doc2"}),
        ]
        chunks = self.chunker.chunk_documents(docs)
        assert len(chunks) >= 2
        # 全局 chunk_id 应该唯一递增
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_force_split_long_paragraph(self):
        """超长无换行的段落应被强制切分。"""
        # 生成一个超长段落（远超 chunk_size）
        long_text = "这是一个超长的句子。" * 50  # 约 450 字符
        chunks = self.chunker.chunk(long_text)
        # 由于 chunk_size=128，应该被分成多个块
        assert len(chunks) >= 2

    def test_overlap_between_chunks(self):
        text = "第一段内容。" * 30 + "\n\n" + "第二段内容。" * 30
        chunks = self.chunker.chunk(text)
        if len(chunks) >= 2:
            overlap_size = self.chunker._chunk_overlap
            last_chunk_text = chunks[-2].content
            current_chunk_text = chunks[-1].content
            tail = last_chunk_text[-overlap_size * 3:] if len(last_chunk_text) > overlap_size * 3 else last_chunk_text
            assert any(tail[j:j + 4] in current_chunk_text for j in range(len(tail) - 3)), \
                "overlap region of previous chunk should appear in current chunk"

    def test_no_content_loss(self):
        """分块后拼接应覆盖所有原文内容。"""
        text = "段落一。" * 20 + "\n\n" + "段落二。" * 20 + "\n\n" + "段落三。" * 20
        chunks = self.chunker.chunk(text)
        # 验证关键内容出现在某些 chunk 中
        all_content = " ".join(c.content for c in chunks)
        assert "段落一" in all_content
        assert "段落二" in all_content
        assert "段落三" in all_content

    def test_chunk_with_config_from_yaml(self):
        """使用与 config.yaml 一致的参数。"""
        chunker = TextChunker(chunk_size=512, chunk_overlap=100, separator="\n\n")
        text = "测试内容。" * 5
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_chunk_metadata_includes_chunk_info(self):
        meta = {"source": "news.txt"}
        text = "内容。" * 100
        chunks = self.chunker.chunk(text, metadata=meta)
        for c in chunks:
            assert c.metadata["source"] == "news.txt"


# ============================================================
# TitleBasedChunker 测试
# ============================================================


class TestTitleBasedChunker:
    def setup_method(self):
        from src.processor.chunker import TitleBasedChunker
        self.chunker = TitleBasedChunker(chunk_size=128)

    def test_markdown_headings(self):
        text = "# 宏观经济分析\n这是宏观分析内容。\n\n## 股票市场\n股票市场表现良好。"
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("宏观经济分析" in c.content for c in chunks)
        assert any("股票市场" in c.content for c in chunks)

    def test_chinese_numbered_sections(self):
        text = "一、经济概况\n2024年GDP增长5.3%。\n\n二、货币政策\n央行维持稳健货币政策。"
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("经济概况" in c.content for c in chunks)
        assert any("货币政策" in c.content for c in chunks)

    def test_chapter_sections(self):
        text = "第一章 研究背景\n研究背景内容。\n\n第二章 数据分析\n数据分析结果。"
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 2

    def test_numbered_sections(self):
        text = "1. 市场综述\n市场综述内容。\n\n2. 行业分析\n行业分析内容。"
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 2

    def test_empty_text(self):
        assert self.chunker.chunk("") == []
        assert self.chunker.chunk("   \n  ") == []

    def test_metadata_preservation(self):
        meta = {"source": "report.pdf", "title": "研报"}
        text = "# 概述\n概述内容。"
        chunks = self.chunker.chunk(text, metadata=meta)
        for c in chunks:
            assert c.metadata["source"] == "report.pdf"
            assert "section_title" in c.metadata

    def test_section_title_in_metadata(self):
        text = "# 市场分析\n分析内容。"
        chunks = self.chunker.chunk(text)
        assert any("市场分析" in c.metadata.get("section_title", "") for c in chunks)

    def test_long_section_fallback(self):
        long_section = "一、详细分析\n" + "这是详细的金融分析内容。" * 50
        chunks = self.chunker.chunk(long_section)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.token_count <= 128 * 2  # 允许一定容差

    def test_mixed_content(self):
        text = "引言部分没有标题。\n\n# 正文\n正文内容。"
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("引言" in c.metadata.get("section_title", "") for c in chunks)

    def test_chunk_documents(self):
        from src.loaders.base_loader import Document
        docs = [
            Document(content="# 第一章\n内容一。" * 20, metadata={"source": "doc1"}),
            Document(content="# 第二章\n内容二。" * 20, metadata={"source": "doc2"}),
        ]
        chunks = self.chunker.chunk_documents(docs)
        assert len(chunks) >= 2
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_no_titles(self):
        text = "纯文本内容，没有任何标题。\n\n另一段纯文本。"
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 1
