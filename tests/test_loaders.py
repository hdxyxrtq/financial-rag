"""文档加载器测试。"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.loaders.base_loader import BaseLoader, Document
from src.loaders.pdf_loader import PDFLoader
from src.loaders.qa_loader import QALoader
from src.loaders.text_loader import TextLoader

# ============================================================
# BaseLoader 测试
# ============================================================


class TestBaseLoader:
    def test_document_creation(self):
        doc = Document(content="测试内容", metadata={"source": "test.txt"})
        assert doc.content == "测试内容"
        assert doc.metadata["source"] == "test.txt"
        assert doc.doc_id  # 应该自动生成

    def test_document_default_metadata(self):
        doc = Document(content="测试")
        assert doc.metadata == {}

    def test_document_str(self):
        doc = Document(content="内容", metadata={"title": "标题"})
        result = str(doc)
        assert "标题" in result

    def test_base_loader_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLoader()  # type: ignore[abstract]

    def test_base_loader_load_batch_default(self):
        class DummyLoader(BaseLoader):
            def load(self, source):
                return [Document(content=f"from {source}")]

        loader = DummyLoader()
        result = loader.load_batch(["a", "b"])
        assert len(result) == 2


# ============================================================
# TextLoader 测试
# ============================================================


class TestTextLoader:
    def setup_method(self):
        self.loader = TextLoader()

    def _write_temp(self, content: str, suffix: str = ".txt", encoding: str = "utf-8"):
        f = tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, mode="w", encoding=encoding
        )
        f.write(content)
        f.close()
        return Path(f.name)

    def test_load_txt_file(self):
        path = self._write_temp("这是一篇测试新闻\n\n第二段内容。")
        try:
            docs = self.loader.load(path)
            assert len(docs) == 1
            assert "测试新闻" in docs[0].content
            assert docs[0].metadata["doc_type"] == "text"
            assert docs[0].metadata["source"] == os.path.basename(path)
        finally:
            path.unlink()

    def test_load_md_file(self):
        path = self._write_temp("# 标题\n\n正文内容", suffix=".md")
        try:
            docs = self.loader.load(path)
            assert len(docs) == 1
            assert docs[0].metadata["title"] == "标题"
        finally:
            path.unlink()

    def test_extract_title_from_markdown(self):
        path = self._write_temp("# 2024年经济报告\n\n正文")
        try:
            docs = self.loader.load(path)
            assert docs[0].metadata["title"] == "2024年经济报告"
        finally:
            path.unlink()

    def test_extract_first_line_as_title(self):
        path = self._write_temp("财经新闻标题\n\n正文内容")
        try:
            docs = self.loader.load(path)
            assert docs[0].metadata["title"] == "财经新闻标题"
        finally:
            path.unlink()

    def test_empty_file(self):
        path = self._write_temp("   \n\n  ")
        try:
            docs = self.loader.load(path)
            assert len(docs) == 0
        finally:
            path.unlink()

    def test_nonexistent_file(self):
        docs = self.loader.load("/nonexistent/path.txt")
        assert len(docs) == 0

    def test_unsupported_format(self):
        path = self._write_temp("内容", suffix=".xlsx")
        try:
            docs = self.loader.load(path)
            assert len(docs) == 0
        finally:
            path.unlink()

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.txt").write_text("文件A", encoding="utf-8")
            (Path(tmpdir) / "b.md").write_text("文件B", encoding="utf-8")
            (Path(tmpdir) / "c.csv").write_text("ignored", encoding="utf-8")

            docs = self.loader.load(Path(tmpdir))
            assert len(docs) == 2
            sources = {d.metadata["source"] for d in docs}
            assert sources == {"a.txt", "b.md"}

    def test_load_batch_with_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.txt").write_text("内容A", encoding="utf-8")
            (Path(tmpdir) / "b.txt").write_text("内容B", encoding="utf-8")

            docs = self.loader.load_batch([Path(tmpdir)])
            assert len(docs) == 2

    def test_file_path_in_metadata(self):
        path = self._write_temp("测试")
        try:
            docs = self.loader.load(path)
            assert "file_path" in docs[0].metadata
            assert str(path.resolve()) in docs[0].metadata["file_path"]
        finally:
            path.unlink()


# ============================================================
# QALoader 测试
# ============================================================


class TestQALoader:
    def setup_method(self):
        self.loader = QALoader()

    def _write_json(self, data):
        f = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        )
        json.dump(data, f, ensure_ascii=False)
        f.close()
        return Path(f.name)

    def _write_csv(self, content: str):
        f = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w", encoding="utf-8-sig"
        )
        f.write(content)
        f.close()
        return Path(f.name)

    def test_load_json_list_format(self):
        data = [
            {"question": "什么是GDP？", "answer": "国内生产总值", "source": "测试", "category": "宏观"},
            {"question": "什么是CPI？", "answer": "居民消费价格指数", "source": "测试", "category": "宏观"},
        ]
        path = self._write_json(data)
        try:
            docs = self.loader.load(path)
            assert len(docs) == 2
            assert "问题：什么是GDP？" in docs[0].content
            assert "答案：国内生产总值" in docs[0].content
            assert docs[0].metadata["doc_type"] == "qa"
            assert docs[0].metadata["category"] == "宏观"
            assert docs[0].metadata["source"] == "测试"
        finally:
            path.unlink()

    def test_load_json_object_format(self):
        data = {"qa_list": [{"question": "问题1", "answer": "答案1"}]}
        path = self._write_json(data)
        try:
            docs = self.loader.load(path)
            assert len(docs) == 1
            assert "问题：问题1" in docs[0].content
        finally:
            path.unlink()

    def test_load_csv_format(self):
        content = "question,answer,source,category\n什么是GDP？,国内生产总值,测试,宏观\n"
        path = self._write_csv(content)
        try:
            docs = self.loader.load(path)
            assert len(docs) == 1
            assert docs[0].metadata["category"] == "宏观"
        finally:
            path.unlink()

    def test_skip_missing_question(self):
        data = [{"answer": "没有问题"}, {"question": "正常问题", "answer": "正常答案"}]
        path = self._write_json(data)
        try:
            docs = self.loader.load(path)
            assert len(docs) == 1
            assert "正常问题" in docs[0].content
        finally:
            path.unlink()

    def test_title_truncation(self):
        long_q = "这是一个非常非常非常非常非常非常非常非常长的测试问题用来验证标题截断功能是否正常工作"
        data = [{"question": long_q, "answer": "答案"}]
        path = self._write_json(data)
        try:
            docs = self.loader.load(path)
            assert docs[0].metadata["title"].endswith("...")
            assert len(docs[0].metadata["title"]) <= 33
        finally:
            path.unlink()

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "qa.json"
            json_path.write_text(
                json.dumps([{"question": "Q1", "answer": "A1"}]),
                encoding="utf-8",
            )
            docs = self.loader.load(Path(tmpdir))
            assert len(docs) == 1

    def test_empty_question_field(self):
        data = [{"question": "", "answer": "答案"}]
        path = self._write_json(data)
        try:
            docs = self.loader.load(path)
            assert len(docs) == 0
        finally:
            path.unlink()


# ============================================================
# PDFLoader 测试
# ============================================================


class TestPDFLoader:
    def setup_method(self):
        self.loader = PDFLoader()

    def test_nonexistent_file(self):
        docs = self.loader.load("/nonexistent/test.pdf")
        assert len(docs) == 0

    def test_unsupported_format(self):
        f = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
        f.write("not a pdf")
        f.close()
        try:
            docs = self.loader.load(Path(f.name))
            assert len(docs) == 0
        finally:
            Path(f.name).unlink()

    def test_pdf_with_pymupdf(self, tmp_path):
        """使用 PyMuPDF 创建测试 PDF 并验证加载。"""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF 未安装")

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Page 1: Financial Report 2024")
        page = doc.new_page()
        page.insert_text((72, 72), "Page 2: Market Analysis Q1")
        doc.save(str(pdf_path))
        doc.close()

        docs = self.loader.load(pdf_path)
        assert len(docs) == 1
        assert "Page 1" in docs[0].content
        assert "Page 2" in docs[0].content
        assert docs[0].metadata["doc_type"] == "pdf"
        assert docs[0].metadata["total_pages"] == "2"
