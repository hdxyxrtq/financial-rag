from __future__ import annotations

import re
from pathlib import Path

_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "吗", "什么", "怎么", "如何", "哪", "哪些", "为什么",
    "请", "问", "告诉", "帮", "可以", "能", "那个", "这个",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "will", "would", "could", "should", "can",
    "what", "how", "why", "who", "which", "where", "when", "please",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _extract_keywords(query: str, top_n: int = 5) -> list:
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", query)
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    return keywords[:top_n]


def _highlight_keywords(text: str, keywords: list) -> str:
    if not keywords:
        return text
    result = text
    for kw in keywords:
        escaped = re.escape(kw)
        result = re.sub(
            f"({escaped})",
            r"<mark style='background-color: #fef08a; padding: 0 2px; border-radius: 2px;'>\1</mark>",
            result,
            flags=re.IGNORECASE,
        )
    return result


def _get_file_extension(name: str) -> str:
    return Path(name).suffix.lower()


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _get_doc_type(ext: str) -> str:
    type_map = {
        ".pdf": "PDF 研报",
        ".txt": "文本",
        ".md": "Markdown",
        ".json": "Q&A (JSON)",
        ".csv": "Q&A (CSV)",
    }
    return type_map.get(ext, "未知")
