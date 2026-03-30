from __future__ import annotations

import re


class TextCleaner:
    """金融领域文本清洗器。"""

    _PUNCTUATION_MAP = {
        "％": "%",
        "＄": "$",
        "￥": "￥",  # 保留（与 ¥ 不同）
        "（": "(",
        "）": ")",
        "，": ",",
        "。": "。",  # 保留中文句号
        "：": "：",  # 保留中文冒号
        "；": "；",  # 保留中文分号
        "？": "？",  # 保留中文问号
        "！": "！",  # 保留中文叹号
        '"': '"',
        "'": "'",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
    }

    def clean(self, text: str) -> str:
        """执行完整的文本清洗流程。"""
        text = self._remove_special_chars(text)
        text = self._normalize_punctuation(text)
        text = self._merge_broken_lines(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def clean_batch(self, texts: list[str]) -> list[str]:
        """批量清洗文本。"""
        return [self.clean(t) for t in texts]

    def _normalize_whitespace(self, text: str) -> str:
        """合并多余空白，但保留段落间换行。"""
        # 将连续 3 个以上换行压缩为 2 个
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 将行内多个空格合并为一个
        text = re.sub(r"[^\S\n]+", " ", text)
        # 去除行首尾空白
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines)

    def _merge_broken_lines(self, text: str) -> str:
        """合并被 PDF 换行打断的句子。"""
        lines = text.split("\n")
        merged: list[str] = []
        i = 0

        while i < len(lines):
            current = lines[i].rstrip()

            if not current:
                merged.append("")
                i += 1
                continue

            # 向后看是否能合并
            while i + 1 < len(lines):
                next_line = lines[i + 1].strip()

                if not next_line:
                    # 空行 = 段落边界，停止合并
                    break

                if self._should_merge(current, next_line):
                    current = current + next_line
                    i += 1
                else:
                    break

            merged.append(current)
            i += 1

        return "\n".join(merged)

    def _should_merge(self, current: str, next_line: str) -> bool:
        """判断当前行是否应与下一行合并。"""
        if not current or not next_line:
            return False

        last_char = current[-1]
        first_char = next_line[0]

        # 当前行以标点结尾 → 不合并
        if last_char in "。！？；：,.!?;:\u3001\u3002":
            return False

        # 下一行是标题（以 # 开头或全大写字母）→ 不合并
        if next_line.startswith("#") or (next_line.isupper() and len(next_line) > 3):
            return False

        # 下一行以数字+标点开头（如 "1." "一、"）→ 不合并
        if re.match(r"^[\d一二三四五六七八九十]+[.、）)]", next_line):
            return False

        # 英文：当前行以字母/数字结尾，下一行以小写字母开头 → 合并
        if current[-1].isascii() and current[-1].isalnum() and first_char.islower():
            return True

        # 中文：当前行无结束标点，下一行也是中文内容 → 合并
        if (
            "\u4e00" <= last_char <= "\u9fff"
            and "\u4e00" <= first_char <= "\u9fff"
        ):
            return True

        # 英文行中间断开（当前以连字符结尾）
        return bool(current.endswith("-") and next_line[0:1].isalpha())

    def _remove_special_chars(self, text: str) -> str:
        """去除乱码和控制字符，保留金融符号和常用标点。"""
        result: list[str] = []

        for char in text:
            code = ord(char)

            # 控制字符（保留换行 \n、制表符 \t）
            if code < 0x20 and char not in "\n\t":
                continue

            # 删除字符区域
            if 0x7F <= code <= 0x9F:
                continue

            # 零宽字符
            if code in (0x200B, 0x200C, 0x200D, 0xFEFF):
                continue

            result.append(char)

        return "".join(result)

    def _normalize_punctuation(self, text: str) -> str:
        """全角半角标点规范化。"""
        result = text
        for full, half in self._PUNCTUATION_MAP.items():
            result = result.replace(full, half)
        return result
