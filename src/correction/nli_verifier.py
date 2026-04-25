from __future__ import annotations

import logging
import re

from src.correction.types import ClaimVerdict

logger = logging.getLogger(__name__)

# Boilerplate phrases to filter out during claim decomposition
_BOILERPLATE_PATTERNS = [
    r"根据(现有|上述|以上)?(资料|数据|信息|分析|文档|报告)",
    r"(以上|上述|如下)?(分析|回答|内容)?(仅?供)?参考",
    r"(建议|提醒您|请注意)",
    r"(如需|如有)(进一步|更多|其他)?(疑问|问题|帮助|信息)",
    r"(希望|希望能)(对您|这些)(有所)?(帮助|用处)",
    r"(总的来说|总而言之|综上)",
    r"^(以上|上述)(就是|便是)?(全部)?(内容|信息)?",
    r"(数据|信息)?(来源|出处|源于)",
    r"(不构成|并非)(任何)?(投资|专业)?(建议|意见)",
]

_SENTENCE_SPLIT_RE = re.compile(r"[。；！？\n]")
_QUESTION_MARKS = re.compile(r"[？?]|(吗|呢|吧)[。！？]?$")
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS))


class NLIVerifier:
    """Layer 3: Claim decomposition + lightweight NLI verification."""

    def __init__(self) -> None:
        pass

    def decompose_claims(self, answer: str) -> list[str]:
        """Decompose answer into atomic claims using regex + heuristic rules.

        Steps:
            1. Split by sentence delimiters (。；！？\\n)
            2. Filter out short sentences (< 10 chars)
            3. Filter out interrogative sentences
            4. Filter out boilerplate phrases
        """
        raw_parts = _SENTENCE_SPLIT_RE.split(answer)
        claims: list[str] = []
        for part in raw_parts:
            stripped = part.strip()
            if not stripped:
                continue
            # Filter short sentences
            if len(stripped) < 10:
                continue
            # Filter interrogative sentences
            if _QUESTION_MARKS.search(stripped):
                continue
            # Filter boilerplate
            if _BOILERPLATE_RE.search(stripped):
                continue
            claims.append(stripped)
        return claims

    def verify(self, claims: list[str], context: str) -> list[ClaimVerdict]:
        """Verify claims against context using character-level n-gram overlap.

        Thresholds:
            overlap > 0.5  → SUPPORTED
            overlap < 0.3  → UNSUPPORTED
            0.3 <= overlap <= 0.5 → UNSUPPORTED (needs further verification)
        """
        if not claims:
            return []

        context_ngrams = self._char_ngrams(context, n=2)
        results: list[ClaimVerdict] = []
        for claim in claims:
            claim_ngrams = self._char_ngrams(claim, n=2)
            if not claim_ngrams:
                results.append(ClaimVerdict(
                    claim=claim,
                    supported=True,
                    evidence="",
                    source="nli_lightweight",
                ))
                continue

            overlap_count = sum(1 for ng in claim_ngrams if ng in context_ngrams)
            overlap_ratio = overlap_count / len(claim_ngrams)

            if overlap_ratio > 0.5:
                results.append(ClaimVerdict(
                    claim=claim,
                    supported=True,
                    evidence=f"n-gram overlap: {overlap_ratio:.2f}",
                    source="nli_lightweight",
                ))
            else:
                results.append(ClaimVerdict(
                    claim=claim,
                    supported=False,
                    evidence=f"n-gram overlap: {overlap_ratio:.2f}",
                    source="nli_lightweight",
                ))

        supported_count = sum(1 for r in results if r.supported)
        logger.info(
            "NLIVerifier: %d/%d claims supported",
            supported_count, len(results),
        )
        return results

    @staticmethod
    def _char_ngrams(text: str, n: int = 2) -> set[str]:
        """Extract character-level n-grams from text, ignoring whitespace."""
        cleaned = re.sub(r"\s+", "", text)
        if len(cleaned) < n:
            return set()
        return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}
