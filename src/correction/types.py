from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalQuality:
    """Layer 0: Retrieval quality assessment result."""
    level: str  # "GOOD" / "MARGINAL" / "WEAK"
    top_score: float
    avg_score: float
    num_sources: int


@dataclass
class ClaimVerdict:
    """Layer 3/4: Individual claim verification result."""
    claim: str
    supported: bool
    evidence: str
    source: str


@dataclass
class CorrectionResult:
    """Aggregated correction result across all layers."""
    passed: bool
    flagged_claims: list[str] = field(default_factory=list)
    layer_results: dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0
