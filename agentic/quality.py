"""
agentic/quality.py

Your VLMResult has no single numeric quality score — it has
overall_quality (str), defects (dict[str, DefectDetail] with
present/confidence/severity as strings), and regeneration_recommended
(bool). This module turns that into a comparable score so the
Validation Agent can decide "did it actually get better".
"""

from schemas import VLMResult

_ORDINAL = {
    "poor": 0, "bad": 0, "low": 0,
    "fair": 1, "medium": 1, "moderate": 1, "unknown": 1,
    "good": 2, "high": 2,
    "excellent": 3, "very high": 3,
}


def _band(value: str) -> int:
    return _ORDINAL.get((value or "").strip().lower(), 1)


def quality_score(vlm: VLMResult) -> float:
    """Higher is better. Penalizes each present defect by its severity band."""
    score = _band(vlm.overall_quality) * 10.0
    for detail in vlm.defects.values():
        if detail.present:
            score -= 2.0 + _band(detail.severity)
    if vlm.regeneration_recommended:
        score -= 5.0
    return score


def improved(before: VLMResult, after: VLMResult) -> bool:
    """True if the after-report is meaningfully better than the before-report."""
    if before.regeneration_recommended and not after.regeneration_recommended:
        return True
    return quality_score(after) > quality_score(before)


def describe(vlm: VLMResult) -> str:
    """One-line human-readable summary for the PDF report and logs."""
    present = [f"{k} ({v.severity})" for k, v in vlm.defects.items() if v.present]
    defect_str = ", ".join(present) if present else "none detected"
    return (f"Overall quality: {vlm.overall_quality} | Defects: {defect_str} | "
            f"Regeneration recommended: {vlm.regeneration_recommended} | {vlm.summary}")
