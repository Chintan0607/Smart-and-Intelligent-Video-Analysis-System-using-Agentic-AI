from typing import Any, Dict, List


class ValidationAgent:
    def __init__(self, quality_threshold: float = 0.5, max_selected_frames: int = 6):
        self.quality_threshold = quality_threshold
        self.max_selected_frames = max_selected_frames

    def decide(self, raw_frame_evaluations: List[Dict[str, Any]], selected_frame_numbers: List[int]) -> Dict[str, Any]:
        if not raw_frame_evaluations:
            return {
                "enhance_video": False,
                "reason": "No frame evaluations available.",
            }

        reviewed_frames = [
            item for item in raw_frame_evaluations if item.get("frame_number") in selected_frame_numbers
        ]

        if not reviewed_frames:
            return {
                "enhance_video": False,
                "reason": "No selected frames matched the evaluations.",
            }

        issue_counts = {
            "Blur": 0,
            "Noise": 0,
            "Compression": 0,
            "Lighting": 0,
        }
        severity_score = 0.0
        total_frames = len(reviewed_frames)

        for frame in reviewed_frames:
            defects = frame.get("defects", {})
            for defect_name, defect_data in defects.items():
                if defect_data.get("present"):
                    issue_counts[defect_name] += 1
                    if defect_data.get("severity") == "high":
                        severity_score += 1.0
                    elif defect_data.get("severity") == "medium":
                        severity_score += 0.5
                    else:
                        severity_score += 0.2

        normalized_score = severity_score / max(1, total_frames)
        issue_summary = [
            name for name, count in issue_counts.items() if count > 0
        ]

        if normalized_score >= self.quality_threshold and total_frames <= self.max_selected_frames:
            return {
                "enhance_video": True,
                "reason": (
                    f"Selected frames show consistent issues {issue_summary} "
                    f"and normalized severity {normalized_score:.2f} >= threshold {self.quality_threshold}."
                ),
                "normalized_score": normalized_score,
                "issue_summary": issue_summary,
            }

        return {
            "enhance_video": False,
            "reason": (
                f"Selected frames do not support enhancement: "
                f"normalized severity {normalized_score:.2f} < threshold {self.quality_threshold} "
                f"or too many selected frames ({total_frames} > {self.max_selected_frames})."
            ),
            "normalized_score": normalized_score,
            "issue_summary": issue_summary,
        }
