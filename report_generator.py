import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import REPORT_FOLDER


def _compute_issue_counts(raw_frame_evaluations: List[Dict[str, Any]]) -> Dict[str, int]:
    issue_counts = {"Blur": 0, "Noise": 0, "Compression": 0, "Lighting": 0}
    for evaluation in raw_frame_evaluations:
        defects = evaluation.get("defects", {})
        for defect_name, defect_info in defects.items():
            if defect_info.get("present"):
                issue_counts[defect_name] += 1
    return issue_counts


def _detect_anomalies(raw_frame_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    issue_counts = _compute_issue_counts(raw_frame_evaluations)
    frame_count = len(raw_frame_evaluations)
    if frame_count == 0:
        return {
            "anomaly_detected": False,
            "anomaly_reasons": [],
            "crime_risk": "unknown",
        }

    repeated_issues = [name for name, count in issue_counts.items() if count >= max(2, frame_count * 0.3)]
    anomaly_reasons = []

    if repeated_issues:
        anomaly_reasons.append(
            f"Consistent issues across sampled frames: {', '.join(repeated_issues)}"
        )

    suspicious_frames = 0
    for evaluation in raw_frame_evaluations:
        defects = evaluation.get("defects", {})
        if (
            defects.get("Lighting", {}).get("present")
            and defects.get("Noise", {}).get("present")
            and defects.get("Compression", {}).get("present")
        ):
            suspicious_frames += 1

    if suspicious_frames > 0:
        anomaly_reasons.append(
            f"{suspicious_frames} sampled frame(s) show combined lighting, noise, and compression issues, which may obscure suspicious activity."
        )

    crime_risk = "low"
    if "Noise" in repeated_issues and "Lighting" in repeated_issues:
        crime_risk = "medium"
    if "Noise" in repeated_issues and "Lighting" in repeated_issues and "Blur" in repeated_issues:
        crime_risk = "high"
    if suspicious_frames > max(1, frame_count * 0.15):
        crime_risk = "high"

    return {
        "anomaly_detected": bool(anomaly_reasons),
        "anomaly_reasons": anomaly_reasons,
        "crime_risk": crime_risk,
    }


def _serialize_selected_frames(
    selected_state: Any,
    raw_frame_evaluations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    frame_map = {item.get("frame_number"): item for item in raw_frame_evaluations}
    serialized = []
    for frame_number in selected_state.get("selected_frame_numbers", []):
        evaluation = frame_map.get(frame_number, {})
        serialized.append(
            {
                "frame_number": frame_number,
                "summary": evaluation.get("summary", ""),
                "overall_quality": evaluation.get("overall_quality", ""),
                "defects": evaluation.get("defects", {}),
                "anomaly_flags": {
                    "high_severity": any(
                        defect.get("severity") == "high"
                        for defect in evaluation.get("defects", {}).values()
                    ),
                },
            }
        )
    return serialized


def generate_analysis_report(
    video_name: str,
    metadata: Dict[str, Any],
    payload: Dict[str, Any],
    selected_state: Any,
    validation_result: Dict[str, Any],
    enhanced_results: List[Dict[str, Any]],
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(REPORT_FOLDER) / f"report_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    issue_counts = _compute_issue_counts(payload.get("raw_frame_evaluations", []))
    anomaly_info = _detect_anomalies(payload.get("raw_frame_evaluations", []))

    report_data = {
        "report_metadata": {
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "sampled_frames": payload.get("total_sampled_frames"),
            "processing_time_seconds": payload.get("processing_time_seconds"),
            "recommended_gan_tool": selected_state.get("recommended_gan"),
            "enhance_video": validation_result.get("enhance_video"),
            "validation_reason": validation_result.get("reason"),
        },
        "video_metadata": metadata,
        "aggregate_assessment": {
            "aggregated_summary": payload.get("aggregated_summary", ""),
            "issue_counts": issue_counts,
            "raw_frame_count": len(payload.get("raw_frame_evaluations", [])),
        },
        "anomaly_detection": anomaly_info,
        "selected_frames": _serialize_selected_frames(selected_state, payload.get("raw_frame_evaluations", [])),
        "enhancement_results": [
            {
                "frame_number": item.get("frame_number"),
                "tool": item.get("tool"),
                "output_path": item.get("output_path"),
            }
            for item in enhanced_results
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return str(report_path)
