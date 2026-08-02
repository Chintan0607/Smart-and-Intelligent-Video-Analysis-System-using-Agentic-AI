"""
agentic/report_generator.py

Builds the downloadable PDF from a VideoReport (your real schemas.py
model, extended with retries/history/vlm_before — see the schemas.py
addition note in the chat). Pure reportlab, runs on Node A.
"""

import logging
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)
from PIL import Image as PILImage

from schemas import VideoReport, FrameReportEntry
from .quality import describe

logger = logging.getLogger("agentic.report_generator")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="SmallGrey", fontSize=8, textColor=colors.grey))
styles.add(ParagraphStyle(name="FrameHeading", fontSize=13, spaceAfter=6, spaceBefore=10))


def _scaled_image(path: str, max_w: float, max_h: float) -> RLImage:
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = min(max_w / w, max_h / h)
    return RLImage(path, width=w * ratio, height=h * ratio)


def _tool_log(history) -> str:
    if not history:
        return "No enhancement attempted."
    return " | ".join(
        f"Iteration {i}: tool='{rec['tool']}', improved={rec['improved']} "
        f"(before={rec['quality_before']:.1f}, after={(rec['quality_after'] or 0):.1f})"
        for i, rec in enumerate(history, start=1)
    )


def _frame_section(entry: FrameReportEntry, story):
    story.append(Paragraph(f"Frame #{entry.frame.frame_number} "
                            f"(t={entry.frame.metadata.timestamp_sec:.2f}s)", styles["FrameHeading"]))

    img_w, img_h = 2.6 * inch, 2.0 * inch
    original_path = entry.frame.filepath
    final_path = entry.enhancement.enhanced_filepath if entry.enhancement and entry.enhancement.applied else original_path
    try:
        before_img = _scaled_image(original_path, img_w, img_h)
        after_img = _scaled_image(final_path, img_w, img_h)
        img_table = Table(
            [[before_img, after_img], ["Original", "Enhanced (final)"]],
            colWidths=[img_w + 10, img_w + 10],
        )
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, 1), 8),
            ("TEXTCOLOR", (0, 1), (-1, 1), colors.grey),
        ]))
        story.append(img_table)
    except Exception as e:
        logger.warning(f"Image preview failed for frame {entry.frame.frame_number}: {e}")
        story.append(Paragraph(f"[Image preview unavailable: {e}]", styles["SmallGrey"]))

    story.append(Spacer(1, 6))

    vlm_before = getattr(entry, "vlm_before", None)
    story.append(Paragraph(f"<b>VLM (before):</b> {describe(vlm_before) if vlm_before else 'n/a'}",
                            styles["Normal"]))
    story.append(Paragraph(f"<b>VLM (after):</b> {describe(entry.vlm)}", styles["Normal"]))

    retries = getattr(entry, "retries", 0)
    history = getattr(entry, "history", [])
    story.append(Paragraph(f"<b>Enhancement log ({retries} iteration(s)):</b> {_tool_log(history)}",
                           styles["Normal"]))

    story.append(HRFlowable(width="100%", color=colors.lightgrey, spaceBefore=8, spaceAfter=8))


def generate_pdf_report(report: VideoReport, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    story = []

    story.append(Paragraph("Video Analysis & Enhancement Report", styles["Title"]))
    story.append(Paragraph(f"Source video: {report.video_name}", styles["Normal"]))
    story.append(Spacer(1, 10))

    accepted = sum(1 for e in report.entries if not (e.vlm and e.vlm.regeneration_recommended))
    summary_data = [
        ["Total frames in video", str(report.total_frames_in_video)],
        ["Keyframes analyzed", str(report.total_keyframes_extracted)],
        ["Frames in acceptable state", f"{accepted} / {len(report.entries)}"],
        ["Total processing time", f"{report.processing_time_seconds:.1f} sec"],
    ]
    t = Table(summary_data, colWidths=[2.4 * inch, 4.1 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(PageBreak())

    story.append(Paragraph("Per-Keyframe Detail", styles["Heading1"]))
    for entry in report.entries:
        _frame_section(entry, story)

    doc.build(story)
    logger.info(f"[report] PDF written to {output_path}")
    return output_path
