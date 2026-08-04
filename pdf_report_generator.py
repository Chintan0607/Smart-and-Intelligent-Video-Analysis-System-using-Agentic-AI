"""
pdf_report.py

Builds the downloadable PDF the agentic pipeline produces once every
frame has either converged or exhausted its retry budget. Pure
report-building — takes an already-populated AgentRunReport and writes
a PDF; no HTTP, no agent logic here.
"""
from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from schemas import AgentFrameResult, AgentRunReport, ToolType

_STYLES = getSampleStyleSheet()
_TITLE_STYLE = _STYLES["Title"]
_HEADING_STYLE = _STYLES["Heading2"]
_BODY_STYLE = _STYLES["Normal"]
_CAPTION_STYLE = ParagraphStyle(
    "Caption", parent=_BODY_STYLE, fontSize=8, textColor=colors.grey, alignment=1
)

_THUMB_SIZE = 2.4 * inch


def _safe_image(path: str) -> Image:
    """Reportlab raises if a referenced file is missing; degrade gracefully."""
    if path and os.path.exists(path):
        img = Image(path, width=_THUMB_SIZE, height=_THUMB_SIZE)
        img.hAlign = "CENTER"
        return img
    return Paragraph("(image unavailable)", _CAPTION_STYLE)


def _tool_log_text(frame_result: AgentFrameResult) -> str:
    if not frame_result.tools_used:
        return "No enhancement applied — VLM approved the original frame on iteration 1."
    tool_names = " → ".join(tool.value for tool in frame_result.tools_used)
    return (
        f"Tools applied ({frame_result.total_tool_applications} pass"
        f"{'es' if frame_result.total_tool_applications != 1 else ''}): {tool_names}"
    )


def _build_frame_section(frame_result: AgentFrameResult, max_iterations: int) -> list:
    story = []

    status_label = "CONVERGED" if frame_result.converged else "MAX ITERATIONS REACHED"
    story.append(
        Paragraph(
            f"Frame #{frame_result.frame_number} — {status_label} "
            f"({len(frame_result.iterations)}/{max_iterations} iterations)",
            _HEADING_STYLE,
        )
    )

    before_after_table = Table(
        [
            [_safe_image(frame_result.original_filepath), _safe_image(frame_result.final_filepath)],
            [Paragraph("Original (extracted)", _CAPTION_STYLE), Paragraph("Final (post-agent)", _CAPTION_STYLE)],
        ],
        colWidths=[_THUMB_SIZE + 10, _THUMB_SIZE + 10],
    )
    before_after_table.setStyle(
        TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")])
    )
    story.append(before_after_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"<b>VLM before:</b> {frame_result.before_summary}", _BODY_STYLE))
    story.append(Paragraph(f"<b>VLM after:</b> {frame_result.after_summary}", _BODY_STYLE))
    story.append(Spacer(1, 4))
    story.append(Paragraph(_tool_log_text(frame_result), _BODY_STYLE))

    iteration_rows = [["Iter", "Tool Used", "Defects Detected", "Regeneration Needed?"]]
    for rec in frame_result.iterations:
        iteration_rows.append(
            [
                str(rec.iteration_number),
                rec.tool_used.value,
                ", ".join(rec.defects_detected) or "-",
                "Yes" if rec.regeneration_recommended else "No",
            ]
        )
    iteration_table = Table(iteration_rows, colWidths=[0.5 * inch, 1.6 * inch, 2.4 * inch, 1.4 * inch])
    iteration_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(Spacer(1, 6))
    story.append(iteration_table)
    story.append(PageBreak())
    return story


def generate_pdf_report(report: AgentRunReport, output_path: str) -> str:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    story = []

    # --- Summary page ---
    story.append(Paragraph("Agentic Enhancement Report", _TITLE_STYLE))
    story.append(Spacer(1, 12))

    converged_count = sum(1 for r in report.frame_results if r.converged)
    maxed_out_count = len(report.frame_results) - converged_count
    total_tool_calls = sum(r.total_tool_applications for r in report.frame_results)

    summary_rows = [
        ["Video", report.video_name],
        ["Total keyframes processed", str(report.total_frames_processed)],
        ["Converged (quality approved)", str(converged_count)],
        ["Hit max iterations", str(maxed_out_count)],
        ["Max iterations per frame", str(report.max_iterations)],
        ["Total tool applications", str(total_tool_calls)],
        ["Total processing time", f"{report.total_processing_time_seconds:.2f} s"],
    ]
    summary_table = Table(summary_rows, colWidths=[2.5 * inch, 3.5 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(summary_table)
    story.append(PageBreak())

    # --- Per-frame sections ---
    for frame_result in report.frame_results:
        story.extend(_build_frame_section(frame_result, report.max_iterations))

    doc.build(story)
    return output_path
    
