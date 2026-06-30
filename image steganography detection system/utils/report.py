"""
Enhanced PDF report generator.
Supports single-image and multi-image (batch) aggregate reports.
Includes: Synopsis, Problem Description, Risk Factors, Image Details,
          Message Extraction results.
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image as RLImage,
    PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
import matplotlib
matplotlib.use("Agg")

# ── Palette ─────────────────────────────────────────────────────────────
C_DARK = colors.HexColor("#0d1117")
C_PANEL = colors.HexColor("#1e2530")
C_ACCENT = colors.HexColor("#00d4aa")
C_RED = colors.HexColor("#e53935")
C_YELLOW = colors.HexColor("#f9a825")
C_BLUE = colors.HexColor("#1565c0")
C_LIGHT = colors.HexColor("#e6edf3")
C_MID = colors.HexColor("#607d8b")
C_WHITE = colors.white
C_BLACK = colors.black
C_BG = colors.HexColor("#f4f6f8")
C_BORDER = colors.HexColor("#cfd8dc")

W_PAGE = A4[0] - 4 * cm   # usable width


def _risk_color(level: str):
    return {"HIGH": C_RED, "MEDIUM": C_YELLOW, "LOW": C_ACCENT}.get(level, C_MID)


def _risk_bg(level: str):
    return {
        "HIGH":   colors.HexColor("#ffebee"),
        "MEDIUM": colors.HexColor("#fffde7"),
        "LOW":    colors.HexColor("#e8f5e9"),
    }.get(level, colors.HexColor("#f5f5f5"))


# ── Style sheet ─────────────────────────────────────────────────────────
def _styles():
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "rpt_title",
            fontSize=24, textColor=C_DARK, alignment=TA_CENTER,
            leading=30, spaceAfter=10, fontName="Helvetica-Bold"),
        "subtitle": ParagraphStyle(
            "rpt_sub",
            fontSize=10, textColor=C_MID, alignment=TA_CENTER,
            leading=14, spaceBefore=6, spaceAfter=16),
        "h1": ParagraphStyle(
            "rpt_h1",
            fontSize=15, textColor=C_DARK, fontName="Helvetica-Bold",
            spaceBefore=18, spaceAfter=6, borderPad=4, borderColor=C_ACCENT),
        "h2": ParagraphStyle(
            "rpt_h2",
            fontSize=12, textColor=C_DARK, fontName="Helvetica-Bold",
            spaceBefore=12, spaceAfter=4),
        "h3": ParagraphStyle(
            "rpt_h3",
            fontSize=10, textColor=C_BLUE, fontName="Helvetica-Bold",
            spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle(
            "rpt_body",
            fontSize=9, textColor=colors.HexColor("#263238"),
            leading=15, alignment=TA_JUSTIFY, fontName="Helvetica"),
        "mono": ParagraphStyle(
            "rpt_mono",
            fontSize=8, textColor=colors.HexColor("#1a237e"),
            fontName="Courier", leading=12,
            backColor=colors.HexColor("#e8eaf6"),
            leftIndent=8, rightIndent=8),
        "small": ParagraphStyle(
            "rpt_small",
            fontSize=7.5, textColor=C_MID, leading=11),
        "caption": ParagraphStyle(
            "rpt_caption",
            fontSize=8, textColor=C_MID, alignment=TA_CENTER, spaceAfter=6),
        "bullet": ParagraphStyle(
            "rpt_bullet",
            fontSize=9, textColor=colors.HexColor("#263238"),
            leading=14, leftIndent=16, bulletIndent=6),
    }


# ── Reusable flowables ───────────────────────────────────────────────────

def _section_header(title: str, st: dict) -> list:
    return [
        HRFlowable(width=W_PAGE, thickness=2, color=C_ACCENT,
                   spaceBefore=10, spaceAfter=4),
        Paragraph(title, st["h1"]),
    ]


def _kv_table(rows: list, col1: float = 4 * cm) -> Table:
    t = Table(rows, colWidths=[col1, W_PAGE - col1])
    t.setStyle(TableStyle([
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",      (0, 0), (0,  -1), C_MID),
        ("TEXTCOLOR",      (1, 0), (1,  -1), C_BLACK),
        ("FONTNAME",       (0, 0), (0,  -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_BG, C_WHITE]),
        ("BOX",            (0, 0), (-1, -1), 0.4, C_BORDER),
        ("INNERGRID",      (0, 0), (-1, -1), 0.25, C_BORDER),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ]))
    return t


def _detection_table(detections: dict, st: dict) -> Table:
    """Enhanced detection table showing which tests triggered."""
    header = [
        Paragraph("<b>Test</b>",   st["small"]),
        Paragraph("<b>Result</b>", st["small"]),
    ]
    rows = [header]
    for name, detected in detections.items():
        result_text = "DETECTED" if detected else "CLEAN"
        color = C_RED if detected else C_ACCENT
        result_para = Paragraph(
            f'<font color="{color.hexval()}"><b>{result_text}</b></font>',
            st["small"],
        )
        # Shorten test names for display
        display_name = name.replace(" Analysis", "").replace(" Test", "")
        rows.append([
            Paragraph(display_name, st["small"]),
            result_para,
        ])

    t = Table(rows, colWidths=[6 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_DARK),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, C_WHITE]),
        ("BOX",            (0, 0), (-1, -1), 0.4, C_BORDER),
        ("INNERGRID",      (0, 0), (-1, -1), 0.25, C_BORDER),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


def _detection_summary_table(batch_results: list, st: dict) -> Table:
    """Summary table showing which tests triggered for each image."""
    test_names = ["LSB", "Chi-Square", "RS", "Histogram", "DCT"]

    header = [Paragraph("<b>Image</b>", st["small"])]
    header.extend([Paragraph(f"<b>{name}</b>", st["small"])
                  for name in test_names])
    header.append(Paragraph("<b>Risk</b>", st["small"]))

    rows = [header]

    for result in batch_results:
        ov = result["overall"]
        detections = ov.get("detections", {})
        fname = os.path.basename(result["image_path"])[:25]

        # Map detection results to test names
        det_map = {
            "LSB Analysis": detections.get("LSB Analysis", False),
            "Chi-Square Test": detections.get("Chi-Square Test", False),
            "RS Analysis": detections.get("RS Analysis", False),
            "Histogram Analysis": detections.get("Histogram Analysis", False),
            "DCT Analysis": detections.get("DCT Analysis", False),
        }

        row = [Paragraph(fname, st["small"])]
        for test in test_names:
            # Match test name to detection key
            key = next((k for k in det_map.keys() if test in k), None)
            detected = det_map.get(key, False) if key else False
            # Use text instead of symbols
            status = "DETECTED" if detected else "CLEAN"
            color = C_RED if detected else C_ACCENT
            row.append(Paragraph(
                f'<font color="{color.hexval()}">{status}</font>',
                st["small"]
            ))

        # Risk level with color
        level = ov["risk_level"]
        color = _risk_color(level)
        row.append(Paragraph(
            f'<font color="{color.hexval()}"><b>{level}</b></font>',
            st["small"]
        ))
        rows.append(row)

    # Calculate column widths
    col_widths = [4.5 * cm] + [2.2 * cm] * 5 + [2.5 * cm]

    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_DARK),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, C_WHITE]),
        ("BOX",            (0, 0), (-1, -1), 0.4, C_BORDER),
        ("INNERGRID",      (0, 0), (-1, -1), 0.25, C_BORDER),
        ("LEFTPADDING",    (0, 0), (-1, -1), 4),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


def _verdict_banner(level: str, verdict: str, score: float, st: dict) -> Table:
    rc = _risk_color(level)
    bg = _risk_bg(level)
    t = Table([[
        Paragraph(
            f'<font color="{rc.hexval()}"><b>{level} RISK</b></font>', st["h2"]),
        Paragraph(verdict, st["body"]),
        Paragraph(f'<b>{score * 100:.0f}%</b>', st["h2"]),
    ]], colWidths=[3 * cm, W_PAGE - 6 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 1.0, rc),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (2, 0), (2,  0),  "CENTER"),
    ]))
    return t


def _pil_to_rl_image(pil_img: Image.Image, max_w: float, max_h: float) -> RLImage:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h, 1.0)
    return RLImage(buf, width=w * scale, height=h * scale)


def _image_thumbnail(path: str, max_w: float = 4 * cm, max_h: float = 3 * cm) -> RLImage:
    img = Image.open(path).convert("RGB")
    return _pil_to_rl_image(img, max_w, max_h)


# ── Aggregate bar chart ──────────────────────────────────────────────────

def _make_aggregate_chart(batch_results: list) -> RLImage:
    names = [os.path.basename(r["image_path"])[:20] for r in batch_results]
    scores = [r["overall"]["risk_score"] * 100 for r in batch_results]
    col_map = {"HIGH": "#e53935", "MEDIUM": "#f9a825", "LOW": "#00d4aa"}
    bar_colors = [col_map.get(r["overall"]["risk_level"], "#607d8b")
                  for r in batch_results]

    fig, ax = plt.subplots(
        figsize=(10, max(3, len(names) * 0.5 + 1.5)), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f6f8")

    bars = ax.barh(names, scores, color=bar_colors,
                   height=0.6, edgecolor="white")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Risk Score (%)", fontsize=9)
    ax.set_title("Risk Score per Image", fontsize=11, fontweight="bold", pad=8)
    ax.axvline(60, color="#e53935", linewidth=0.8,
               linestyle="--", alpha=0.5, label="High threshold")
    ax.axvline(40, color="#f9a825", linewidth=0.8, linestyle="--",
               alpha=0.5, label="Medium threshold")
    ax.legend(fontsize=7, loc="lower right")

    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}%", va="center", fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.4)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return _pil_to_rl_image(Image.open(buf), W_PAGE, 12 * cm)


# ── Synopsis content ───────────────────────────────────────────────

SYNOPSIS = """
Steganography is the practice of concealing secret information within ordinary,
non-secret files or messages to avoid detection. Unlike cryptography, which
makes data unreadable, steganography hides the very existence of the data.
Digital image steganography is among the most prevalent techniques, where hidden
payloads are embedded within image pixels, often targeting the least-significant
bits (LSBs) of colour channels, discrete cosine transform (DCT) coefficients, or
other statistical properties of the image.

This report presents the results of a multi-algorithm forensic analysis performed
by the Stego Detector system. Each image has been subjected to five independent
detection tests: LSB randomness analysis, Chi-square statistical testing, RS
(Regular-Singular) payload estimation, histogram comb-pattern detection, and
DCT/FFT frequency-domain analysis. The combination of these tests provides a
comprehensive, layered assessment of the likelihood that steganographic content
is present in each image.
"""


def _build_image_description(fname: str, overall: dict, ar: dict, st: dict) -> list:
    """Build description for a single image."""
    flowables = []

    # Which tests detected something
    detections = overall.get("detections", {})
    triggered = [k.replace(" Analysis", "").replace(" Test", "")
                 for k, v in detections.items() if v]
    clean_tests = [k.replace(" Analysis", "").replace(" Test", "")
                   for k, v in detections.items() if not v]

    # Build description
    if triggered:
        flowables.append(Paragraph(
            f"<b>Detection Summary:</b> The following tests detected anomalies: "
            f"<b>{', '.join(triggered)}</b>. "
            f"Tests that returned clean results: {', '.join(clean_tests) if clean_tests else 'None'}.",
            st["body"]
        ))
    else:
        flowables.append(Paragraph(
            "<b>Detection Summary:</b> None of the five detection tests identified "
            "statistical anomalies consistent with steganographic embedding.",
            st["body"]
        ))

    flowables.append(Spacer(1, 0.2 * cm))

    # Key metrics
    rs_payload = ar.get("rs_analysis", {}).get(
        "average_payload_estimate", 0) * 100
    lsb_score = ar.get("lsb", {}).get("avg_correlation", 0)
    chi_p = ar.get("chi_square", {}).get("average_p_value", 1.0)
    comb_score = ar.get("histogram", {}).get("avg_comb_score", 1.0)

    flowables.append(Paragraph(
        f"<b>Key Metrics:</b> RS payload estimate: <b>{rs_payload:.2f}%</b> | "
        f"LSB correlation: <b>{lsb_score:.4f}</b> | "
        f"Chi-square p-value: <b>{chi_p:.4f}</b> | "
        f"Histogram comb score: <b>{comb_score:.3f}</b>",
        st["body"]
    ))

    return flowables


def _build_extraction_section(extraction: dict, st: dict) -> list:
    """Build message extraction display section with text-based indicators."""
    flowables = []

    if not extraction or not extraction.get("extraction_attempted"):
        flowables.append(
            Paragraph("No message extraction was performed.", st["body"]))
        return flowables

    flowables.append(
        Paragraph("<b>LSB Message Extraction Results</b>", st["h3"]))

    best = extraction.get("best_result", {})
    best_ch = extraction.get("best_channel", "—")
    likely = best.get("likely_message", False)

    # Use text indicators
    likely_text = "YES" if likely else "NO"
    likely_color = C_ACCENT if likely else C_MID

    flowables.append(Paragraph(
        f"Best channel: <b>{best_ch}</b> | "
        f"Printable ratio: <b>{best.get('printable_ratio', 0):.1%}</b> | "
        f"Characters: <b>{best.get('length', 0)}</b> | "
        f"Likely message: <font color='{likely_color.hexval()}'><b>{likely_text}</b></font>",
        st["body"]
    ))
    flowables.append(Spacer(1, 0.15 * cm))

    if likely:
        txt = best.get("text", "")[:500]
        flowables.append(
            Paragraph("<b>Extracted content (first 500 chars):</b>", st["body"]))
        flowables.append(Spacer(1, 0.1 * cm))
        # Split into lines for readability
        for i in range(0, len(txt), 80):
            flowables.append(Paragraph(txt[i:i + 80], st["mono"]))
        flowables.append(Spacer(1, 0.15 * cm))

    # Per-channel table - using "YES"/"NO" text instead of symbols
    ch_rows = [
        ["Channel", "Length", "Printable Ratio", "Likely Message"]
    ]
    for ch_name, ch_data in extraction.get("channels", {}).items():
        is_likely = ch_data.get("likely_message", False)
        ch_rows.append([
            ch_name,
            str(ch_data.get("length", 0)),
            f"{ch_data.get('printable_ratio', 0):.1%}",
            "YES" if is_likely else "NO",
        ])

    ct = Table(ch_rows, colWidths=[3.5 * cm, 3 * cm, 3.5 * cm, 3.5 * cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_DARK),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, C_WHITE]),
        ("BOX",            (0, 0), (-1, -1), 0.4, C_BORDER),
        ("INNERGRID",      (0, 0), (-1, -1), 0.25, C_BORDER),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("ALIGN",          (3, 0), (-1, -1), "CENTER"),
    ]))

    # Color the YES/NO cells
    for i, row in enumerate(ch_rows[1:], start=1):
        is_yes = row[3] == "YES"
        color = C_ACCENT if is_yes else C_MID
        ct.setStyle(TableStyle([
            ("TEXTCOLOR", (3, i), (3, i), color),
            ("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"),
        ]))

    flowables.append(ct)

    return flowables


def _build_aggregate_description(batch_results: list, st: dict) -> list:
    """Short aggregate description for the batch summary section."""
    n_img = len(batch_results)
    high = sum(
        1 for r in batch_results if r["overall"]["risk_level"] == "HIGH")
    medium = sum(
        1 for r in batch_results if r["overall"]["risk_level"] == "MEDIUM")
    low = n_img - high - medium
    avg_score = sum(r["overall"]["risk_score"]
                    for r in batch_results) / n_img * 100

    # Count images with extracted messages
    msg_count = sum(
        1 for r in batch_results
        if r.get("extraction", {}) and
        r["extraction"].get("best_result", {}).get("likely_message")
    )

    # Count images with any positive detection
    any_detected = sum(
        1 for r in batch_results
        if any(r["overall"].get("detections", {}).values())
    )

    summary = []

    if high > 0:
        summary.append(Paragraph(
            f"<b>Critical Findings:</b> <b>{high} of {n_img}</b> image(s) flagged as HIGH RISK, "
            f"indicating probable steganographic embedding requiring immediate investigation.",
            st["body"]
        ))
    elif medium > 0:
        summary.append(Paragraph(
            f"<b>Findings:</b> <b>{medium} of {n_img}</b> image(s) returned MEDIUM RISK signals, "
            f"suggesting possible partial or low-density embedding that warrants further review.",
            st["body"]
        ))
    else:
        summary.append(Paragraph(
            f"<b>Findings:</b> All <b>{n_img}</b> analysed images returned LOW RISK results, "
            f"with no statistical evidence of steganographic embedding detected.",
            st["body"]
        ))

    summary.append(Spacer(1, 0.1 * cm))

    # Additional statistics
    extra_stats = []
    if msg_count > 0:
        extra_stats.append(
            f"<b>{msg_count}</b> image(s) with extracted messages")
    if any_detected > 0:
        extra_stats.append(
            f"<b>{any_detected}</b> image(s) with positive detections")

    if extra_stats:
        summary.append(Paragraph(
            f"<b>Summary Statistics:</b> {' | '.join(extra_stats)}. "
            f"Average risk score: <b>{avg_score:.0f}%</b> "
            f"({high} HIGH · {medium} MEDIUM · {low} LOW).",
            st["body"]
        ))

    return summary


# ── Main report builder ──────────────────────────────────────────────────

def generate_pdf_report(
    image_path: str,
    analysis_results: dict,
    overall: dict,
    output_path: str,
    extraction_result: dict = None,
) -> str:
    """Single-image report (delegates to generate_batch_report)."""
    batch = [{
        "image_path":       image_path,
        "analysis_results": analysis_results,
        "overall":          overall,
        "extraction":       extraction_result,
    }]
    return generate_batch_report(batch, output_path)


def generate_batch_report(batch_results: list, output_path: str) -> str:
    """
    batch_results: list of dicts with keys:
        image_path, analysis_results, overall, extraction (optional)
    """
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2.2 * cm, bottomMargin=2 * cm,
        title="Steganography Analysis Report",
        author="Stego Detector",
    )
    st = _styles()
    story = []
    n_img = len(batch_results)
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # ── Cover ────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.5 * cm),
        Paragraph("STEGANOGRAPHY ANALYSIS REPORT", st["title"]),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "Forensic Image Intelligence System  ·  Stego Detector", st["subtitle"]),
        HRFlowable(width=W_PAGE, thickness=2.5, color=C_ACCENT, spaceAfter=10),
        Spacer(1, 0.3 * cm),
    ]

    high = sum(
        1 for r in batch_results if r["overall"]["risk_level"] == "HIGH")
    medium = sum(
        1 for r in batch_results if r["overall"]["risk_level"] == "MEDIUM")
    low = sum(1 for r in batch_results if r["overall"]["risk_level"] == "LOW")

    # Count messages
    msg_count = sum(
        1 for r in batch_results
        if r.get("extraction", {}) and
        r["extraction"].get("best_result", {}).get("likely_message")
    )

    cover_data = [
        ["Report Generated", now],
        ["Images Analysed",  str(n_img)],
        ["HIGH Risk",        str(high)],
        ["MEDIUM Risk",      str(medium)],
        ["LOW Risk",         str(low)],
        ["Messages Found",   str(msg_count)],
        ["Detection Tests",
            "5 per image (LSB, Chi-Square, RS, Histogram, DCT/FFT)"],
    ]
    story.append(_kv_table(cover_data, col1=4.5 * cm))
    story.append(Spacer(1, 0.8 * cm))

    # Aggregate bar chart (batch only)
    if n_img > 1:
        story.append(Paragraph("Risk Score Overview", st["h2"]))
        story.append(_make_aggregate_chart(batch_results))
        story.append(Spacer(1, 0.4 * cm))

        # Detection summary table
        story.append(Paragraph("Detection Summary by Image", st["h2"]))
        story.append(_detection_summary_table(batch_results, st))
        story.append(Spacer(1, 0.6 * cm))

    story.append(PageBreak())

    # ── Synopsis ─────────────────────────────────────────────────────
    story += _section_header("1. Synopsis", st)
    for para in SYNOPSIS.strip().split("\n\n"):
        story.append(Paragraph(para.strip(), st["body"]))
        story.append(Spacer(1, 0.2 * cm))

    # Add batch summary to synopsis
    if n_img > 1:
        story.append(Paragraph("<b>Batch Summary</b>", st["h3"]))
        story.extend(_build_aggregate_description(batch_results, st))
        story.append(Spacer(1, 0.2 * cm))

    story.append(PageBreak())

    # ── Per-image results ────────────────────────────────────────────
    for idx, result in enumerate(batch_results):
        img_path = result["image_path"]
        ar = result["analysis_results"]
        overall = result["overall"]
        extraction = result.get("extraction")
        fname = os.path.basename(img_path)

        story += _section_header(f"Image {idx + 1}: {fname}", st)

        # Thumbnail + file info
        try:
            thumb = _image_thumbnail(img_path, max_w=5 * cm, max_h=4 * cm)
            size = os.path.getsize(img_path)
            info_rows = [
                ["Filename",   fname],
                ["Format",     overall.get("image_format", "?")],
                ["Mode",       overall.get("image_mode", "?")],
                ["Dimensions",
                    f"{overall['image_size'][0]} × {overall['image_size'][1]} px"],
                ["File size",  f"{size / 1024:.1f} KB"],
            ]
            info_t = _kv_table(info_rows, col1=3 * cm)
            layout = Table([[thumb, info_t]], colWidths=[
                           5.5 * cm, W_PAGE - 5.5 * cm])
            layout.setStyle(TableStyle([
                ("VALIGN",      (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (1, 0), (1,  0),  12),
            ]))
            story.append(layout)
        except Exception:
            story.append(_kv_table([
                ["Filename",   fname],
                ["Dimensions",
                    f"{overall['image_size'][0]} × {overall['image_size'][1]} px"],
            ]))

        story.append(Spacer(1, 0.4 * cm))

        # Verdict banner
        story.append(_verdict_banner(
            overall["risk_level"], overall["verdict"], overall["risk_score"], st))
        story.append(Spacer(1, 0.4 * cm))

        # Stats row
        pos = overall["positive_count"]
        tot = overall["total_tests"]
        payload = overall.get("estimated_payload_pct", 0)
        story.append(Paragraph(
            f"<b>Tests positive:</b> {pos}/{tot}   "
            f"<b>Estimated payload:</b> {payload:.2f}% of pixels   "
            f"<b>Risk score:</b> {overall['risk_score'] * 100:.0f}%",
            st["body"],
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Detection table (enhanced)
        story.append(Paragraph("Detection Test Results", st["h3"]))
        story.append(_detection_table(overall["detections"], st))
        story.append(Spacer(1, 0.4 * cm))

        # Image-specific description (which tests detected)
        story.append(Paragraph("Analysis Summary", st["h3"]))
        story.extend(_build_image_description(fname, overall, ar, st))
        story.append(Spacer(1, 0.4 * cm))

        # Message extraction
        if extraction:
            story.append(Paragraph("Message Extraction", st["h3"]))
            story.extend(_build_extraction_section(extraction, st))
        else:
            story.append(Paragraph(
                "<i>Message extraction was not performed for this image.</i>",
                st["body"]
            ))

        story.append(Spacer(1, 0.5 * cm))
        if idx < n_img - 1:
            story.append(PageBreak())

    # ── Aggregate summary (batch only) ──────────────────────────────
    if n_img > 1:
        story.append(PageBreak())
        story += _section_header("Aggregate Summary", st)

        # Summary statistics
        high_imgs = [os.path.basename(r["image_path"]) for r in batch_results
                     if r["overall"]["risk_level"] == "HIGH"]
        med_imgs = [os.path.basename(r["image_path"]) for r in batch_results
                    if r["overall"]["risk_level"] == "MEDIUM"]
        msg_imgs = [os.path.basename(r["image_path"]) for r in batch_results
                    if r.get("extraction", {}) and
                    r["extraction"].get("best_result", {}).get("likely_message")]

        # Images with any positive detection
        pos_imgs = [os.path.basename(r["image_path"]) for r in batch_results
                    if any(r["overall"].get("detections", {}).values())]

        if high_imgs:
            story.append(Paragraph(
                f"<b>High-risk images (require immediate investigation):</b> "
                f"{', '.join(high_imgs)}.", st["body"]))
            story.append(Spacer(1, 0.1 * cm))

        if med_imgs:
            story.append(Paragraph(
                f"<b>Medium-risk images (warrant further review):</b> "
                f"{', '.join(med_imgs)}.", st["body"]))
            story.append(Spacer(1, 0.1 * cm))

        if msg_imgs:
            story.append(Paragraph(
                f"<b>Images with probable extracted messages:</b> "
                f"{', '.join(msg_imgs)}.", st["body"]))
            story.append(Spacer(1, 0.1 * cm))

        if pos_imgs and not high_imgs and not med_imgs:
            story.append(Paragraph(
                f"<b>Images with positive detections but low risk:</b> "
                f"{', '.join(pos_imgs)}.", st["body"]))
            story.append(Spacer(1, 0.1 * cm))

        if not high_imgs and not med_imgs and not msg_imgs and not pos_imgs:
            story.append(Paragraph(
                "All analysed images appear consistent with unmodified content. "
                "No steganographic embedding detected across the batch.", st["body"]))

        story.append(Spacer(1, 0.4 * cm))

        # Detailed aggregate table
        story.append(Paragraph("Detailed Batch Results", st["h2"]))
        agg_rows = [["Image", "Risk", "Score",
                     "Tests +ve", "Payload", "Message"]]
        for r in batch_results:
            ov = r["overall"]
            ext = r.get("extraction") or {}
            msg_found = ext.get("best_result", {}).get("likely_message", False)
            pos_count = ov["positive_count"]
            if pos_count > 0:
                # Show which tests triggered
                detections = ov.get("detections", {})
                triggered = [k.replace(" Analysis", "").replace(" Test", "")[:4]
                             for k, v in detections.items() if v]
                tests_str = f"{pos_count}/{ov['total_tests']} ({', '.join(triggered)})"
            else:
                tests_str = f"{pos_count}/{ov['total_tests']}"

            agg_rows.append([
                os.path.basename(r["image_path"])[:28],
                ov["risk_level"],
                f"{ov['risk_score'] * 100:.0f}%",
                tests_str,
                f"{ov.get('estimated_payload_pct', 0):.1f}%",
                "YES" if msg_found else "NO",
            ])

        col_map = {"HIGH": C_RED, "MEDIUM": C_YELLOW, "LOW": C_ACCENT}
        at = Table(agg_rows, colWidths=[
                   5 * cm, 2.5 * cm, 2 * cm, 3.5 * cm, 2.5 * cm, 2 * cm])
        ts = [
            ("BACKGROUND",     (0, 0), (-1, 0),  C_DARK),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 7.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, C_WHITE]),
            ("BOX",            (0, 0), (-1, -1), 0.4, C_BORDER),
            ("INNERGRID",      (0, 0), (-1, -1), 0.25, C_BORDER),
            ("LEFTPADDING",    (0, 0), (-1, -1), 4),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ]
        for row_i, r in enumerate(batch_results, start=1):
            level = r["overall"]["risk_level"]
            c = col_map.get(level, C_MID)
            ts.append(("TEXTCOLOR", (1, row_i), (1, row_i), c))
            ts.append(("FONTNAME",  (1, row_i), (1, row_i), "Helvetica-Bold"))
        at.setStyle(TableStyle(ts))
        story.append(at)

    # ── Footer disclaimer ────────────────────────────────────────────
    story += [
        Spacer(1, 1 * cm),
        HRFlowable(width=W_PAGE, thickness=0.5, color=C_MID, spaceBefore=8),
        Paragraph(
            "This report is generated automatically by Stego Detector. Results are probabilistic "
            "and based on statistical analysis; they are not legally conclusive. No single test "
            "is definitive. Findings should be interpreted alongside domain expertise and "
            "corroborated with additional forensic tools before any decision is made.",
            st["small"],
        ),
    ]

    doc.build(story)
    return output_path
