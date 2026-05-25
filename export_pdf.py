"""
export_pdf.py
=============
Reads all output CSVs and compiles them into a single PDF report
suitable for uploading to NotebookLM or sharing as a research document.
"""

import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUTPUT_DIR = "output"
PDF_PATH   = os.path.join(OUTPUT_DIR, "research_results_full.pdf")

# ── Colour palette ──────────────────────────────────────────────────────────
C_NAVY    = colors.HexColor("#1a3a5c")
C_BLUE    = colors.HexColor("#2e6fad")
C_LIGHT   = colors.HexColor("#dce9f5")
C_HEADER  = colors.HexColor("#2e6fad")
C_ALT     = colors.HexColor("#f0f6fc")
C_WHITE   = colors.white
C_BLACK   = colors.black
C_RED     = colors.HexColor("#c0392b")
C_GREEN   = colors.HexColor("#27ae60")

styles    = getSampleStyleSheet()

# ── Custom paragraph styles ─────────────────────────────────────────────────
TITLE_STYLE = ParagraphStyle(
    "Title2", parent=styles["Title"],
    fontSize=22, textColor=C_NAVY, spaceAfter=6,
    alignment=TA_CENTER, fontName="Helvetica-Bold",
)
H1_STYLE = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=14, textColor=C_NAVY, spaceBefore=14, spaceAfter=4,
    fontName="Helvetica-Bold",
)
H2_STYLE = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=11, textColor=C_BLUE, spaceBefore=8, spaceAfter=3,
    fontName="Helvetica-Bold",
)
BODY_STYLE = ParagraphStyle(
    "Body2", parent=styles["Normal"],
    fontSize=8.5, leading=13, spaceAfter=4,
)
SMALL_STYLE = ParagraphStyle(
    "Small", parent=styles["Normal"],
    fontSize=7.5, leading=11,
)
CENTER_STYLE = ParagraphStyle(
    "Center", parent=styles["Normal"],
    fontSize=8.5, alignment=TA_CENTER,
)


# ── Helper: build a ReportLab Table from a DataFrame ───────────────────────

def df_to_table(df: pd.DataFrame, col_widths=None,
                font_size=7, max_col_w=5.5*cm) -> Table:
    """Convert a DataFrame to a styled ReportLab Table."""

    # Header + data rows
    header = list(df.columns)
    rows   = [header] + [list(r) for r in df.itertuples(index=False, name=None)]

    # Format numbers
    def fmt(v):
        if isinstance(v, float):
            if abs(v) >= 100:
                return f"{v:,.1f}"
            elif abs(v) >= 1:
                return f"{v:.4f}"
            else:
                return f"{v:.6f}"
        return str(v) if v is not None else ""

    rows = [[fmt(c) for c in row] for row in rows]
    rows[0] = header   # keep header as-is

    # Column widths: auto if not provided
    if col_widths is None:
        page_w = landscape(A4)[0] - 2 * cm
        n      = len(header)
        col_widths = [min(max_col_w, page_w / n)] * n

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)

    # Alternating row colours
    row_cmds = []
    for i in range(1, len(rows)):
        bg = C_ALT if i % 2 == 0 else C_WHITE
        row_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",  (0, 0), (-1, 0), C_HEADER),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), font_size),
        ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        # Data
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), font_size),
        ("ALIGN",       (0, 1), (-1, -1), "CENTER"),
        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
        ("LINEBELOW",   (0, 0), (-1, 0), 1.2, C_NAVY),
        *row_cmds,
    ]))
    return tbl


def section(story, title, level=1):
    style = H1_STYLE if level == 1 else H2_STYLE
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(title, style))
    if level == 1:
        story.append(HRFlowable(width="100%", thickness=1,
                                color=C_BLUE, spaceAfter=4))


def load(filename) -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, filename)
    return pd.read_csv(path)


# ── Build PDF ───────────────────────────────────────────────────────────────

def build_pdf():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=landscape(A4),
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.8*cm,  bottomMargin=1.8*cm,
        title="Vietnamese Bank Stock Portfolio Research Results",
        author="NCKH Sinh Vien",
    )

    story = []
    PW = landscape(A4)[0] - 3*cm   # usable page width

    # ════════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph(
        "Tối ưu hóa Danh mục Cổ phiếu Ngân hàng Việt Nam",
        TITLE_STYLE,
    ))
    story.append(Paragraph(
        "bằng Mô hình Black-Litterman kết hợp Phân cụm K-means và Monte Carlo",
        ParagraphStyle("Sub", parent=styles["Normal"],
                       fontSize=13, alignment=TA_CENTER,
                       textColor=C_BLUE, spaceAfter=10),
    ))
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="60%", thickness=2, color=C_NAVY,
                             hAlign="CENTER", spaceAfter=10))
    story.append(Spacer(1, 0.4*cm))

    meta_lines = [
        "Universe: 25 cổ phiếu ngân hàng niêm yết tại HOSE / HNX",
        "Dữ liệu: Giá đóng cửa tháng &amp; Vốn hóa thị trường  |  2014-06 → 2026-05",
        "Cửa sổ huấn luyện: 36 tháng (rolling)  |  OOS: 106 tháng (2017-08 → 2026-05)",
        "Lãi suất phi rủi ro: VN10Y / 12 (time-varying)",
        "Danh mục: MKT · TAN · MV · BL · EW (1/N) · RP (Risk Parity)",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, CENTER_STYLE))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.8*cm))

    # Mini summary box
    summary_data = [
        ["Danh mục", "Ann. Return", "Ann. Vol", "Sharpe", "Sortino", "MDD"],
        ["MKT", "+23.40%", "27.05%", "0.729", "0.763", "-32.51%"],
        ["TAN", "+19.20%", "27.81%", "0.558", "0.549", "-51.54%"],
        ["MV",  "+16.24%", "25.40%", "0.494", "0.539", "-38.44%"],
        ["BL",  "+25.29%", "29.77%", "0.726", "0.715", "-51.13%"],
        ["EW",  "+26.62%", "27.38%", "0.837", "0.904", "-36.55%"],
        ["RP",  "+26.62%", "27.38%", "0.837", "0.904", "-36.55%"],
    ]
    col_w = [3*cm, 2.8*cm, 2.8*cm, 2.5*cm, 2.5*cm, 2.8*cm]
    tbl = Table(summary_data, colWidths=col_w, hAlign="CENTER")
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("BACKGROUND",   (0, 3), (-1, 3), C_LIGHT),  # BL row highlight
        ("BACKGROUND",   (0, 5), (-1, 6), colors.HexColor("#e8f5e9")),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#90a4ae")),
        ("LINEBELOW",    (0, 0), (-1, 0), 1.5, C_NAVY),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 – PERFORMANCE SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    section(story, "1. Hiệu suất Out-of-Sample (Tóm tắt)", 1)
    story.append(Paragraph(
        "Lợi nhuận năm hoá, biến động, Sharpe, Sortino, MDD, Calmar và tỷ lệ thắng "
        "của 6 danh mục trên 106 tháng OOS (08/2017 – 05/2026). "
        "Lãi suất phi rủi ro sử dụng VN10Y / 12 theo từng năm.", BODY_STYLE,
    ))
    df_perf = load("performance_summary_v2.csv")
    story.append(df_to_table(df_perf, col_widths=[2.5*cm]*len(df_perf.columns), font_size=8))

    story.append(Spacer(1, 0.5*cm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 – STATISTICAL TESTS
    # ════════════════════════════════════════════════════════════════════════
    section(story, "2. Kiểm định Thống kê", 1)

    section(story, "2a. Paired t-test  (H₀: E[r_BL] = E[r_other])", 2)
    story.append(Paragraph(
        "Kiểm định t ghép cặp hai phía so sánh lợi nhuận tháng OOS của BL "
        "với từng danh mục còn lại.", BODY_STYLE,
    ))
    df_tt = load("ttest_results_v2.csv")
    story.append(df_to_table(df_tt, col_widths=[4*cm, 3.5*cm, 3.5*cm, 4*cm], font_size=8))

    story.append(Spacer(1, 0.3*cm))
    section(story, "2b. Jobson-Korkie Test  (H₀: SR_BL = SR_other)", 2)
    story.append(Paragraph(
        "Kiểm định Jobson & Korkie (1981) so sánh trực tiếp hai hệ số Sharpe "
        "(dùng thống kê Sharpe theo tháng, không annualise). "
        "Thống kê z tiệm cận phân phối chuẩn N(0,1).", BODY_STYLE,
    ))
    df_jk = load("jobson_korkie_results.csv")
    story.append(df_to_table(df_jk, col_widths=[4*cm, 3.5*cm, 3.5*cm, 4*cm], font_size=8))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 – DISTRIBUTION TESTS
    # ════════════════════════════════════════════════════════════════════════
    section(story, "3. Kiểm định Phân phối Lợi nhuận", 1)

    section(story, "3a. Jarque-Bera  (H₀: phân phối chuẩn)", 2)
    story.append(Paragraph(
        "Kiểm tra liệu lợi nhuận tháng OOS có phân phối chuẩn không. "
        "Bác bỏ H₀ (p < 0.05) nghĩa là lợi nhuận có độ lệch hoặc đuôi dày.", BODY_STYLE,
    ))
    df_jb = load("distribution_jb_tests.csv")
    story.append(df_to_table(df_jb, col_widths=[2.5*cm]*len(df_jb.columns), font_size=8))

    story.append(Spacer(1, 0.3*cm))
    section(story, "3b. Ljung-Box  (H₀: không có tự tương quan)", 2)
    story.append(Paragraph(
        "Kiểm tra tự tương quan chuỗi lợi nhuận tháng ở các độ trễ 5, 10, 20. "
        "p > 0.05 nghĩa là không có bằng chứng tự tương quan.", BODY_STYLE,
    ))
    df_lb = load("distribution_lb_tests.csv")
    story.append(df_to_table(df_lb, col_widths=[2.2*cm]*len(df_lb.columns), font_size=7.5))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 – DRAWDOWN ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    section(story, "4. Phân tích Drawdown", 1)

    section(story, "4a. Tóm tắt Drawdown", 2)
    story.append(Paragraph(
        "Số đợt sụt giảm, độ sâu trung bình, thời gian phục hồi trung bình, "
        "số tháng danh mục vượt ngưỡng -20%.", BODY_STYLE,
    ))
    df_dds = load("drawdown_summary.csv")
    story.append(df_to_table(df_dds, col_widths=[2.5*cm]*len(df_dds.columns), font_size=8))

    story.append(Spacer(1, 0.3*cm))
    section(story, "4b. Chi tiết từng đợt Drawdown", 2)
    story.append(Paragraph(
        "Ngày bắt đầu, ngày đáy, ngày phục hồi, độ sâu (%), "
        "số tháng đến đáy và số tháng phục hồi cho từng đợt sụt giảm.", BODY_STYLE,
    ))
    df_ddp = load("drawdown_periods.csv")
    cw_ddp = [2.2*cm, 2.5*cm, 2.5*cm, 2.5*cm,
               2.2*cm, 2.5*cm, 2.5*cm, 2.5*cm]
    story.append(df_to_table(df_ddp, col_widths=cw_ddp, font_size=7.5))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 – SENSITIVITY ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    section(story, "5. Phân tích Độ nhạy (One-at-a-time)", 1)
    story.append(Paragraph(
        "Mỗi tham số thay đổi độc lập, các tham số còn lại giữ nguyên giá trị "
        "cơ sở (lookback=36, max_weight=30%, k=3, rf=VN10Y động). "
        "Chỉ số BL Sharpe được dùng để đánh giá mức độ nhạy cảm.", BODY_STYLE,
    ))
    df_sen = load("sensitivity_results.csv")

    for param in df_sen["Parameter"].unique():
        section(story, f"Tham số: {param}", 2)
        sub = df_sen[df_sen["Parameter"] == param].drop(columns=["Parameter"])
        cw  = [2.5*cm] + [2.4*cm] * (len(sub.columns) - 1)
        story.append(df_to_table(sub, col_widths=cw, font_size=7.5))
        story.append(Spacer(1, 0.2*cm))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 – MONTE CARLO SENSITIVITY
    # ════════════════════════════════════════════════════════════════════════
    section(story, "6. Monte Carlo – Độ bền vững Sharpe trước sai số dự báo", 1)
    story.append(Paragraph(
        "Với mỗi mức nhiễu δ ∈ {-10%, -5%, 0%, +5%, +10%}, view vector q "
        "được nhiễu hoá 2000 lần theo phân phối chuẩn. "
        "μ_BL được tính closed-form (tuyến tính trong q), "
        "Sharpe của w_BL cố định được đánh giá dưới mỗi q nhiễu. "
        "Cột TAN là Sharpe cố định (không dùng views).", BODY_STYLE,
    ))
    df_mc = load("monte_carlo_v2.csv")
    # Summarise: mean Expected_Sharpe by (Delta_Noise, Port_Type)
    mc_pivot = (df_mc.groupby(["Delta_Noise", "Port_Type"])["Expected_Sharpe"]
                .agg(["mean", "std", "min", "max"])
                .round(4)
                .reset_index())
    mc_pivot.columns = ["Delta_Noise", "Port_Type",
                        "Mean_Sharpe", "Std_Sharpe", "Min_Sharpe", "Max_Sharpe"]
    story.append(df_to_table(mc_pivot,
                              col_widths=[3*cm, 2.5*cm, 3*cm, 3*cm, 3*cm, 3*cm],
                              font_size=8))

    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Full Monte Carlo data (first 60 rows shown):", H2_STYLE))
    df_mc_head = df_mc.head(60)
    story.append(df_to_table(df_mc_head,
                              col_widths=[3*cm, 2.8*cm, 2.5*cm, 3.5*cm],
                              font_size=7))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7 – OOS RETURNS SERIES
    # ════════════════════════════════════════════════════════════════════════
    section(story, "7. Chuỗi Lợi nhuận Out-of-Sample (106 tháng)", 1)
    story.append(Paragraph(
        "Lợi nhuận tháng (simple return) của 6 danh mục và lãi suất phi rủi ro "
        "cho từng tháng OOS.", BODY_STYLE,
    ))
    df_oos = load("oos_returns_v2.csv")
    n_cols = len(df_oos.columns)
    cw_oos = [2.8*cm] + [2.2*cm] * (n_cols - 1)
    story.append(df_to_table(df_oos, col_widths=cw_oos, font_size=7.5))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8 – CLUSTER SIGNALS
    # ════════════════════════════════════════════════════════════════════════
    section(story, "8. Tín hiệu Phân cụm K-means", 1)
    story.append(Paragraph(
        "Nhãn cụm và vị thế view (Long / Short / Neutral) của từng cổ phiếu "
        "tại mỗi kỳ tái cân bằng. Phân cụm dựa trên (lợi nhuận năm hoá, "
        "biến động năm hoá) được chuẩn hoá trước khi đưa vào K-means (k=3).", BODY_STYLE,
    ))
    df_cl = load("cluster_signals_v2.csv")
    story.append(Paragraph(f"Tổng số quan sát: {len(df_cl):,} dòng", SMALL_STYLE))
    story.append(Spacer(1, 0.15*cm))
    story.append(df_to_table(df_cl,
                              col_widths=[2.8*cm, 2.2*cm, 3*cm, 4*cm],
                              font_size=7))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 9 – WEIGHTS HISTORY
    # ════════════════════════════════════════════════════════════════════════
    section(story, "9. Lịch sử Tỷ trọng Danh mục", 1)
    story.append(Paragraph(
        "Tỷ trọng của 6 danh mục cho từng cổ phiếu tại mỗi kỳ tái cân bằng. "
        "w_mkt = vốn hóa thị trường; w_TAN = Tangency; w_MV = Min-Variance; "
        "w_BL = Black-Litterman; w_EW = Equal-Weight; w_RP = Risk-Parity.", BODY_STYLE,
    ))
    df_wt = load("weights_v2.csv")
    story.append(Paragraph(f"Tổng số quan sát: {len(df_wt):,} dòng", SMALL_STYLE))
    story.append(Spacer(1, 0.15*cm))
    cw_wt = [2.8*cm, 2*cm] + [2.2*cm] * (len(df_wt.columns) - 2)
    story.append(df_to_table(df_wt, col_widths=cw_wt, font_size=6.5))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # APPENDIX – RF RATE TABLE
    # ════════════════════════════════════════════════════════════════════════
    section(story, "Phụ lục – Lãi suất Phi rủi ro VN10Y theo Năm", 1)
    story.append(Paragraph(
        "Lãi suất trái phiếu chính phủ Việt Nam kỳ hạn 10 năm (VN10Y) "
        "được dùng làm lãi suất phi rủi ro. "
        "Lãi suất tháng = VN10Y_năm / 12.", BODY_STYLE,
    ))
    rf_data = [
        ["Năm", "VN10Y (% năm)", "RF tháng (%)"],
        ["2014", "6.50", "0.5417"],
        ["2015", "6.50", "0.5417"],
        ["2016", "6.20", "0.5167"],
        ["2017", "5.80", "0.4833"],
        ["2018", "4.58", "0.3817"],
        ["2019", "4.58", "0.3817"],
        ["2020", "3.00", "0.2500"],
        ["2021", "2.53", "0.2108"],
        ["2022", "4.00", "0.3333"],
        ["2023", "3.50", "0.2917"],
        ["2024", "3.06", "0.2550"],
        ["2025", "3.10", "0.2583"],
        ["2026", "4.35", "0.3625"],
    ]
    rf_tbl = Table(rf_data, colWidths=[3*cm, 4*cm, 4*cm], hAlign="LEFT")
    rf_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#90a4ae")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_ALT]),
    ]))
    story.append(rf_tbl)

    # ── Build ────────────────────────────────────────────────────────────────
    doc.build(story)
    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"[✓] PDF saved: {PDF_PATH}  ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    print("Building PDF report...")
    build_pdf()
