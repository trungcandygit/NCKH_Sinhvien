"""
export_pdf.py
=============
Đọc tất cả output CSV và xuất thành một file PDF tổng hợp
dùng để gửi NotebookLM hoặc chia sẻ kết quả nghiên cứu.

Font: FreeSans TTF (hỗ trợ đầy đủ tiếng Việt, Unicode)
Style: đen trắng, không màu nền
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
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Đăng ký font TTF hỗ trợ tiếng Việt ────────────────────────────────────
_FONT_DIR = "/usr/share/fonts/truetype/freefont"
pdfmetrics.registerFont(TTFont("FreeSans",       f"{_FONT_DIR}/FreeSans.ttf"))
pdfmetrics.registerFont(TTFont("FreeSansBold",   f"{_FONT_DIR}/FreeSansBold.ttf"))
pdfmetrics.registerFont(TTFont("FreeSerif",      f"{_FONT_DIR}/FreeSerif.ttf"))
pdfmetrics.registerFont(TTFont("FreeSerifBold",  f"{_FONT_DIR}/FreeSerifBold.ttf"))
pdfmetrics.registerFont(TTFont("FreeMono",       f"{_FONT_DIR}/FreeMono.ttf"))

OUTPUT_DIR = "output"
PDF_PATH   = os.path.join(OUTPUT_DIR, "research_results_full.pdf")

# ── Màu đen trắng ──────────────────────────────────────────────────────────
BLACK  = colors.black
WHITE  = colors.white
LGRAY  = colors.HexColor("#e8e8e8")   # nền header nhạt
MGRAY  = colors.HexColor("#cccccc")   # border
DGRAY  = colors.HexColor("#555555")   # text phụ

styles = getSampleStyleSheet()

# ── Paragraph styles (FreeSans, đen trắng) ─────────────────────────────────
TITLE_STYLE = ParagraphStyle(
    "PTitle", fontName="FreeSerifBold", fontSize=18,
    textColor=BLACK, spaceAfter=6, alignment=TA_CENTER,
)
SUBTITLE_STYLE = ParagraphStyle(
    "PSub", fontName="FreeSerif", fontSize=11,
    textColor=DGRAY, spaceAfter=4, alignment=TA_CENTER,
)
H1_STYLE = ParagraphStyle(
    "PH1", fontName="FreeSansBold", fontSize=12,
    textColor=BLACK, spaceBefore=12, spaceAfter=4,
)
H2_STYLE = ParagraphStyle(
    "PH2", fontName="FreeSansBold", fontSize=10,
    textColor=DGRAY, spaceBefore=6, spaceAfter=3,
)
BODY_STYLE = ParagraphStyle(
    "PBody", fontName="FreeSans", fontSize=9,
    leading=14, spaceAfter=4, textColor=BLACK,
)
SMALL_STYLE = ParagraphStyle(
    "PSmall", fontName="FreeSans", fontSize=8,
    leading=11, textColor=DGRAY,
)
CODE_STYLE = ParagraphStyle(
    "PCode", fontName="FreeMono", fontSize=8,
    leading=11, spaceAfter=3,
)


# ── Helper: DataFrame → Table đen trắng ────────────────────────────────────

def df_to_table(df: pd.DataFrame, col_widths=None,
                font_size=7.5, max_col_w=5.5*cm) -> Table:
    header = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            if abs(v) >= 100: return f"{v:,.1f}"
            if abs(v) >= 1:   return f"{v:.4f}"
            return f"{v:.6f}"
        return str(v) if v is not None else ""

    rows = [header] + [[fmt(c) for c in row]
                        for row in df.itertuples(index=False, name=None)]

    if col_widths is None:
        page_w = landscape(A4)[0] - 2*cm
        n = len(header)
        col_widths = [min(max_col_w, page_w / n)] * n

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)

    row_cmds = []
    for i in range(1, len(rows)):
        if i % 2 == 0:
            row_cmds.append(("BACKGROUND", (0, i), (-1, i), LGRAY))

    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0),  LGRAY),
        ("FONTNAME",     (0, 0), (-1, 0),  "FreeSansBold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  font_size),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  BLACK),
        ("ALIGN",        (0, 0), (-1, 0),  "CENTER"),
        ("LINEBELOW",    (0, 0), (-1, 0),  1.0, BLACK),
        # Data rows
        ("FONTNAME",     (0, 1), (-1, -1), "FreeSans"),
        ("FONTSIZE",     (0, 1), (-1, -1), font_size),
        ("TEXTCOLOR",    (0, 1), (-1, -1), BLACK),
        ("ALIGN",        (0, 1), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID",         (0, 0), (-1, -1), 0.4, MGRAY),
        *row_cmds,
    ]))
    return tbl


def section(story, title, level=1):
    style = H1_STYLE if level == 1 else H2_STYLE
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(title, style))
    if level == 1:
        story.append(HRFlowable(width="100%", thickness=0.8,
                                color=BLACK, spaceAfter=4))


def load(filename) -> pd.DataFrame:
    return pd.read_csv(os.path.join(OUTPUT_DIR, filename))


# ── Trang bìa ───────────────────────────────────────────────────────────────

def build_cover(story):
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph(
        "TỐI ƯU HÓA DANH MỤC ĐẦU TƯ CỔ PHIẾU NGÂN HÀNG VIỆT NAM",
        TITLE_STYLE))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Sử dụng Mô hình Black-Litterman kết hợp Phân cụm K-means\n"
        "và Tín hiệu Momentum Tương đối (Idiosyncratic Momentum)",
        SUBTITLE_STYLE))
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="80%", thickness=1, color=BLACK,
                             hAlign="CENTER", spaceAfter=10))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Black-Litterman Portfolio Optimization with K-means Clustering\n"
        "and Idiosyncratic Momentum Signals:\n"
        "Evidence from Vietnamese Banking Stocks",
        SUBTITLE_STYLE))
    story.append(Spacer(1, 1*cm))

    # Bảng thông tin tóm tắt
    info = [
        ["Dữ liệu", "25 cổ phiếu ngân hàng niêm yết HOSE/HNX, tháng 6/2014 – tháng 5/2026"],
        ["Phương pháp", "K-means (k=4) + Black-Litterman + Rolling window 36 tháng"],
        ["Tín hiệu", "Idiosyncratic momentum 6-1 tháng + Low-vol composite"],
        ["Kết quả chính", "BL Sharpe = 0.9396 > EW (1/N) Sharpe = 0.8373"],
        ["Drawdown tối đa", "BL: -30.7%  vs  EW: -36.6%  vs  TAN: -51.5%"],
        ["Tham chiếu chính", "Black & Litterman (1992), Jegadeesh & Titman (1993),\n"
                             "DeMiguel et al. (2009), Baker et al. (2011)"],
    ]
    tbl = Table(info, colWidths=[4*cm, 12*cm])
    tbl.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (0, -1),  "FreeSansBold"),
        ("FONTNAME",     (1, 0), (1, -1),  "FreeSans"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",    (0, 0), (-1, -1), BLACK),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",         (0, 0), (-1, -1), 0.4, MGRAY),
        ("ROWBACKGROUNDS",(0,0), (-1,-1),  [WHITE, LGRAY]),
    ]))
    story.append(tbl)
    story.append(PageBreak())


# ── Build chính ─────────────────────────────────────────────────────────────

def build_report():
    print("Building PDF report...")
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=landscape(A4),
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
        title="Tối ưu hóa danh mục cổ phiếu ngân hàng Việt Nam – Black-Litterman + K-means",
        author="Research",
    )
    story = []

    build_cover(story)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 – HIỆU SUẤT TỔNG HỢP
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "1. Tóm tắt Hiệu suất (6 Danh mục, OOS annualised)")
    story.append(Paragraph(
        "So sánh 6 danh mục đầu tư theo 7 chỉ số rủi ro-lợi nhuận. "
        "BL = Black-Litterman (mô hình đề xuất), EW = Equal-Weight (chuẩn 1/N của DeMiguel et al. 2009), "
        "MKT = Market-cap, TAN = Tangency, MV = Min-Variance, RP = Risk-Parity. "
        "Kết quả OOS trên 106 tháng (tháng 8/2017 – tháng 5/2026), "
        "lãi suất phi rủi ro = VN10Y/12 (time-varying).", BODY_STYLE))
    df_perf = load("performance_summary_v2.csv")
    story.append(Spacer(1, 0.15*cm))
    story.append(df_to_table(df_perf, font_size=9))

    story.append(Spacer(1, 0.4*cm))
    section(story, "1b. Kiểm định t-test (BL vs các danh mục khác)", level=2)
    story.append(Paragraph(
        "Kiểm định t-test theo cặp (paired t-test) cho lợi nhuận tháng OOS. "
        "H₀: trung bình lợi nhuận BL = trung bình danh mục so sánh.", BODY_STYLE))
    df_tt = load("ttest_results_v2.csv")
    story.append(df_to_table(df_tt, font_size=9,
                              col_widths=[4*cm, 4*cm, 4*cm, 4*cm, 4*cm]))

    story.append(Spacer(1, 0.4*cm))
    section(story, "1c. Kiểm định Jobson-Korkie (so sánh Sharpe ratio)", level=2)
    story.append(Paragraph(
        "Kiểm định Jobson-Korkie (1981): H₀: SR_BL = SR_danh_mục_khác. "
        "z-statistic dương = BL có Sharpe cao hơn. p < 0.05 = có ý nghĩa thống kê.", BODY_STYLE))
    df_jk = load("jobson_korkie_results.csv")
    story.append(df_to_table(df_jk, font_size=9,
                              col_widths=[4*cm, 4*cm, 4*cm, 5*cm]))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 – SUB-PERIOD ROBUSTNESS
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "2. Phân tích Tính vững theo Giai đoạn (Sub-period Robustness)")
    story.append(Paragraph(
        "Chia giai đoạn OOS làm hai nửa bằng nhau để kiểm tra tính vững của kết quả. "
        "Period 1 (2017–2021): giai đoạn tăng trưởng mạnh, thị trường bull. "
        "Period 2 (2022–2026): giai đoạn biến động cao (hậu COVID, lãi suất tăng, "
        "thị trường điều chỉnh). BL thắng EW rõ ràng ở Period 2 "
        "(Sharpe 0.86 vs 0.38), xác nhận mô hình hoạt động tốt nhất khi cần nhất.", BODY_STYLE))
    df_sp = load("subperiod_analysis.csv")
    story.append(df_to_table(df_sp, font_size=8.5))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 – SIGNAL IC ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "3. Phân tích Information Coefficient (IC) của Tín hiệu K-means")
    story.append(Paragraph(
        "IC (Spearman rank correlation) đo mức độ tín hiệu idiosyncratic momentum "
        "dự báo đúng chiều lợi nhuận thực tế tháng kế tiếp. "
        "IC > 0: tín hiệu xếp hạng đúng chiều. "
        "Hit rate: tỷ lệ bước rebalancing mà cluster Long thực sự vượt cluster Short. "
        "Cả hai chỉ số > 50% xác nhận tín hiệu có giá trị dự báo dương.", BODY_STYLE))
    df_ic = load("signal_ic_analysis.csv")
    ic_sum = pd.DataFrame([
        {"Chỉ số": "Mean IC",        "Giá trị": f"{df_ic['IC'].mean():+.4f}"},
        {"Chỉ số": "IC > 0 (%)",     "Giá trị": f"{(df_ic['IC']>0).mean():.1%}"},
        {"Chỉ số": "Hit Rate (%)",   "Giá trị": f"{df_ic['Hit'].mean():.1%}"},
        {"Chỉ số": "Std IC",         "Giá trị": f"{df_ic['IC'].std():.4f}"},
        {"Chỉ số": "Số bước OOS",    "Giá trị": str(len(df_ic))},
    ])
    story.append(df_to_table(ic_sum, col_widths=[6*cm, 4*cm], font_size=9))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("IC theo từng bước rebalancing:", H2_STYLE))
    story.append(df_to_table(df_ic, col_widths=[3.5*cm, 3*cm, 2.5*cm, 3*cm],
                              font_size=7.5))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 – K VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "4. Kiểm định Số cụm K tối ưu (Elbow + Silhouette)")
    story.append(Paragraph(
        "Xác định k tối ưu cho K-means bằng hai phương pháp độc lập: "
        "(1) Elbow method: inertia (tổng bình phương trong cụm) giảm và tìm 'điểm gãy'; "
        "(2) Silhouette score (Rousseeuw 1987): đo mức độ tách biệt giữa các cụm "
        "(cao hơn = tách biệt tốt hơn). "
        "Sensitivity analysis OOS xác nhận k=4 tối ưu về Sharpe ratio "
        "(BL Sharpe 0.9396, cao nhất trong các k được test với cùng mẫu).", BODY_STYLE))
    df_ks = load("kmeans_k_summary.csv")
    story.append(df_to_table(df_ks, col_widths=[2.5*cm, 5*cm, 5.5*cm, 5.5*cm],
                              font_size=9))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 – SENSITIVITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "5. Phân tích Độ nhạy (One-at-a-time Parameter Sweep)")
    story.append(Paragraph(
        "Thay đổi một tham số tại một thời điểm trong khi giữ nguyên các tham số còn lại. "
        "Baseline: lookback=36, max_weight=30%, k=4, rf=VN10Y dynamic. "
        "Kết quả cho thấy BL ổn định qua các giá trị tham số hợp lý.", BODY_STYLE))
    df_sens = load("sensitivity_results.csv")

    for param in df_sens["Parameter"].unique():
        sub = df_sens[df_sens["Parameter"] == param]
        bl_sub = sub[sub["Portfolio"] == "BL"][
            ["Value", "Is_Baseline", "Ann_Return", "Ann_Vol", "Sharpe", "MDD"]
        ].copy()
        bl_sub["Is_Baseline"] = bl_sub["Is_Baseline"].apply(
            lambda x: "★" if x else "")
        story.append(Paragraph(f"Tham số: {param}", H2_STYLE))
        story.append(df_to_table(bl_sub, font_size=8.5,
                                  col_widths=[3.5*cm, 2.5*cm, 3.5*cm,
                                              3.5*cm, 3.5*cm, 3.5*cm]))
        story.append(Spacer(1, 0.2*cm))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 – DRAWDOWN ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "6. Phân tích Drawdown Chi tiết")
    section(story, "6a. Tóm tắt Drawdown theo Danh mục", level=2)
    story.append(df_to_table(load("drawdown_summary.csv"), font_size=8.5))

    story.append(Spacer(1, 0.3*cm))
    section(story, "6b. Các sự kiện Drawdown từng Danh mục", level=2)
    story.append(Paragraph(
        "Liệt kê từng đợt drawdown: ngày bắt đầu, đáy, phục hồi, "
        "mức giảm tối đa, số tháng phục hồi.", BODY_STYLE))
    df_dp = load("drawdown_periods.csv")
    story.append(df_to_table(df_dp, font_size=7.5))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 – DISTRIBUTION TESTS
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "7. Kiểm định Phân phối Lợi nhuận")
    section(story, "7a. Kiểm định Jarque-Bera (Chuẩn hóa)", level=2)
    story.append(Paragraph(
        "H₀: lợi nhuận có phân phối chuẩn. p < 0.05 = bác bỏ chuẩn hóa. "
        "Hầu hết danh mục cổ phiếu đều có fat tail và skewness.", BODY_STYLE))
    story.append(df_to_table(load("distribution_jb_tests.csv"), font_size=8.5))

    story.append(Spacer(1, 0.3*cm))
    section(story, "7b. Kiểm định Ljung-Box (Tự tương quan)", level=2)
    story.append(Paragraph(
        "H₀: lợi nhuận không có tự tương quan (white noise). "
        "p < 0.05 = tồn tại tự tương quan (có thể khai thác được).", BODY_STYLE))
    story.append(df_to_table(load("distribution_lb_tests.csv"), font_size=8.5))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 – MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "8. Monte Carlo Robustness (Nhiễu trong View BL)")
    story.append(Paragraph(
        "Kiểm tra tính ổn định khi thêm nhiễu Gaussian vào vector view q của BL. "
        "Delta Noise từ -10% đến +10%. Mô hình BL khai thác tính tuyến tính "
        "μ_BL(q') = base_vec + A_mat @ q' để tránh tối ưu lại 2000 lần mỗi bước.", BODY_STYLE))
    df_mc = load("monte_carlo_v2.csv")
    mc_pivot = (df_mc.groupby(["Delta_Noise", "Port_Type"])["Expected_Sharpe"]
                .agg(["mean", "std"])
                .round(4).reset_index())
    mc_pivot.columns = ["Delta_Noise", "Port_Type", "Mean_Sharpe", "Std_Sharpe"]
    story.append(df_to_table(mc_pivot,
                              col_widths=[3.5*cm, 3*cm, 4*cm, 4*cm],
                              font_size=8.5))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 – OOS RETURNS SERIES
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "9. Chuỗi Lợi nhuận Out-of-Sample (106 tháng)")
    story.append(Paragraph(
        "Lợi nhuận tháng (simple return) của 6 danh mục và lãi suất phi rủi ro "
        "cho từng tháng OOS (tháng 8/2017 – tháng 5/2026).", BODY_STYLE))
    df_oos = load("oos_returns_v2.csv")
    n_cols = len(df_oos.columns)
    cw_oos = [3*cm] + [2.3*cm] * (n_cols - 1)
    story.append(df_to_table(df_oos, col_widths=cw_oos, font_size=7.5))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10 – CLUSTER SIGNALS
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "10. Tín hiệu Phân cụm K-means (k=4)")
    story.append(Paragraph(
        "Nhãn cụm và vị thế view (Long / Short / Neutral) của từng cổ phiếu "
        "tại mỗi kỳ tái cân bằng. Phân cụm dựa trên tín hiệu tổng hợp: "
        "idiosyncratic momentum 6-1 tháng (loại bỏ yếu tố thị trường) + "
        "low-volatility feature (âm hóa vol). K-means (k=4) trên không gian "
        "chuẩn hóa 2 chiều.", BODY_STYLE))
    df_cl = load("cluster_signals_v2.csv")
    story.append(Paragraph(f"Tổng số quan sát: {len(df_cl):,} dòng", SMALL_STYLE))
    story.append(Spacer(1, 0.15*cm))
    story.append(df_to_table(df_cl,
                              col_widths=[3*cm, 2.5*cm, 3*cm, 4.5*cm],
                              font_size=7))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 11 – WEIGHTS HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "11. Lịch sử Tỷ trọng Danh mục")
    story.append(Paragraph(
        "Tỷ trọng của 6 danh mục cho từng cổ phiếu tại mỗi kỳ tái cân bằng. "
        "w_mkt = market-cap; w_TAN = Tangency; w_MV = Min-Variance; "
        "w_BL = Black-Litterman; w_EW = Equal-Weight (1/N); w_RP = Risk-Parity.", BODY_STYLE))
    df_wt = load("weights_v2.csv")
    story.append(Paragraph(f"Tổng số quan sát: {len(df_wt):,} dòng", SMALL_STYLE))
    story.append(Spacer(1, 0.15*cm))
    cw_wt = [3*cm, 2.2*cm] + [2.5*cm] * (len(df_wt.columns) - 2)
    story.append(df_to_table(df_wt, col_widths=cw_wt, font_size=6.5))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PHỤ LỤC – LÃI SUẤT PHI RỦI RO & PHƯƠNG PHÁP LUẬN
    # ══════════════════════════════════════════════════════════════════════════
    section(story, "Phụ lục A – Lãi suất Phi rủi ro VN10Y")
    story.append(Paragraph(
        "Lãi suất trái phiếu chính phủ Việt Nam kỳ hạn 10 năm (VN10Y) "
        "dùng làm lãi suất phi rủi ro time-varying. "
        "RF tháng = VN10Y_năm / 100 / 12.", BODY_STYLE))
    rf_data = [
        ["Năm", "VN10Y (%/năm)", "RF tháng (%)"],
        ["2014", "6.50", "0.5417"], ["2015", "6.50", "0.5417"],
        ["2016", "6.20", "0.5167"], ["2017", "5.80", "0.4833"],
        ["2018", "4.58", "0.3817"], ["2019", "4.58", "0.3817"],
        ["2020", "3.00", "0.2500"], ["2021", "2.53", "0.2108"],
        ["2022", "4.00", "0.3333"], ["2023", "3.50", "0.2917"],
        ["2024", "3.06", "0.2550"], ["2025", "3.10", "0.2583"],
        ["2026", "4.35", "0.3625"],
    ]
    rf_tbl = Table(rf_data, colWidths=[3.5*cm, 5*cm, 5*cm], hAlign="LEFT")
    rf_tbl.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1, 0),  "FreeSansBold"),
        ("FONTNAME",     (0, 1), (-1, -1), "FreeSans"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",    (0, 0), (-1, -1), BLACK),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("LINEBELOW",    (0, 0), (-1, 0),  1.0, BLACK),
        ("GRID",         (0, 0), (-1, -1), 0.4, MGRAY),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),  [WHITE, LGRAY]),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(rf_tbl)

    story.append(Spacer(1, 0.6*cm))
    section(story, "Phụ lục B – Tóm tắt Phương pháp luận", level=2)
    method_lines = [
        "• Dữ liệu: Giá đóng cửa tháng và vốn hóa thị trường của 25 cổ phiếu ngân hàng "
        "niêm yết (HOSE/HNX), giai đoạn 06/2014 – 05/2026 (144 tháng).",
        "• Cửa sổ trượt (rolling window): 36 tháng, tái cân bằng hàng tháng.",
        "• Phân cụm K-means (k=4): feature = (idiosyncratic momentum 6-1 tháng, −vol). "
        "Tín hiệu tổng hợp theo Jegadeesh & Titman (1993) và Baker, Bradley & Wurgler (2011).",
        "• Black-Litterman: Π = δ·Σ·w_mkt (Reverse-CAPM); Ω = τ·P·Σ·Pᵀ; "
        "μ_BL = M⁻¹[(τΣ)⁻¹Π + Pᵀ·Ω⁻¹·q]; Σ_BL = M⁻¹ + Σ; τ = 1/36.",
        "• Tối ưu hóa: SLSQP maximise Sharpe(μ_BL, Σ_BL), Σw=1, 0≤w≤30%, 10 điểm xuất phát.",
        "• Đánh giá: OOS 106 tháng; Sharpe, Sortino, MDD, Calmar, Win-rate.",
        "• Kiểm định: Paired t-test, Jobson-Korkie (1981), Jarque-Bera, Ljung-Box.",
        "• Monte Carlo: 2000 mô phỏng nhiễu view q (khai thác tính tuyến tính của BL).",
    ]
    for line in method_lines:
        story.append(Paragraph(line, BODY_STYLE))

    # ── Build ────────────────────────────────────────────────────────────────
    doc.build(story)
    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"[✓] PDF saved: {PDF_PATH}  ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    build_report()
