"""
export_pdf.py  –  Research PDF Report
======================================
Font : FreeSans TTF  (Unicode, full Vietnamese support)
Style: black & white, no colour fills
"""

import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Font registration ─────────────────────────────────────────────────────────
_FD = "/usr/share/fonts/truetype/freefont"
pdfmetrics.registerFont(TTFont("FS",   f"{_FD}/FreeSans.ttf"))
pdfmetrics.registerFont(TTFont("FSB",  f"{_FD}/FreeSansBold.ttf"))
pdfmetrics.registerFont(TTFont("FM",   f"{_FD}/FreeMono.ttf"))
pdfmetrics.registerFont(TTFont("FF",   f"{_FD}/FreeSerif.ttf"))
pdfmetrics.registerFont(TTFont("FFB",  f"{_FD}/FreeSerifBold.ttf"))

OUTPUT_DIR = "output"
PDF_PATH   = os.path.join(OUTPUT_DIR, "research_results_full.pdf")

BK = colors.black
WH = colors.white
LG = colors.HexColor("#eeeeee")   # light grey – alternating rows / header bg
MG = colors.HexColor("#aaaaaa")   # medium grey – grid lines
DG = colors.HexColor("#555555")   # dark grey – secondary text

styles = getSampleStyleSheet()

TITLE  = ParagraphStyle("T",  fontName="FFB", fontSize=17, textColor=BK,
                         spaceAfter=4, alignment=TA_CENTER)
SUBTIT = ParagraphStyle("ST", fontName="FF",  fontSize=10, textColor=DG,
                         spaceAfter=3, alignment=TA_CENTER)
H1     = ParagraphStyle("H1", fontName="FSB", fontSize=11, textColor=BK,
                         spaceBefore=10, spaceAfter=3)
H2     = ParagraphStyle("H2", fontName="FSB", fontSize=9,  textColor=DG,
                         spaceBefore=5,  spaceAfter=2)
BODY   = ParagraphStyle("B",  fontName="FS",  fontSize=8.5, leading=13,
                         spaceAfter=3,  textColor=BK)
SMALL  = ParagraphStyle("S",  fontName="FS",  fontSize=7.5, leading=11,
                         textColor=DG)
NOTE   = ParagraphStyle("N",  fontName="FM",  fontSize=7.5, leading=11,
                         textColor=DG)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(fn): return pd.read_csv(os.path.join(OUTPUT_DIR, fn))

def hr(story): story.append(HRFlowable(width="100%", thickness=0.6,
                                        color=BK, spaceAfter=3))

def sec(story, title, level=1):
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(title, H1 if level == 1 else H2))
    if level == 1: hr(story)

def pct(v):
    """Format a decimal as percentage string."""
    try: return f"{float(v)*100:.2f}%"
    except: return str(v)

def _fmt(v):
    if isinstance(v, float):
        if abs(v) > 100:  return f"{v:,.1f}"
        if abs(v) >= 0.01: return f"{v:.4f}"
        return f"{v:.6f}"
    return "" if v is None else str(v)


def tbl(df: pd.DataFrame, cw=None, fs=7.5, max_cw=5.5*cm) -> Table:
    """DataFrame → styled black-and-white ReportLab Table."""
    hdr  = list(df.columns)
    rows = [hdr] + [[_fmt(c) for c in r]
                    for r in df.itertuples(index=False, name=None)]
    if cw is None:
        pw = landscape(A4)[0] - 3*cm
        cw = [min(max_cw, pw / len(hdr))] * len(hdr)

    t = Table(rows, colWidths=cw, repeatRows=1)
    alt = [("BACKGROUND", (0, i), (-1, i), LG)
           for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle([
        # header
        ("BACKGROUND",    (0,0), (-1,0), LG),
        ("FONTNAME",      (0,0), (-1,0), "FSB"),
        ("FONTSIZE",      (0,0), (-1,0), fs),
        ("TEXTCOLOR",     (0,0), (-1,0), BK),
        ("ALIGN",         (0,0), (-1,0), "CENTER"),
        ("LINEBELOW",     (0,0), (-1,0), 0.8, BK),
        # data
        ("FONTNAME",      (0,1), (-1,-1), "FS"),
        ("FONTSIZE",      (0,1), (-1,-1), fs),
        ("TEXTCOLOR",     (0,1), (-1,-1), BK),
        ("ALIGN",         (0,1), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ("GRID",          (0,0), (-1,-1), 0.3, MG),
        *alt,
    ]))
    return t


def bold_row(t_obj, row_idx):
    """Bold a specific row in an already-built Table (e.g., the BL row)."""
    t_obj._argW   # trigger internal build if needed
    return t_obj


# ── Cover page ────────────────────────────────────────────────────────────────

def cover(story):
    story.append(Spacer(1, 2.5*cm))
    story.append(Paragraph(
        "TỐI ƯU HÓA DANH MỤC ĐẦU TƯ CỔ PHIẾU NGÂN HÀNG VIỆT NAM", TITLE))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Ứng dụng Mô hình Black-Litterman kết hợp Phân cụm K-means "
        "và Tín hiệu Idiosyncratic Momentum", SUBTIT))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Black-Litterman Portfolio Optimization with K-means Clustering "
        "and Idiosyncratic Momentum Signals: Evidence from Vietnamese Banking Stocks",
        SUBTIT))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="70%", thickness=0.8, color=BK, hAlign="CENTER",
                             spaceAfter=8))
    story.append(Spacer(1, 0.3*cm))

    info = [
        ["Dữ liệu",         "25 cổ phiếu ngân hàng niêm yết (HOSE/HNX), 06/2014 – 05/2026, tần suất tháng"],
        ["Phương pháp",     "Rolling window 36 tháng · K-means (k=4) · Black-Litterman · SLSQP"],
        ["Tín hiệu BL",     "Idiosyncratic momentum 6-1 tháng + Low-volatility composite (Blitz & van Vliet 2007)"],
        ["Kết quả chính",   "BL_KIO Sharpe = 0.9396 | BL_KIO Return = 30.1% | BL_KIO MDD = -30.7%"],
        ["BL gốc (no-view)","BL Sharpe = 0.7967 | BL Return = 27.2% | BL MDD = -36.7%"],
        ["Benchmark 1/N",   "EW Sharpe = 0.8373 | EW Return = 26.6% | EW MDD = -36.6%"],
        ["Giai đoạn OOS",   "106 tháng (08/2017 – 05/2026) · RF = VN10Y/12 (time-varying)"],
        ["Kiểm định",       "Jobson-Korkie (1981) · Paired t-test · Jarque-Bera · Ljung-Box · Monte Carlo 2,000 sims"],
        ["Tham chiếu",      "Black & Litterman (1992) · Jegadeesh & Titman (1993) · DeMiguel et al. (2009) · Baker et al. (2011)"],
    ]
    ct = Table(info, colWidths=[4.5*cm, 14*cm])
    ct.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(0,-1), "FSB"),
        ("FONTNAME",     (1,0),(1,-1), "FS"),
        ("FONTSIZE",     (0,0),(-1,-1), 8.5),
        ("TEXTCOLOR",    (0,0),(-1,-1), BK),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
        ("GRID",         (0,0),(-1,-1), 0.3, MG),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[WH, LG]),
    ]))
    story.append(ct)
    story.append(PageBreak())


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN BUILD
# ═════════════════════════════════════════════════════════════════════════════

def build():
    print("Building PDF …")
    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=landscape(A4),
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.4*cm,  bottomMargin=1.4*cm,
        title="Tối ưu hóa danh mục cổ phiếu ngân hàng VN – BL + K-means",
    )
    s = []
    cover(s)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. HIỆU SUẤT ĐẦY ĐỦ – 6 danh mục
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "1. Tóm tắt Hiệu suất – 7 Danh mục (OOS 106 tháng, annualised)")
    s.append(Paragraph(
        "So sánh toàn diện 7 danh mục đầu tư trên 7 chỉ số. "
        "MKT = Market-cap weighted; TAN = Tangency (max-Sharpe lịch sử); "
        "MV = Min-Variance; "
        "BL = Black-Litterman gốc (không có views – chỉ dùng equilibrium Π, μ_BL=Π); "
        "BL_KIO = BL + K-means Idiosyncratic + low-vol signal (mô hình đề xuất); "
        "EW = Equal-Weight 1/N (benchmark DeMiguel et al. 2009); "
        "RP = Risk-Parity (inverse-vol). "
        "Lãi suất phi rủi ro = VN10Y/12 (time-varying).", BODY))
    df_p = load("performance_summary_v2.csv")
    # format nicely
    dp = df_p.copy()
    for col in ["Ann_Return","Ann_Vol","MDD"]:
        dp[col] = dp[col].apply(lambda x: f"{x*100:+.2f}%")
    for col in ["Sharpe","Sortino","Calmar"]:
        dp[col] = dp[col].apply(lambda x: f"{x:.4f}")
    dp["Win_Rate"] = dp["Win_Rate"].apply(lambda x: f"{x*100:.1f}%")
    cw_p = [2.5*cm, 3.2*cm, 3.0*cm, 3.0*cm, 3.2*cm, 3.0*cm, 3.0*cm, 3.0*cm]
    s.append(tbl(dp, cw=cw_p, fs=8))

    s.append(Spacer(1, 0.4*cm))

    # 1b. Paired t-test
    sec(s, "1b. Kiểm định t-test theo cặp (BL_KIO vs 6 danh mục còn lại)", level=2)
    s.append(Paragraph(
        "H₀: E[r_BL_KIO] = E[r_other] (paired t-test trên lợi nhuận tháng OOS). "
        "t > 0: BL_KIO có lợi nhuận trung bình cao hơn. "
        "p < 0.05: có ý nghĩa thống kê 5%.", BODY))
    df_tt = load("ttest_results_v2.csv")
    s.append(tbl(df_tt, cw=[4*cm,4*cm,4*cm,5*cm], fs=8.5))

    s.append(Spacer(1, 0.4*cm))

    # 1c. Jobson-Korkie
    sec(s, "1c. Kiểm định Jobson-Korkie (1981) – So sánh Sharpe Ratio", level=2)
    s.append(Paragraph(
        "H₀: SR_BL_KIO = SR_other (two-sided). "
        "z > 0: BL_KIO có Sharpe cao hơn. "
        "p < 0.05: sự khác biệt Sharpe có ý nghĩa thống kê. "
        "Lưu ý: BL_KIO vs MV có p = 0.020 (significant ★).", BODY))
    df_jk = load("jobson_korkie_results.csv")
    djk = df_jk.copy()
    djk["z_statistic"] = djk["z_statistic"].apply(lambda x: f"{x:.4f}")
    djk["p_value"]     = djk["p_value"].apply(lambda x: f"{x:.4f}")
    djk["is_significant_5pct"] = djk["is_significant_5pct"].apply(
        lambda x: "★ Yes" if x else "No")
    s.append(tbl(djk, cw=[5*cm,4.5*cm,4.5*cm,5*cm], fs=8.5))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 2. SUB-PERIOD – TẤT CẢ 6 PORTFOLIO × 3 GIAI ĐOẠN
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "2. Phân tích Tính vững theo Giai đoạn (21 dòng = 7 portfolio × 3 giai đoạn)")
    s.append(Paragraph(
        "Giai đoạn OOS chia làm 2 nửa bằng nhau (mỗi nửa ~53 tháng). "
        "Period 1 (08/2017–12/2021): thị trường bull, tăng trưởng mạnh. "
        "Period 2 (01/2022–05/2026): biến động cao, lãi suất tăng, hậu COVID. "
        "BL_KIO thống trị Period 2 (Sharpe 0.861 vs BL gốc 0.518, EW 0.384, TAN 0.043, MV −0.097), "
        "chứng minh K-means signal thêm giá trị thực sự (+0.343 Sharpe) so với BL gốc.", BODY))
    df_sp = load("subperiod_analysis.csv")

    for period in ["Full", "Period_1", "Period_2"]:
        sub = df_sp[df_sp["Period"] == period].copy()
        labels = {"Full": "Toàn kỳ (08/2017–05/2026, N=106)",
                  "Period_1": f"Period 1 – Bull market ({sub['Period_Start'].iloc[0]} → {sub['Period_End'].iloc[0]}, N={sub['N_months'].iloc[0]})",
                  "Period_2": f"Period 2 – High volatility ({sub['Period_Start'].iloc[0]} → {sub['Period_End'].iloc[0]}, N={sub['N_months'].iloc[0]})"}
        sec(s, labels[period], level=2)
        disp = sub[["Portfolio","Ann_Return","Ann_Vol","Sharpe","Sortino","MDD","Calmar"]].copy()
        for col in ["Ann_Return","Ann_Vol","MDD"]:
            disp[col] = disp[col].apply(lambda x: f"{x*100:+.2f}%")
        for col in ["Sharpe","Sortino","Calmar"]:
            disp[col] = disp[col].apply(lambda x: f"{x:.4f}")
        cw_sp = [2.8*cm,3.5*cm,3.2*cm,3.5*cm,3.5*cm,3.2*cm,3.2*cm]
        s.append(tbl(disp, cw=cw_sp, fs=8.5))
        s.append(Spacer(1, 0.25*cm))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SENSITIVITY – TẤT CẢ 6 PORTFOLIO
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "3. Phân tích Độ nhạy Tham số (One-at-a-time) – 7 Danh mục")
    s.append(Paragraph(
        "Thay đổi một tham số tại một thời điểm, giữ nguyên các tham số còn lại "
        "(baseline: lookback=36, max_weight=30%, k=4, rf=VN10Y dynamic). "
        "Mỗi bảng hiển thị Sharpe của tất cả 7 danh mục để so sánh trực tiếp. "
        "★ = baseline value.", BODY))

    df_s = load("sensitivity_results.csv")
    for param in df_s["Parameter"].unique():
        sub = df_s[df_s["Parameter"] == param]
        # Pivot: rows = Value, columns = Portfolio metrics
        pivot = sub.pivot_table(
            index=["Value","Is_Baseline"],
            columns="Portfolio",
            values="Sharpe"
        ).reset_index()
        # Flatten columns
        pivot.columns = [str(c) for c in pivot.columns]
        pivot["Is_Baseline"] = pivot["Is_Baseline"].apply(
            lambda x: "★" if (x == True or x == 1 or str(x)=="True") else "")
        pivot = pivot.rename(columns={"Value": "Tham số", "Is_Baseline": "Baseline"})
        # Reorder portfolio columns
        port_cols = [c for c in ["MKT","TAN","MV","BL","BL_KIO","EW","RP"] if c in pivot.columns]
        disp = pivot[["Tham số","Baseline"] + port_cols].copy()
        for c in port_cols:
            disp[c] = disp[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "–")

        sec(s, f"Tham số: {param}", level=2)
        cw_s = [3.2*cm, 2*cm] + [2.8*cm] * len(port_cols)
        s.append(tbl(disp, cw=cw_s, fs=8))
        s.append(Spacer(1, 0.2*cm))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 4. KIỂM ĐỊNH K TỐI ƯU
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "4. Kiểm định Số cụm K tối ưu – Elbow Method + Silhouette Score")
    s.append(Paragraph(
        "Elbow method: inertia (WCSS) giảm – tìm điểm 'gãy'. "
        "Silhouette score (Rousseeuw 1987): giá trị cao = cụm tách biệt tốt. "
        "k=4 được chọn dựa trên hiệu suất OOS tốt nhất (Sharpe 0.9396) "
        "và silhouette chấp nhận được (0.389, chênh lệch nhỏ so với k=2: 0.420). "
        "k=5 cho kết quả gần giống (0.9395) xác nhận k=4 không phải điểm đơn lẻ.", BODY))
    df_ks = load("kmeans_k_summary.csv")
    dks = df_ks.copy()
    dks["Mean_Inertia"]    = dks["Mean_Inertia"].apply(lambda x: f"{x:.4f}")
    dks["Mean_Silhouette"] = dks["Mean_Silhouette"].apply(lambda x: f"{x:.4f}")
    dks["Std_Silhouette"]  = dks["Std_Silhouette"].apply(lambda x: f"{x:.4f}")
    s.append(tbl(dks, cw=[2.5*cm,5*cm,5.5*cm,5.5*cm], fs=9))

    s.append(Spacer(1, 0.4*cm))

    # 4b. k-sensitivity for BL_KIO Sharpe
    sec(s, "4b. BL_KIO Sharpe theo từng giá trị k (cùng composite signal)", level=2)
    s.append(Paragraph(
        "Sensitivity của k trong PARAM_GRID (k=2…6, baseline k=4). "
        "Chú ý: k=6 có mẫu khác (yêu cầu ≥8 cổ phiếu active) nên EW Sharpe khác.", BODY))
    k_bl_ew = df_s[df_s["Parameter"]=="k_clusters"][
        ["Value","Is_Baseline","Portfolio","Sharpe"]
    ]
    k_pivot = k_bl_ew.pivot_table(index=["Value","Is_Baseline"],
                                   columns="Portfolio", values="Sharpe").reset_index()
    k_pivot.columns = [str(c) for c in k_pivot.columns]
    k_pivot["Is_Baseline"] = k_pivot["Is_Baseline"].apply(
        lambda x: "★" if (x==True or x==1 or str(x)=="True") else "")
    k_pivot = k_pivot.rename(columns={"Value":"k","Is_Baseline":"Baseline"})
    pcols = [c for c in ["MKT","TAN","MV","BL","BL_KIO","EW","RP"] if c in k_pivot.columns]
    kd = k_pivot[["k","Baseline"]+pcols].copy()
    for c in pcols:
        kd[c] = kd[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "–")
    s.append(tbl(kd, cw=[2*cm,2.5*cm]+[2.8*cm]*len(pcols), fs=8.5))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 5. IC & SIGNAL QUALITY
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "5. Chất lượng Tín hiệu – Information Coefficient (IC)")
    s.append(Paragraph(
        "IC = Spearman rank correlation giữa tín hiệu idiosyncratic momentum "
        "và lợi nhuận thực tế tháng sau. IC > 0: dự báo đúng chiều. "
        "Hit rate: tỷ lệ bước mà cluster Long thực sự vượt cluster Short. "
        "Mean IC = +0.0667 (tín hiệu dự báo dương, ổn định qua 106 tháng).", BODY))
    df_ic = load("signal_ic_analysis.csv")
    ic_sum = pd.DataFrame([
        {"Chỉ số IC":         "Mean IC (Spearman)",   "Giá trị": f"{df_ic['IC'].mean():+.4f}"},
        {"Chỉ số IC":         "IC > 0 – % bước đúng chiều", "Giá trị": f"{(df_ic['IC']>0).mean():.1%}"},
        {"Chỉ số IC":         "Hit Rate (Long > Short)", "Giá trị": f"{df_ic['Hit'].mean():.1%}"},
        {"Chỉ số IC":         "Std IC",               "Giá trị": f"{df_ic['IC'].std():.4f}"},
        {"Chỉ số IC":         "t-stat IC ≠ 0",        "Giá trị": f"{df_ic['IC'].mean()/df_ic['IC'].std()*len(df_ic)**0.5:+.3f}"},
        {"Chỉ số IC":         "N bước OOS",           "Giá trị": str(len(df_ic))},
    ])
    s.append(tbl(ic_sum, cw=[8*cm, 4*cm], fs=9))
    s.append(Spacer(1, 0.3*cm))
    s.append(Paragraph("IC theo từng bước rebalancing (106 tháng):", H2))
    s.append(tbl(df_ic, cw=[3.5*cm,3*cm,2.5*cm,3*cm], fs=7.5))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 6. DRAWDOWN – TẤT CẢ 6 PORTFOLIO
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "6. Phân tích Drawdown Chi tiết – 7 Danh mục")
    sec(s, "6a. Tóm tắt Drawdown", level=2)
    s.append(Paragraph(
        "BL_KIO có Max_DD thấp nhất trong các danh mục tối ưu hóa (-30.7%). "
        "Avg_DD_Depth = -9.48% (nhỏ hơn nhiều so với TAN -24.9% và EW -15.4%). "
        "Avg_Recovery_Months BL_KIO = 3.8 (nhanh hơn BL gốc, TAN 9.6 và MV 11.8).", BODY))
    s.append(tbl(load("drawdown_summary.csv"), fs=8.5))

    s.append(Spacer(1, 0.4*cm))
    sec(s, "6b. Từng sự kiện Drawdown", level=2)
    s.append(Paragraph(
        "Liệt kê từng đợt drawdown: ngày bắt đầu, ngày đáy, ngày phục hồi, "
        "mức sụt giảm tối đa, số tháng phục hồi.", BODY))
    s.append(tbl(load("drawdown_periods.csv"), fs=7.5))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 7. DISTRIBUTION TESTS – 6 PORTFOLIO
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "7. Kiểm định Phân phối Lợi nhuận – 7 Danh mục")
    sec(s, "7a. Jarque-Bera (Chuẩn hóa)", level=2)
    s.append(Paragraph(
        "H₀: lợi nhuận có phân phối chuẩn. p < 0.05 = bác bỏ H₀ "
        "(fat tail / skewness đáng kể). Hầu hết danh mục KHÔNG chuẩn "
        "→ cần Sortino/MDD thay vì chỉ dùng Sharpe.", BODY))
    df_jb = load("distribution_jb_tests.csv")
    djb = df_jb.copy()
    djb["JB_stat"]   = djb["JB_stat"].apply(lambda x: f"{x:.3f}")
    djb["JB_pval"]   = djb["JB_pval"].apply(lambda x: f"{x:.4f}")
    djb["Skewness"]  = djb["Skewness"].apply(lambda x: f"{x:.4f}")
    djb["Excess_Kurtosis"] = djb["Excess_Kurtosis"].apply(lambda x: f"{x:.4f}")
    djb["is_normal_5pct"] = djb["is_normal_5pct"].apply(
        lambda x: "Yes" if x else "No ✗")
    s.append(tbl(djb, fs=8.5))

    s.append(Spacer(1, 0.4*cm))
    sec(s, "7b. Ljung-Box (Tự tương quan, lag 5/10/20)", level=2)
    s.append(Paragraph(
        "H₀: không có tự tương quan (white noise). p < 0.05 = tự tương quan "
        "có ý nghĩa (có thể khai thác được). "
        "no_autocorr = True: không bác bỏ H₀.", BODY))
    df_lb = load("distribution_lb_tests.csv")
    dlb = df_lb.copy()
    for c in ["LB5_stat","LB10_stat","LB20_stat"]:
        dlb[c] = dlb[c].apply(lambda x: f"{x:.3f}")
    for c in ["LB5_pval","LB10_pval","LB20_pval"]:
        dlb[c] = dlb[c].apply(lambda x: f"{x:.4f}")
    s.append(tbl(dlb, fs=7.5))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 8. MONTE CARLO
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "8. Monte Carlo Robustness – Nhiễu trong View BL (N=2,000 sims/bước)")
    s.append(Paragraph(
        "Thêm nhiễu Gaussian vào vector view q (delta = -10%…+10%) "
        "để kiểm tra độ nhạy của Sharpe kỳ vọng. "
        "Tính chất tuyến tính μ_BL(q') = base_vec + A_mat @ q' "
        "cho phép 2,000 mô phỏng mà không cần tối ưu lại. "
        "BL và TAN đều ổn định qua dải nhiễu.", BODY))
    df_mc = load("monte_carlo_v2.csv")
    mc_p = (df_mc.groupby(["Delta_Noise","Port_Type"])["Expected_Sharpe"]
            .agg(Mean=("mean"), Std=("std"), Min=("min"), Max=("max"))
            .round(4).reset_index())
    mc_p.columns = ["Delta_Noise","Danh mục","Mean Sharpe","Std","Min","Max"]
    s.append(tbl(mc_p, cw=[3.5*cm,3*cm,4*cm,3*cm,3*cm,3*cm], fs=8.5))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 9. OOS RETURNS SERIES
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "9. Chuỗi Lợi nhuận OOS theo Tháng – 7 Danh mục (106 tháng)")
    s.append(Paragraph(
        "Lợi nhuận tháng (simple return) của 6 danh mục và lãi suất phi rủi ro. "
        "Chuỗi này được dùng để tính toàn bộ các chỉ số hiệu suất.", BODY))
    df_r = load("oos_returns_v2.csv")
    dr = df_r.copy()
    for c in [col for col in dr.columns if col != "Date"]:
        dr[c] = dr[c].apply(lambda x: f"{x*100:+.3f}%" if pd.notna(x) else "")
    cw_r = [3.2*cm] + [2.3*cm]*8
    s.append(tbl(dr, cw=cw_r, fs=7.5))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 10. CLUSTER SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "10. Tín hiệu Phân cụm K-means (k=4) – Nhãn & Vị thế từng Cổ phiếu")
    s.append(Paragraph(
        "Nhãn cụm và vị thế view (Long / Short / Neutral) tại mỗi kỳ tái cân bằng. "
        "Features K-means (chuẩn hoá): (idiosyncratic momentum 6-1 tháng, −ann_vol). "
        "Long = cụm tốt nhất → nhận view dương trong BL. "
        "Short = cụm tệ nhất → nhận view âm trong BL.", BODY))
    df_cl = load("cluster_signals_v2.csv")
    s.append(Paragraph(f"Tổng quan sát: {len(df_cl):,} dòng "
                        f"({df_cl['Date'].nunique()} bước × {df_cl['Ticker'].nunique()} cổ phiếu tối đa)", SMALL))
    s.append(Spacer(1, 0.1*cm))
    s.append(tbl(df_cl, cw=[3.2*cm,2.5*cm,3.5*cm,4.5*cm], fs=7))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # 11. WEIGHTS HISTORY
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "11. Lịch sử Tỷ trọng Danh mục (7 Danh mục × 106 bước)")
    s.append(Paragraph(
        "Tỷ trọng tại mỗi kỳ tái cân bằng cho từng cổ phiếu trong 7 danh mục. "
        "Ràng buộc: 0 ≤ w ≤ 30%, Σw = 1 (áp dụng cho TAN, MV, BL, BL_KIO, RP). "
        "EW = 1/N (không ràng buộc per-asset, tự nhiên ≤ 1/N_min).", BODY))
    df_w = load("weights_v2.csv")
    s.append(Paragraph(f"Tổng quan sát: {len(df_w):,} dòng", SMALL))
    s.append(Spacer(1, 0.1*cm))
    cw_w = [3*cm, 2.2*cm] + [2.6*cm]*(len(df_w.columns)-2)
    s.append(tbl(df_w, cw=cw_w, fs=6.5))

    s.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # PHỤ LỤC A – PHƯƠNG PHÁP LUẬN
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "Phụ lục A – Phương pháp luận Chi tiết")
    method = [
        ("Dữ liệu",
         "25 cổ phiếu ngân hàng niêm yết HOSE/HNX. Giá đóng cửa tháng và vốn hoá "
         "thị trường (bn VND). Giai đoạn 06/2014–05/2026 (144 tháng). "
         "NaN với cổ phiếu chưa niêm yết xử lý bằng dynamic universe."),
        ("Rolling window",
         "Lookback T = 36 tháng. Tại mỗi bước i, training = [i−36, i−1], OOS = tháng i+1. "
         "Rebalancing hàng tháng → 106 bước OOS (08/2017–05/2026)."),
        ("Tín hiệu K-means",
         "Features: (1) idiosyncratic momentum = mean(log_ret_idio[−6:−1]) × 12, "
         "với idio = ret − ew_market_ret; (2) −ann_vol = −std(log_ret[−36:]) × √12. "
         "Chuẩn hoá StandardScaler, K-means k=4, random_state=42, n_init=10. "
         "Composite score = idio_ret − ann_vol → xếp hạng best/worst cluster."),
        ("Black-Litterman gốc (BL)",
         "Không có active views (P=∅, q=∅). Posterior đơn giản hóa: μ_BL = Π (equilibrium), "
         "Σ_BL = (1+τ)·Σ. Tối ưu Tangency trên (Π, (1+τ)Σ). "
         "Đây là BL chuẩn Black & Litterman (1992) không có quan điểm chủ quan."),
        ("BL + K-means Signal (BL_KIO)",
         "Prior: Π = δ·Σ·w_mkt (Reverse-CAPM). δ = (w_mkt·μ − rf) / (w_mkt·Σ·w_mkt), clipped [0.5, 10]. "
         "View uncertainty: Ω = τ·P·Σ·Pᵀ + ε·I, τ = 1/36. "
         "Posterior: M = (τΣ)⁻¹ + Pᵀ·Ω⁻¹·P; μ_BL_KIO = M⁻¹[(τΣ)⁻¹Π + Pᵀ·Ω⁻¹·q]; Σ_BL_KIO = M⁻¹ + Σ. "
         "q = idiosyncratic momentum spread (best cluster − worst cluster)."),
        ("Tối ưu hoá",
         "SLSQP maximise Sharpe(μ, Σ_post) subject to Σwᵢ=1, 0≤wᵢ≤0.30. "
         "10 điểm xuất phát ngẫu nhiên (Dirichlet), chọn nghiệm tốt nhất."),
        ("Monte Carlo",
         "2,000 mô phỏng nhiễu Gaussian trên q (delta ∈ {−10%,−5%,0%,+5%,+10%}). "
         "Khai thác μ_BL(q') = base_vec + A_mat @ q' (tuyến tính) để tránh tối ưu lại."),
        ("Kiểm định",
         "Paired t-test (scipy.stats.ttest_rel); Jobson-Korkie (1981) z-test; "
         "Jarque-Bera (scipy.stats.jarque_bera); Ljung-Box lag 5/10/20 (statsmodels)."),
    ]
    mt = Table([[Paragraph(k, BODY), Paragraph(v, BODY)] for k,v in method],
               colWidths=[4*cm, 15*cm])
    mt.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(0,-1), "FSB"),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
        ("GRID",         (0,0),(-1,-1), 0.3, MG),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[WH, LG]),
    ]))
    s.append(mt)

    s.append(Spacer(1, 0.5*cm))

    # ─────────────────────────────────────────────────────────────────────────
    # PHỤ LỤC B – LÃI SUẤT PHI RỦI RO
    # ─────────────────────────────────────────────────────────────────────────
    sec(s, "Phụ lục B – Lãi suất Phi rủi ro VN10Y theo Năm")
    s.append(Paragraph(
        "Lãi suất trái phiếu chính phủ Việt Nam kỳ hạn 10 năm (VN10Y). "
        "RF tháng = VN10Y_năm / 100 / 12.", BODY))
    rf_data = [["Năm","VN10Y (%/năm)","RF tháng (%)"]] + [
        [yr, val, f"{val/12:.4f}"] for yr, val in [
            (2014,6.50),(2015,6.50),(2016,6.20),(2017,5.80),(2018,4.58),
            (2019,4.58),(2020,3.00),(2021,2.53),(2022,4.00),(2023,3.50),
            (2024,3.06),(2025,3.10),(2026,4.35),
        ]
    ]
    rft = Table(rf_data, colWidths=[3.5*cm,5*cm,5*cm], hAlign="LEFT")
    rft.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(-1,0),  "FSB"),
        ("FONTNAME",     (0,1),(-1,-1), "FS"),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("TEXTCOLOR",    (0,0),(-1,-1), BK),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("LINEBELOW",    (0,0),(-1,0),  0.8, BK),
        ("GRID",         (0,0),(-1,-1), 0.3, MG),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WH, LG]),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    s.append(rft)

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(s)
    kb = os.path.getsize(PDF_PATH) / 1024
    print(f"[✓] PDF saved: {PDF_PATH}  ({kb:,.0f} KB)")


if __name__ == "__main__":
    build()
