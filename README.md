# NCKH_Sinhvien — Tối ưu hoá danh mục cổ phiếu ngân hàng Việt Nam

Nghiên cứu kết hợp phân cụm chuỗi thời gian K-means và mô hình Black-Litterman  
để xây dựng danh mục đầu tư tối ưu từ 25 cổ phiếu ngân hàng niêm yết tại HOSE/HNX.

## Dữ liệu (`data/`)
| File | Mô tả |
|---|---|
| `bank_monthly_close.csv` | Giá đóng cửa hàng tháng, 25 NHTM, 2015–2026 |
| `bank_monthly_mktcap_bn.csv` | Vốn hoá thị trường (tỷ VND), hàng tháng |

## Danh sách 25 ngân hàng
VCB · BID · CTG · MBB · TCB · ACB · VPB · HDB · VIB · TPB  
STB · SHB · LPB · SSB · MSB · OCB · NAB · BAB · KLB · PGB  
VAB · VBB · SGB · ABB · BVB

## Nguồn dữ liệu
vnstock (VCI) — crawl tháng 05/2026
