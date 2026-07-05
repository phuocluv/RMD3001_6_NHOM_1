# Ứng dụng mô hình học sâu GNN để dự báo lây lan rủi ro thanh lý cấp độ vault

**Thực nghiệm từ giao thức cho vay phi tập trung Aave**

Bài tập nhóm học phần Phương pháp nghiên cứu khoa học — Trường Đại học Kinh tế, Đại học Đà Nẵng
Nhóm 1 — 49K33 / 50K06.5

## Mô tả dự án

Dự án ứng dụng mô hình mạng nơ-ron đồ thị (Graph Convolutional Network - GCN) để đo lường và
dự báo xác suất một vị thế vay trên giao thức Aave (V2 & V3) bị thanh lý trong vòng 3 ngày
tiếp theo (`liquidated_next_3d`). Mô hình tận dụng cấu trúc mạng lưới giữa các ví (shared
collateral, co-liquidation) thay vì chỉ dùng đặc trưng tĩnh của từng vị thế như mô hình nền
MLP. Kết quả được giải thích bằng KernelSHAP để xác định mức đóng góp của từng đặc trưng
(on-chain vs. cấu trúc mạng lưới) vào rủi ro thanh lý.

Toàn bộ chi tiết học thuật (bối cảnh, giả thuyết, phương pháp, kết quả) được trình bày trong
báo cáo `RMD3001_6_NHOM_1.pdf` (xem mục Dữ liệu bên dưới).

## Cấu trúc repo

Toàn bộ file nằm ở thư mục gốc (không có thư mục con):

- `data_processing.py` — pipeline xử lý dữ liệu thô (raw) thành file model-ready
  (`aave_model_ready_analysis_v2.csv`): dựng lại trạng thái user-day, tính Health Factor,
  Liquidation Threshold, Distance to Liquidation, gắn nhãn `liquidated_next_3d`.
- `train_gnn.py` — xây dựng đồ thị (shared collateral / co-liquidation edges), huấn luyện mô
  hình GCN, mô hình nền MLP, và giải thích bằng SHAP.
- `requirements.txt` — danh sách thư viện Python cần thiết.
- `dataset_link.txt` — link Google Drive chứa dữ liệu thô, dữ liệu đã xử lý và báo cáo PDF.

**Lưu ý:** dữ liệu raw (`aave_borrow.csv`, `aave_repay.csv`, `aave_supply.csv`,
`aave_liquidationcall.csv`, `threshold_final.csv`, `token_prices_daily.csv`,
`Fear & Greed Index.csv`), file dữ liệu đã xử lý (`data_processed.csv`) và báo cáo đầy đủ
(`RMD3001_6_NHOM_1.pdf`) **không nằm trong repo** — tải về theo link trong `dataset_link.txt`.

## Cách chạy

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Tải dữ liệu thô từ link trong `dataset_link.txt`, đặt các file CSV raw vào cùng thư mục gốc
với `data_processing.py` (script tự động tìm file trong `data/raw/` hoặc ngay tại thư mục
gốc nếu `data/raw/` không tồn tại):

- `aave_supply.csv`
- `aave_borrow.csv`
- `aave_repay.csv`
- `aave_liquidationcall.csv`
- `threshold_final.csv`
- `token_prices_daily.csv`
- `Fear & Greed Index.csv`

### 3. Chạy pipeline xử lý dữ liệu

```bash
python data_processing.py
```

Kết quả sinh ra:
- `data/processed/aave_model_ready_analysis_v2.csv` — file dữ liệu model-ready cuối cùng
- `data/processed/aave_user_day_reserve_positions_for_hf_v2.csv` — chi tiết vị thế theo reserve
- `reports/aave_model_ready_analysis_v2_report.txt` — báo cáo thống kê pipeline

### 4. Huấn luyện mô hình GNN

```bash
python train_gnn.py
```

Script sẽ dựng đồ thị từ dữ liệu model-ready, huấn luyện GCN và MLP baseline, đánh giá bằng
AUC-ROC/PR-AUC/Precision/Recall/F1, và sinh giải thích SHAP (global feature importance +
local waterfall plot).

## Yêu cầu hệ thống

Xem `requirements.txt` — bao gồm `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`,
`imbalanced-learn`, `shap`, `torch`, `torch-geometric`.

## Tham khảo thêm

Xem báo cáo đầy đủ `RMD3001_6_NHOM_1.pdf` (link trong `dataset_link.txt`) để biết chi tiết về
cơ sở lý thuyết, thiết kế đồ thị (shared_collateral_edge, co_liquidation_edge), kiến trúc mô
hình GCN, giả thuyết nghiên cứu (H1–H3) và kết quả thực nghiệm.
