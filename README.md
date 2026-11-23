# Đồ án CSC14005 — KAN: Kolmogorov–Arnold Networks

Kho mã nguồn thực nghiệm, trực quan hóa Manim và báo cáo LaTeX cho đồ án môn học **Nhập môn Học máy (CSC14005)**, Trường Đại học Khoa học Tự nhiên — ĐHQG TP.HCM.

- **Bài báo gốc:** Liu và cộng sự, *KAN: Kolmogorov–Arnold Networks*, `arXiv:2404.19756v5` (ngày 09/02/2025, được chấp nhận tại **ICLR 2025**).
- **Phạm vi thực nghiệm:** Tái lập cốt lõi một panel so sánh chính ở Mục 3.1 / Hình 3.1 trên hàm $f(x,y)=\exp(\sin(\pi x)+y^2)$ (KAN `[2, 1, 1]` đối chứng với MLP theo `test RMSE` và số tham số), kèm khảo sát bổ sung về kích thước lưới $G$.

---

## 1. Cấu trúc thư mục (Tách code theo module, `report/` chứa trọn bộ LaTeX)

```text
KAN-project/
├── README.md
├── experiment_spec.md              # Hợp đồng dữ liệu, kiến trúc, API (TV1–TV3)
├── results_schema.md               # Chuẩn cột results/runs.csv và results/summary.json
├── Source/                         # Mã nguồn tái lập thực nghiệm
│   ├── README.md
│   ├── requirements.txt
│   ├── configs/                    # Cấu hình YAML cho dataset & 6 MLP đối chứng (TV1)
│   ├── src/
│   │   ├── data.py                 # [TV1] Sinh dữ liệu, chia split & metadata
│   │   └── mlp_baseline.py         # [TV1] MLP đối chứng cấu hình được & đếm tham số
│   ├── scripts/
│   │   ├── prepare_data.py         # [TV1] CLI kiểm tra giải tích & xuất split dùng chung
│   │   └── smoke_tv1.py            # [TV1] Smoke test dữ liệu + 6 MLP configs
│   └── tests/                      # Unit tests bằng pytest cho phần TV1
├── analysis/                       # Thư mục chờ code phân tích & vẽ đồ thị (TV3/TV4)
├── Manim/                          # Mã nguồn hoạt hình Manim
│   ├── README.md
│   ├── scenes/
│   │   ├── style.py                # [TV1] Giao diện dùng chung (màu, font tiếng Việt, helper)
│   │   ├── 01_problem_mlp.py       # [TV1] Cảnh 1 (0:00–1:50): Bài toán & MLP
│   │   └── tv1_scene01_notes.md    # [TV1] Ghi chú timecode & nguồn cho nhóm video
│   └── assets/
├── results/                        # Log thực nghiệm chính thức (runs.csv hiện chỉ có header)
├── fixtures/                       # Output smoke test phát triển (KHÔNG nộp như kết quả thật)
├── video/                          # Thư mục chờ kịch bản, phụ đề & link video (TV4–TV5)
└── report/                         # Toàn bộ các folder & source LaTeX biên dịch Report.pdf
```

---

## 2. Thành viên và phạm vi công việc

| TV | Họ và tên | MSSV | Phần việc chính theo kế hoạch | Trạng thái hiện tại trong repo |
|---|---|---|---|---|
| **TV1** | **Tạ Bỉnh Phong** | `24127100` | Dữ liệu & MLP (`data.py`, `mlp_baseline.py`, `prepare_data.py`, `smoke_tv1.py`, `tests/`), Manim Cảnh 1 (`style.py`, `01_problem_mlp.py`), Báo cáo Chương 1 + Tóm tắt nháp + `data_pipeline.pdf` | **Đã hoàn thiện phần TV1** |
| **TV2** | **Trịnh Khánh Linh** | `[MSSV 2]` | Adapter KAN (`kan_model.py`, `check_kan.py`), Manim Cảnh 2 (`02_kan_spline.py`), Báo cáo Chương 2 | Chờ TV2 bổ sung |
| **TV3** | **Nguyễn Trần Phương Thuý** | `[MSSV 3]` | Huấn luyện LBFGS & lưới $G$ (`train_grid.py`, `metrics.py`, `run_core.py`), `analysis/evaluate.py`, Báo cáo Chương 3 | Chờ TV3 bổ sung |
| **TV4** | **Lâm Gia Hưng** | `[MSSV 4]` | Chạy kiểm `evaluate.py`, Manim Cảnh 3 (`03_results.py`), `video/` storyboard & phụ đề, Báo cáo Chương 4 | Chờ TV4 bổ sung |
| **TV5** | **Đỗ Đăng Nhật Tiến** | `[MSSV 5]` | Audio, hậu kỳ 1080p & đăng video công khai (`video/url.txt`), Báo cáo Chương 5 & Phụ lục A–B | Chờ TV5 bổ sung |

---

## 3. Quy ước làm việc Git

1. **Mỗi thành viên tự thêm và sửa đúng các file thuộc phần mình:** tránh sửa chéo file của thành viên khác khi chưa thống nhất.
2. **Không ghi số liệu giả vào `results/runs.csv`:** hiện tại file chỉ chứa dòng tiêu đề. Mọi lượt chạy thử phát triển ghi vào `fixtures/`.
3. **Điền MSSV:** TV2–TV5 cập nhật MSSV vào bảng trên và `report/config/metadata.tex`.

---

## 4. Lệnh chạy phần TV1

```powershell
# 1) Cài thư viện và chạy unit tests / smoke test TV1
pip install -r Source/requirements.txt
pytest -v Source/tests
python Source/scripts/prepare_data.py
python Source/scripts/smoke_tv1.py

# 2) Render preview Cảnh 1 Manim (TV1)
manim -ql Manim/scenes/01_problem_mlp.py ProblemAndMLPScene
```

Thư mục `report/` chứa trọn bộ template và cấu trúc LaTeX (bản Overleaf dùng **Compiler: XeLaTeX**, **Main document: main.tex**; trên Windows chạy `powershell -ExecutionPolicy Bypass -File report/build.ps1` khi MiKTeX có đủ gói trong preamble).
