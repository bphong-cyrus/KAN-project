# Đồ án KAN: Kolmogorov–Arnold Networks

- **Bài báo gốc:** Liu et al., *KAN: Kolmogorov–Arnold Networks*, `arXiv:2404.19756v5` (ICLR 2025).
- **Mục tiêu thực nghiệm:** Tái lập panel chính của Hình 3.1 trên hàm $f(x,y)=\exp(\sin(\pi x)+y^2)$ (KAN `[2, 1, 1]` so với MLP theo `test RMSE` và số tham số), kèm khảo sát kích thước lưới $G$.

## Cấu trúc thư mục

```text
KAN-project/
├── README.md
├── experiment_spec.md          # Thiết lập dữ liệu, tham số & API dùng chung
├── results_schema.md           # Cấu trúc CSV log và summary.json
├── Source/
│   ├── requirements.txt
│   ├── configs/                # Cấu hình dataset & các cấu hình MLP
│   ├── src/
│   │   ├── data.py             # Sinh dữ liệu và chia tập train/val/test cố định
│   │   └── mlp_baseline.py     # Định nghĩa MLP đối chứng & đếm tham số
│   ├── scripts/
│   │   ├── prepare_data.py     # Kiểm tra và xuất split cố định
│   │   └── smoke_tv1.py        # Chạy thử pipeline dữ liệu + MLP
│   └── tests/                  # Unit tests (pytest)
├── analysis/                   # Script phân tích & vẽ đồ thị (chờ TV3/TV4)
├── Manim/
│   └── scenes/
│       ├── style.py            # Màu sắc, font tiếng Việt & helper dùng chung
│       ├── 01_problem_mlp.py   # Cảnh 1 (0:00–1:50): bài toán & MLP
│       └── tv1_scene01_notes.md
├── results/runs.csv            # Bảng log thực nghiệm chính thức (chưa có run chính)
├── fixtures/                   # Kết quả smoke test tạm khi dev
├── video/                      # Kịch bản, phụ đề & link video (chờ TV4–TV5)
└── report/                     # Source LaTeX biên dịch Report.pdf
```

## Phân công nhóm

| Thành viên | MSSV | Phụ trách |
|---|---|---|
| Tạ Bỉnh Phong (TV1) | `24127100` | Dữ liệu & MLP (`Source/`), Manim Cảnh 1, Báo cáo Chương 1 & Tóm tắt |
| Trịnh Khánh Linh (TV2) | `[MSSV 2]` | KAN model (`pykan`), Manim Cảnh 2, Báo cáo Chương 2 |
| Nguyễn Trần Phương Thuý (TV3) | `[MSSV 3]` | Training LBFGS & grid schedule, `analysis/`, Báo cáo Chương 3 |
| Lâm Gia Hưng (TV4) | `[MSSV 4]` | Tổng hợp kết quả & biểu đồ, Manim Cảnh 3, kịch bản/phụ đề video, Báo cáo Chương 4 |
| Đỗ Đăng Nhật Tiến (TV5) | `[MSSV 5]` | Audio, hậu kỳ & đăng video, Báo cáo Chương 5 & Phụ lục |

## Chạy thử phần hiện có

```powershell
pip install -r Source/requirements.txt
pytest -v Source/tests
python Source/scripts/prepare_data.py
python Source/scripts/smoke_tv1.py
```

Render Manim Cảnh 1:

```powershell
# Preview nhanh (480p15)
manim -ql Manim/scenes/01_problem_mlp.py ProblemAndMLPScene --media_dir Manim/media

# Bản xuất 1080p60
manim -qh Manim/scenes/01_problem_mlp.py ProblemAndMLPScene --media_dir Manim/media
```

Biên dịch báo cáo trong `report/`: mở [report/main.tex](report/main.tex) trên Overleaf với **XeLaTeX**, hoặc chạy cục bộ trên Windows bằng `powershell -ExecutionPolicy Bypass -File report/build.ps1`.
