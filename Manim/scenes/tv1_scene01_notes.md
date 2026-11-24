# Ghi chú Cảnh 1 (`01_problem_mlp.py`)

- **Class:** `ProblemAndMLPScene` (TV1 — Tạ Bỉnh Phong, `24127100`)
- **Đầu ra 1080p60:** `Manim/media/videos/01_problem_mlp/1080p60/ProblemAndMLPScene.mp4` (`110.00s`, `1920x1080`, `60 fps`)

| Nhịp | Thời gian thực (`ffprobe`) | Nội dung hình ảnh | Gợi ý thuyết minh (cho TV4 biên tập) | Nguồn trên màn hình |
|---|---|---|---|---|
| 1 | `0:00–0:25` | Tiêu đề, 36 điểm đầu từ tập train trên $[-1,1]^2$ và điểm mới $(x^*,y^*)$. Cuối nhịp giữ hình để đọc câu hỏi. | Mô hình học từ một tập điểm hữu hạn và dự đoán tại vị trí chưa gặp. | `Liu et al., ICLR 2025 (arXiv:2404.19756v5)` |
| 2 | `0:25–0:55` | Công thức $f(x,y)=\exp(\sin(\pi x)+y^2)$ và lát cắt $y_0=0.40$. Đường dự đoán của MLP 45 tham số chuyển qua các trạng thái LBFGS đã ghi, từ bước 0 đến 200 với `max_iter=1`. Bảng bên phải hiển thị train RMSE trên 1000 mẫu. | Giải thích phần dư trên lát cắt và RMSE trên toàn bộ tập train. Bước 200 là điểm dừng của minh họa. | `Bài toán Mục 3.1 — Liu et al. (2025)` |
| 3 | `0:55–1:30` | Sơ đồ MLP $[2,4,1]$, trọng số $w_{ij}$ trên cạnh, đồ thị SiLU ở nút ẩn, công thức affine $\mathcal A_\ell(\mathbf z)=W_\ell\mathbf z+\mathbf b_\ell$ và công thức hợp thành hai tầng. | Theo dõi tín hiệu từ đầu vào qua trọng số và bias. Nút ẩn áp dụng SiLU trước khi tầng cuối tính đầu ra. | `Hornik et al., 1989; Liu et al., 2025` |
| 4 | `1:30–1:50` | Phóng to một cạnh và nút nhận. Cạnh $w$ chuyển thành hàm một biến $\phi(x)$ và nút chuyển thành phép cộng $\sum$. | Dẫn sang Cảnh 2 bằng câu hỏi điều gì thay đổi khi cạnh học cả một hàm. | `Liu et al., Hình 0.1 (2025)` |
