# Ghi chú Cảnh 1 (`01_problem_mlp.py`)

- **Class:** `ProblemAndMLPScene` (TV1 — Tạ Bỉnh Phong, `24127100`)
- **Đầu ra 1080p60:** `Manim/media/videos/01_problem_mlp/1080p60/ProblemAndMLPScene.mp4`

| Nhịp | Thời gian ước lượng | Nội dung hình ảnh | Gợi ý thuyết minh (cho TV4 biên tập) | Nguồn trên màn hình |
|---|---|---|---|---|
| 1 | `0:00–0:24` | Tiêu đề; lưới $[-1,1]^2$ vẽ 36 điểm đầu trích từ tập `train` thật (`seed=42`) và 1 điểm `test` $(\mathbf{x}^*, ?)$ nhấp nháy. | Đặt bài toán học máy: từ tập điểm quan sát hữu hạn trên miền $[-1,1]^2$, mô hình cần dự đoán chính xác giá trị tại điểm mới chưa gặp. | `Liu et al., ICLR 2025 (arXiv:2404.19756v5)` |
| 2 | `0:24–0:53` | Công thức $f(x,y)=\exp(\sin(\pi x)+y^2)$; lát cắt 1D trượt theo $y_0$; đường dự đoán **thực tế** của `MLPBaseline([2, 11, 1], SiLU)` qua các checkpoint `LBFGS` (`0 -> 20 -> 40 -> 100 -> 200` iters) cùng công thức MSE / RMSE. | Giới thiệu hàm mục tiêu Mục 3.1 và minh họa trực tiếp quá trình MLP tối ưu bằng LBFGS thu nhỏ dần phần dư trên tập train 1000 mẫu. | `Bài toán Mục 3.1 — Liu et al. (2025)` |
| 3 | `0:53–1:22` | Sơ đồ minh họa MLP `[2, 4, 1]` (ghi rõ thực nghiệm quét 6 cấu hình 25–501 tham số): trọng số cạnh $w_{ij}$, hàm $\mathrm{SiLU}(z)$ cố định tại nút và công thức $(W_2 \circ \sigma \circ W_1)(\mathbf{x})$. | Giải thích cơ chế MLP: cạnh mang trọng số tuyến tính, nút mang hàm kích hoạt cố định nên không tối ưu trực tiếp hình dạng từng hàm 1D. | `Hornik et al., 1989; Liu et al., 2025` |
| 4 | `1:22–1:39` | Tách một cặp nút–cạnh ra giữa màn hình: minh họa khái niệm chuyển trọng số $w$ thành hàm 1D $\phi(x)$ trên cạnh, nút $\sigma$ thành nút cộng $\sum$, kết bằng câu hỏi chuyển tiếp sang KAN. | Đặt câu hỏi chuyển cảnh: nếu đưa hàm phi tuyến học được từ nút lên ngay trên từng cạnh nối, còn nút chỉ làm phép cộng thì sao? | `Liu et al., Hình 0.1 (2025)` |
