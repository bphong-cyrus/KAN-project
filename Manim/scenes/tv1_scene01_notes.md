# Ghi chú Cảnh 1 (`01_problem_mlp.py`)

- **Class:** `ProblemAndMLPScene` (TV1 — Tạ Bỉnh Phong, `24127100`)
- **Đầu ra 1080p60:** `Manim/media/videos/01_problem_mlp/1080p60/ProblemAndMLPScene.mp4` (`75.50s`, `1920x1080`, `60 fps`)

| Nhịp | Thời gian thực (`ffprobe`) | Nội dung hình ảnh | Gợi ý thuyết minh (cho TV4 biên tập) | Nguồn trên màn hình |
|---|---|---|---|---|
| 1 | `0:00–0:20` | Tiêu đề; lưới $[-1,1]^2$ với biến thống nhất $(x, y)$ vẽ 36 điểm đầu từ tập `train` (`seed=42`, `float64`) và 1 điểm `test` $(x^*, y^*), ?$ nhấp nháy. | Đặt bài toán học máy: từ tập điểm quan sát hữu hạn trên $[-1,1]^2$, mô hình $\hat{f}_\theta(x,y)$ cần tổng quát hóa tới điểm mới chưa gặp. | `Liu et al., ICLR 2025 (arXiv:2404.19756v5)` |
| 2 | `0:20–0:44` | Công thức $f(x,y)=\exp(\sin(\pi x)+y^2)$; lát cắt 1D quét $y_0$; **từng bước LBFGS thật** (`max_iter=1`, `0..200`, không nội suy giả) của `MLPBaseline([2,11,1], float64)` trên 1000 mẫu cùng `train RMSE` động và 11 điểm đối chiếu lát cắt $y_0=0.40$; cuối nhịp nổi bật ngưỡng bão hòa bậc $10^{-1}$ so với mục tiêu $\lesssim 10^{-5}$. | Dù đường cong nhìn gần khớp về mặt đại thể, ở bài toán khoa học MLP 45 tham số vẫn bão hòa quanh $\mathrm{RMSE}\sim 10^{-1}$, cách rất xa chuẩn $10^{-5}$. | `Bài toán Mục 3.1 — Liu et al. (2025)` |
| 3 | `0:44–1:02` | Sơ đồ minh họa MLP `[2, 4, 1]` (đầu vào $x, y$, đầu ra $\hat{f}$; ghi rõ thực nghiệm quét 10 cấu hình $25\text{--}2241$ tham số): trọng số cạnh $w_{ij}$, $\mathrm{SiLU}(z)$ cố định tại nút và công thức $(W_2 \circ \sigma \circ W_1)(\mathbf{x})$. | Lý giải nguyên nhân: MLP đặt trọng số tuyến tính trên cạnh và cố định $\sigma(z)$ tại nút, không tối ưu trực tiếp hình dạng từng hàm 1D. | `Hornik et al., 1989; Liu et al., 2025` |
| 4 | `1:02–1:15.5` | Tách cặp nút $x$–cạnh–nút ra giữa: chuyển trọng số $w$ thành hàm 1D $\phi(x)$ trên cạnh, nút $\sigma$ thành nút cộng $\sum$, kết bằng câu hỏi chuyển tiếp sang KAN. | Đặt câu hỏi chuyển cảnh: nếu đưa hàm phi tuyến học được $\phi(x)$ lên ngay trên cạnh và chỉ giữ phép cộng tại nút thì điều gì thay đổi? | `Liu et al., Hình 0.1 (2025)` |
