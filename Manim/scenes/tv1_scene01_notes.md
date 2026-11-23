# Ghi chú Cảnh 1 Manim (`01_problem_mlp.py`) — Bàn giao TV1 -> TV4 & TV5

- **Người dựng cảnh:** TV1 — Tạ Bỉnh Phong (`24127100`)
- **Lệnh preview:** `manim -ql Manim/scenes/01_problem_mlp.py ProblemAndMLPScene`
- **Lệnh xuất 1080p60:** `manim -qh Manim/scenes/01_problem_mlp.py ProblemAndMLPScene`
- **Tổng thời lượng mục tiêu:** ~1 phút 50 giây (`0:00–1:50`)

---

## Bảng timecode & gợi ý thuyết minh cho TV4 (biên tập lời/phụ đề)

| Nhịp | Timecode | Chuyển động trên màn hình | Ý thuyết minh gợi ý cho TV4 | Nguồn hiển thị lúc dùng |
|---|---|---|---|---|
| **1** | `0:00–0:25` | Màn hình mở đầu tên đề tài; sau đó hiện lưới $[-1,1]^2$ bên trái với các điểm màu $(\mathbf{x}_i, y_i)$ và điểm truy vấn nhấp nháy $(x^*, y^*) \mapsto ?$; bên phải hiện $\mathcal{D}=\{(\mathbf{x}_i, y_i)\}_{i=1}^n$. | *"Trong học máy, một bài toán kinh điển là xấp xỉ hàm nhiều biến từ một tập hữu hạn điểm quan sát. Cho dữ liệu đầu vào trên miền $[-1,1]^2$, làm thế nào mô hình dự đoán chính xác giá trị tại một điểm mới chưa từng thấy?"* | `Liu et al., ICLR 2025 (arXiv:2404.19756v5)` |
| **2** | `0:25–0:55` | Hiện công thức $f(x,y)=\exp(\sin(\pi x)+y^2)$; lát cắt 1D của hàm thay đổi mượt khi $y_0$ trượt; xuất hiện 11 điểm dữ liệu mẫu và đường dự đoán $\hat{f}_\theta$ màu hổ phách co dần các đoạn sai số dọc về sát đường cong mục tiêu cùng công thức MSE / RMSE. | *"Xét bài toán chuẩn trong Mục 3.1 của bài báo: $f(x,y)=\exp(\sin(\pi x)+y^2)$. Khi cố định hoặc thay đổi một biến, hàm tạo thành mặt cong phi tuyến trơn. Mô hình học tham số $\theta$ sao cho sai số bình phương trung bình trên dữ liệu huấn luyện nhỏ nhất."* | `Bài toán Mục 3.1 — Liu et al. (2025)` |
| **3** | `0:55–1:30` | Sơ đồ mạng MLP `[2, 4, 1]` tự dựng: xung tín hiệu chạy từ $x_1, x_2$ qua các cạnh màu hổ phách tới nút ẩn $\sigma$ và đầu ra $\hat{y}$. Bên phải phóng to đồ thị hàm kích hoạt cố định $\mathrm{SiLU}(z)$ tại nút và công thức $\mathrm{MLP}(\mathbf{x})=(W_2 \circ \sigma \circ W_1)(\mathbf{x})$. | *"Cách tiếp cận quen thuộc nhất là Multi-Layer Perceptron. Ở MLP, các cạnh mang trọng số tuyến tính học được $w_{ij}$, còn hàm phi tuyến như $\mathrm{SiLU}$ được đặt cố định tại mỗi nút. Để ghép lại những đường cong giải tích mượt mà, MLP thường phải phối hợp nhiều nút ẩn và tham số."* | `Hornik et al., 1989; Liu et al., 2025` |
| **4** | `1:30–1:50` | Làm mờ các nhánh xung quanh, giữ lại một cạnh nối giữa nút đầu vào và nút ẩn; phóng to cặp cạnh–nút này: trọng số vô hướng $w$ trên cạnh biến hình thành hộp đường cong spline học được $\phi(x)$ màu ngọc, còn nút $\sigma$ biến thành nút cộng $\sum$. Hiện câu hỏi chuyển cảnh. | *"Nhưng hãy đặt một câu hỏi đảo ngược lại: điều gì sẽ xảy ra nếu ta không giữ hàm phi tuyến cố định tại nút nữa, mà đưa hẳn một hàm một biến học được lên từng cạnh nối, còn nút chỉ làm phép cộng?"* -> Nối thẳng sang Cảnh 2 của TV2. | `Ý tưởng cốt lõi — Liu et al., Fig. 0.1 (2025)` |

---

## Kiểm tra chấp nhận trước khi ghép
- Không có số RMSE giả nào xuất hiện trong Cảnh 1.
- Toàn bộ đồ thị và sơ đồ mạng đều được vẽ bằng Manim thuần.
- Nhãn trích dẫn xuất hiện đúng thời điểm trình bày nội dung tương ứng.
