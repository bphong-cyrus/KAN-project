# Đặc tả thực nghiệm (`experiment_spec.md`)

Tài liệu thống nhất dữ liệu đầu vào, cấu hình mô hình, ngân sách tối ưu và giao diện code giữa TV1 (dữ liệu & MLP), TV2 (KAN) và TV3 (huấn luyện & đánh giá).

## 1. Đối chiếu thiết lập với bài báo `2404.19756v5` và mã `pykan`

Ngoài văn bản PDF bài báo, nhóm đã đối chiếu trực tiếp mã nguồn tác giả (`KindXiaoming/pykan`, hàm `kan.utils.create_dataset` và `MultKAN.fit`). Các mục không nêu trong cả PDF lẫn mặc định `pykan` được chuẩn hóa theo quy ước chung của nhóm.

| Hạng mục | Trong bài báo (`2404.19756v5`) & `pykan` | Thiết lập nhóm |
|---|---|---|
| Hàm chính (P1) | PDF Mục 3.1, Hình 3.1 (tr. 14): $f(x,y)=\exp(\sin(\pi x)+y^2)$ | `task="exp_sin_y2"` |
| Kiến trúc KAN | PDF Mục 3.1 (tr. 14): `[2, 1, 1]` | `[2, 1, 1]` |
| Hàm cạnh & bậc spline | PDF Mục 2.2 (2.10)–(2.12): $w_b\,\mathrm{SiLU}(x)+w_s\sum_i c_i B_i(x)$, $k=3$ | `SiLU` + B-spline $k=3$ |
| Miền & số mẫu train/test | PDF Mục 2.4 (`1000` train); `pykan` `create_dataset` mặc định `ranges=[-1, 1]`, `train_num=1000`, `test_num=1000` | `[-1, 1]^d` đều ngẫu nhiên; `n_train=1000`, `n_test=1000` |
| Tập validation | Không tách riêng trong `pykan` | Bổ sung `n_val = 200` độc lập (không đánh giá `test` khi chọn cấu hình) |
| Lịch lưới KAN | PDF Mục 3.1: tăng mỗi 200 bước LBFGS qua `{3,5,10,20,50,100,200,500,1000}` | Dùng 5 giai đoạn `{3,5,10,20,50}`, mỗi giai đoạn 200 lần gọi optimizer |
| Optimizer | Ví dụ fitting của tác giả gọi `fit(steps=200)` theo từng giai đoạn. `MultKAN.fit` gọi LBFGS với `max_iter=20` và `strong_wolfe`. | `torch.optim.LBFGS`, khớp số lần gọi optimizer tích lũy và `max_iter=20` tại từng mốc KAN–MLP |
| Cấu hình MLP | Hình 3.1 vẽ depth 2–5 tới $10^4$–$10^5$ tham số; không liệt kê width từng điểm | 10 cấu hình depth 2–4 (`25–2241` tham số), `SiLU` + 2 điểm `Tanh`, `xavier_uniform` |
| Seed & độ chính xác | Không khóa seed toàn cục trong PDF | Split chính `seed=42`; lặp `{42,43,44}` ở cặp trọng tâm; `float64`, `normalize_x/y=False` |

## 2. Danh sách bài toán

| `task` | Công thức | KAN | Miền | Ưu tiên |
|---|---|---|---|---|
| `exp_sin_y2` | $f(x, y) = \exp(\sin(\pi x) + y^2)$ | `[2, 1, 1]` | $[-1, 1]^2$ | P1 (bắt buộc) |
| `bessel` | $f(x) = J_0(20x)$ (`scipy.special.j0`) | `[1, 1]` | $[-1, 1]$ | P2 (mở rộng) |
| `multiply` | $f(x, y) = xy$ | `[2, 2, 1]` | $[-1, 1]^2$ | P2 (mở rộng) |

## 3. Module dữ liệu (`Source/src/data.py`)

- **Độ chính xác `float64`:** Trên $f(x,y)=\exp(\sin(\pi x)+y^2) \in [0.37, 7.39]$, bước lượng tử hóa của `float32` quanh $7.39$ xấp xỉ $4.77 \times 10^{-7}$. Nhóm dùng `float64` cho dữ liệu, tham số khởi tạo và tính toán để giảm ảnh hưởng của làm tròn khi đo RMSE nhỏ.
- **Split & chống rò rỉ:** Ba tập `train`/`val`/`test` dùng `np.random.SeedSequence(seed).spawn(3)`, tự kiểm tra rời nhau từng hàng và hữu hạn, trả về `split_id` xác định (`exp_sin_y2_s42_1000_200_1000_9f2a4341` cho split chính `float64`). Cả đầu vào lẫn đầu ra không chuẩn hóa (`normalize_x=False`, `normalize_y=False`) để giữ đúng thang `RMSE` của bài báo. Hàm `DatasetBundle.to()` tự cập nhật `metadata["dtype"]` và gắn cờ `upcast_from` nếu nạp từ `float32` lên `float64`.

```python
from Source.src.data import make_dataset, load_dataset_from_yaml, evaluate_target

bundle = make_dataset(task="exp_sin_y2", n_train=1000, n_val=200, n_test=1000, seed=42, dtype="float64")
# bundle.x_train: (1000, 2), bundle.y_train: (1000, 1)
# bundle.to_pykan_dict() -> dict đầu vào cho pykan
```

## 4. Cấu hình MLP đối chứng (`Source/src/mlp_baseline.py`)

Mười cấu hình trong `Source/configs/mlp_panel.yaml` được khóa trước theo công thức tham số: $4h+1$ (depth 2, 1 lớp ẩn), $h^2+5h+1$ (depth 3, 2 lớp ẩn) và $2h^2+6h+1$ (depth 4, 3 lớp ẩn). Thang `SiLU` phủ từ 25 tới 2241 tham số. Hai đối chứng `Tanh` ở mức 45 và 177 tham số cho phép kiểm tra ảnh hưởng của hàm kích hoạt ở cùng kiến trúc. Xem kiểm tra độ nhạy trên train/validation với ba seed trong [fixture phát triển](fixtures/tv1_smoke_report.json). Xavier được khởi tạo trực tiếp ở dtype yêu cầu trên CPU, rồi mô hình được đưa tới thiết bị chạy.

| `config_id` | Kiến trúc | Độ sâu ($L$) | Activation | Tham số | Mốc đối chứng |
|---|---|---|---|---|---|
| `mlp_d2_w6` | `[2, 6, 1]` | 2 | `silu` | 25 | KAN `G=3` |
| `mlp_d2_w11` | `[2, 11, 1]` | 2 | `silu` | 45 | KAN `G=10` (cặp lặp 3 seed) |
| `mlp_tanh_d2_w11` | `[2, 11, 1]` | 2 | `tanh` | 45 | Đối chứng `Tanh` (45 tham số) |
| `mlp_d2_w19` | `[2, 19, 1]` | 2 | `silu` | 77 | KAN `G=20` |
| `mlp_d2_w41` | `[2, 41, 1]` | 2 | `silu` | 165 | KAN `G=50` |
| `mlp_d3_w11` | `[2, 11, 11, 1]` | 3 | `silu` | 177 | Depth 3 gần `G=50` |
| `mlp_tanh_d3_w11` | `[2, 11, 11, 1]` | 3 | `tanh` | 177 | Đối chứng `Tanh` depth 3 |
| `mlp_d3_w20` | `[2, 20, 20, 1]` | 3 | `silu` | 501 | Depth 3 trung bình |
| `mlp_d4_w24` | `[2, 24, 24, 24, 1]` | 4 | `silu` | 1297 | Depth 4 dung lượng cao |
| `mlp_d4_w32` | `[2, 32, 32, 32, 1]` | 4 | `silu` | 2241 | Depth 4 mở rộng (>2k tham số) |

```python
from Source.src.mlp_baseline import build_mlp, count_trainable_params

model = build_mlp(input_dim=2, hidden_widths=[11], activation="silu", seed=42, dtype="float64")
n_params = count_trainable_params(model)  # 45
```

## 5. Hợp đồng ngân sách LBFGS & đánh giá chung

- **Đơn vị đếm:** Mục 3.1 báo cáo 1.800 bước LBFGS cho 9 giai đoạn. Trong [ví dụ fitting của tác giả](https://github.com/KindXiaoming/pykan/blob/master/docs/Example/Example_1_function_fitting.rst), mỗi giai đoạn truyền `steps=200` vào `MultKAN.fit` và mỗi vòng gọi `optimizer.step(closure)`. Nhóm dùng đơn vị này. Cột `steps` đếm số lần gọi optimizer tích lũy, `max_iter` ghi giới hạn vòng lặp nội bộ mỗi lần gọi và `closure_evals` đếm số lần chạy closure thực tế, kể cả dò đường.
- **Ngân sách so sánh:** Panel thu gọn dùng 200 lần gọi optimizer ở mỗi $G$ trong lịch `{3,5,10,20,50}`. Mỗi MLP huấn luyện liên tục và lưu checkpoint tại `{200,400,600,800,1000}` theo `mlp_panel.yaml`. So sánh checkpoint KAN và MLP có cùng `steps`, `max_iter=20` và `split_id` trong cùng seed. Ví dụ cặp trọng tâm tại $G=10$ dùng mốc 600 cho cả hai. Ghi số lần chạy closure và thời gian để người đọc thấy chênh lệch chi phí thực tế.
- **Grid cố định:** Nếu chạy đối chứng khởi tạo từ đầu, ghi riêng dưới `kan_fixed_grid` và khóa ngân sách so với checkpoint tương ứng của `kan_grid_extension`.
- **Lặp seed:** Chạy 3 seed (`42, 43, 44`) ít nhất trên cặp trọng tâm; chỉ vẽ `mean ± std` tại điểm có $\ge 2$ seed.
- **Bấm giờ & chống rò rỉ:** Đo `time_s` trên cùng máy (`torch.cuda.synchronize()` nếu dùng GPU); chỉ dùng `val_rmse` khi tinh chỉnh/kiểm tra độ nhạy, giữ `test_rmse` thuần túy cho đánh giá cuối.
