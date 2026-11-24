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
| Lịch lưới KAN | PDF Mục 3.1: tăng mỗi 200 bước LBFGS qua `{3,5,10,20,50,100,200,500,1000}` | Bắt đầu với `{3,5,10,20,50}` (mở rộng thêm nếu đủ thời gian) |
| Optimizer | `pykan` dùng `LBFGS` (`max_iter=20`, `line_search_fn="strong_wolfe"`) | `torch.optim.LBFGS`, khớp ngân sách `(outer_steps, max_iter)` cho cả KAN & MLP |
| Cấu hình MLP | Hình 3.1 vẽ depth 2–5 tới $10^4$–$10^5$ tham số; không liệt kê width từng điểm | 10 cấu hình depth 2–4 (`25–2241` tham số), `SiLU` + 2 điểm `Tanh`, `xavier_uniform` |
| Seed & độ chính xác | Không khóa seed toàn cục trong PDF | Split chính `seed=42`; lặp `{42,43,44}` ở cặp trọng tâm; `float64`, `normalize_x/y=False` |

## 2. Danh sách bài toán

| `task` | Công thức | KAN | Miền | Ưu tiên |
|---|---|---|---|---|
| `exp_sin_y2` | $f(x, y) = \exp(\sin(\pi x) + y^2)$ | `[2, 1, 1]` | $[-1, 1]^2$ | P1 (bắt buộc) |
| `bessel` | $f(x) = J_0(20x)$ (`scipy.special.j0`) | `[1, 1]` | $[-1, 1]$ | P2 (mở rộng) |
| `multiply` | $f(x, y) = xy$ | `[2, 2, 1]` | $[-1, 1]^2$ | P2 (mở rộng) |

## 3. Module dữ liệu (`Source/src/data.py`)

- **Độ chính xác `float64`:** Trên $f(x,y)=\exp(\sin(\pi x)+y^2) \in [0.37, 7.39]$, kiểu `float32` có bước lượng tử hóa $\mathrm{ULP}_{32}(7.39) \approx 4.77 \times 10^{-7}$ ($\varepsilon_{32} \approx 1.19 \times 10^{-7}$), gây sàn nhiễu số học sớm cho `LBFGS` khi KAN/MLP nhắm tới `RMSE` bậc $10^{-5}$–$10^{-6}$ (và $10^{-8}$ trên `multiply`). Do đó toàn bộ dataset và mô hình mặc định dùng `float64` ($\varepsilon_{64} \approx 2.22 \times 10^{-16}$).
- **Split & chống rò rỉ:** Ba tập `train`/`val`/`test` dùng `np.random.SeedSequence(seed).spawn(3)`, tự kiểm tra rời nhau từng hàng và hữu hạn, trả về `split_id` xác định (`exp_sin_y2_s42_1000_200_1000_9f2a4341` cho split chính `float64`). Cả đầu vào lẫn đầu ra không chuẩn hóa (`normalize_x=False`, `normalize_y=False`) để giữ đúng thang `RMSE` của bài báo. Hàm `DatasetBundle.to()` tự cập nhật `metadata["dtype"]` và gắn cờ `upcast_from` nếu nạp từ `float32` lên `float64`.

```python
from Source.src.data import make_dataset, load_dataset_from_yaml, evaluate_target

bundle = make_dataset(task="exp_sin_y2", n_train=1000, n_val=200, n_test=1000, seed=42, dtype="float64")
# bundle.x_train: (1000, 2), bundle.y_train: (1000, 1)
# bundle.to_pykan_dict() -> dict đầu vào cho pykan
```

## 4. Cấu hình MLP đối chứng (`Source/src/mlp_baseline.py`)

Mười cấu hình trong `Source/configs/mlp_panel.yaml` được khóa trước theo công thức tham số: $4h+1$ (depth 2, 1 lớp ẩn), $h^2+5h+1$ (depth 3, 2 lớp ẩn) và $2h^2+6h+1$ (depth 4, 3 lớp ẩn). Bên cạnh thang `SiLU` (từ 25 tới 2241 tham số để không cắt cụt miền dung lượng cao), bảng giữ 2 điểm đối chứng `Tanh` ở mức 45 và 177 tham số vì kiểm tra phát triển `float64` trên `mlp_d2_w11` (3 seed `{42,43,44}`, 10x20 bước `LBFGS`, chỉ đo trên `train/val`) cho thấy `Tanh` (`val_rmse = 0.1011 ± 0.0310`) và `SiLU` (`0.1207 ± 0.0411`) đều vượt xa `ReLU` (`0.2379 ± 0.0563`).

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

- **Khớp cách đếm `steps` với `pykan`:** Trong `pykan` (`MultKAN.fit` + `kan/LBFGS.py`), mỗi vòng lặp `for _ in range(steps)` gọi `optimizer.step(closure)` với `max_iter=20`, `line_search_fn="strong_wolfe"`. Do đó trong `results/runs.csv`, cột `steps` là **số lần gọi `optimizer.step(closure)` (outer steps)**, cột `optimizer` ghi rõ `LBFGS(max_iter=20,strong_wolfe)`, và cột `closure_evals` đếm **tổng số lần đánh giá `closure`**. TV3 phải áp dụng đúng cùng cặp `(outer_steps, max_iter)` cho cả KAN và MLP (vd. `outer_steps=90, max_iter=20` cho tối đa `1800` bước nội tại LBFGS, hoặc khớp đúng cấu hình đã chạy cho KAN).
- **Lịch lưới nối tiếp vs. cố định:** Tách riêng `kan_grid_extension` và `kan_fixed_grid`, không gộp chung hai đường.
- **Lặp seed:** Chạy 3 seed (`42, 43, 44`) ít nhất trên cặp trọng tâm; chỉ vẽ `mean ± std` tại điểm có $\ge 2$ seed.
- **Bấm giờ & chống rò rỉ:** Đo `time_s` trên cùng máy (`torch.cuda.synchronize()` nếu dùng GPU); chỉ dùng `val_rmse` khi tinh chỉnh/kiểm tra độ nhạy, giữ `test_rmse` thuần túy cho đánh giá cuối.
