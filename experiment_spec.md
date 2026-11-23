# Đặc tả thực nghiệm (`experiment_spec.md`)

Tài liệu thống nhất dữ liệu đầu vào, cấu hình mô hình và giao diện code giữa TV1 (dữ liệu & MLP), TV2 (KAN) và TV3 (huấn luyện & đánh giá).

## 1. Đối chiếu thiết lập với bài báo `2404.19756v5`

Các mục không ghi chi tiết trong PDF bài báo được đánh dấu *Chưa nêu trong PDF* và dùng quy ước chung của nhóm.

| Hạng mục | Trong bài báo (`2404.19756v5`) | Thiết lập nhóm |
|---|---|---|
| Hàm chính (P1) | Mục 3.1, Hình 3.1 (tr. 14): $f(x,y)=\exp(\sin(\pi x)+y^2)$ | `task="exp_sin_y2"` |
| Kiến trúc KAN | Mục 3.1 (tr. 14): `[2, 1, 1]` | `[2, 1, 1]` |
| Hàm cạnh & bậc spline | Mục 2.2 (2.10)–(2.12): $w_b\,\mathrm{SiLU}(x)+w_s\sum_i c_i B_i(x)$, $k=3$ | `SiLU` + B-spline $k=3$ |
| Số mẫu train | Mục 2.4 (tr. 10): `1000` mẫu train | `n_train = 1000` |
| Lịch lưới KAN | Mục 3.1: tăng mỗi 200 bước LBFGS qua `{3,5,10,20,50,100,200,500,1000}` | Bắt đầu với `{3,5,10,20,50}` (mở rộng thêm nếu đủ thời gian) |
| Optimizer | Mục 3.1: `LBFGS` cho cả KAN và MLP | `torch.optim.LBFGS`, log cả `steps` và `closure_evals` |
| Miền & cách lấy mẫu | *Chưa nêu trong PDF* ở Mục 3.1 (PDE Mục 3.4 dùng $[-1,1]^2$) | $[-1, 1]^d$, lấy mẫu đều ngẫu nhiên độc lập |
| Cỡ tập val / test | *Chưa nêu trong PDF* | `n_val = 200`, `n_test = 1000` |
| Cấu hình MLP | *Chưa nêu chi tiết width từng điểm* (Hình 3.1 vẽ depth 2–5) | 6 cấu hình depth 2–3, `activation="silu"`, `bias=True`, `xavier_uniform` |
| Seed & dtype | *Chưa nêu trong PDF* | Split chính `seed=42`; lặp `{42,43,44}` ở cặp trọng tâm; `float32`, `normalize_x=False` |

## 2. Danh sách bài toán

| `task` | Công thức | KAN | Miền | Ưu tiên |
|---|---|---|---|---|
| `exp_sin_y2` | $f(x, y) = \exp(\sin(\pi x) + y^2)$ | `[2, 1, 1]` | $[-1, 1]^2$ | P1 (bắt buộc) |
| `bessel` | $f(x) = J_0(20x)$ (`scipy.special.j0`) | `[1, 1]` | $[-1, 1]$ | P2 (mở rộng) |
| `multiply` | $f(x, y) = xy$ | `[2, 2, 1]` | $[-1, 1]^2$ | P2 (mở rộng) |

## 3. Module dữ liệu (`Source/src/data.py`)

Split `train` / `val` / `test` được sinh bằng `np.random.SeedSequence(seed).spawn(3)` ở độ chính xác `float64` rồi chuyển sang `float32`. Module tự kiểm tra không có điểm trùng giữa ba tập, mọi giá trị hữu hạn và trả về `split_id` dạng `{task}_s{seed}_{n_train}_{n_val}_{n_test}_{sha256[:8]}`. Cả đầu vào lẫn đầu ra đều không chuẩn hóa (`normalize_x=False`, `normalize_y=False`) để giữ đúng miền $[-1,1]^2$ và thang $\RMSE$ gốc; các module TV2/TV3 dùng chung split này và không tự chuẩn hóa thêm. (`DatasetBundle.load()` nạp bằng `weights_only=True`.)

```python
from Source.src.data import make_dataset, load_dataset_from_yaml, evaluate_target

bundle = make_dataset(task="exp_sin_y2", n_train=1000, n_val=200, n_test=1000, seed=42)
# bundle.x_train: (1000, 2), bundle.y_train: (1000, 1)
# bundle.to_pykan_dict() -> dict đầu vào cho pykan
```

## 4. Cấu hình MLP đối chứng (`Source/src/mlp_baseline.py`)

Sáu cấu hình trong `Source/configs/mlp_panel.yaml` được chọn theo ngân sách tham số học được ($4h+1$ với 1 lớp ẩn, $h^2+5h+1$ với 2 lớp ẩn) để bám quanh các mức $G \in \{3, 5, 10, 20, 50\}$ của KAN `[2, 1, 1]`, chốt trước khi xem kết quả test. Đây là đối chứng theo ngân sách tham số, không hàm ý tương đương cấu trúc giữa MLP và KAN. Dùng `SiLU` vì có đạo hàm trơn $C^\infty$ phù hợp với `LBFGS` và khớp nhánh cơ sở của KAN; kiểm tra phát triển 3 seed (200 bước) cho thấy `SiLU` (`0.1275 ± 0.0320`) và `Tanh` (`0.1009 ± 0.0247`) vượt trội `ReLU` (`0.2239 ± 0.0185`), nên cần xét thêm độ nhạy `Tanh` khi chạy đủ 1800 bước.

| `config_id` | Kiến trúc | Tham số | Mốc KAN gần nhất |
|---|---|---|---|
| `mlp_d2_w6` | `[2, 6, 1]` | 25 | KAN $G=3$ |
| `mlp_d2_w11` | `[2, 11, 1]` | 45 | KAN $G=10$ (cặp lặp 3 seed) |
| `mlp_d2_w19` | `[2, 19, 1]` | 77 | KAN $G=20$ |
| `mlp_d2_w41` | `[2, 41, 1]` | 165 | KAN $G=50$ |
| `mlp_d3_w11` | `[2, 11, 11, 1]` | 177 | Depth 3 gần $G=50$ |
| `mlp_d3_w20` | `[2, 20, 20, 1]` | 501 | Depth 3 mở rộng |

```python
from Source.src.mlp_baseline import build_mlp, count_trainable_params

model = build_mlp(input_dim=2, hidden_widths=[11], activation="silu", seed=42)
n_params = count_trainable_params(model)  # 45
```

*(Số tham số thực tế của KAN `[2, 1, 1]` sẽ được TV2 xác nhận từ `pykan` bằng `requires_grad=True`.)*

## 5. Quy ước huấn luyện & đánh giá chung

- **Lịch lưới nối tiếp vs. cố định:** Tách riêng điểm từ quá trình grid extension nối tiếp (`kan_grid_extension`) và mô hình khởi tạo lại ở từng $G$ cố định (`kan_fixed_grid`), không gộp chung hai đường.
- **Lặp seed:** Chạy 3 seed (`42, 43, 44`) ít nhất trên cặp KAN/MLP gần tham số nhất; chỉ báo `mean ± std` hoặc error bar ở điểm có $\ge 2$ seed.
- **Bấm giờ LBFGS:** Đo `time_s` trên cùng máy, gọi `torch.cuda.synchronize()` nếu chạy GPU, và lưu cả `steps` lẫn `closure_evals`.
- **Validation vs. Test:** Dùng `val_rmse` để kiểm tra cấu hình/dừng sớm; đồ thị `test_rmse` theo số tham số là đường đánh giá cuối giống Hình 3.1 của bài báo.
