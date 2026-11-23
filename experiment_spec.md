# Đặc tả thực nghiệm & hợp đồng giao diện (`experiment_spec.md`)

Tài liệu này là hợp đồng kỹ thuật giữa **TV1 (Dữ liệu & MLP)**, **TV2 (KAN)** và **TV3 (Huấn luyện, Lưới & Độ đo)** nhằm bảo đảm toàn bộ mô hình được đánh giá công bằng trên cùng bộ dữ liệu, cùng split và cùng chuẩn ghi log.

---

## 1. Phân tách thiết lập từ bài báo gốc vs. Lựa chọn của nhóm

Mọi thông tin gán nhãn **“Xác định từ bài báo”** đều đối chiếu trực tiếp từ `2404.19756v5.pdf` (*KAN: Kolmogorov–Arnold Networks*, accepted at **ICLR 2025**, `arXiv:2404.19756v5`, 09/02/2025). Những chi tiết không được nêu tường minh trong PDF được ghi rõ **“Chưa xác định từ bài báo gốc”** và đi kèm quy tắc khóa của nhóm.

| Hạng mục | Nguồn bài báo `2404.19756v5` | Trạng thái xác định | Lựa chọn khóa của nhóm |
|---|---|---|---|
| Hàm mục tiêu chính (P1) | Mục 3.1, Hình 3.1 (tr. 14): $f(x,y)=\exp(\sin(\pi x)+y^2)$ | Xác định từ bài báo | Giữ nguyên hàm chính: `task="exp_sin_y2"` |
| Kiến trúc KAN chính | Mục 3.1 (tr. 14): KAN `[2, 1, 1]` | Xác định từ bài báo | Giữ nguyên `[2, 1, 1]` |
| Hàm cạnh KAN & bậc spline | Mục 2.2, Công thức (2.10)–(2.12) (tr. 6); Mục 2.3 (tr. 7): $\phi(x)=w_b\,\mathrm{SiLU}(x)+w_s\sum_i c_i B_i(x)$, $k=3$ | Xác định từ bài báo | Giữ nguyên `SiLU` + B-spline bậc $k=3$ |
| Số mẫu huấn luyện | Mục 2.4 (tr. 10): *"our training samples are 1000"* cho bài toán đồ chơi $\exp(\sin(\pi x)+y^2)$ | Xác định từ bài báo | `n_train = 1000` |
| Lịch mở rộng lưới KAN | Mục 3.1 (tr. 14): tăng lưới mỗi 200 bước LBFGS qua $G \in \{3,5,10,20,50,100,200,500,1000\}$ (tổng 1800 bước) | Xác định từ bài báo | Thu nhỏ có kiểm soát: $G \in \{3,5,10,20,50\}$ (tối thiểu 4 mức đầu; mở rộng thêm nếu đủ ngân sách tính toán) |
| Optimizer | Mục 3.1 (tr. 14): LBFGS cho cả KAN và MLP | Xác định từ bài báo | Dùng `torch.optim.LBFGS` cho cả KAN và MLP; ghi rõ `steps` và `closure_evals` |
| Miền lấy mẫu & cách lấy mẫu | Không ghi tường minh trong Mục 3.1 (Mục 3.4 PDE dùng $[-1,1]^2$); phụ thuộc hàm `create_dataset` trong repo `pykan` | **Chưa xác định từ bài báo gốc** (chưa đối chiếu code `pykan` trực tiếp tại thời điểm viết spec) | Miền chuẩn $[-1, 1]^d$, lấy mẫu ngẫu nhiên đều (uniform) độc lập cho từng split |
| Số mẫu validation / test | Mục 3.1 không nêu kích thước test cụ thể trong văn bản PDF | **Chưa xác định từ bài báo gốc** | `n_val = 200`, `n_test = 1000` |
| Cấu hình width/depth/activation/init của MLP đối chứng | Mục 3.1 ghi chung *"different depths and widths"* (Hình 3.1 vẽ depth 2–5); không liệt kê chi tiết width từng điểm trên panel $f(x,y)=\exp(\sin(\pi x)+y^2)$ | **Chưa xác định từ bài báo gốc** | Khóa trước 6 cấu hình MLP (depth 2 và 3), `activation="silu"`, `bias=True`, `init="xavier_uniform"` |
| Seed, dtype và chuẩn hóa đầu vào | Không nêu trong PDF | **Chưa xác định từ bài báo gốc** | `seed=42` cho split chính; lặp `{42, 43, 44}` cho cặp cấu hình trọng tâm; `dtype="float32"`; `normalize_x=False` |

---

## 2. Danh sách bài toán & quy ước hàm

| `task` | Công thức | Kiến trúc KAN | Miền mặc định | Mức ưu tiên |
|---|---|---|---|---|
| `exp_sin_y2` | $f(x, y) = \exp(\sin(\pi x) + y^2)$ | `[2, 1, 1]` | $[-1, 1]^2$ | **P1 — Bắt buộc (panel chính)** |
| `bessel` | $f(x) = J_0(20x)$ (`scipy.special.j0`) | `[1, 1]` | $[-1, 1]$ | P2 — Bổ sung nếu panel chính đã xong |
| `multiply` | $f(x, y) = xy$ | `[2, 2, 1]` | $[-1, 1]^2$ | P2 — Bổ sung nếu panel chính đã xong |

---

## 3. Hợp đồng dữ liệu dùng chung (`Source/src/data.py` — TV1)

1. **Tính tách biệt và tái lập split:**
   - Dùng `np.random.SeedSequence(seed).spawn(3)` tạo 3 luồng RNG độc lập cho `train`, `val`, `test`.
   - Sinh tọa độ và tính nhãn mục tiêu ở `float64` rồi mới ép kiểu sang `dtype` mục tiêu (`float32` mặc định) để tránh sai số lượng tử hóa.
   - Kiểm tra tự động: tất cả tensor hữu hạn, hình dạng chuẩn `x: (N, d)`, `y: (N, 1)`, không có điểm trùng lặp chính xác giữa `train`, `val`, `test`.
   - Sinh `split_id` xác định theo dạng `{task}_s{seed}_{n_train}_{n_val}_{n_test}_{sha256[:8]}` từ byte dữ liệu các tập.
2. **Không rò rỉ dữ liệu:** Nếu bật `normalize_x=True`, giá trị `mean` và `std` chỉ được tính trên `x_train` và áp dụng lại cho `x_val`, `x_test`. Mặc định `normalize_x=False` vì miền $[-1, 1]^d$ đã cân bằng.
3. **API bắt buộc:**

```python
from Source.src.data import make_dataset, load_dataset_from_yaml, evaluate_target, DatasetBundle

bundle: DatasetBundle = make_dataset(
    task="exp_sin_y2",
    n_train=1000,
    n_val=200,
    n_test=1000,
    seed=42,
    domain=(-1.0, 1.0),
    dtype="float32",
    sampling="uniform",
    normalize_x=False,
    device="cpu",
)
# bundle.x_train: (1000, 2), bundle.y_train: (1000, 1)
# bundle.x_val:   (200, 2),  bundle.y_val:   (200, 1)
# bundle.x_test:  (1000, 2), bundle.y_test:  (1000, 1)
# bundle.to_pykan_dict() -> {"train_input": ..., "train_label": ..., "test_input": ..., "test_label": ...}
```

---

## 4. Hợp đồng MLP baseline (`Source/src/mlp_baseline.py` — TV1)

- **API bắt buộc:**

```python
from Source.src.mlp_baseline import build_mlp, count_trainable_params, expected_mlp_params, load_mlp_panel_configs

model = build_mlp(
    input_dim=2,
    hidden_widths=[6],
    activation="silu",
    output_dim=1,
    bias=True,
    init="xavier_uniform",
    seed=42,
    dtype="float32",
    device="cpu",
)
n_params: int = count_trainable_params(model)
```

- **Quy tắc khóa cấu hình trước khi xem test RMSE:** Sáu cấu hình MLP trong `Source/configs/mlp_panel.yaml` được chọn thuần túy dựa trên số tham số học được để phủ quanh các mức $G \in \{3, 5, 10, 20, 50\}$ của KAN `[2, 1, 1]` (theo công thức đếm tham số lý thuyết $E(G+k+2)$ với $E=3, k=3$ cho ra khoảng 24–168 tham số):

| `config_id` | Kiến trúc | Số hidden layers | Công thức đếm tham số | `params` thực tế | Mốc KAN tham chiếu |
|---|---|---|---|---|---|
| `mlp_d2_w6` | `[2, 6, 1]` | 1 (depth 2) | $4h + 1$ | **25** | Gần KAN $G=3$ |
| `mlp_d2_w11` | `[2, 11, 1]` | 1 (depth 2) | $4h + 1$ | **45** | Gần KAN $G=10$ |
| `mlp_d2_w19` | `[2, 19, 1]` | 1 (depth 2) | $4h + 1$ | **77** | Gần KAN $G=20$ |
| `mlp_d2_w41` | `[2, 41, 1]` | 1 (depth 2) | $4h + 1$ | **165** | Gần KAN $G=50$ |
| `mlp_d3_w11` | `[2, 11, 11, 1]` | 2 (depth 3) | $h^2 + 5h + 1$ | **177** | Đối chứng depth 3 gần $G=50$ |
| `mlp_d3_w20` | `[2, 20, 20, 1]` | 2 (depth 3) | $h^2 + 5h + 1$ | **501** | Mở rộng capacity depth 3 |

*Ghi chú:* Số tham số thực tế của KAN `[2, 1, 1]` theo phiên bản `pykan` cụ thể sẽ được **TV2** xác nhận và ghi lại bằng `count_trainable_params` (không đoán trước).

---

## 5. Giao diện kỳ vọng cho TV2 & TV3

- **TV2 (`Source/src/kan_model.py`):**
  - `build_kan(width: list[int], grid: int = 3, k: int = 3, seed: int = 42, device: str = "cpu") -> torch.nn.Module`
  - `count_kan_params(model: torch.nn.Module) -> int` (đếm tham số học được thực tế từ `pykan`, tách riêng buffer grid không học được).
- **TV3 (`Source/src/train_grid.py`, `Source/src/metrics.py`):**
  - `compute_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float`: $\sqrt{\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2}$.
  - `train_lbfgs(model, dataset, steps, ...)`: huấn luyện với `LBFGS`, ghi lại đồng thời `steps` (outer optimizer steps) và `closure_evals` (số lần gọi hàm closure/line search), đồng bộ thiết bị (`torch.cuda.synchronize()` nếu dùng CUDA) trước khi bấm giờ `time_s`.
  - **Tách bạch hai thí nghiệm về $G$:**
    - **Thí nghiệm A (`kan_grid_extension`):** Các checkpoint tại $G \in \{3, 5, 10, 20, 50\}$ trên một quá trình huấn luyện nối tiếp (mỗi mức $G$ kế thừa hàm spline từ mức trước đó).
    - **Thí nghiệm B (`kan_fixed_grid` — đối chứng):** Khởi tạo lại từ đầu tại từng mức $G$ cố định với cùng ngân sách optimizer và seed, không trộn chung số liệu với Thí nghiệm A.
- **Quy tắc lặp seed & chọn mô hình:**
  - Chạy tối thiểu **3 seed** (`[42, 43, 44]`) trên ít nhất một cặp cấu hình KAN và MLP có số tham số gần nhau; chỉ tính trung bình $\pm$ độ lệch chuẩn (và vẽ thanh sai số) tại các điểm thực sự có $n_{\text{seed}} \ge 2$.
  - Sử dụng `val_rmse` để kiểm tra chọn cấu hình / điểm dừng; đường cong `test_rmse` theo số tham số trong Hình 3.1 được ghi nhận rõ là **đường đánh giá (evaluation curve)** giống quy ước trực quan hóa của bài báo gốc, không dùng `test_rmse` để chọn lại siêu tham số.
