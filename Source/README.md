# Source (`Source/`)

## TV3: huấn luyện KAN nối tiếp qua các mức lưới

Phần TV3 dùng lại dữ liệu YAML và hàm đếm tham số của TV1, chuẩn CSV và
nhận diện commit từ `scripts/smoke_tv1.py`; dùng adapter `build_kan` và
`refine_kan` của TV2. Không chạy lại phần smoke MLP của TV1 khi import.

Chạy từ thư mục gốc repo:

```powershell
# Kiểm tra nhanh hai mức G=3 -> 5
python -m Source.scripts.check_kan_training

# Kiểm tra cả lịch, ngân sách nhỏ (vẫn là log phát triển)
python -m Source.scripts.check_kan_training --grids 3 5 10 20 50 --steps-per-grid 2 --max-iter 5
```

`src/train_grid.py:train_kan_schedule` huấn luyện tiếp từ mô hình cũ, chỉ
refine bằng dữ liệu train và tạo optimizer mới sau mỗi lần thay lưới.
Thiết lập LBFGS lấy theo TV1: lr=1, history_size=100, tolerance_grad và
tolerance_change bằng 1e-12, strong_wolfe. Không thêm regularization hoặc
cập nhật vị trí lưới trong các bước optimizer; việc refine vẫn dùng cơ chế
chuyển lưới/hệ số của adapter TV2. Cần ghi nhận khác biệt này khi đối chiếu
với lịch cập nhật lưới trong `pykan.fit`.

Kết quả ghi vào `fixtures/tv3_grid_smoke_runs.csv` và
`fixtures/tv3_grid_smoke_report.json`, không ghi `results/runs.csv`.
CSV giữ schema TV1; `test_rmse` để trống vì chỉ đo train/validation khi dev.
`steps` và `closure_evals` tích lũy qua các mức lưới. `time_s` gồm huấn
luyện, refine và đánh giá train/validation, không gồm khởi tạo hay ghi file.
Các file fixture được thay thế mỗi lần chạy; JSON lưu cấu hình CLI thực tế.
Runner đọc trực tiếp `configs/dataset_exp_sin_y2.yaml` của TV1, không cần
YAML riêng cho TV3. Lịch lưới và ngân sách lấy từ các đối số dòng lệnh.

Đây là pipeline kiểm tra phần tăng lưới của TV3, chưa phải panel KAN–MLP
chính thức. Chưa có runner MLP theo checkpoint, lưu trọng số để resume,
đánh giá test cuối hoặc script tổng hợp biểu đồ.

Mã nguồn thực nghiệm cho đồ án KAN. Tham khảo quy ước tham số tại [../experiment_spec.md](../experiment_spec.md) và định dạng log tại [../results_schema.md](../results_schema.md).

## Cài đặt

Thiết lập kiểm tra ban đầu: Python 3.10.0, PyTorch 2.5.1+cu121,
NumPy 1.26.4, SciPy 1.15.2. Requirements hiện ghim NumPy 2.1.3 để dùng chung
với Manim 0.19.1. Khi chạy thực nghiệm mới, ghi phiên bản NumPy thực tế vào
log môi trường; các fixture/kết quả cũ không tự được xem là đã chạy với NumPy mới.

```powershell
pip install -r Source/requirements.txt
```

Để cài cả Source và Manim vào cùng `.venv` trên Python 3.12:

```powershell
uv --no-cache pip install --python .venv/Scripts/python.exe -r Manim/requirements.txt
```

## Chạy kiểm tra phần TV1 (Dữ liệu & MLP đối chứng)

```powershell
# Unit tests
pytest -v Source/tests

# Kiểm tra phân chia dữ liệu (thêm --save-out để xuất file .pt dùng chung)
python Source/scripts/prepare_data.py

# Smoke test dữ liệu + 10 cấu hình MLP (xuất log tạm vào fixtures/)
python Source/scripts/smoke_tv1.py
```

## Bàn giao cho trainer TV3

Đọc `optimizer.checkpoint_outer_steps` trong `configs/mlp_panel.yaml` để lưu MLP tại các mốc tích lũy tương ứng với KAN. Quy ước cột `steps`, `max_iter`, `closure_evals` và cách phân biệt SiLU/Tanh nằm trong [schema kết quả](../results_schema.md).

### KAN: khởi tạo và tăng lưới (TV2)

Với `pykan==0.2.8`, dùng adapter cho cả khởi tạo và tăng lưới:

```python
import torch
from Source.src.kan_model import build_kan, refine_kan

model = build_kan(seed=42)  # CPU, khởi tạo trực tiếp float64, G=3
for grid in (3, 5, 10, 20, 50):
    if grid != 3:
        model = refine_kan(model, grid, bundle.x_train)
    # Model mới sau refine: tạo lại optimizer và closure dùng model hiện tại.
    optimizer = torch.optim.LBFGS(
        [p for p in model.parameters() if p.requires_grad],
        max_iter=20, line_search_fn="strong_wolfe",
    )
    # TV3 thực hiện 200 optimizer.step(closure) cho mỗi grid tại đây.
```

`build_kan` và `refine_kan` tạm đặt dtype mặc định thành `float64` rồi khôi phục,
đồng thời giữ nguyên RNG của PyTorch, NumPy và Python (kể cả khi có exception).
Không chạy đồng thời adapter với code khác thay đổi dtype/RNG toàn cục.
`refine_kan` nhận tensor train CPU/float64, nạp lại cache bằng train ngay trước
refine và tắt `auto_save` trên cả model nguồn lẫn model mới. Validation/test
chỉ dùng đánh giá; không truyền vào bước tăng lưới. Tránh gọi trực tiếp
`model.refine()` vì pykan dùng cache forward gần nhất và bật lại `auto_save`.

Kiểm tra bàn giao (không thay thế chạy huấn luyện panel):

```powershell
python Source/scripts/check_kan.py
python -m pytest -v Source/tests/test_kan_model.py
```

Script kiểm tra forward/backward tại cả 5 lưới, số tham số tương ứng
`24, 30, 45, 75, 165`, dtype và cache train sau refine. Tests bổ sung kiểm tra
dtype mặc định, RNG, thứ tự đánh giá và việc không sinh checkpoint tự động.

## Các file đã cài đặt

- `src/data.py`: `TASK_SPECS`, `evaluate_target()`, `DatasetBundle`, `make_dataset()`, `load_dataset_from_yaml()`
- `src/mlp_baseline.py`: `MLPBaseline`, `build_mlp()`, `count_trainable_params()`, `expected_mlp_params()`, `load_mlp_panel_configs()`
- `configs/dataset_{exp_sin_y2,bessel,multiply}.yaml` và `configs/mlp_panel.yaml`
- `scripts/prepare_data.py` và `scripts/smoke_tv1.py`
- `tests/test_data.py` và `tests/test_mlp_baseline.py`
