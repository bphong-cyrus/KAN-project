# Mã nguồn tái lập thực nghiệm (`Source/`)

Thư mục mã nguồn tái lập thực nghiệm cho đồ án **KAN: Kolmogorov–Arnold Networks** (CSC14005).  
Đặc tả dữ liệu, API và quy tắc chạy xem tại [../experiment_spec.md](../experiment_spec.md); chuẩn cột log xem tại [../results_schema.md](../results_schema.md).

---

## 1. Cài đặt môi trường

Khuyến nghị Python 3.10+ (đã kiểm chứng trên Python `3.10.0`, PyTorch `2.5.1+cu121`, NumPy `1.26.4`, SciPy `1.15.2`).

```powershell
# Từ thư mục gốc KAN-project/
pip install -r Source/requirements.txt
```

---

## 2. Lệnh chạy phần TV1 (Data + MLP Baseline)

### 2.1. Chạy unit tests bằng `pytest`

```powershell
pytest -v Source/tests
```

### 2.2. Chuẩn bị / kiểm tra bộ dữ liệu dùng chung (`prepare_data.py`)

```powershell
python Source/scripts/prepare_data.py
# Hoặc xuất sẵn split cố định ra file .pt:
python Source/scripts/prepare_data.py --save-out fixtures/split_exp_sin_y2_s42.pt
```

### 2.3. Chạy smoke test end-to-end TV1 (`smoke_tv1.py`)

```powershell
python Source/scripts/smoke_tv1.py
```

Ghi kết quả chạy thử vào `fixtures/tv1_smoke_runs.csv` (`experiment_id="smoke_tv1_dev_only"`) và `fixtures/tv1_smoke_report.json` để xác nhận đường ống dữ liệu và 6 cấu hình MLP (25, 45, 77, 165, 177, 501 tham số). **Không** dùng dữ liệu trong `fixtures/` thay cho `results/runs.csv`.

---

## 3. Thành phần TV1 đã cài đặt

- `Source/src/data.py`: `TASK_SPECS`, `evaluate_target()`, `DatasetBundle`, `make_dataset()`, `load_dataset_from_yaml()`.
- `Source/src/mlp_baseline.py`: `MLPBaseline`, `build_mlp()`, `count_trainable_params()`, `expected_mlp_params()`, `load_mlp_panel_configs()`.
- `Source/configs/dataset_{exp_sin_y2,bessel,multiply}.yaml` và `Source/configs/mlp_panel.yaml`.
- `Source/scripts/prepare_data.py` và `Source/scripts/smoke_tv1.py`.
- `Source/tests/test_data.py` và `Source/tests/test_mlp_baseline.py`.
