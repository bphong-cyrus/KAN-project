# Source (`Source/`)

Mã nguồn thực nghiệm cho đồ án KAN. Tham khảo quy ước tham số tại [../experiment_spec.md](../experiment_spec.md) và định dạng log tại [../results_schema.md](../results_schema.md).

## Cài đặt

Đã kiểm tra trên Python 3.10.0, PyTorch 2.5.1+cu121, NumPy 1.26.4, SciPy 1.15.2:

```powershell
pip install -r Source/requirements.txt
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

## Các file đã cài đặt

- `src/data.py`: `TASK_SPECS`, `evaluate_target()`, `DatasetBundle`, `make_dataset()`, `load_dataset_from_yaml()`
- `src/mlp_baseline.py`: `MLPBaseline`, `build_mlp()`, `count_trainable_params()`, `expected_mlp_params()`, `load_mlp_panel_configs()`
- `configs/dataset_{exp_sin_y2,bessel,multiply}.yaml` và `configs/mlp_panel.yaml`
- `scripts/prepare_data.py` và `scripts/smoke_tv1.py`
- `tests/test_data.py` và `tests/test_mlp_baseline.py`
