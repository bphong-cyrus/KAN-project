# Chuẩn định dạng kết quả (`results_schema.md`)

Quy ước cấu trúc file `results/runs.csv` và `results/summary.json` dùng chung giữa code thực nghiệm, script phân tích và biểu đồ báo cáo/Manim.

## 1. Cột của `results/runs.csv`

Mỗi dòng ứng với một cấu hình, một checkpoint huấn luyện và một seed. Không ghi kết quả smoke test (`smoke_*`) vào `results/runs.csv`.

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `run_id` | `string` | Mã duy nhất cho từng dòng, gồm cấu hình, seed và checkpoint, vd: `mlp_d2_w11_s42_step600`, `kan_g10_s42_seq_step600` |
| `experiment_id` | `string` | Tên đợt chạy (`core_panel_mlp`, `kan_grid_extension`, `kan_fixed_grid`, `p2_bessel`...) |
| `task` | `string` | `exp_sin_y2`, `bessel`, hoặc `multiply` |
| `model` | `string` | `MLP` hoặc `KAN` |
| `config_id` | `string` | Mã cấu hình ổn định qua các seed và checkpoint. MLP dùng đúng mã từ YAML, vd: `mlp_d2_w11` và `mlp_tanh_d2_w11`. |
| `activation` | `string` / rỗng | Kích hoạt lớp ẩn của MLP, vd: `silu` hoặc `tanh`. Để trống với KAN. |
| `architecture` | `string` | Dạng chuỗi list độ rộng, vd: `"[2, 11, 1]"` hoặc `"[2, 1, 1]"` |
| `G` | `int` / rỗng | Số khoảng lưới của KAN; để trống với MLP |
| `k` | `int` / rỗng | Bậc B-spline của KAN (mặc định `3`); để trống với MLP |
| `seed` | `int` | Hạt giống ngẫu nhiên (`42`, `43`, `44`...) |
| `split_id` | `string` | Mã `DatasetBundle.split_id` trả về từ `Source/src/data.py` |
| `params` | `int` | Số tham số `requires_grad=True` đếm trực tiếp từ mô hình |
| `train_rmse` | `float` | RMSE trên tập `train` |
| `val_rmse` | `float` | RMSE trên tập `val` |
| `test_rmse` | `float` | RMSE trên tập `test` |
| `time_s` | `float` | Thời gian chạy tích lũy (giây) trên cùng máy |
| `optimizer` | `string` | Tên & tham số optimizer, vd: `LBFGS(max_iter=20,strong_wolfe)` |
| `steps` | `int` | Số lần gọi `optimizer.step(closure)` (outer steps) tích lũy |
| `max_iter` | `int` | Giới hạn số vòng lặp nội bộ trong mỗi lần gọi `optimizer.step` |
| `closure_evals` | `int` | Tổng số lần gọi closure (bao gồm cả Strong-Wolfe line search) |
| `config_path` | `string` | Đường dẫn tương đối từ gốc repo tới file YAML cấu hình |
| `code_commit` | `string` | Git short hash (`git rev-parse --short HEAD`) |

## 2. Cấu trúc `results/summary.json`

File JSON tổng hợp theo `(experiment_id, task, model, config_id, activation, architecture, G, k, steps, max_iter)` để vẽ đồ thị và dùng cho Manim Cảnh 3. Mỗi điểm nhận tối đa một dòng từ mỗi seed. Dùng `steps` và `max_iter` để chọn các điểm có cùng ngân sách, đồng thời giữ `config_id` và `activation` để phân biệt các MLP có cùng kiến trúc.

```json
{
  "schema_version": "2.0",
  "is_demo_fixture": false,
  "generated_at": "YYYY-MM-DDTHH:MM:SSZ",
  "source_csv": "results/runs.csv",
  "code_commit": "<git-short-hash>",
  "task": "exp_sin_y2",
  "split_id": "<split-id>",
  "scaling_points": [
    {
      "model": "KAN",
      "experiment_id": "kan_grid_extension",
      "config_id": "kan_d2_w1_seq",
      "activation": null,
      "architecture": "[2, 1, 1]",
      "G": 10,
      "k": 3,
      "steps": 600,
      "max_iter": 20,
      "params": 0,
      "n_seeds": 1,
      "train_rmse_mean": 0.0,
      "train_rmse_std": null,
      "test_rmse_mean": 0.0,
      "test_rmse_std": null,
      "time_s_mean": 0.0,
      "run_ids": ["..."]
    }
  ],
  "grid_extension_curve": []
}
```

Khi `n_seeds == 1`, các trường `*_std` để giá trị `null`. Mọi điểm hiển thị trên báo cáo và video phải truy ngược được về danh sách `run_ids` trong `results/runs.csv`.
