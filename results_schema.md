# Đặc tả định dạng dữ liệu đầu ra (`results_schema.md`)

Tài liệu quy định chuẩn schema cho `results/runs.csv` (nguồn dữ liệu duy nhất cho báo cáo và đồ thị) và `results/summary.json` (dùng bởi cảnh Manim 3 của TV4).

---

## 1. Schema cho `results/runs.csv`

Mỗi dòng tương ứng với **một cấu hình / một checkpoint lưới $G$ / một seed duy nhất**. Không ghi đè hoặc gộp nhiều seed thành một dòng trong `runs.csv`.

| Thứ tự | Tên cột | Kiểu | Bắt buộc | Mô tả & quy ước |
|---|---|---|---|---|
| 1 | `run_id` | `string` | Có | Định danh duy nhất cho mỗi dòng (ví dụ: `mlp_d2_w11_s42` hoặc `kan_g10_s42_seq`). |
| 2 | `experiment_id` | `string` | Có | Mã đợt chạy (`core_panel_mlp`, `kan_grid_extension`, `kan_fixed_grid`, `p2_bessel`, `smoke_tv1_dev_only`...). Không trộn dòng `smoke_*` vào `results/runs.csv`. |
| 3 | `task` | `string` | Có | Tên bài toán: `exp_sin_y2`, `bessel`, hoặc `multiply`. |
| 4 | `model` | `string` | Có | Tên họ mô hình: `MLP` hoặc `KAN`. |
| 5 | `architecture` | `string` | Có | Chuỗi kiến trúc theo danh sách độ rộng, ví dụ `"[2, 11, 1]"` hoặc `"[2, 1, 1]"`. |
| 6 | `G` | `int / empty` | Với KAN | Số khoảng lưới của KAN (`3, 5, 10, 20, 50, ...`). **Để trống với MLP.** |
| 7 | `k` | `int / empty` | Với KAN | Bậc B-spline (mặc định `3`). **Để trống với MLP.** |
| 8 | `seed` | `int` | Có | Seed khởi tạo/huấn luyện (`42`, `43`, `44`, ...). |
| 9 | `split_id` | `string` | Có | Mã split trả về từ `DatasetBundle.split_id` trong `Source/src/data.py`. |
| 10 | `params` | `int` | Có | Tổng số tham số có `requires_grad=True` đếm trực tiếp từ mô hình. |
| 11 | `train_rmse` | `float` | Có | RMSE trên tập `train` tại checkpoint đang ghi. |
| 12 | `val_rmse` | `float` | Có | RMSE trên tập `val` tại checkpoint đang ghi (dùng kiểm tra chọn cấu hình/early stop). |
| 13 | `test_rmse` | `float` | Có | RMSE trên tập `test` dùng chung cho đối chiếu cuối. |
| 14 | `time_s` | `float` | Có | Thời gian huấn luyện tích lũy (giây) trên cùng máy đo; phải đồng bộ GPU trước khi bấm giờ nếu dùng CUDA. |
| 15 | `optimizer` | `string` | Có | Tên optimizer thực tế (ví dụ `LBFGS`). |
| 16 | `steps` | `int` | Có | Số bước optimizer tích lũy đến checkpoint hiện tại. |
| 17 | `closure_evals` | `int` | Khuyến nghị | Tổng số lần gọi `closure()` thực tế trong `LBFGS` (phân biệt với outer steps). |
| 18 | `config_path` | `string` | Có | Đường dẫn tương đối đến file YAML cấu hình đã dùng (ví dụ `Source/configs/mlp_panel.yaml`). |
| 19 | `code_commit` | `string` | Có | Git commit hash ngắn (`git rev-parse --short HEAD`) của repo tại thời điểm chạy (kèm hậu tố `-dirty` nếu còn thay đổi chưa commit). |

*(17 cột chuẩn từ `Project-Plan.md` Mục 3.1 được giữ nguyên tên chính xác; bổ sung thêm `val_rmse` và `closure_evals` theo Mục 6.1 để phục vụ validation policy và đối chiếu chi phí LBFGS.)*

---

## 2. Schema cho `results/summary.json` (đầu vào Manim Cảnh 3 — TV4)

`analysis/evaluate.py` (TV3 viết, TV4 chạy/kiểm) đọc `results/runs.csv`, tổng hợp theo `(experiment_id, task, model, architecture, G)` và xuất ra `results/summary.json` với cấu trúc sau:

```json
{
  "schema_version": "1.0",
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
      "architecture": "[2, 1, 1]",
      "G": 3,
      "k": 3,
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

### Quy tắc kiểm tra hợp lệ:
- `is_demo_fixture` bắt buộc phải là `false` trong `results/summary.json` chính thức.
- Chỉ điền `test_rmse_std` khi `n_seeds >= 2`; nếu `n_seeds == 1`, giá trị chuẩn phải là `null`.
- Mọi điểm trong `Manim/scenes/03_results.py` và `report/tables/core_results.tex` phải truy ngược được về danh sách `run_ids` trong `results/runs.csv`.
