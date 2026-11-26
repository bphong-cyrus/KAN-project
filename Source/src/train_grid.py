import time
import math

import torch

from .metrics import rmse
from .kan_model import build_kan, refine_kan
from .mlp_baseline import count_trainable_params

def train_lbfgs(model, data, steps=5, max_iter=20):
    """LBFGS theo thiết lập chạy thử TV1, dùng lại cho từng giai đoạn KAN."""
    if steps < 1 or max_iter < 1:
        raise ValueError("steps và max_iter phải >= 1.")
    optimizer = torch.optim.LBFGS(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.0,
        max_iter=max_iter,
        history_size=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    closure_evals = 0

    def closure():
        nonlocal closure_evals
        closure_evals += 1

        optimizer.zero_grad()
        prediction = model(data.x_train)

        if prediction.shape != data.y_train.shape:
            raise ValueError("Shape dự đoán và đáp án không khớp.")

        loss = (prediction - data.y_train).square().mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Loss chứa NaN hoặc Inf.")

        loss.backward()
        return loss

    history = []
    start = time.perf_counter()
    model.train()

    for step in range(1, steps + 1):
        optimizer.step(closure)

        # Chấm lại bằng tham số sau cập nhật.
        train_error = rmse(model, data.x_train, data.y_train)
        val_error = rmse(model, data.x_val, data.y_val)

        if not all(math.isfinite(v) for v in (train_error, val_error)):
            raise RuntimeError("RMSE chứa NaN hoặc Inf.")

        row = {
            "step": step,
            "train_rmse": train_error,
            "val_rmse": val_error,
            "closure_evals": closure_evals,
            "time_s": time.perf_counter() - start,
        }
        history.append(row)

        print(
            f"Step {step}/{steps}: "
            f"train={train_error:.6f}, val={val_error:.6f}"
        )

    return {
        "history": history,
        "steps": steps,
        "max_iter": max_iter,
        "closure_evals": closure_evals,
        # Thời gian này gồm cả việc chấm RMSE sau mỗi bước.
        "time_s": time.perf_counter() - start,
    }


def train_kan_schedule(data, grids=(3, 5, 10, 20, 50), *, seed=42,
                       steps_per_grid=200, max_iter=20, k=3):
    """Train nối tiếp; chỉ dùng train để refine, tạo optimizer mới mỗi mức.

    Không cập nhật vị trí lưới trong bước optimizer. time_s tích lũy gồm
    train, refine và đánh giá train/val; không gồm khởi tạo. Không đọc test.
    """
    grids = tuple(grids)
    if (not grids or any(type(g) is not int or g < 1 for g in grids)
            or any(a >= b for a, b in zip(grids, grids[1:]))):
        raise ValueError("grids phải là các số nguyên dương tăng nghiêm ngặt.")
    if steps_per_grid < 1 or max_iter < 1:
        raise ValueError("Ngân sách huấn luyện phải >= 1.")
    if data.x_train.dtype != torch.float64 or data.x_train.device.type != 'cpu':
        raise ValueError("Adapter KAN yêu cầu dữ liệu CPU/float64.")

    model = build_kan(width=[data.x_train.shape[1], 1, 1],
                      grid=grids[0], k=k, seed=seed)
    initial_rmse = rmse(model, data.x_train, data.y_train)
    checkpoints = []
    total_evals = 0
    start = time.perf_counter()
    for index, grid in enumerate(grids):
        if index:
            model = refine_kan(model, grid, data.x_train)
        print(f"\nKAN: G={grid}", flush=True)
        result = train_lbfgs(model, data, steps_per_grid, max_iter)
        total_evals += result['closure_evals']
        last = result['history'][-1]
        checkpoints.append({
            'G': grid, 'k': k,
            'params': count_trainable_params(model),
            'train_rmse': last['train_rmse'], 'val_rmse': last['val_rmse'],
            'steps': (index + 1) * steps_per_grid, 'max_iter': max_iter,
            'closure_evals': total_evals,
            'time_s': time.perf_counter() - start,
        })
    return model, {'initial_train_rmse': initial_rmse, 'checkpoints': checkpoints}
