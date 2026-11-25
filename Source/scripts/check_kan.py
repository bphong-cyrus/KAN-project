"""Kiểm tra forward/backward và lịch tăng lưới KAN trên dữ liệu train P1."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Source.src.data import load_dataset_from_yaml
from Source.src.kan_model import build_kan, refine_kan
from Source.src.mlp_baseline import count_trainable_params


def check_forward_backward(model, inputs, targets) -> None:
    model.zero_grad(set_to_none=True)
    predictions = model(inputs)

    if predictions.shape != targets.shape:
        raise RuntimeError(f"Sai shape: {predictions.shape}, cần {targets.shape}.")
    if predictions.dtype != torch.float64:
        raise RuntimeError(f"Sai dtype: {predictions.dtype}, cần torch.float64.")
    if not torch.isfinite(predictions).all():
        raise RuntimeError("Forward trả về NaN hoặc Inf.")

    loss = (predictions - targets).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("MSE trả về NaN hoặc Inf.")
    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Gradient bị thiếu hoặc không hữu hạn: {name}.")

    if model.auto_save:
        raise RuntimeError("auto_save phải tắt sau khi khởi tạo/tăng lưới.")
    if any(p.dtype != torch.float64 for p in model.parameters()):
        raise RuntimeError("Tất cả tham số phải có dtype float64.")
    expected_params = 3 * (model.grid + model.k + 2)
    params = count_trainable_params(model)
    if params != expected_params:
        raise RuntimeError(f"Sai số tham số: {params}, cần {expected_params}.")
    print(f"G={model.grid} | forward/backward OK | dtype={predictions.dtype} | params={params}")


def main() -> None:
    config_path = REPO_ROOT / "Source/configs/dataset_exp_sin_y2.yaml"
    bundle = load_dataset_from_yaml(config_path)
    model = build_kan()
    check_forward_backward(model, bundle.x_train[:16], bundle.y_train[:16])
    for grid in (5, 10, 20, 50):
        # Mô phỏng trainer vừa đánh giá validation trước khi tăng lưới.
        with torch.no_grad():
            model(bundle.x_val)
        model = refine_kan(model, grid, bundle.x_train)
        torch.testing.assert_close(model.cache_data, bundle.x_train, rtol=0, atol=0)
        check_forward_backward(model, bundle.x_train[:16], bundle.y_train[:16])


if __name__ == "__main__":
    main()
