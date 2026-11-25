"""Kiểm tra KAN trên dữ liệu P1 bằng một lượt forward và backward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Source.src.data import load_dataset_from_yaml
from Source.src.kan_model import build_kan
from Source.src.mlp_baseline import count_trainable_params


def main() -> None:
    config_path = REPO_ROOT / "Source/configs/dataset_exp_sin_y2.yaml"
    bundle = load_dataset_from_yaml(config_path)
    model = build_kan()
    inputs = bundle.x_train[:16]
    targets = bundle.y_train[:16]
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

    print(f"Forward/backward OK | shape={tuple(predictions.shape)} | dtype={predictions.dtype}")
    print(f"params={count_trainable_params(model)}")


if __name__ == "__main__":
    main()