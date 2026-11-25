"""Khởi tạo mô hình KAN số cho thực nghiệm."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import random
from threading import RLock

import numpy as np
import torch

from kan import KAN


_PYKAN_LOCK = RLock()


@contextmanager
def _isolated_pykan_state():
    """Cô lập các mặc định toàn cục mà pykan dùng khi tạo lớp spline.

    Lock tuần tự hóa các lời gọi adapter; không dùng adapter song song với
    code khác tự thay đổi dtype hoặc RNG toàn cục.
    """
    with _PYKAN_LOCK, torch.random.fork_rng():
        previous_dtype = torch.get_default_dtype()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            torch.set_default_dtype(torch.float64)
            yield
        finally:
            torch.set_default_dtype(previous_dtype)
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def build_kan(
    width: Sequence[int] = (2, 1, 1),
    *,
    grid: int = 3,
    k: int = 3,
    seed: int = 42,
) -> KAN:
    """Tạo KAN chạy CPU với tham số float64 và nhánh cơ sở SiLU."""
    with _isolated_pykan_state():
        model = KAN(
            width=list(width),
            grid=grid,
            k=k,
            seed=seed,
            device="cpu",
            base_fun="silu",
            affine_trainable=False,
            symbolic_enabled=False,
            auto_save=False,
        )
        # pykan vẫn đăng ký tham số symbolic khi tắt nhánh này.
        model.symbolic_fun.requires_grad_(False)
    return model


def refine_kan(model: KAN, new_grid: int, x_train: torch.Tensor) -> KAN:
    """Tăng lưới KAN CPU/float64 dùng riêng dữ liệu train.

    Trả về model mới; caller phải tạo lại optimizer từ model.parameters().
    Chỉ dành cho KAN số do build_kan tạo, với save_act=True.
    """
    if isinstance(new_grid, bool) or not isinstance(new_grid, int) or new_grid <= model.grid:
        raise ValueError("new_grid phải là số nguyên lớn hơn lưới hiện tại.")
    if x_train.ndim != 2 or x_train.shape[0] == 0 or x_train.shape[1] != model.width_in[0]:
        raise ValueError("x_train phải có shape (N, input_dim) với N > 0.")
    if x_train.device.type != "cpu" or x_train.dtype != torch.float64:
        raise ValueError("x_train phải là tensor CPU có dtype float64.")
    if any(p.device.type != "cpu" or p.dtype != torch.float64 for p in model.parameters()):
        raise ValueError("model phải chạy CPU với tham số float64.")
    if not torch.isfinite(x_train).all():
        raise ValueError("x_train chứa NaN hoặc Inf.")
    if model.symbolic_enabled or not model.save_act:
        raise ValueError("Cần KAN số với symbolic_enabled=False và save_act=True.")

    with _isolated_pykan_state(), torch.no_grad():
        # refine() dùng cache_data và log_history() của model nguồn.
        model.auto_save = False
        model(x_train.detach())
        refined = model.refine(new_grid)
        # pykan 0.2.8 bật lại auto_save khi tạo model trong refine().
        refined.auto_save = False
        refined.symbolic_fun.requires_grad_(False)
        refined.train(model.training)
    return refined
