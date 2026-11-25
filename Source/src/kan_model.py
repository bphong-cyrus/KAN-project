"""Khởi tạo mô hình KAN số cho thực nghiệm."""

from __future__ import annotations

from collections.abc import Sequence

from kan import KAN


def build_kan(
    width: Sequence[int] = (2, 1, 1),
    *,
    grid: int = 3,
    k: int = 3,
    seed: int = 42,
) -> KAN:
    """Tạo KAN chạy CPU với tham số float64 và nhánh cơ sở SiLU."""
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
    return model.double()