"""Mô hình MLP đối chứng cấu hình được và hàm đếm tham số học được.

Phụ trách: Tạ Bỉnh Phong (TV1 - 24127100).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import yaml

from .data import _resolve_dtype, _to_repo_rel_posix

SUPPORTED_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

SUPPORTED_INITS = {"xavier_uniform", "xavier_normal", "kaiming_uniform", "default"}


def expected_mlp_params(
    input_dim: int,
    hidden_widths: Sequence[int],
    output_dim: int = 1,
    bias: bool = True,
) -> int:
    """Tính số tham số lý thuyết của MLP dạng [input_dim, *hidden_widths, output_dim]."""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive integers.")
    widths = [int(input_dim), *[int(w) for w in hidden_widths], int(output_dim)]
    if any(w <= 0 for w in widths):
        raise ValueError(f"All layer widths must be positive, got {widths}.")
    total = 0
    for d_in, d_out in zip(widths[:-1], widths[1:]):
        total += d_in * d_out + (d_out if bias else 0)
    return total


def count_trainable_params(model: nn.Module) -> int:
    """Đếm chính xác tổng số phần tử trong các tensor có ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLPBaseline(nn.Module):
    """Multi-Layer Perceptron chuẩn hóa làm mô hình đối chứng với KAN."""

    def __init__(
        self,
        input_dim: int,
        hidden_widths: Sequence[int],
        activation: str = "silu",
        output_dim: int = 1,
        bias: bool = True,
        init: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        act_key = str(activation).lower()
        if act_key not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. Choose from {list(SUPPORTED_ACTIVATIONS)}."
            )
        init_key = str(init).lower()
        if init_key not in SUPPORTED_INITS:
            raise ValueError(
                f"Unsupported init '{init}'. Choose from {sorted(SUPPORTED_INITS)}."
            )

        hidden = [int(w) for w in hidden_widths]
        if not hidden or any(w <= 0 for w in hidden):
            raise ValueError(f"hidden_widths must be a non-empty list of positive ints, got {hidden}.")

        self.input_dim = int(input_dim)
        self.hidden_widths = hidden
        self.output_dim = int(output_dim)
        self.activation_name = act_key
        self.bias = bool(bias)
        self.init_name = init_key
        self.architecture = [self.input_dim, *self.hidden_widths, self.output_dim]
        self.architecture_str = str(self.architecture)

        act_cls = SUPPORTED_ACTIVATIONS[act_key]
        layers: list[nn.Module] = []
        dims = self.architecture
        for idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out, bias=self.bias))
            if idx < len(dims) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)
        self._apply_init()

    def _apply_init(self) -> None:
        if self.init_name == "default":
            return
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                if self.init_name == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.init_name == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.init_name == "kaiming_uniform":
                    nonlinearity = "relu" if self.activation_name == "relu" else "leaky_relu"
                    nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input tensor of shape (N, {self.input_dim}), got {tuple(x.shape)}."
            )
        return self.net(x)


def build_mlp(
    input_dim: int,
    hidden_widths: Sequence[int],
    activation: str = "silu",
    output_dim: int = 1,
    bias: bool = True,
    init: str = "xavier_uniform",
    seed: int | None = None,
    dtype: str | torch.dtype = "float64",
    device: str | torch.device = "cpu",
) -> MLPBaseline:
    """Khởi tạo ``MLPBaseline`` có kiểm soát seed độc lập với trạng thái RNG toàn cục."""
    _, torch_dtype = _resolve_dtype(dtype)
    if seed is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            model = MLPBaseline(
                input_dim=input_dim,
                hidden_widths=hidden_widths,
                activation=activation,
                output_dim=output_dim,
                bias=bias,
                init=init,
            )
    else:
        model = MLPBaseline(
            input_dim=input_dim,
            hidden_widths=hidden_widths,
            activation=activation,
            output_dim=output_dim,
            bias=bias,
            init=init,
        )
    return model.to(device=device, dtype=torch_dtype)


@dataclass(frozen=True)
class MLPConfig:
    """Một cấu hình MLP đối chứng đọc từ YAML."""

    config_id: str
    experiment_id: str
    task: str
    input_dim: int
    hidden_widths: list[int]
    output_dim: int
    activation: str
    bias: bool
    init: str
    dtype: str
    default_seed: int
    expected_params: int
    kan_reference: str
    config_path: str

    @property
    def architecture(self) -> list[int]:
        return [self.input_dim, *self.hidden_widths, self.output_dim]

    @property
    def architecture_str(self) -> str:
        return str(self.architecture)


def load_mlp_panel_configs(config_path: str | Path) -> list[MLPConfig]:
    """Nạp danh sách các cấu hình MLP đối chứng đã khóa từ file YAML."""
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    norm_path = _to_repo_rel_posix(cfg_path)
    default_activation = str(raw.get("activation", "silu")).lower()
    configs: list[MLPConfig] = []
    for entry in raw["models"]:
        hidden = [int(w) for w in entry["hidden_widths"]]
        input_dim = int(raw["input_dim"])
        output_dim = int(raw.get("output_dim", 1))
        bias = bool(raw.get("bias", True))
        expected = int(entry["expected_params"])
        computed = expected_mlp_params(input_dim, hidden, output_dim=output_dim, bias=bias)
        if computed != expected:
            raise ValueError(
                f"Config '{entry['config_id']}' expected_params={expected} disagrees with formula={computed}."
            )
        configs.append(
            MLPConfig(
                config_id=str(entry["config_id"]),
                experiment_id=str(raw["experiment_id"]),
                task=str(raw["task"]),
                input_dim=input_dim,
                hidden_widths=hidden,
                output_dim=output_dim,
                activation=str(entry.get("activation", default_activation)).lower(),
                bias=bias,
                init=str(entry.get("init", raw.get("init", "xavier_uniform"))),
                dtype=str(raw.get("dtype", "float64")),
                default_seed=int(raw.get("default_seed", 42)),
                expected_params=expected,
                kan_reference=str(entry.get("kan_reference", "")),
                config_path=norm_path,
            )
        )
    return configs
