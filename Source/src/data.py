"""Sinh dữ liệu tổng hợp, chia tập cố định và lưu metadata split.

Phụ trách: Tạ Bỉnh Phong (TV1 - 24127100).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from scipy.special import j0 as scipy_j0

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_repo_rel_posix(path: str | Path) -> str:
    """Chuẩn hóa đường dẫn cấu hình về dạng tương đối từ gốc repo để tránh rò rỉ đường dẫn máy cá nhân."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()

TASK_SPECS: dict[str, dict[str, Any]] = {
    "exp_sin_y2": {
        "input_dim": 2,
        "output_dim": 1,
        "formula": "f(x, y) = exp(sin(pi * x) + y^2)",
        "kan_architecture": [2, 1, 1],
        "default_domain": [(-1.0, 1.0), (-1.0, 1.0)],
        "priority": "P1",
    },
    "bessel": {
        "input_dim": 1,
        "output_dim": 1,
        "formula": "f(x) = J_0(20 * x)",
        "kan_architecture": [1, 1],
        "default_domain": [(-1.0, 1.0)],
        "priority": "P2",
    },
    "multiply": {
        "input_dim": 2,
        "output_dim": 1,
        "formula": "f(x, y) = x * y",
        "kan_architecture": [2, 2, 1],
        "default_domain": [(-1.0, 1.0), (-1.0, 1.0)],
        "priority": "P2",
    },
}


def _resolve_dtype(dtype: str | torch.dtype) -> tuple[str, torch.dtype]:
    if isinstance(dtype, torch.dtype):
        for name, dt in DTYPE_MAP.items():
            if dt == dtype:
                return name, dt
        raise ValueError(f"Unsupported torch.dtype: {dtype}. Expected float32 or float64.")
    key = str(dtype).lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Expected one of {list(DTYPE_MAP)}.")
    return key, DTYPE_MAP[key]


def _normalize_domain(
    domain: Sequence[float] | Sequence[Sequence[float]],
    input_dim: int,
) -> list[tuple[float, float]]:
    if (
        len(domain) == 2
        and isinstance(domain[0], (int, float))
        and isinstance(domain[1], (int, float))
    ):
        low, high = float(domain[0]), float(domain[1])
        if not low < high:
            raise ValueError(f"Domain lower bound must be < upper bound, got {(low, high)}.")
        return [(low, high) for _ in range(input_dim)]

    if len(domain) != input_dim:
        raise ValueError(
            f"Domain must specify {input_dim} intervals, got {len(domain)}: {domain}."
        )
    normalized: list[tuple[float, float]] = []
    for dim_idx, bounds in enumerate(domain):
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Domain entry at dim {dim_idx} must be [low, high], got {bounds}.")
        low, high = float(bounds[0]), float(bounds[1])
        if not low < high:
            raise ValueError(
                f"Domain lower bound must be < upper bound at dim {dim_idx}, got {(low, high)}."
            )
        normalized.append((low, high))
    return normalized


def evaluate_target(
    task: str,
    x: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Tính giá trị hàm mục tiêu chính xác tại các điểm đầu vào ``x`` dạng ``(N, d)``.

    Trả về mảng/tensor có cùng kiểu với ``x`` và shape chuẩn ``(N, 1)``.
    """
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task '{task}'. Supported tasks: {list(TASK_SPECS)}.")
    expected_dim = TASK_SPECS[task]["input_dim"]
    is_tensor = isinstance(x, torch.Tensor)

    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise ValueError(
            f"Task '{task}' expects input shape (N, {expected_dim}), got {tuple(x.shape)}."
        )

    if is_tensor:
        x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        x_np = np.asarray(x, dtype=np.float64)

    if task == "exp_sin_y2":
        y_np = np.exp(np.sin(np.pi * x_np[:, 0]) + np.square(x_np[:, 1]))
    elif task == "bessel":
        y_np = scipy_j0(20.0 * x_np[:, 0])
    elif task == "multiply":
        y_np = x_np[:, 0] * x_np[:, 1]
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unhandled task '{task}'.")

    y_np = np.reshape(y_np, (-1, 1))
    if not np.all(np.isfinite(y_np)):
        raise ValueError(f"Non-finite target values encountered for task '{task}'.")

    if is_tensor:
        return torch.as_tensor(y_np, dtype=x.dtype, device=x.device)
    return y_np.astype(np.asarray(x).dtype if np.issubdtype(np.asarray(x).dtype, np.floating) else np.float64)


def _sample_split(
    rng: np.random.Generator,
    n_samples: int,
    domain: list[tuple[float, float]],
    sampling: str,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError(f"Split size must be positive, got {n_samples}.")
    if sampling != "uniform":
        raise ValueError(f"Unsupported sampling method '{sampling}'. Expected 'uniform'.")
    cols = [rng.uniform(low, high, size=n_samples) for low, high in domain]
    return np.column_stack(cols).astype(np.float64)


def _check_disjoint_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> None:
    def _row_set(arr: np.ndarray) -> set[bytes]:
        contiguous = np.ascontiguousarray(arr)
        return {row.tobytes() for row in contiguous}

    train_set = _row_set(x_train)
    val_set = _row_set(x_val)
    test_set = _row_set(x_test)

    if train_set & val_set:
        raise RuntimeError("Data leakage check failed: overlap detected between train and val.")
    if train_set & test_set:
        raise RuntimeError("Data leakage check failed: overlap detected between train and test.")
    if val_set & test_set:
        raise RuntimeError("Data leakage check failed: overlap detected between val and test.")


def _compute_split_digest(
    task: str,
    seed: int,
    arrays: Sequence[np.ndarray],
) -> tuple[str, dict[str, str]]:
    split_names = ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")
    fingerprints: dict[str, str] = {}
    master = hashlib.sha256()
    master.update(f"{task}:{seed}".encode("utf-8"))
    for name, arr in zip(split_names, arrays):
        raw = np.ascontiguousarray(arr).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        fingerprints[name] = digest[:16]
        master.update(raw)
    short_hash = master.hexdigest()[:8]
    return short_hash, fingerprints


@dataclass
class DatasetBundle:
    """Gói dữ liệu dùng chung cho mọi mô hình MLP và KAN trên cùng một split."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    split_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tuple(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        )

    def to_pykan_dict(self, include_val: bool = False) -> dict[str, torch.Tensor]:
        """Chuyển đổi sang định dạng dictionary mà ``pykan`` kỳ vọng."""
        payload = {
            "train_input": self.x_train,
            "train_label": self.y_train,
            "test_input": self.x_test,
            "test_label": self.y_test,
        }
        if include_val:
            payload["val_input"] = self.x_val
            payload["val_label"] = self.y_val
        return payload

    def to(
        self,
        device: str | torch.device,
        dtype: str | torch.dtype | None = None,
    ) -> DatasetBundle:
        if dtype is not None:
            target_dtype_name, target_dtype = _resolve_dtype(dtype)
        else:
            target_dtype_name, target_dtype = _resolve_dtype(self.x_train.dtype)

        new_meta = dict(self.metadata)
        prev_dtype_name = str(self.metadata.get("dtype", _resolve_dtype(self.x_train.dtype)[0]))
        new_meta["dtype"] = target_dtype_name
        if prev_dtype_name == "float32" and target_dtype == torch.float64:
            new_meta["upcast_from"] = "float32"
            new_meta["precision_warning"] = (
                "Upcast from float32; float32 quantization floor (~4.8e-7 near y=7.39) remains."
            )

        return DatasetBundle(
            x_train=self.x_train.to(device=device, dtype=target_dtype),
            y_train=self.y_train.to(device=device, dtype=target_dtype),
            x_val=self.x_val.to(device=device, dtype=target_dtype),
            y_val=self.y_val.to(device=device, dtype=target_dtype),
            x_test=self.x_test.to(device=device, dtype=target_dtype),
            y_test=self.y_test.to(device=device, dtype=target_dtype),
            split_id=self.split_id,
            metadata=new_meta,
        )

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "x_train": self.x_train.cpu(),
                "y_train": self.y_train.cpu(),
                "x_val": self.x_val.cpu(),
                "y_val": self.y_val.cpu(),
                "x_test": self.x_test.cpu(),
                "y_test": self.y_test.cpu(),
                "split_id": self.split_id,
                "metadata": self.metadata,
            },
            out_path,
        )
        return out_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> DatasetBundle:
        """Nạp split đã lưu bằng ``weights_only=True`` (chỉ dùng file split nội bộ tin cậy)."""
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        return cls(
            x_train=payload["x_train"],
            y_train=payload["y_train"],
            x_val=payload["x_val"],
            y_val=payload["y_val"],
            x_test=payload["x_test"],
            y_test=payload["y_test"],
            split_id=payload["split_id"],
            metadata=payload["metadata"],
        )


def make_dataset(
    task: str = "exp_sin_y2",
    n_train: int = 1000,
    n_val: int = 200,
    n_test: int = 1000,
    seed: int = 42,
    domain: Sequence[float] | Sequence[Sequence[float]] | None = None,
    dtype: str | torch.dtype = "float64",
    sampling: str = "uniform",
    normalize_x: bool = False,
    device: str | torch.device = "cpu",
) -> DatasetBundle:
    """Sinh 3 tập train/val/test cố định, tách biệt và có ``split_id`` xác định."""
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task '{task}'. Supported tasks: {list(TASK_SPECS)}.")

    spec = TASK_SPECS[task]
    input_dim = int(spec["input_dim"])
    dtype_name, torch_dtype = _resolve_dtype(dtype)
    np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

    resolved_domain = (
        list(spec["default_domain"])
        if domain is None
        else _normalize_domain(domain, input_dim=input_dim)
    )

    child_seeds = np.random.SeedSequence(int(seed)).spawn(3)
    rng_train = np.random.default_rng(child_seeds[0])
    rng_val = np.random.default_rng(child_seeds[1])
    rng_test = np.random.default_rng(child_seeds[2])

    x_train_raw = _sample_split(rng_train, n_train, resolved_domain, sampling)
    x_val_raw = _sample_split(rng_val, n_val, resolved_domain, sampling)
    x_test_raw = _sample_split(rng_test, n_test, resolved_domain, sampling)

    y_train_f64 = evaluate_target(task, x_train_raw)
    y_val_f64 = evaluate_target(task, x_val_raw)
    y_test_f64 = evaluate_target(task, x_test_raw)

    norm_stats: dict[str, list[float]] | None = None
    if normalize_x:
        mean = x_train_raw.mean(axis=0, keepdims=True)
        std = x_train_raw.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        x_train_f64 = (x_train_raw - mean) / std
        x_val_f64 = (x_val_raw - mean) / std
        x_test_f64 = (x_test_raw - mean) / std
        norm_stats = {
            "x_mean": mean.squeeze(0).tolist(),
            "x_std": std.squeeze(0).tolist(),
        }
    else:
        x_train_f64, x_val_f64, x_test_f64 = x_train_raw, x_val_raw, x_test_raw

    x_train_np = x_train_f64.astype(np_dtype)
    y_train_np = y_train_f64.astype(np_dtype)
    x_val_np = x_val_f64.astype(np_dtype)
    y_val_np = y_val_f64.astype(np_dtype)
    x_test_np = x_test_f64.astype(np_dtype)
    y_test_np = y_test_f64.astype(np_dtype)

    for name, arr in (
        ("x_train", x_train_np),
        ("y_train", y_train_np),
        ("x_val", x_val_np),
        ("y_val", y_val_np),
        ("x_test", x_test_np),
        ("y_test", y_test_np),
    ):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Non-finite values generated in '{name}'.")

    _check_disjoint_splits(x_train_np, x_val_np, x_test_np)

    short_hash, fingerprints = _compute_split_digest(
        task,
        int(seed),
        (x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np),
    )
    split_id = f"{task}_s{int(seed)}_{n_train}_{n_val}_{n_test}_{short_hash}"

    metadata: dict[str, Any] = {
        "task": task,
        "formula": spec["formula"],
        "priority": spec["priority"],
        "input_dim": input_dim,
        "output_dim": int(spec["output_dim"]),
        "domain": [[lo, hi] for lo, hi in resolved_domain],
        "sampling": sampling,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "seed": int(seed),
        "dtype": dtype_name,
        "normalize_x": bool(normalize_x),
        "normalize_y": False,
        "normalization": norm_stats,
        "split_id": split_id,
        "fingerprints_sha256_16": fingerprints,
    }

    return DatasetBundle(
        x_train=torch.from_numpy(x_train_np).to(device=device),
        y_train=torch.from_numpy(y_train_np).to(device=device),
        x_val=torch.from_numpy(x_val_np).to(device=device),
        y_val=torch.from_numpy(y_val_np).to(device=device),
        x_test=torch.from_numpy(x_test_np).to(device=device),
        y_test=torch.from_numpy(y_test_np).to(device=device),
        split_id=split_id,
        metadata=metadata,
    )


def load_dataset_from_yaml(
    config_path: str | Path,
    device: str | torch.device = "cpu",
) -> DatasetBundle:
    """Đọc cấu hình dataset YAML và gọi ``make_dataset`` tương ứng."""
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    bundle = make_dataset(
        task=cfg["task"],
        n_train=int(cfg["n_train"]),
        n_val=int(cfg["n_val"]),
        n_test=int(cfg["n_test"]),
        seed=int(cfg["seed"]),
        domain=cfg.get("domain"),
        dtype=cfg.get("dtype", "float64"),
        sampling=cfg.get("sampling", "uniform"),
        normalize_x=bool(cfg.get("normalize_x", False)),
        device=device,
    )
    bundle.metadata["config_path"] = _to_repo_rel_posix(cfg_path)
    return bundle
