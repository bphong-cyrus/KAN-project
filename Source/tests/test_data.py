"""Unit tests cho Source/src/data.py (TV1 - Tạ Bỉnh Phong)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from Source.src.data import (
    DatasetBundle,
    evaluate_target,
    load_dataset_from_yaml,
    make_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "Source" / "configs"


def test_analytical_values_main_task() -> None:
    pts = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    vals = evaluate_target("exp_sin_y2", pts)
    expected = np.array(
        [[1.0], [math.e], [1.0], [math.e], [math.e]],
        dtype=np.float64,
    )
    assert vals.shape == (5, 1)
    np.testing.assert_allclose(vals, expected, rtol=1e-12, atol=1e-12)


def test_analytical_values_secondary_tasks_and_tensor_preservation() -> None:
    x_1d = torch.tensor([[0.0], [0.1], [-0.5]], dtype=torch.float64)
    y_bessel = evaluate_target("bessel", x_1d)
    assert isinstance(y_bessel, torch.Tensor)
    assert y_bessel.dtype == torch.float64
    assert y_bessel.shape == (3, 1)
    assert math.isclose(float(y_bessel[0, 0]), 1.0, rel_tol=1e-12)

    x_2d = torch.tensor([[0.5, -4.0], [-1.0, 1.0]], dtype=torch.float32)
    y_mul = evaluate_target("multiply", x_2d)
    assert isinstance(y_mul, torch.Tensor)
    assert y_mul.dtype == torch.float32
    assert y_mul.shape == (2, 1)
    torch.testing.assert_close(y_mul, torch.tensor([[-2.0], [-1.0]], dtype=torch.float32))


def test_evaluate_target_rejects_bad_shape_or_task() -> None:
    with pytest.raises(ValueError):
        evaluate_target("unknown_task", np.zeros((4, 2)))
    with pytest.raises(ValueError):
        evaluate_target("exp_sin_y2", np.zeros((4,)))
    with pytest.raises(ValueError):
        evaluate_target("exp_sin_y2", np.zeros((4, 3)))


def test_make_dataset_shapes_bounds_finiteness_and_disjoint() -> None:
    bundle = make_dataset(
        task="exp_sin_y2",
        n_train=1000,
        n_val=200,
        n_test=1000,
        seed=42,
        domain=(-1.0, 1.0),
        dtype="float32",
    )
    assert bundle.x_train.shape == (1000, 2)
    assert bundle.y_train.shape == (1000, 1)
    assert bundle.x_val.shape == (200, 2)
    assert bundle.y_val.shape == (200, 1)
    assert bundle.x_test.shape == (1000, 2)
    assert bundle.y_test.shape == (1000, 1)

    for tensor in bundle.as_tuple():
        assert tensor.dtype == torch.float32
        assert torch.all(torch.isfinite(tensor))

    for x_split in (bundle.x_train, bundle.x_val, bundle.x_test):
        assert float(x_split.min()) >= -1.0
        assert float(x_split.max()) <= 1.0

    def rows(t: torch.Tensor) -> set[tuple[float, ...]]:
        return {tuple(row) for row in t.tolist()}

    assert not (rows(bundle.x_train) & rows(bundle.x_val))
    assert not (rows(bundle.x_train) & rows(bundle.x_test))
    assert not (rows(bundle.x_val) & rows(bundle.x_test))


def test_seed_reproducibility_and_sensitivity() -> None:
    b1 = make_dataset(task="exp_sin_y2", n_train=64, n_val=16, n_test=64, seed=42)
    b2 = make_dataset(task="exp_sin_y2", n_train=64, n_val=16, n_test=64, seed=42)
    b3 = make_dataset(task="exp_sin_y2", n_train=64, n_val=16, n_test=64, seed=43)

    assert b1.split_id == b2.split_id
    assert b1.split_id != b3.split_id
    for t1, t2 in zip(b1.as_tuple(), b2.as_tuple()):
        torch.testing.assert_close(t1, t2)
    assert not torch.equal(b1.x_train, b3.x_train)


def test_standardization_uses_train_only_stats() -> None:
    bundle = make_dataset(
        task="exp_sin_y2",
        n_train=500,
        n_val=100,
        n_test=100,
        seed=7,
        domain=[(2.0, 6.0), (-3.0, 1.0)],
        dtype="float64",
        normalize_x=True,
    )
    train_mean = bundle.x_train.mean(dim=0)
    train_std = bundle.x_train.std(dim=0, unbiased=False)
    torch.testing.assert_close(train_mean, torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(train_std, torch.ones(2, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    assert bundle.metadata["normalization"] is not None
    assert len(bundle.metadata["normalization"]["x_mean"]) == 2


def test_pykan_dict_save_load_and_yaml_loading(tmp_path: Path) -> None:
    bundle = load_dataset_from_yaml(CONFIGS_DIR / "dataset_exp_sin_y2.yaml")
    assert bundle.x_train.shape == (1000, 2)
    assert bundle.x_val.shape == (200, 2)
    assert bundle.x_test.shape == (1000, 2)

    pykan_dict = bundle.to_pykan_dict(include_val=True)
    assert set(pykan_dict) == {
        "train_input",
        "train_label",
        "test_input",
        "test_label",
        "val_input",
        "val_label",
    }

    ckpt_path = tmp_path / "split.pt"
    bundle.save(ckpt_path)
    loaded = DatasetBundle.load(ckpt_path)
    assert loaded.split_id == bundle.split_id
    assert loaded.metadata["normalize_x"] is False
    assert loaded.metadata["normalize_y"] is False
    for t1, t2 in zip(bundle.as_tuple(), loaded.as_tuple()):
        torch.testing.assert_close(t1, t2)
