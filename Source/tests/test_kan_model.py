"""Regression checks cho adapter pykan CPU/float64 của TV2."""

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from Source.src.data import make_dataset
from Source.src.kan_model import build_kan, refine_kan
from Source.src.mlp_baseline import count_trainable_params


def snapshot_state():
    return (
        torch.get_default_dtype(),
        torch.get_rng_state().clone(),
        np.random.get_state(),
        random.getstate(),
    )


def assert_state_unchanged(before):
    dtype, torch_state, numpy_state, python_state = before
    assert torch.get_default_dtype() == dtype
    assert torch.equal(torch.get_rng_state(), torch_state)
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_initialization_independent_of_default_dtype_and_preserves_rng():
    previous_dtype = torch.get_default_dtype()
    models = []
    try:
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            before = snapshot_state()
            models.append(build_kan(seed=42))
            assert_state_unchanged(before)
    finally:
        torch.set_default_dtype(previous_dtype)
    for name, parameter in models[0].state_dict().items():
        torch.testing.assert_close(parameter, models[1].state_dict()[name], rtol=1e-12, atol=1e-12)
    assert all(p.dtype == torch.float64 for model in models for p in model.parameters())


def test_global_state_restored_on_constructor_error():
    before = snapshot_state()

    def fail(**kwargs):
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        raise RuntimeError("constructor failed")

    with patch("Source.src.kan_model.KAN", side_effect=fail):
        with pytest.raises(RuntimeError, match="constructor failed"):
            build_kan()
    assert_state_unchanged(before)


def test_full_grid_schedule_without_checkpoint_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundle = make_dataset(task="exp_sin_y2", seed=42, dtype="float64")
    model = build_kan()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        for grid in (3, 5, 10, 20, 50):
            if grid != 3:
                with torch.no_grad():
                    model(bundle.x_val)
                before = snapshot_state()
                old_model = model
                model = refine_kan(model, grid, bundle.x_train)
                assert model is not old_model
                assert_state_unchanged(before)
                torch.testing.assert_close(model.cache_data, bundle.x_train, rtol=0, atol=0)
            assert not model.auto_save
            assert all(p.dtype == torch.float64 for p in model.parameters())
            assert count_trainable_params(model) == 3 * (grid + 3 + 2)
            model.zero_grad(set_to_none=True)
            prediction = model(bundle.x_train[:32])
            assert prediction.shape == bundle.y_train[:32].shape
            loss = (prediction - bundle.y_train[:32]).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            for parameter in model.parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None
                    assert torch.isfinite(parameter.grad).all()
    finally:
        torch.set_default_dtype(previous_dtype)
    assert list(tmp_path.iterdir()) == []


def test_refinement_independent_of_last_evaluation_and_default_dtype():
    bundle = make_dataset(task="exp_sin_y2", seed=42, dtype="float64")
    models = [build_kan(seed=42), build_kan(seed=42)]
    previous_dtype = torch.get_default_dtype()
    try:
        refined = []
        for model, dtype, inputs in zip(
            models, (torch.float32, torch.float64), (bundle.x_val, bundle.x_train)
        ):
            torch.set_default_dtype(dtype)
            model.eval()
            with torch.no_grad():
                model(inputs)
            refined.append(refine_kan(model, 5, bundle.x_train))
            assert not refined[-1].training
    finally:
        torch.set_default_dtype(previous_dtype)
    for name, parameter in refined[0].state_dict().items():
        torch.testing.assert_close(parameter, refined[1].state_dict()[name], rtol=1e-10, atol=1e-10)


def test_refinement_rejects_invalid_inputs():
    model = build_kan()
    with pytest.raises(ValueError, match="new_grid"):
        refine_kan(model, 3, torch.zeros(32, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="float64"):
        refine_kan(model, 5, torch.zeros(32, 2, dtype=torch.float32))
    with pytest.raises(ValueError, match="NaN"):
        refine_kan(model, 5, torch.full((32, 2), float("nan"), dtype=torch.float64))
