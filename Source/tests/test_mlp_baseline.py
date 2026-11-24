"""Unit tests cho Source/src/mlp_baseline.py (TV1 - Tạ Bỉnh Phong)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from Source.src.data import make_dataset
from Source.src.mlp_baseline import (
    build_mlp,
    count_trainable_params,
    expected_mlp_params,
    load_mlp_panel_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MLP_PANEL_YAML = REPO_ROOT / "Source" / "configs" / "mlp_panel.yaml"


def test_single_hidden_layer_param_count_matches_4h_plus_1() -> None:
    for h in (4, 6, 8, 11, 16, 19, 32, 41):
        model = build_mlp(input_dim=2, hidden_widths=[h], activation="silu", seed=42)
        expected = 4 * h + 1
        assert expected_mlp_params(2, [h], 1, bias=True) == expected
        assert count_trainable_params(model) == expected


def test_multi_hidden_layer_param_count_and_forward_backward() -> None:
    model = build_mlp(input_dim=2, hidden_widths=[11, 11], activation="silu", seed=42)
    # 2*11+11 + 11*11+11 + 11*1+1 = 33 + 132 + 12 = 177
    assert count_trainable_params(model) == 177
    assert model.architecture == [2, 11, 11, 1]

    m_depth4 = build_mlp(input_dim=2, hidden_widths=[24, 24, 24], activation="silu", seed=42)
    # 2*24+24 + 2*(24*24+24) + 24*1+1 = 72 + 1200 + 25 = 1297
    assert count_trainable_params(m_depth4) == 1297

    x = torch.randn(16, 2, dtype=torch.float64)
    y_hat = model(x)
    assert y_hat.shape == (16, 1)
    assert torch.all(torch.isfinite(y_hat))

    loss = torch.mean(y_hat ** 2)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)


def test_seed_determinism_and_non_interference_with_global_rng() -> None:
    torch.manual_seed(999)
    probe_before = torch.rand(4)

    torch.manual_seed(999)
    m1 = build_mlp(input_dim=2, hidden_widths=[11], seed=42)
    probe_after = torch.rand(4)
    torch.testing.assert_close(probe_before, probe_after)

    m2 = build_mlp(input_dim=2, hidden_widths=[11], seed=42)
    m3 = build_mlp(input_dim=2, hidden_widths=[11], seed=43)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1, p2)

    diffs = [not torch.equal(p1, p3) for p1, p3 in zip(m1.parameters(), m3.parameters())]
    assert any(diffs)


def test_locked_yaml_configs_have_distinct_increasing_params() -> None:
    configs = load_mlp_panel_configs(MLP_PANEL_YAML)
    assert len(configs) == 10
    silu_params = []
    seen_variants = set()
    for cfg in configs:
        assert cfg.dtype == "float64"
        assert cfg.config_path == "Source/configs/mlp_panel.yaml"
        model = build_mlp(
            input_dim=cfg.input_dim,
            hidden_widths=cfg.hidden_widths,
            activation=cfg.activation,
            output_dim=cfg.output_dim,
            bias=cfg.bias,
            init=cfg.init,
            seed=cfg.default_seed,
            dtype=cfg.dtype,
        )
        assert model.activation_name == cfg.activation
        n_params = count_trainable_params(model)
        assert n_params == cfg.expected_params
        seen_variants.add((cfg.activation, n_params))
        if cfg.activation == "silu":
            silu_params.append(n_params)

    assert len(seen_variants) == len(configs)
    assert len(set(silu_params)) == len(silu_params)
    assert silu_params == sorted(silu_params)
    assert silu_params[-1] >= 2000
    assert any(cfg.activation == "tanh" for cfg in configs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_build_mlp_initializes_in_requested_dtype(dtype: torch.dtype) -> None:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model_a = build_mlp(2, [11], seed=42, dtype=dtype)
        torch.set_default_dtype(torch.float64)
        model_b = build_mlp(2, [11], seed=42, dtype=dtype)
    finally:
        torch.set_default_dtype(previous_dtype)

    for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters()):
        assert parameter_a.dtype == parameter_b.dtype == dtype
        torch.testing.assert_close(parameter_a, parameter_b, atol=0, rtol=0)


def test_short_lbfgs_smoke_decreases_loss() -> None:
    bundle = make_dataset(task="exp_sin_y2", n_train=120, n_val=40, n_test=40, seed=42)
    model = build_mlp(input_dim=2, hidden_widths=[11], activation="silu", seed=42)
    criterion = nn.MSELoss()

    with torch.no_grad():
        initial_loss = float(criterion(model(bundle.x_train), bundle.y_train))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=15, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(model(bundle.x_train), bundle.y_train)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        final_loss = float(criterion(model(bundle.x_train), bundle.y_train))

    assert torch.isfinite(torch.tensor(final_loss))
    assert final_loss < initial_loss


def test_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError):
        build_mlp(input_dim=2, hidden_widths=[], activation="silu")
    with pytest.raises(ValueError):
        build_mlp(input_dim=2, hidden_widths=[8], activation="unknown")
    model = build_mlp(input_dim=2, hidden_widths=[8], activation="silu")
    with pytest.raises(ValueError):
        model(torch.randn(5, 3, dtype=torch.float64))
