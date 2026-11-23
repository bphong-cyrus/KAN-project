"""Smoke test tổng hợp phần TV1 (Data + 6 MLP Baselines).

Chạy thử quy trình sinh dữ liệu, kiểm chứng giải tích, đếm tham số MLP và huấn luyện
LBFGS ngắn nhằm xác nhận pipeline dữ liệu -> MLP hoạt động thông suốt trước khi
ghép với KAN của TV2 và trainer chính thức của TV3.

Toàn bộ kết quả thử nghiệm được lưu vào `fixtures/` với nhãn `smoke_tv1_dev_only`
và tuyệt đối KHÔNG ghi vào `results/runs.csv`.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Source.src.data import evaluate_target, load_dataset_from_yaml
from Source.src.mlp_baseline import (
    build_mlp,
    count_trainable_params,
    load_mlp_panel_configs,
)

CSV_COLUMNS = [
    "run_id",
    "experiment_id",
    "task",
    "model",
    "architecture",
    "G",
    "k",
    "seed",
    "split_id",
    "params",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "time_s",
    "optimizer",
    "steps",
    "closure_evals",
    "config_path",
    "code_commit",
]


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", str(REPO_ROOT), "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        return f"{out}-dirty" if dirty != 0 else out
    except Exception:
        return "uncommitted"


def _rmse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(x)
        return float(torch.sqrt(torch.mean((pred - y) ** 2)).item())


def _run_analytical_checks() -> dict[str, float]:
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    expected = np.array([1.0, math.e, 1.0, math.e, math.e], dtype=np.float64)
    actual = evaluate_target("exp_sin_y2", points).ravel()
    max_abs_err = float(np.max(np.abs(actual - expected)))
    assert max_abs_err < 1e-12, f"Analytical check failed with err={max_abs_err}"
    return {
        "f(0,0)": float(actual[0]),
        "f(0,1)": float(actual[1]),
        "f(1,0)": float(actual[2]),
        "f(0.5,0)": float(actual[3]),
        "f(-1,-1)": float(actual[4]),
        "max_abs_error": max_abs_err,
    }


def main() -> None:
    dataset_yaml = REPO_ROOT / "Source" / "configs" / "dataset_exp_sin_y2.yaml"
    mlp_yaml = REPO_ROOT / "Source" / "configs" / "mlp_panel.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_dataset_from_yaml(dataset_yaml, device=device)
    configs = load_mlp_panel_configs(mlp_yaml)

    analytical = _run_analytical_checks()
    code_commit = _get_git_commit()

    print("=== TV1 Smoke Verification (Data + MLP Baseline) ===")
    print(f"Split ID      : {bundle.split_id}")
    print(f"Device        : {device}")
    print(
        f"Shapes        : train={tuple(bundle.x_train.shape)}, "
        f"val={tuple(bundle.x_val.shape)}, test={tuple(bundle.x_test.shape)}"
    )
    print(f"Analytical err: {analytical['max_abs_error']:.2e} (PASS)")
    print()

    smoke_rows: list[dict[str, object]] = []
    criterion = nn.MSELoss()
    smoke_steps = 20

    header_fmt = "{:<12} {:<16} {:>7} {:>11} {:>11} {:>11} {:>8}"
    print(
        header_fmt.format(
            "config_id", "architecture", "params", "init_rmse", "train_rmse", "val_rmse", "time_s"
        )
    )
    print("-" * 82)

    for cfg in configs:
        model = build_mlp(
            input_dim=cfg.input_dim,
            hidden_widths=cfg.hidden_widths,
            activation=cfg.activation,
            output_dim=cfg.output_dim,
            bias=cfg.bias,
            init=cfg.init,
            seed=cfg.default_seed,
            dtype=cfg.dtype,
            device=device,
        )
        n_params = count_trainable_params(model)
        assert n_params == cfg.expected_params

        init_train_rmse = _rmse(model, bundle.x_train, bundle.y_train)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=smoke_steps,
            line_search_fn="strong_wolfe",
        )
        closure_counter = {"evals": 0}

        def closure() -> torch.Tensor:
            closure_counter["evals"] += 1
            optimizer.zero_grad()
            loss = criterion(model(bundle.x_train), bundle.y_train)
            loss.backward()
            return loss

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step(closure)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        train_rmse = _rmse(model, bundle.x_train, bundle.y_train)
        val_rmse = _rmse(model, bundle.x_val, bundle.y_val)
        test_rmse = _rmse(model, bundle.x_test, bundle.y_test)

        assert all(math.isfinite(v) for v in (init_train_rmse, train_rmse, val_rmse, test_rmse))
        assert train_rmse < init_train_rmse, f"{cfg.config_id} failed to decrease smoke loss"

        print(
            f"{cfg.config_id:<12} {cfg.architecture_str:<16} {n_params:>7d} "
            f"{init_train_rmse:>11.5f} {train_rmse:>11.5f} {val_rmse:>11.5f} {elapsed:>8.3f}"
        )

        smoke_rows.append(
            {
                "run_id": f"smoke_{cfg.config_id}_s{cfg.default_seed}",
                "experiment_id": "smoke_tv1_dev_only",
                "task": cfg.task,
                "model": "MLP",
                "architecture": cfg.architecture_str,
                "G": "",
                "k": "",
                "seed": cfg.default_seed,
                "split_id": bundle.split_id,
                "params": n_params,
                "train_rmse": round(train_rmse, 6),
                "val_rmse": round(val_rmse, 6),
                "test_rmse": round(test_rmse, 6),
                "time_s": round(elapsed, 4),
                "optimizer": f"LBFGS(smoke_max_iter={smoke_steps})",
                "steps": smoke_steps,
                "closure_evals": closure_counter["evals"],
                "config_path": cfg.config_path,
                "code_commit": code_commit,
            }
        )

    fixtures_dir = REPO_ROOT / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    csv_out = fixtures_dir / "tv1_smoke_runs.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(smoke_rows)

    env_info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "code_commit": code_commit,
    }
    report_out = fixtures_dir / "tv1_smoke_report.json"
    with report_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "note": "TV1 smoke output for pipeline verification only - NOT official results.",
                "split_id": bundle.split_id,
                "dataset_metadata": bundle.metadata,
                "analytical_checks": analytical,
                "environment": env_info,
                "runs_count": len(smoke_rows),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Environment   : Python {env_info['python']}, PyTorch {env_info['torch']}, {env_info['device_used']}")
    print(f"Wrote smoke   : {csv_out.relative_to(REPO_ROOT)} & {report_out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
