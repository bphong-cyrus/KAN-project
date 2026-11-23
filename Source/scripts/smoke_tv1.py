"""Smoke test cho module dữ liệu và 6 cấu hình MLP đối chứng.

Chạy kiểm tra nhanh trên tập dữ liệu chính bằng LBFGS ngắn hạn và xuất log phát triển
vào `fixtures/` (`experiment_id="smoke_tv1_dev_only"`), không ghi vào `results/runs.csv`.
"""

from __future__ import annotations

import argparse
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


def _run_focal_sensitivity_check(
    bundle,
    device: torch.device,
    seeds: tuple[int, ...] = (42, 43, 44),
    activations: tuple[str, ...] = ("silu", "tanh", "relu"),
    outer_steps: int = 10,
    max_iter_per_step: int = 20,
) -> dict[str, object]:
    """Đo độ nhạy activation & độ ổn định 3 seed trên cấu hình trọng tâm mlp_d2_w11 ([2, 11, 1], 45 params)."""
    criterion = nn.MSELoss()
    summary: dict[str, object] = {}

    print()
    print("=== TV1 Focal Sensitivity Check: mlp_d2_w11 [2, 11, 1] (45 params, 3 seeds, 200 LBFGS iters) ===")
    print(f"{'activation':<10} {'val_rmse (mean+-sd)':<24} {'test_rmse (mean+-sd)':<24} {'avg_evals':>9} {'avg_s':>7}")
    print("-" * 82)

    for act in activations:
        seed_records: list[dict[str, float | int]] = []
        for s in seeds:
            model = build_mlp(
                input_dim=2,
                hidden_widths=[11],
                activation=act,
                output_dim=1,
                bias=True,
                init="xavier_uniform",
                seed=s,
                dtype="float32",
                device=device,
            )
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=max_iter_per_step,
                history_size=100,
                tolerance_grad=1e-12,
                tolerance_change=1e-12,
                line_search_fn="strong_wolfe",
            )
            evals = {"count": 0}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(outer_steps):
                def closure() -> torch.Tensor:
                    evals["count"] += 1
                    optimizer.zero_grad()
                    loss = criterion(model(bundle.x_train), bundle.y_train)
                    loss.backward()
                    return loss

                optimizer.step(closure)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            train_rmse = _rmse(model, bundle.x_train, bundle.y_train)
            val_rmse = _rmse(model, bundle.x_val, bundle.y_val)
            test_rmse = _rmse(model, bundle.x_test, bundle.y_test)
            assert all(math.isfinite(v) for v in (train_rmse, val_rmse, test_rmse))

            seed_records.append(
                {
                    "seed": int(s),
                    "train_rmse": round(train_rmse, 6),
                    "val_rmse": round(val_rmse, 6),
                    "test_rmse": round(test_rmse, 6),
                    "closure_evals": int(evals["count"]),
                    "time_s": round(elapsed, 4),
                }
            )

        val_vals = [float(r["val_rmse"]) for r in seed_records]
        test_vals = [float(r["test_rmse"]) for r in seed_records]
        eval_vals = [int(r["closure_evals"]) for r in seed_records]
        time_vals = [float(r["time_s"]) for r in seed_records]

        val_mean = float(np.mean(val_vals))
        val_std = float(np.std(val_vals, ddof=1))
        test_mean = float(np.mean(test_vals))
        test_std = float(np.std(test_vals, ddof=1))
        avg_evals = float(np.mean(eval_vals))
        avg_time = float(np.mean(time_vals))

        print(
            f"{act:<10} {val_mean:.4f} +- {val_std:.4f}         "
            f"{test_mean:.4f} +- {test_std:.4f}         {avg_evals:>9.1f} {avg_time:>7.3f}"
        )
        summary[act] = {
            "architecture": "[2, 11, 1]",
            "params": 45,
            "seeds": list(seeds),
            "lbfgs_max_iters": outer_steps * max_iter_per_step,
            "val_rmse_mean": round(val_mean, 6),
            "val_rmse_std": round(val_std, 6),
            "test_rmse_mean": round(test_mean, 6),
            "test_rmse_std": round(test_std, 6),
            "avg_closure_evals": round(avg_evals, 2),
            "avg_time_s": round(avg_time, 4),
            "per_seed": seed_records,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test & focal sensitivity verification (TV1).")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Thiết bị chạy thử (mặc định 'cpu' vì LBFGS mạng 25-501 tham số nhanh hơn GPU).",
    )
    args = parser.parse_args()

    dataset_yaml = REPO_ROOT / "Source" / "configs" / "dataset_exp_sin_y2.yaml"
    mlp_yaml = REPO_ROOT / "Source" / "configs" / "mlp_panel.yaml"

    device = torch.device(args.device)
    bundle = load_dataset_from_yaml(dataset_yaml, device=device)
    configs = load_mlp_panel_configs(mlp_yaml)

    analytical = _run_analytical_checks()
    code_commit = _get_git_commit()

    print("=== TV1 Smoke Verification (Data + MLP Baseline) ===", flush=True)
    print(f"Split ID      : {bundle.split_id}", flush=True)
    print(f"Device        : {device}", flush=True)
    print(
        f"Shapes        : train={tuple(bundle.x_train.shape)}, "
        f"val={tuple(bundle.x_val.shape)}, test={tuple(bundle.x_test.shape)}",
        flush=True,
    )
    print(f"Analytical err: {analytical['max_abs_error']:.2e} (PASS)", flush=True)
    print(flush=True)

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

    focal_sensitivity = _run_focal_sensitivity_check(bundle, device=device)

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
                "note": "TV1 smoke & focal sensitivity verification - NOT official panel results.",
                "split_id": bundle.split_id,
                "dataset_metadata": bundle.metadata,
                "analytical_checks": analytical,
                "focal_activation_sensitivity_dev_only": focal_sensitivity,
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
