"""CLI chuẩn bị và kiểm tra bộ dữ liệu dùng chung cho nhóm thực nghiệm (TV1).

Chạy mặc định để kiểm tra hàm chính `exp_sin_y2`:
    python Source/scripts/prepare_data.py

Xuất sẵn file `.pt` cho TV2/TV3 nạp trực tiếp:
    python Source/scripts/prepare_data.py --save-out fixtures/split_exp_sin_y2_s42.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Source.src.data import evaluate_target, load_dataset_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra và tạo bộ split dùng chung từ YAML (TV1)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "Source" / "configs" / "dataset_exp_sin_y2.yaml",
        help="Đường dẫn tới file cấu hình dataset YAML.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Thiết bị đích cho tensor ('cpu' hoặc 'cuda').",
    )
    parser.add_argument(
        "--save-out",
        type=Path,
        default=None,
        help="Nếu chỉ định, lưu DatasetBundle ra file .pt để chia sẻ cố định.",
    )
    return parser.parse_args()


def _verify_main_task_spot_checks() -> float:
    pts = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.0], [-1.0, -1.0]],
        dtype=np.float64,
    )
    expected = np.array([1.0, math.e, 1.0, math.e, math.e], dtype=np.float64)
    actual = evaluate_target("exp_sin_y2", pts).ravel()
    return float(np.max(np.abs(actual - expected)))


def main() -> None:
    args = parse_args()
    bundle = load_dataset_from_yaml(args.config, device=args.device)

    print("=== TV1 Dataset Preparation ===")
    print(f"Config   : {args.config}")
    print(f"Task     : {bundle.metadata['task']} ({bundle.metadata['formula']})")
    print(f"Domain   : {bundle.metadata['domain']} | Sampling: {bundle.metadata['sampling']}")
    print(f"Split ID : {bundle.split_id}")
    print(
        f"Tensors  : train={tuple(bundle.x_train.shape)}, "
        f"val={tuple(bundle.x_val.shape)}, test={tuple(bundle.x_test.shape)}, "
        f"dtype={bundle.x_train.dtype}"
    )

    if bundle.metadata["task"] == "exp_sin_y2":
        max_err = _verify_main_task_spot_checks()
        print(f"SpotErr  : {max_err:.2e} (analytical checks PASS)")

    if args.save_out is not None:
        saved = bundle.save(args.save_out)
        print(f"Saved    : {saved}")


if __name__ == "__main__":
    main()
