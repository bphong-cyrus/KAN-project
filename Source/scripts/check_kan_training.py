"""TV3: chạy thử tăng lưới KAN; tái sử dụng dữ liệu và schema CSV của TV1.

Chạy từ gốc repo: python -m Source.scripts.check_kan_training
Không ghi kết quả phát triển vào results/runs.csv.
"""

import argparse
import csv
import json
import platform
from pathlib import Path

import torch
import yaml

from Source.scripts.smoke_tv1 import CSV_COLUMNS, _get_git_commit
from Source.src.data import load_dataset_from_yaml
from Source.src.train_grid import train_kan_schedule

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(
        description='TV3 development check: train KAN with sequential grid refinement.'
    )
    parser.add_argument('--steps-per-grid', type=int, default=2)
    parser.add_argument('--max-iter', type=int, default=5)
    parser.add_argument('--grids', type=int, nargs='+', default=[3, 5])
    args = parser.parse_args()
    if args.steps_per_grid < 1 or args.max_iter < 1:
        parser.error('steps-per-grid and max-iter must be >= 1')

    config_rel = 'Source/configs/dataset_exp_sin_y2.yaml'
    dataset_path = ROOT / config_rel
    with dataset_path.open(encoding='utf-8') as handle:
        dataset_config = yaml.safe_load(handle)
    seed = int(dataset_config['seed'])
    data = load_dataset_from_yaml(dataset_path)
    _, result = train_kan_schedule(
        data, args.grids, seed=seed,
        steps_per_grid=args.steps_per_grid, max_iter=args.max_iter,
    )

    commit = _get_git_commit()
    rows = []
    for point in result['checkpoints']:
        rows.append({
            **point,
            'run_id': f"smoke_kan_g{point['G']}_s{seed}_seq_step{point['steps']}",
            'experiment_id': 'smoke_tv3_grid_dev_only',
            'task': 'exp_sin_y2', 'model': 'KAN',
            'config_id': 'kan_d2_w1_seq', 'activation': '',
            'architecture': '[2, 1, 1]', 'seed': seed,
            'split_id': data.split_id, 'test_rmse': '',
            'optimizer': f'LBFGS(max_iter={args.max_iter},strong_wolfe)',
            'config_path': config_rel, 'code_commit': commit,
        })

    fixture_dir = ROOT / 'fixtures'
    fixture_dir.mkdir(exist_ok=True)
    with (fixture_dir / 'tv3_grid_smoke_runs.csv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (fixture_dir / 'tv3_grid_smoke_report.json').open('w', encoding='utf-8') as handle:
        json.dump({
            'note': 'Development only; test not evaluated; not official panel results.',
            'config': {
                **vars(args),
                'dataset': config_rel,
                'dataset_config': dataset_config,
                'grid_position_updates': False,
                'regularization_lambda': 0,
                'time_scope': 'train_plus_refine_plus_train_val_evaluation_excludes_initialization',
            },
            'split_id': data.split_id,
            'code_commit': commit,
            'environment': {'python': platform.python_version(), 'torch': torch.__version__},
            **result,
        }, handle, indent=2)
    print('\nSaved fixtures/tv3_grid_smoke_runs.csv and tv3_grid_smoke_report.json')


if __name__ == '__main__':
    main()
