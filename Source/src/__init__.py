"""Package mã nguồn thí nghiệm KAN & MLP đối chứng (CSC14005)."""

from .data import (
    TASK_SPECS,
    DatasetBundle,
    evaluate_target,
    load_dataset_from_yaml,
    make_dataset,
)
from .mlp_baseline import (
    MLPBaseline,
    MLPConfig,
    build_mlp,
    count_trainable_params,
    expected_mlp_params,
    load_mlp_panel_configs,
)

__all__ = [
    "TASK_SPECS",
    "DatasetBundle",
    "MLPBaseline",
    "MLPConfig",
    "build_mlp",
    "count_trainable_params",
    "evaluate_target",
    "expected_mlp_params",
    "load_dataset_from_yaml",
    "load_mlp_panel_configs",
    "make_dataset",
]
