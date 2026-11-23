"""Sinh sơ đồ quy trình dữ liệu và đối chứng KAN--MLP do TV1 tự dựng.

Xuất file vector PDF tại `report/figures/tv1/data_pipeline.pdf`.
Chạy:
    python report/figures/tv1/generate_data_pipeline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    edge_color: str = "#153F68",
    fill_color: str = "#F0F5FA",
    header_color: str = "#153F68",
    title_size: float = 9.0,
    body_size: float = 7.6,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=edge_color,
        facecolor=fill_color,
        zorder=2,
    )
    ax.add_patch(box)

    header_h = min(0.08, h * 0.27)
    header = FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0,
        facecolor=header_color,
        zorder=3,
    )
    ax.add_patch(header)

    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=title_size,
        fontweight="bold",
        zorder=4,
    )

    body_text = "\n".join(lines)
    ax.text(
        x + w / 2,
        y + (h - header_h) / 2,
        body_text,
        ha="center",
        va="center",
        color="#0F172A",
        fontsize=body_size,
        linespacing=1.42,
        zorder=4,
    )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#334155",
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_length=5.5,head_width=3.5",
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=5,
    )
    ax.add_patch(arrow)


def build_figure(pdf_out: Path, preview_png: Path | None = None) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5,
        0.945,
        "Quy trình dữ liệu dùng chung và đối chứng công bằng MLP — KAN (Hàm chính Mục 3.1)",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#153F68",
    )

    # Stage 1: Sinh dữ liệu
    _draw_box(
        ax,
        x=0.02,
        y=0.31,
        w=0.205,
        h=0.39,
        title="1. Sinh dữ liệu (TV1)",
        lines=[
            r"$f(x,y)=\exp(\sin(\pi x)+y^2)$",
            r"Miền $[-1, 1]^2$, đều ngẫu nhiên",
            "SeedSequence(seed).spawn(3)",
            "Tính chuẩn float64 -> float32",
            "Source/src/data.py",
        ],
    )

    # Stage 2: Split dùng chung
    _draw_box(
        ax,
        x=0.255,
        y=0.31,
        w=0.20,
        h=0.39,
        title="2. Split cố định (TV1)",
        lines=[
            "Train: 1000 điểm",
            "Val: 200 (chọn cấu hình)",
            "Test: 1000 (đánh giá cuối)",
            "Kiểm tra hữu hạn & tách biệt",
            "Khóa mã split_id dùng chung",
        ],
    )

    # Stage 3a: MLP Baseline (trên)
    _draw_box(
        ax,
        x=0.495,
        y=0.54,
        w=0.235,
        h=0.32,
        title="3a. MLP đối chứng (TV1)",
        lines=[
            "Depth 2 & 3 (SiLU, Xavier)",
            "6 cấu hình khóa theo tham số:",
            "25, 45, 77, 165, 177, 501 params",
            "Source/src/mlp_baseline.py",
        ],
        edge_color="#B45309",
        fill_color="#FFFBEB",
        header_color="#B45309",
    )

    # Stage 3b: KAN Model (dưới)
    _draw_box(
        ax,
        x=0.495,
        y=0.14,
        w=0.235,
        h=0.32,
        title="3b. Mô hình KAN (TV2)",
        lines=[
            "Kiến trúc [2, 1, 1], bậc k = 3",
            "Cạnh: w_b SiLU(x) + w_s B-spline(x)",
            "Lịch lưới G ∈ {3, 5, 10, 20, 50}",
            "Source/src/kan_model.py",
        ],
        edge_color="#0F766E",
        fill_color="#F0FDFA",
        header_color="#0F766E",
    )

    # Stage 4: Trainer + Metrics chung
    _draw_box(
        ax,
        x=0.77,
        y=0.51,
        w=0.21,
        h=0.35,
        title="4. Huấn luyện chung (TV3)",
        lines=[
            "Optimizer: LBFGS dùng chung",
            "Ghi steps & closure_evals",
            "Đo RMSE (train / val / test)",
            "Đo thời gian cùng máy",
            "train_grid.py & metrics.py",
        ],
        edge_color="#1E3A8A",
        fill_color="#EFF6FF",
        header_color="#1E3A8A",
    )

    # Stage 5: Results & Evaluation
    _draw_box(
        ax,
        x=0.77,
        y=0.14,
        w=0.21,
        h=0.29,
        title="5. Đối chiếu (TV4)",
        lines=[
            "results/runs.csv (run_id duy nhất)",
            "results/summary.json",
            "Đồ thị RMSE–Params / G / Time",
            "Đối chiếu Hình 3.1 bài báo",
        ],
        edge_color="#475569",
        fill_color="#F8FAFC",
        header_color="#334155",
    )

    # Mũi tên nối thoáng, không chạm góc hộp
    _draw_arrow(ax, (0.227, 0.505), (0.252, 0.505))
    _draw_arrow(ax, (0.457, 0.57), (0.492, 0.70), color="#B45309", rad=-0.08)
    _draw_arrow(ax, (0.457, 0.44), (0.492, 0.30), color="#0F766E", rad=0.08)
    _draw_arrow(ax, (0.732, 0.70), (0.767, 0.70), color="#B45309")
    _draw_arrow(ax, (0.732, 0.34), (0.767, 0.58), color="#0F766E", rad=0.06)
    _draw_arrow(ax, (0.875, 0.507), (0.875, 0.435), color="#1E3A8A")

    ax.text(
        0.5,
        0.045,
        "Sơ đồ tự dựng theo đặc tả experiment_spec.md và thiết kế Source/src/ (TV1 — Tạ Bỉnh Phong). Không chứa kết quả thực nghiệm.",
        ha="center",
        va="center",
        fontsize=7.8,
        style="italic",
        color="#475569",
    )

    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_out, bbox_inches="tight", dpi=300)
    if preview_png is not None:
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_png, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TV1 data & baseline pipeline figure.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data_pipeline.pdf",
        help="Output vector PDF path.",
    )
    parser.add_argument(
        "--preview-png",
        type=Path,
        default=None,
        help="Optional preview PNG path for visual inspection.",
    )
    args = parser.parse_args()
    build_figure(args.out, args.preview_png)
    print(f"Generated {args.out}")


if __name__ == "__main__":
    main()
