"""Cảnh 1 (0:00–1:50): Bài toán xấp xỉ hàm & trực giác MLP truyền thống.

Phụ trách: Tạ Bỉnh Phong (TV1 - 24127100).
Sử dụng dữ liệu thật từ `make_dataset` và quỹ đạo huấn luyện thực của `MLPBaseline([2, 11, 1])`.

Chạy preview:
    manim -ql Manim/scenes/01_problem_mlp.py ProblemAndMLPScene --media_dir Manim/media
Xuất 1080p60:
    manim -qh Manim/scenes/01_problem_mlp.py ProblemAndMLPScene --media_dir Manim/media
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from manim import (
    Axes,
    Circle,
    Create,
    DashedVMobject,
    DecimalNumber,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LEFT,
    Line,
    ManimColor,
    MathTex,
    MoveAlongPath,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Transform,
    UP,
    VGroup,
    ValueTracker,
    Write,
    always_redraw,
    interpolate_color,
)

SCENES_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCENES_DIR.parents[1]
for _p in (SCENES_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Source.src.data import make_dataset
from Source.src.mlp_baseline import build_mlp
from style import (
    CARD_BG,
    DATA_CORAL,
    GRID_COLOR,
    KAN_TEAL,
    MLP_AMBER,
    PRIMARY_BLUE,
    TARGET_GREEN,
    TEXT_LIGHT,
    TEXT_MUTED,
    apply_shared_theme,
    fade_out_all,
    make_citation_badge,
    make_label,
    make_section_header,
)

apply_shared_theme()


def _target_fn(x: float, y: float) -> float:
    return math.exp(math.sin(math.pi * x) + y * y)


def _prepare_tv1_scene_data(y0_fixed: float = 0.40, num_x: int = 81) -> dict:
    """Sinh dữ liệu split thật (seed=42) và quỹ đạo huấn luyện thật của MLP [2, 11, 1]."""
    bundle = make_dataset(
        task="exp_sin_y2",
        n_train=1000,
        n_val=200,
        n_test=1000,
        seed=42,
        dtype="float32",
        device="cpu",
    )
    sample_xy = bundle.x_train[:36].numpy()
    sample_vals = bundle.y_train[:36].squeeze(1).numpy()

    # Chọn 1 điểm thật từ tập test nằm gọn trong vùng trống để đặt nhãn (x*, ?) không bị đè
    test_candidates = bundle.x_test[:100].numpy()
    best_pt = (0.45, 0.65)
    best_dist = -1.0
    for tx, ty in test_candidates:
        if -0.78 <= tx <= 0.78 and -0.78 <= ty <= 0.72:
            d = float(np.min(np.linalg.norm(sample_xy - np.array([tx, ty]), axis=1)))
            if d > best_dist:
                best_dist = d
                best_pt = (float(tx), float(ty))

    # Huấn luyện thật MLPBaseline([2, 11, 1], SiLU) trên 1000 mẫu train bằng LBFGS
    model = build_mlp(
        input_dim=2,
        hidden_widths=[11],
        activation="silu",
        output_dim=1,
        bias=True,
        init="xavier_uniform",
        seed=42,
        dtype="float32",
        device="cpu",
    )
    x_grid = np.linspace(-1.0, 1.0, num_x)
    slice_inputs = torch.tensor(
        np.column_stack([x_grid, np.full_like(x_grid, y0_fixed)]),
        dtype=torch.float32,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    capture_outer = {0, 1, 2, 5, 10}
    ckpt_iters: list[int] = []
    ckpt_preds: list[np.ndarray] = []

    with torch.no_grad():
        ckpt_iters.append(0)
        ckpt_preds.append(model(slice_inputs).squeeze(1).numpy().copy())

    for outer in range(1, 11):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(model(bundle.x_train), bundle.y_train)
            loss.backward()
            return loss

        optimizer.step(closure)
        if outer in capture_outer:
            with torch.no_grad():
                ckpt_iters.append(outer * 20)
                ckpt_preds.append(model(slice_inputs).squeeze(1).numpy().copy())

    return {
        "sample_xy": sample_xy,
        "sample_vals": sample_vals,
        "query_xy": best_pt,
        "x_grid": x_grid,
        "ckpt_iters": ckpt_iters,
        "ckpt_preds": ckpt_preds,
    }


class ProblemAndMLPScene(Scene):
    """Cảnh mở đầu của TV1 (~1m40s): giới thiệu bài toán, MLP và câu nối sang KAN."""

    def construct(self) -> None:
        self._scene_data = _prepare_tv1_scene_data(y0_fixed=0.40)

        self._beat1_problem_and_samples()
        fade_out_all(self, run_time=0.6)

        self._beat2_target_and_fitting()
        fade_out_all(self, run_time=0.6)

        mlp_bundle = self._beat3_mlp_architecture()
        self._beat4_transition_to_kan(mlp_bundle)

    # ------------------------------------------------------------------
    # Beat 1 (0:00 - 0:25): Tiêu đề & dữ liệu thực từ split train
    # ------------------------------------------------------------------
    def _beat1_problem_and_samples(self) -> None:
        title = make_label(
            "KAN: Kolmogorov–Arnold Networks",
            font_size=40,
            color=PRIMARY_BLUE,
            weight="BOLD",
        )
        subtitle = make_label(
            "Bài toán xấp xỉ hàm nhiều biến và cách tiếp cận MLP",
            font_size=24,
            color=TEXT_LIGHT,
        ).next_to(title, DOWN, buff=0.35)
        author_tag = make_label(
            "CSC14005 — Nhập môn Học máy",
            font_size=18,
            color=TEXT_MUTED,
        ).next_to(subtitle, DOWN, buff=0.35)
        badge_open = make_citation_badge("Liu et al., ICLR 2025 (arXiv:2404.19756v5)")

        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2), run_time=1.8)
        self.play(FadeIn(author_tag), FadeIn(badge_open), run_time=1.0)
        self.wait(2.2)
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(author_tag),
            FadeOut(badge_open),
            run_time=0.7,
        )

        header = make_section_header(
            "1. Bài toán học hàm từ dữ liệu hữu hạn",
            "Dự đoán giá trị đầu ra trên miền liên tục",
        )
        badge = make_citation_badge("Liu et al., ICLR 2025 (arXiv:2404.19756v5)")
        self.play(Write(header), FadeIn(badge), run_time=1.2)

        domain_axes = Axes(
            x_range=[-1.15, 1.15, 0.5],
            y_range=[-1.15, 1.15, 0.5],
            x_length=4.4,
            y_length=4.4,
            axis_config={"color": TEXT_MUTED, "stroke_width": 1.8, "include_tip": True},
        ).shift(LEFT * 3.2 + DOWN * 0.30)

        x_lbl = MathTex("x_1", font_size=24, color=TEXT_LIGHT).next_to(
            domain_axes.x_axis.get_end(), RIGHT, buff=0.12
        )
        y_lbl = MathTex("x_2", font_size=24, color=TEXT_LIGHT).next_to(
            domain_axes.y_axis.get_end(), UP, buff=0.12
        )
        domain_box = RoundedRectangle(
            corner_radius=0.05,
            width=domain_axes.c2p(1, 0)[0] - domain_axes.c2p(-1, 0)[0],
            height=domain_axes.c2p(0, 1)[1] - domain_axes.c2p(0, -1)[1],
            stroke_color=PRIMARY_BLUE,
            stroke_width=1.6,
            fill_color=PRIMARY_BLUE,
            fill_opacity=0.06,
        ).move_to(domain_axes.c2p(0, 0))
        domain_tag = MathTex(r"[-1, 1]^2", font_size=22, color=PRIMARY_BLUE).next_to(
            domain_box, UP, buff=0.10, aligned_edge=RIGHT
        )
        sample_caption = make_label(
            "36 điểm mẫu đầu từ tập train (n = 1000, seed = 42)",
            font_size=15,
            color=TEXT_MUTED,
        ).next_to(domain_axes, DOWN, buff=0.18)

        self.play(
            Create(domain_axes),
            FadeIn(x_lbl),
            FadeIn(y_lbl),
            Create(domain_box),
            FadeIn(domain_tag),
            run_time=1.8,
        )

        pts_xy = self._scene_data["sample_xy"]
        vals = self._scene_data["sample_vals"]
        dots = VGroup()
        low_col = ManimColor(PRIMARY_BLUE)
        high_col = ManimColor(DATA_CORAL)
        for (px, py), val in zip(pts_xy, vals):
            alpha = max(0.0, min(1.0, (float(val) - 0.5) / 6.5))
            dot_color = interpolate_color(low_col, high_col, alpha)
            dots.add(Dot(domain_axes.c2p(float(px), float(py)), radius=0.055, color=dot_color))

        self.play(FadeIn(dots, lag_ratio=0.04), FadeIn(sample_caption), run_time=2.5)

        card = RoundedRectangle(
            corner_radius=0.15,
            width=5.7,
            height=4.0,
            stroke_color=GRID_COLOR,
            stroke_width=1.5,
            fill_color=CARD_BG,
            fill_opacity=0.88,
        ).shift(RIGHT * 3.1 + DOWN * 0.35)

        eq_data = MathTex(
            r"\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}, \quad y_i = f(\mathbf{x}_i)",
            font_size=28,
            color=TEXT_LIGHT,
        ).move_to(card.get_top() + DOWN * 0.85)
        line1 = make_label(
            "• Quan sát hữu hạn trên miền [-1, 1]²",
            font_size=21,
            color=TEXT_LIGHT,
        ).next_to(eq_data, DOWN, buff=0.45, aligned_edge=LEFT)
        line2 = make_label(
            "• Cần học mô hình f_θ(x) tổng quát",
            font_size=21,
            color=TEXT_LIGHT,
        ).next_to(line1, DOWN, buff=0.30, aligned_edge=LEFT)
        line3 = make_label(
            "• Dự đoán tại điểm test chưa gặp?",
            font_size=21,
            color=TARGET_GREEN,
            weight="BOLD",
        ).next_to(line2, DOWN, buff=0.35, aligned_edge=LEFT)

        card_text = VGroup(eq_data, line1, line2, line3)
        card_text.move_to(card.get_center())

        self.play(FadeIn(card), Write(eq_data), run_time=1.8)
        self.play(FadeIn(line1), FadeIn(line2), run_time=2.0)

        qx, qy = self._scene_data["query_xy"]
        query_pt = domain_axes.c2p(qx, qy)
        query_dot = Dot(query_pt, radius=0.09, color=TARGET_GREEN)
        query_ring = Circle(radius=0.18, color=TARGET_GREEN, stroke_width=2.5).move_to(query_pt)
        query_lbl = MathTex(r"(\mathbf{x}^*, ?)", font_size=22, color=TARGET_GREEN).next_to(
            query_ring, UP, buff=0.08
        )

        self.play(GrowFromCenter(query_dot), Create(query_ring), FadeIn(query_lbl), FadeIn(line3), run_time=1.8)
        self.play(query_ring.animate.scale(1.45).set_stroke(opacity=0.25), run_time=1.3)
        self.play(query_ring.animate.scale(1 / 1.45).set_stroke(opacity=1.0), run_time=1.3)
        self.wait(5.5)

    # ------------------------------------------------------------------
    # Beat 2 (0:25 - 0:55): Hàm mục tiêu & quỹ đạo huấn luyện thực của MLP
    # ------------------------------------------------------------------
    def _beat2_target_and_fitting(self) -> None:
        header = make_section_header(
            "2. Hàm mục tiêu chuẩn & tối thiểu hóa sai số",
            "Quá trình học thực tế của MLP [2, 11, 1] trên lát cắt 1D",
        )
        badge = make_citation_badge("Bài toán Mục 3.1 — Liu et al. (2025)")
        target_eq = MathTex(
            r"f(x, y) = \exp\!\left(\sin(\pi x) + y^2\right)",
            font_size=31,
            color=TARGET_GREEN,
        ).to_edge(UP, buff=0.45).to_edge(RIGHT, buff=0.50)

        self.play(Write(header), Write(target_eq), FadeIn(badge), run_time=1.8)

        axes = Axes(
            x_range=[-1.05, 1.05, 0.5],
            y_range=[-1.2, 7.8, 2.0],
            x_length=6.8,
            y_length=4.4,
            axis_config={"color": TEXT_MUTED, "stroke_width": 1.8},
        ).shift(LEFT * 2.5 + DOWN * 0.50)

        ax_x = MathTex("x", font_size=24, color=TEXT_LIGHT).next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        ax_y = MathTex("f(x, y_0)", font_size=24, color=TEXT_LIGHT).next_to(axes.y_axis.get_end(), UP, buff=0.1)

        y0_tracker = ValueTracker(0.0)
        y0_prefix = MathTex("y_0 =", font_size=26, color=PRIMARY_BLUE)
        y0_num = DecimalNumber(0.0, num_decimal_places=2, font_size=26, color=PRIMARY_BLUE)
        y0_badge = VGroup(y0_prefix, y0_num).arrange(RIGHT, buff=0.10)
        y0_badge.next_to(axes, UP, buff=0.12, aligned_edge=RIGHT).shift(LEFT * 0.25)

        def _update_y0(mob: DecimalNumber) -> None:
            mob.set_value(y0_tracker.get_value())
            mob.next_to(y0_prefix, RIGHT, buff=0.10)

        y0_num.add_updater(_update_y0)

        true_curve = always_redraw(
            lambda: axes.plot(
                lambda x: _target_fn(x, y0_tracker.get_value()),
                x_range=[-1.0, 1.0],
                color=TARGET_GREEN,
                stroke_width=3.2,
            )
        )

        self.play(Create(axes), FadeIn(ax_x), FadeIn(ax_y), FadeIn(y0_badge), run_time=1.5)
        self.play(Create(true_curve), run_time=1.8)

        self.play(y0_tracker.animate.set_value(0.80), run_time=2.6)
        self.play(y0_tracker.animate.set_value(-0.55), run_time=2.6)
        self.play(y0_tracker.animate.set_value(0.40), run_time=2.0)
        y0_num.clear_updaters()

        y0_fixed = 0.40
        sample_xs = np.linspace(-0.92, 0.92, 11)
        sample_dots = VGroup(
            *[
                Dot(axes.c2p(float(sx), _target_fn(float(sx), y0_fixed)), radius=0.065, color=TEXT_LIGHT)
                for sx in sample_xs
            ]
        )
        self.play(FadeIn(sample_dots, lag_ratio=0.06), run_time=1.2)

        x_grid = self._scene_data["x_grid"]
        ckpt_iters = self._scene_data["ckpt_iters"]
        ckpt_preds = self._scene_data["ckpt_preds"]
        ckpt_tracker = ValueTracker(0.0)

        def _interp_ckpt_pred(x_val: float, progress: float) -> float:
            idx = min(len(ckpt_preds) - 2, max(0, int(math.floor(progress))))
            frac = progress - idx
            v0 = float(np.interp(x_val, x_grid, ckpt_preds[idx]))
            v1 = float(np.interp(x_val, x_grid, ckpt_preds[idx + 1]))
            return (1.0 - frac) * v0 + frac * v1

        def _interp_iter_count(progress: float) -> int:
            idx = min(len(ckpt_iters) - 2, max(0, int(math.floor(progress))))
            frac = progress - idx
            return int(round((1.0 - frac) * ckpt_iters[idx] + frac * ckpt_iters[idx + 1]))

        pred_curve = always_redraw(
            lambda: axes.plot(
                lambda x: _interp_ckpt_pred(x, ckpt_tracker.get_value()),
                x_range=[-1.0, 1.0],
                color=MLP_AMBER,
                stroke_width=3.0,
            )
        )
        residual_lines = always_redraw(
            lambda: VGroup(
                *[
                    Line(
                        axes.c2p(float(sx), _interp_ckpt_pred(float(sx), ckpt_tracker.get_value())),
                        axes.c2p(float(sx), _target_fn(float(sx), y0_fixed)),
                        color=DATA_CORAL,
                        stroke_width=2.2,
                    )
                    for sx in sample_xs
                ]
            )
        )

        iter_prefix = make_label("LBFGS iter =", font_size=18, color=MLP_AMBER, weight="BOLD")
        iter_num = DecimalNumber(0, num_decimal_places=0, font_size=21, color=MLP_AMBER)
        iter_badge = VGroup(iter_prefix, iter_num).arrange(RIGHT, buff=0.10)
        iter_badge.next_to(axes, UP, buff=0.12, aligned_edge=LEFT).shift(RIGHT * 0.05)

        def _update_iter(mob: DecimalNumber) -> None:
            mob.set_value(_interp_iter_count(ckpt_tracker.get_value()))
            mob.next_to(iter_prefix, RIGHT, buff=0.10)

        iter_num.add_updater(_update_iter)

        loss_card = RoundedRectangle(
            corner_radius=0.15,
            width=4.9,
            height=4.2,
            stroke_color=GRID_COLOR,
            stroke_width=1.5,
            fill_color=CARD_BG,
            fill_opacity=0.88,
        ).shift(RIGHT * 3.8 + DOWN * 0.48)

        legend_true = make_label("• Đường hàm thực f(x, y₀)", font_size=18, color=TARGET_GREEN)
        legend_pred = make_label("• Dự đoán thực MLP [2, 11, 1]", font_size=18, color=MLP_AMBER).next_to(
            legend_true, DOWN, buff=0.18, aligned_edge=LEFT
        )
        legend_err = make_label("• Phần dư cần tối thiểu", font_size=18, color=DATA_CORAL).next_to(
            legend_pred, DOWN, buff=0.18, aligned_edge=LEFT
        )
        eq_mse = MathTex(
            r"\mathcal{L}_{\mathrm{MSE}}(\theta) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2",
            font_size=23,
            color=TEXT_LIGHT,
        ).next_to(legend_err, DOWN, buff=0.38)
        eq_rmse = MathTex(
            r"\mathrm{RMSE} = \sqrt{\mathcal{L}_{\mathrm{MSE}}}",
            font_size=23,
            color=PRIMARY_BLUE,
        ).next_to(eq_mse, DOWN, buff=0.24)
        train_note = make_label(
            "Huấn luyện thực trên 1000 mẫu (45 tham số)",
            font_size=15,
            color=TEXT_MUTED,
        ).next_to(eq_rmse, DOWN, buff=0.25)

        loss_group = VGroup(legend_true, legend_pred, legend_err, eq_mse, eq_rmse, train_note)
        loss_group.move_to(loss_card.get_center())

        self.play(
            FadeIn(loss_card),
            FadeIn(loss_group),
            FadeIn(iter_badge),
            Create(pred_curve),
            Create(residual_lines),
            run_time=1.8,
        )
        # Di chuyển qua các checkpoint LBFGS thật: 0 -> 20 -> 40 -> 100 -> 200
        self.play(ckpt_tracker.animate.set_value(1.0), run_time=1.8)
        self.play(ckpt_tracker.animate.set_value(2.0), run_time=1.5)
        self.play(ckpt_tracker.animate.set_value(3.0), run_time=1.8)
        self.play(ckpt_tracker.animate.set_value(4.0), run_time=1.8)
        iter_num.clear_updaters()
        self.wait(5.0)

    # ------------------------------------------------------------------
    # Beat 3 (0:55 - 1:30): Kiến trúc MLP (sơ đồ giản lược [2, 4, 1])
    # ------------------------------------------------------------------
    def _build_mlp_diagram(self) -> dict:
        in_x, hid_x, out_x = -4.6, -2.0, 0.5
        in_ys = [0.7, -1.3]
        hid_ys = [1.5, 0.3, -0.9, -2.1]
        out_ys = [-0.3]

        in_nodes = VGroup()
        for idx, y in enumerate(in_ys, start=1):
            c = Circle(radius=0.34, color=PRIMARY_BLUE, stroke_width=2.5, fill_color=CARD_BG, fill_opacity=1.0)
            c.move_to([in_x, y, 0])
            lbl = MathTex(f"x_{idx}", font_size=24, color=TEXT_LIGHT).move_to(c.get_center())
            in_nodes.add(VGroup(c, lbl))

        hid_nodes = VGroup()
        for y in hid_ys:
            c = Circle(radius=0.34, color=MLP_AMBER, stroke_width=2.5, fill_color=CARD_BG, fill_opacity=1.0)
            c.move_to([hid_x, y, 0])
            lbl = MathTex(r"\sigma", font_size=26, color=MLP_AMBER).move_to(c.get_center())
            hid_nodes.add(VGroup(c, lbl))

        out_circle = Circle(radius=0.34, color=TARGET_GREEN, stroke_width=2.5, fill_color=CARD_BG, fill_opacity=1.0)
        out_circle.move_to([out_x, out_ys[0], 0])
        out_lbl = MathTex(r"\hat{y}", font_size=25, color=TARGET_GREEN).move_to(out_circle.get_center())
        out_node = VGroup(out_circle, out_lbl)

        edges_l1 = VGroup()
        for src in in_nodes:
            for dst in hid_nodes:
                edges_l1.add(
                    Line(
                        src[0].get_right(),
                        dst[0].get_left(),
                        color=TEXT_MUTED,
                        stroke_width=2.0,
                        z_index=-1,
                    )
                )

        edges_l2 = VGroup()
        for src in hid_nodes:
            edges_l2.add(
                Line(
                    src[0].get_right(),
                    out_node[0].get_left(),
                    color=TEXT_MUTED,
                    stroke_width=2.0,
                    z_index=-1,
                )
            )

        return {
            "in_nodes": in_nodes,
            "hid_nodes": hid_nodes,
            "out_node": out_node,
            "edges_l1": edges_l1,
            "edges_l2": edges_l2,
        }

    def _beat3_mlp_architecture(self) -> dict:
        header = make_section_header(
            "3. Kiến trúc MLP: trọng số cạnh & activation tại nút",
            "Sơ đồ minh họa [2, 4, 1] (thực nghiệm dùng 6 cấu hình 25–501 tham số)",
        )
        badge = make_citation_badge("Hornik et al., 1989; Liu et al., 2025")
        self.play(Write(header), FadeIn(badge), run_time=1.4)

        mlp = self._build_mlp_diagram()
        all_edges = VGroup(mlp["edges_l1"], mlp["edges_l2"])
        all_nodes = VGroup(mlp["in_nodes"], mlp["hid_nodes"], mlp["out_node"])

        self.play(Create(all_edges), FadeIn(all_nodes), run_time=2.4)

        pulses_l1 = [Dot(radius=0.06, color=PRIMARY_BLUE) for _ in mlp["edges_l1"]]
        self.play(*[MoveAlongPath(p, e) for p, e in zip(pulses_l1, mlp["edges_l1"])], run_time=1.8)
        self.play(*[FadeOut(p) for p in pulses_l1], run_time=0.3)

        pulses_l2 = [Dot(radius=0.06, color=MLP_AMBER) for _ in mlp["edges_l2"]]
        self.play(*[MoveAlongPath(p, e) for p, e in zip(pulses_l2, mlp["edges_l2"])], run_time=1.6)
        self.play(*[FadeOut(p) for p in pulses_l2], run_time=0.3)

        edge_note = make_label(
            "Cạnh: trọng số tuyến tính w_ij (học được)",
            font_size=20,
            color=MLP_AMBER,
            weight="BOLD",
        ).next_to(mlp["hid_nodes"], DOWN, buff=0.45).shift(LEFT * 0.8)
        w_sample = MathTex("w_{ij}", font_size=22, color=MLP_AMBER).move_to(
            mlp["edges_l1"][0].get_center() + UP * 0.25
        )
        self.play(
            mlp["edges_l1"].animate.set_color(MLP_AMBER),
            mlp["edges_l2"].animate.set_color(MLP_AMBER),
            FadeIn(w_sample),
            FadeIn(edge_note),
            run_time=1.8,
        )
        self.wait(2.5)

        node_box = RoundedRectangle(
            corner_radius=0.15,
            width=5.1,
            height=4.8,
            stroke_color=GRID_COLOR,
            stroke_width=1.5,
            fill_color=CARD_BG,
            fill_opacity=0.90,
        ).shift(RIGHT * 3.7 + DOWN * 0.45)

        box_title = make_label(
            "Nút ẩn: Activation cố định σ(z)",
            font_size=21,
            color=PRIMARY_BLUE,
            weight="BOLD",
        ).move_to(node_box.get_top() + DOWN * 0.42)

        silu_axes = Axes(
            x_range=[-4.0, 4.0, 2.0],
            y_range=[-0.6, 3.8, 1.0],
            x_length=3.9,
            y_length=1.9,
            axis_config={"color": TEXT_MUTED, "stroke_width": 1.4},
        ).next_to(box_title, DOWN, buff=0.25)
        silu_curve = silu_axes.plot(
            lambda z: z / (1.0 + math.exp(-z)),
            x_range=[-4.0, 4.0],
            color=MLP_AMBER,
            stroke_width=3.0,
        )
        silu_eq = MathTex(
            r"\mathrm{SiLU}(z) = \frac{z}{1 + e^{-z}}",
            font_size=22,
            color=MLP_AMBER,
        ).next_to(silu_axes, DOWN, buff=0.18)

        mlp_eq = MathTex(
            r"\mathrm{MLP}(\mathbf{x}) = (W_2 \circ \sigma \circ W_1)(\mathbf{x})",
            font_size=24,
            color=TEXT_LIGHT,
        ).next_to(silu_eq, DOWN, buff=0.25)
        mlp_lim = make_label(
            "Không tối ưu hàm 1D trực tiếp trên từng cạnh",
            font_size=17,
            color=TEXT_MUTED,
        ).next_to(mlp_eq, DOWN, buff=0.22)

        focus_ring = SurroundingRectangle(mlp["hid_nodes"][1], color=PRIMARY_BLUE, buff=0.10, corner_radius=0.1)

        self.play(Create(focus_ring), FadeIn(node_box), Write(box_title), run_time=1.8)
        self.play(Create(silu_axes), Create(silu_curve), Write(silu_eq), run_time=2.5)
        self.play(Write(mlp_eq), FadeIn(mlp_lim), run_time=2.2)
        self.wait(6.0)

        mlp["header"] = header
        mlp["badge"] = badge
        mlp["edge_note"] = edge_note
        mlp["w_sample"] = w_sample
        mlp["focus_ring"] = focus_ring
        mlp["right_panel"] = VGroup(node_box, box_title, silu_axes, silu_curve, silu_eq, mlp_eq, mlp_lim)
        return mlp

    # ------------------------------------------------------------------
    # Beat 4 (1:30 - 1:50): Phóng to cạnh–nút & câu nối sang KAN
    # ------------------------------------------------------------------
    def _beat4_transition_to_kan(self, mlp: dict) -> None:
        new_header = make_section_header(
            "4. Ý tưởng chuyển đổi: Đưa hàm học được lên cạnh",
            "Chuyển từ trọng số vô hướng sang hàm một biến",
        )
        new_badge = make_citation_badge("Liu et al., Hình 0.1 (2025)")

        keep_in = mlp["in_nodes"][0]
        keep_hid = mlp["hid_nodes"][0]
        keep_edge = mlp["edges_l1"][0]

        dim_targets = [
            mlp["right_panel"],
            mlp["focus_ring"],
            mlp["edge_note"],
            mlp["w_sample"],
            mlp["in_nodes"][1],
            mlp["hid_nodes"][1],
            mlp["hid_nodes"][2],
            mlp["hid_nodes"][3],
            mlp["out_node"],
            mlp["edges_l2"],
            *[e for idx, e in enumerate(mlp["edges_l1"]) if idx != 0],
        ]

        self.play(
            ReplacementTransform(mlp["header"], new_header),
            ReplacementTransform(mlp["badge"], new_badge),
            *[FadeOut(m) for m in dim_targets],
            run_time=1.5,
        )

        src_target = Line(LEFT * 3.0 + UP * 0.2, RIGHT * 3.0 + UP * 0.2)
        self.play(
            FadeOut(keep_edge),
            keep_in.animate.scale(1.25).move_to(src_target.get_start()),
            keep_hid.animate.scale(1.25).move_to(src_target.get_end()),
            run_time=1.5,
        )
        zoomed_edge = Line(
            keep_in[0].get_right(),
            keep_hid[0].get_left(),
            color=MLP_AMBER,
            stroke_width=4.0,
        )
        self.play(Create(zoomed_edge), run_time=0.8)

        w_badge = RoundedRectangle(
            corner_radius=0.12,
            width=1.7,
            height=0.85,
            stroke_color=MLP_AMBER,
            stroke_width=2.2,
            fill_color=CARD_BG,
            fill_opacity=1.0,
        ).move_to(zoomed_edge.get_center())
        w_text = MathTex("w", font_size=34, color=MLP_AMBER).move_to(w_badge.get_center())
        w_group = VGroup(w_badge, w_text)
        mlp_tag = make_label("MLP: trọng số w trên cạnh, σ cố định tại nút", font_size=21, color=MLP_AMBER).next_to(
            w_group, UP, buff=0.65
        )
        self.play(FadeIn(w_group), FadeIn(mlp_tag), run_time=1.5)
        self.wait(1.8)

        phi_box = RoundedRectangle(
            corner_radius=0.14,
            width=2.5,
            height=1.25,
            stroke_color=KAN_TEAL,
            stroke_width=2.8,
            fill_color=CARD_BG,
            fill_opacity=1.0,
        ).move_to(zoomed_edge.get_center())
        mini_ax = Axes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            x_length=1.45,
            y_length=0.75,
            tips=False,
            axis_config={"color": GRID_COLOR, "stroke_width": 1.0},
        ).move_to(phi_box.get_center() + LEFT * 0.35)
        mini_spline = mini_ax.plot(
            lambda t: 0.65 * math.sin(2.5 * t) + 0.25 * math.cos(4.0 * t),
            x_range=[-1.0, 1.0],
            color=KAN_TEAL,
            stroke_width=2.6,
        )
        phi_tex = MathTex(r"\phi(x)", font_size=28, color=KAN_TEAL).next_to(mini_ax, RIGHT, buff=0.12)
        kan_edge_card = VGroup(phi_box, mini_ax, mini_spline, phi_tex)

        sum_lbl = MathTex(r"\sum", font_size=34, color=KAN_TEAL).move_to(keep_hid[0].get_center())
        dashed_glow = DashedVMobject(
            SurroundingRectangle(kan_edge_card, color=KAN_TEAL, buff=0.12, corner_radius=0.16),
            num_dashes=26,
        )
        kan_tag = make_label(
            "KAN (khái niệm): hàm 1D φ(x) trên cạnh, nút chỉ cộng ∑",
            font_size=20,
            color=KAN_TEAL,
            weight="BOLD",
        ).next_to(kan_edge_card, UP, buff=0.55)
        concept_note = make_label(
            "Minh họa khái niệm hàm 1D học được φ(x) trên cạnh",
            font_size=15,
            color=TEXT_MUTED,
        ).next_to(kan_edge_card, DOWN, buff=0.35)

        self.play(
            Transform(mlp_tag, kan_tag),
            ReplacementTransform(w_group, kan_edge_card),
            zoomed_edge.animate.set_color(KAN_TEAL),
            keep_hid[0].animate.set_stroke(color=KAN_TEAL),
            ReplacementTransform(keep_hid[1], sum_lbl),
            Create(dashed_glow),
            FadeIn(concept_note),
            run_time=2.8,
        )

        bridge_q = make_label(
            "Điều gì xảy ra nếu hàm phi tuyến học được nằm ngay trên cạnh?",
            font_size=24,
            color=TEXT_LIGHT,
            weight="BOLD",
        ).to_edge(DOWN, buff=1.15)
        next_cue = make_label(
            "Tiếp theo: Biểu diễn B-spline & cơ chế mở rộng lưới KAN →",
            font_size=19,
            color=KAN_TEAL,
        ).next_to(bridge_q, DOWN, buff=0.20)

        self.play(Write(bridge_q), FadeIn(next_cue), run_time=2.5)
        self.wait(6.5)
