"""TV2: KAN, hàm cạnh, B-spline và trực giác tăng lưới (150 giây).

Preview: python -m manim -ql Manim/scenes/02_kan_spline.py KANSplineScene --media_dir Manim/media
Final:   python -m manim -qh Manim/scenes/02_kan_spline.py KANSplineScene --media_dir Manim/media

Đồ thị là ví dụ spline xác định, không dùng checkpoint hoặc số liệu panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from manim import (
    Axes, Circle, Create, DashedVMobject, DecimalNumber, Dot, DOWN,
    FadeIn, FadeOut, Indicate, LEFT, Line, MathTex, MoveAlongPath,
    RIGHT, Scene, TexTemplateLibrary, Transform,
    UP, ValueTracker, VGroup, VMobject, Write, always_redraw, linear,
)

SCENES_DIR = Path(__file__).resolve().parent
if str(SCENES_DIR) not in sys.path:
    sys.path.insert(0, str(SCENES_DIR))

from style import (
    DATA_CORAL, GRID_COLOR, KAN_TEAL, MLP_AMBER, PRIMARY_BLUE,
    TARGET_GREEN, TEXT_LIGHT, TEXT_MUTED, apply_shared_theme,
    fade_out_all, make_citation_badge, make_label, make_section_header,
)

apply_shared_theme()

DEGREE = 3
SAMPLE_X = np.linspace(-1.0, 1.0, 401)


def _math(expression: str, size: int = 34, color=TEXT_LIGHT) -> MathTex:
    return MathTex(
        expression, font_size=size, color=color,
        tex_template=TexTemplateLibrary.simple,
    )


def _knots(grid: int) -> np.ndarray:
    """G khoảng trên [-1,1] và k knot mở rộng về mỗi phía."""
    h = 2.0 / grid
    return -1.0 + np.arange(-DEGREE, grid + DEGREE + 1) * h


def _target(x):
    return 0.5 * np.sin(2 * np.pi * x) + 0.25 * np.cos(5 * np.pi * x)


def _fit_demo_spline(knots: np.ndarray) -> BSpline:
    """Hồi quy spline 1D để minh họa; không mô phỏng LBFGS/KAN."""
    design = BSpline.design_matrix(SAMPLE_X, knots, DEGREE).toarray()
    coefficients = np.linalg.lstsq(design, _target(SAMPLE_X), rcond=None)[0]
    return BSpline(knots, coefficients, DEGREE)


def _double_grid(spline: BSpline) -> BSpline:
    """Chèn knot chính xác: đổi biểu diễn, giữ nguyên đường spline cũ."""
    breaks = np.unique(spline.t[(spline.t >= -1.0) & (spline.t <= 1.0)])
    for midpoint in (breaks[:-1] + breaks[1:]) / 2:
        spline = spline.insert_knot(float(midpoint))
    return spline


def _curve(axes: Axes, x, y, color=KAN_TEAL, width: float = 4) -> VMobject:
    points = np.array([axes.c2p(float(a), float(b)) for a, b in zip(x, y)])
    return VMobject(color=color, stroke_width=width).set_points_as_corners(points)


def _axes(width: float = 8.0, height: float = 3.3) -> Axes:
    return Axes(
        x_range=[-1, 1, 0.5], y_range=[-1, 1.5, 0.5],
        x_length=width, y_length=height, tips=False,
        axis_config={"color": GRID_COLOR, "stroke_width": 2, "include_ticks": True},
    )


class KANSplineScene(Scene):
    """Năm đoạn độc lập; thời gian tích lũy 35,65,100,135,150 giây."""

    def construct(self):
        self._network()
        self._edge_function()
        self._basis()
        self._grid_extension()
        self._recap()

    def _start(self, name, title, source):
        self.next_section(name)
        start = self.time
        self.play(
            FadeIn(make_section_header(title)),
            FadeIn(make_citation_badge(source)), run_time=0.6,
        )
        return start

    def _finish(self, start: float, duration: float):
        # Dành 0.6 giây chuyển cảnh, giữ tổng thời lượng độc lập tốc độ render.
        remaining = duration - (self.time - start) - 0.6
        if remaining < -0.01:
            raise RuntimeError("Hoạt hình vượt thời lượng đoạn; cập nhật timecode trước khi bàn giao.")
        if remaining > 0:
            self.wait(remaining)
        fade_out_all(self, run_time=0.6)

    def _network(self):
        start = self._start(
            "01_network", "Từ một cạnh đến cả mạng KAN",
            "Liu et al. (2025), Mục 2.2",
        )
        inputs = VGroup(
            Circle(radius=0.38, color=PRIMARY_BLUE).move_to([-4.8, 1.25, 0]),
            Circle(radius=0.38, color=PRIMARY_BLUE).move_to([-4.8, -0.85, 0]),
        )
        hidden = Circle(radius=0.43, color=KAN_TEAL).move_to([0, 0.2, 0])
        output = Circle(radius=0.43, color=KAN_TEAL).move_to([4.6, 0.2, 0])
        symbols = VGroup(
            _math("x_1", 28).move_to(inputs[0]),
            _math("x_2", 28).move_to(inputs[1]),
            _math(r"\sum", 32).move_to(hidden),
            _math(r"\sum", 32).move_to(output),
        )
        edges = VGroup(
            Line(inputs[0].get_right(), hidden.get_left(), color=KAN_TEAL),
            Line(inputs[1].get_right(), hidden.get_left(), color=DATA_CORAL),
            Line(hidden.get_right(), output.get_left(), color=TARGET_GREEN),
        )
        labels = VGroup(
            _math(r"\phi_{0,1,1}", 29, KAN_TEAL).move_to([-2.6, 1.35, 0]),
            _math(r"\phi_{0,1,2}", 29, DATA_CORAL).move_to([-2.6, -0.95, 0]),
            _math(r"\phi_{1,1,1}", 29, TARGET_GREEN).move_to([2.3, 0.85, 0]),
        )
        architecture = make_label("Kiến trúc [2, 1, 1]", 23, TEXT_MUTED).move_to([0, 2.35, 0])
        self.play(FadeIn(inputs), FadeIn(hidden), FadeIn(output), Write(symbols), run_time=1.4)
        self.play(Create(edges), FadeIn(labels), FadeIn(architecture), run_time=1.6)
        self.wait(3)
        signals = VGroup(
            Dot(edges[0].get_start(), color=KAN_TEAL),
            Dot(edges[1].get_start(), color=DATA_CORAL),
        )
        self.play(FadeIn(signals), run_time=0.5)
        self.play(
            MoveAlongPath(signals[0], edges[0]), MoveAlongPath(signals[1], edges[1]),
            run_time=3, rate_func=linear,
        )
        hidden_formula = _math(
            r"h=\phi_{0,1,1}(x_1)+\phi_{0,1,2}(x_2)", 34
        ).move_to([0, -2.0, 0])
        self.play(FadeOut(signals), Indicate(hidden), Write(hidden_formula), run_time=1.5)
        self.wait(5)
        signal = Dot(edges[2].get_start(), color=TARGET_GREEN)
        self.play(FadeIn(signal), run_time=0.4)
        self.play(MoveAlongPath(signal, edges[2]), run_time=3, rate_func=linear)
        result = _math(r"\hat y=\phi_{1,1,1}(h)", 34).move_to([0, -2.8, 0])
        self.play(FadeOut(signal), Indicate(output), Write(result), run_time=1.5)
        self.wait(4)
        self.play(Indicate(labels), run_time=1.5)
        self._finish(start, 35)

    def _edge_function(self):
        start = self._start(
            "02_edge", "Một hàm cạnh được học như thế nào?",
            "Liu et al. (2025), (2.10)-(2.12)",
        )
        formula = _math(r"\phi(x)=w_b\,\mathrm{SiLU}(x)+w_s\,s(x)", 40).move_to([0, 2.25, 0])
        self.play(Write(formula), run_time=1.8)
        self.wait(3)
        axes = VGroup(*[_axes(3.3, 2.7).move_to([x, -0.3, 0]) for x in (-4.55, 0, 4.55)])
        spline = BSpline(_knots(5), np.array([-0.3, 0.2, 0.9, -0.9, 0.7, 0.2, -0.5, 0.3]), DEGREE)
        base = 0.6 * SAMPLE_X / (1 + np.exp(-SAMPLE_X))
        residual = 0.8 * spline(SAMPLE_X)
        curves = VGroup(
            _curve(axes[0], SAMPLE_X, base, PRIMARY_BLUE),
            _curve(axes[1], SAMPLE_X, residual, KAN_TEAL),
            _curve(axes[2], SAMPLE_X, base + residual, TARGET_GREEN),
        )
        headings = VGroup(*[
            make_label(text, 23, color).move_to([x, 1.45, 0])
            for text, color, x in zip(
                ("Nhánh cơ sở SiLU", "Spline học được", "Hàm trên cạnh"),
                (PRIMARY_BLUE, KAN_TEAL, TARGET_GREEN), (-4.55, 0, 4.55),
            )
        ])
        plus = _math("+", 38).move_to([-2.3, -0.3, 0])
        equals = _math("=", 38).move_to([2.3, -0.3, 0])
        note = make_label("Hệ số minh họa; không phải checkpoint huấn luyện", 21, TEXT_MUTED).move_to([0, -2.65, 0])
        self.play(Create(axes[0]), FadeIn(headings[0]), Create(curves[0]), run_time=2)
        self.wait(3)
        self.play(Create(axes[1]), FadeIn(headings[1]), Create(curves[1]), Write(plus), run_time=2)
        self.wait(4)
        self.play(Create(axes[2]), FadeIn(headings[2]), Create(curves[2]), Write(equals), FadeIn(note), run_time=2)
        self.wait(3)
        self.play(Indicate(curves[2]), run_time=1.5)
        self._finish(start, 30)

    def _basis(self):
        start = self._start(
            "03_basis", "Spline: tổng có trọng số của các hàm cơ sở",
            "Liu et al. (2025), Mục 2.2",
        )
        formula = _math(r"s(x)=\sum_{r=0}^{G+k-1} c_r B_r(x),\qquad k=3", 36).move_to([0, 2.25, 0])
        axes = _axes(8.1, 3.5).move_to([-1.65, -0.25, 0])
        knots = _knots(8)
        basis = BSpline.design_matrix(SAMPLE_X, knots, DEGREE).toarray()
        basis_curves = VGroup(*[
            _curve(axes, SAMPLE_X, basis[:, r], PRIMARY_BLUE, 2).set_stroke(opacity=0.4)
            for r in range(basis.shape[1])
        ])
        selected = 5
        coefficient = ValueTracker(-0.4)
        fixed_coefficients = 0.25 * np.sin(np.arange(basis.shape[1]))
        fixed_coefficients[selected] = 0.0
        fixed = basis @ fixed_coefficients
        weighted_sum = always_redraw(lambda: _curve(
            axes, SAMPLE_X, fixed + coefficient.get_value() * basis[:, selected], KAN_TEAL,
        ))
        selected_basis = _curve(axes, SAMPLE_X, basis[:, selected], MLP_AMBER, 4)
        dots = VGroup(*[Dot(axes.c2p(float(x), 0), radius=0.045, color=TEXT_MUTED) for x in np.linspace(-1, 1, 9)])
        control_label = _math("c_5=", 34, KAN_TEAL).move_to([4.35, 0.6, 0])
        basis_label = _math("B_5(x)", 29, MLP_AMBER).move_to([4.55, 1.5, 0])
        number = DecimalNumber(-0.4, num_decimal_places=2, font_size=34, color=KAN_TEAL)
        number.next_to(control_label, RIGHT, buff=0.15)
        number.add_updater(lambda value: value.set_value(coefficient.get_value()))
        control_note = VGroup(
            make_label("Đổi một hệ số", 24),
            make_label("Chỉ tác động trong", 21, TEXT_MUTED),
            make_label("miền hỗ trợ của B-spline", 21, TEXT_MUTED),
        ).arrange(DOWN, buff=0.2).move_to([4.55, -0.7, 0])
        note = make_label("Minh họa spline; các đường xanh mờ là hàm cơ sở", 21, TEXT_MUTED).move_to([0, -2.85, 0])
        self.play(Write(formula), Create(axes), run_time=1.8)
        self.play(Create(basis_curves), FadeIn(dots), FadeIn(note), run_time=2.5)
        self.wait(4)
        self.play(Create(selected_basis), FadeIn(basis_label), run_time=1.5)
        self.wait(3)
        self.play(FadeIn(control_label), FadeIn(number), FadeIn(control_note), FadeIn(weighted_sum), run_time=1)
        self.play(coefficient.animate.set_value(0.9), run_time=4, rate_func=linear)
        self.wait(3)
        self.play(coefficient.animate.set_value(-0.6), run_time=4, rate_func=linear)
        self.wait(3)
        self.play(coefficient.animate.set_value(0.3), run_time=3, rate_func=linear)
        weighted_sum.clear_updaters()
        number.clear_updaters()
        self._finish(start, 35)

    def _grid_extension(self):
        start = self._start(
            "04_grid", "Tăng lưới: chuyển biểu diễn rồi tiếp tục tối ưu",
            "Liu et al. (2025), Mục 2.4, (2.16)",
        )
        axes = _axes(8.0, 3.6).move_to([-1.6, -0.15, 0])
        target = DashedVMobject(_curve(axes, SAMPLE_X, _target(SAMPLE_X), TEXT_LIGHT, 2), num_dashes=60)
        spline = _fit_demo_spline(_knots(3))
        curve = _curve(axes, SAMPLE_X, spline(SAMPLE_X))
        dots = VGroup(*[Dot(axes.c2p(float(x), 0), radius=0.055, color=MLP_AMBER) for x in np.linspace(-1, 1, 4)])
        grid_label = _math("G=3", 48, KAN_TEAL).move_to([4.6, 1.2, 0])
        degree_label = _math("k=3", 30, TEXT_MUTED).move_to([4.6, 0.4, 0])
        phase = make_label("Fit spline trên lưới thưa", 23).move_to([0, 2.25, 0])
        legend = VGroup(
            make_label("Nét đứt: hàm mục tiêu", 20, TEXT_LIGHT),
            make_label("Xanh: spline minh họa", 20, KAN_TEAL),
            make_label("Chấm: điểm lưới", 20, MLP_AMBER),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([4.55, -1.05, 0])
        note = make_label("Ví dụ spline 1D; không phải log huấn luyện KAN", 21, TEXT_MUTED).move_to([0, -2.85, 0])
        self.play(Create(axes), Create(target), FadeIn(grid_label), FadeIn(degree_label), FadeIn(phase), FadeIn(legend), FadeIn(note), run_time=1.8)
        self.play(Create(curve), FadeIn(dots), run_time=1.8)
        self.wait(3)
        for grid in (6, 12):
            transferred = _double_grid(spline)
            if not np.allclose(transferred(SAMPLE_X), spline(SAMPLE_X), rtol=1e-12, atol=1e-12):
                raise RuntimeError("Chèn knot phải giữ nguyên đường spline minh họa.")
            new_dots = VGroup(*[Dot(axes.c2p(float(x), 0), radius=0.055, color=MLP_AMBER) for x in np.linspace(-1, 1, grid + 1)])
            self.play(
                Transform(phase, make_label("Chuyển sang lưới mới: giữ đường spline cũ", 23).move_to(phase)),
                Transform(grid_label, _math(f"G={grid}", 48, KAN_TEAL).move_to(grid_label)),
                Transform(dots, new_dots), run_time=2,
            )
            self.wait(3)
            # Chèn knot là trường hợp lưới lồng nhau của bài toán chuyển spline.
            # Bước tiếp dùng least squares 1D minh họa, không giả là LBFGS KAN.
            spline = _fit_demo_spline(transferred.t)
            self.play(
                Transform(phase, make_label("Fit lại spline trên lưới mới (minh họa)", 23).move_to(phase)),
                Transform(curve, _curve(axes, SAMPLE_X, spline(SAMPLE_X))), run_time=3,
            )
            self.wait(3)
        self._finish(start, 35)

    def _recap(self):
        start = self._start(
            "05_recap", "Ba ý chính về KAN",
            "Liu et al., ICLR 2025 | arXiv:2404.19756v5",
        )
        rows = VGroup(*[
            make_label(text, 32, color)
            for text, color in (
                ("Cạnh học hàm một biến", KAN_TEAL),
                ("Nút cộng các tín hiệu", PRIMARY_BLUE),
                ("Lưới điều chỉnh độ linh hoạt", MLP_AMBER),
            )
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to([0, 0.65, 0])
        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.8)
            self.wait(1.2)
        question = make_label("Độ chính xác đổi lấy bao nhiêu tham số?", 29).move_to([0, -1.8, 0])
        self.play(Write(question), run_time=1.5)
        self._finish(start, 15)
