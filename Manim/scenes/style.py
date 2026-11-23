"""Giao diện và các thành phần dùng chung cho các cảnh Manim (TV1, TV2, TV4).

Chủ sở hữu: TV1 (Tạ Bỉnh Phong - 24127100).
Dùng chung bảng màu, font hỗ trợ tiếng Việt và helper trích dẫn nguồn ngay trên màn hình.
"""

from __future__ import annotations

import platform
from manim import (
    BLUE_C,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeOut,
    Line,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    config,
)

# Nền tối chuẩn 3Blue1Brown / tài liệu giáo dục hiện đại
BG_COLOR = "#0F172A"
CARD_BG = "#1E293B"
GRID_COLOR = "#334155"

PRIMARY_BLUE = "#38BDF8"
ACCENT_BLUE = BLUE_C
MLP_AMBER = "#F59E0B"
KAN_TEAL = "#2DD4BF"
DATA_CORAL = "#FB7185"
TARGET_GREEN = "#34D399"
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

FONT_SANS = "Segoe UI" if platform.system() == "Windows" else "Noto Sans"


def apply_shared_theme() -> None:
    """Áp dụng màu nền thống nhất cho toàn bộ các scene."""
    config.background_color = BG_COLOR


def make_label(
    text: str,
    font_size: int = 24,
    color: str = TEXT_LIGHT,
    weight: str = "NORMAL",
) -> Text:
    """Tạo text tiếng Việt với font nhất quán giữa các máy."""
    return Text(text, font=FONT_SANS, font_size=font_size, color=color, weight=weight)


def make_section_header(title: str, subtitle: str | None = None) -> VGroup:
    """Tạo tiêu đề mục cố định ở góc trên bên trái khung hình 16:9."""
    heading = make_label(title, font_size=29, color=TEXT_LIGHT, weight="BOLD")
    line_len = max(heading.width, 4.6)
    underline = Line(
        heading.get_bottom() + DOWN * 0.10,
        heading.get_bottom() + DOWN * 0.10,
        color=PRIMARY_BLUE,
        stroke_width=3.0,
    )
    underline.put_start_and_end_on(
        [heading.get_left()[0], heading.get_bottom()[1] - 0.10, 0.0],
        [heading.get_left()[0] + line_len, heading.get_bottom()[1] - 0.10, 0.0],
    )
    items: list = [heading, underline]
    if subtitle:
        sub = make_label(subtitle, font_size=19, color=TEXT_MUTED)
        sub.next_to(underline, DOWN, buff=0.12, aligned_edge=LEFT)
        items.append(sub)
    group = VGroup(*items)
    group.to_edge(UP, buff=0.35).to_edge(LEFT, buff=0.55)
    return group


def make_citation_badge(text: str) -> VGroup:
    """Tạo nhãn nguồn trích dẫn gọn ở góc dưới bên phải màn hình."""
    label = make_label(text, font_size=16, color=TEXT_MUTED)
    box = RoundedRectangle(
        corner_radius=0.1,
        width=label.width + 0.36,
        height=label.height + 0.20,
        stroke_color=GRID_COLOR,
        stroke_width=1.2,
        fill_color=CARD_BG,
        fill_opacity=0.85,
    )
    label.move_to(box.get_center())
    badge = VGroup(box, label)
    badge.to_edge(DOWN, buff=0.22).to_edge(RIGHT, buff=0.35)
    return badge


def fade_out_all(scene: Scene, run_time: float = 0.6) -> None:
    """Xóa mượt toàn bộ đối tượng hiện có trên màn hình giữa các phân đoạn."""
    if scene.mobjects:
        scene.play(*[FadeOut(mob) for mob in scene.mobjects], run_time=run_time)
