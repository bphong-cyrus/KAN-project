# Manim (`Manim/`)

Mã nguồn dựng video minh họa bằng **Manim Community v0.19.1**.

## Cài đặt môi trường chạy độc lập

Scene Cảnh 1 sử dụng trực tiếp dữ liệu từ `Source/src/data.py` và huấn luyện `MLPBaseline` trong PyTorch nên cần đủ cả dependency của `Source/` và `manim==0.19.1`:

```powershell
pip install -r Manim/requirements.txt
```

## Hiện có (TV1)

- `scenes/style.py`: Bảng màu nền tối (`#0F172A`), font tiếng Việt (`Segoe UI` trên Windows, `Noto Sans` trên Linux) và các helper dựng tiêu đề/nguồn trích dẫn.
- `scenes/01_problem_mlp.py`: `ProblemAndMLPScene` (**1 phút 50 giây** ở `1080p60`) giới thiệu bài toán xấp xỉ hàm trên $[-1,1]^2$, quỹ đạo học của `MLPBaseline([2, 11, 1], float64)` trong 200 bước demo, sơ đồ MLP `[2, 4, 1]` và câu nối sang KAN.
- `scenes/tv1_scene01_notes.md`: Bảng timecode đo chính xác và ghi chú lời thoại cho Cảnh 1.

## Lưu ý khi dựng thêm cảnh

- Dùng `apply_shared_theme()` và `make_citation_badge()` từ `style.py` để đồng bộ khung hình 16:9 và nguồn trích dẫn.
- Đặt chữ tiếng Việt trong `Text(..., font=FONT_SANS)` và ghép `MathTex(...)` cho mọi ký hiệu toán học có chỉ số dưới/trên (không viết plain-text như `f_θ` hay `w_ij`).
- Mọi sơ đồ và đồ thị đều dựng trực tiếp bằng Manim, không cắt ảnh từ PDF bài báo.

## Lệnh render Cảnh 1

Các công thức cần `latex` và `dvisvgm` trên PATH. Chạy từ thư mục gốc `KAN-project/` và giữ `--media_dir` là đường dẫn tương đối. Trên Windows với MiKTeX, đường dẫn tuyệt đối có thể khiến `dvisvgm` báo lỗi 87 khi chuyển DVI sang SVG.

```powershell
# Preview 480p15
manim -ql Manim/scenes/01_problem_mlp.py ProblemAndMLPScene --media_dir Manim/media

# Xuất 1080p60
manim -qh Manim/scenes/01_problem_mlp.py ProblemAndMLPScene --media_dir Manim/media
```

## Cảnh 2: KAN và spline (TV2)

- `scenes/02_kan_spline.py`, class `KANSplineScene`: 150 giây thiết kế, gồm kiến trúc `[2,1,1]`, hàm cạnh SiLU + spline, cơ sở B-spline có hỗ trợ cục bộ và trực giác tăng lưới.
- Dùng cùng `style.py` với Cảnh 1. Các đồ thị là minh họa xác định, không dùng log hoặc checkpoint huấn luyện KAN.
- `scenes/tv2_scene02_notes.md`: timecode, lời thoại gợi ý, nguồn và giới hạn của ví dụ.
- `scenes/tv2_scene02.srt`: phụ đề nháp để TV4–TV5 hiệu chỉnh khi ghép lời đọc.

```powershell
.\.venv\Scripts\python.exe -m manim -ql Manim/scenes/02_kan_spline.py KANSplineScene --media_dir Manim/media
.\.venv\Scripts\python.exe -m manim -qh Manim/scenes/02_kan_spline.py KANSplineScene --media_dir Manim/media
```

Đầu ra cuối: `Manim/media/videos/02_kan_spline/1080p60/KANSplineScene.mp4`.
Ví dụ tăng lưới dùng `3→6→12` để chèn knot chính xác trên lưới lồng nhau;
lịch thực nghiệm `3→5→10→20→50` vẫn theo đặc tả chung. Video render chưa có audio và chưa nhúng phụ đề.
