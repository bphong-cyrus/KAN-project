# Manim (`Manim/`)

Mã nguồn dựng video minh họa bằng **Manim Community v0.19.1**.

## Cài đặt môi trường chạy độc lập

Scene Cảnh 1 sử dụng trực tiếp dữ liệu từ `Source/src/data.py` và huấn luyện `MLPBaseline` trong PyTorch nên cần đủ cả dependency của `Source/` và `manim==0.19.1`:

```powershell
pip install -r Manim/requirements.txt
```

## Hiện có (TV1)

- `scenes/style.py`: Bảng màu nền tối (`#0F172A`), font tiếng Việt (`Segoe UI` trên Windows, `Noto Sans` trên Linux) và các helper dựng tiêu đề/nguồn trích dẫn.
- `scenes/01_problem_mlp.py`: `ProblemAndMLPScene` (**1 phút 15.5 giây**, chính xác `75.50s` ở `1080p60`) giới thiệu bài toán xấp xỉ hàm trên $[-1,1]^2$, quỹ đạo học từng bước thật của `MLPBaseline([2, 11, 1], float64)`, sơ đồ kiến trúc MLP `[2, 4, 1]` và câu nối chuyển sang hàm học được trên cạnh.
- `scenes/tv1_scene01_notes.md`: Bảng timecode đo chính xác và ghi chú lời thoại cho Cảnh 1.

## Lưu ý khi dựng thêm cảnh

- Dùng `apply_shared_theme()` và `make_citation_badge()` từ `style.py` để đồng bộ khung hình 16:9 và nguồn trích dẫn.
- Đặt chữ tiếng Việt trong `Text(..., font=FONT_SANS)` và ghép `MathTex(...)` cho mọi ký hiệu toán học có chỉ số dưới/trên (không viết plain-text như `f_θ` hay `w_ij`).
- Mọi sơ đồ và đồ thị đều dựng trực tiếp bằng Manim, không cắt ảnh từ PDF bài báo.

## Lệnh render Cảnh 1

Chạy từ thư mục gốc `KAN-project/`:

```powershell
# Preview 480p15
manim -ql Manim/scenes/01_problem_mlp.py ProblemAndMLPScene --media_dir Manim/media

# Xuất 1080p60
manim -qh Manim/scenes/01_problem_mlp.py ProblemAndMLPScene --media_dir Manim/media
```
