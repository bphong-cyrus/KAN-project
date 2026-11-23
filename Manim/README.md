# Hoạt hình trực quan hóa bằng Manim (`Manim/`)

Thư mục mã nguồn Manim Community (`v0.19.1`) phục vụ video minh họa ý tưởng chính của đồ án **KAN: Kolmogorov–Arnold Networks** (CSC14005).

---

## 1. Thành phần hiện có (Phần TV1)

- `Manim/scenes/style.py`: Bảng màu nền tối (`#0F172A`), font hỗ trợ tiếng Việt, hàm `make_section_header()`, `make_citation_badge()`, `make_label()`, `fade_out_all()`.
- `Manim/scenes/01_problem_mlp.py`: Cảnh mở đầu `ProblemAndMLPScene` (`0:00–1:50`) trình bày bài toán xấp xỉ hàm trên $[-1,1]^2$, mục tiêu $f(x,y)=\exp(\sin(\pi x)+y^2)$, kiến trúc MLP với trọng số cạnh & activation cố định tại nút, và câu nối chuyển phi tuyến lên cạnh.
- `Manim/scenes/tv1_scene01_notes.md`: Bảng timecode, mô tả khung hình và nguồn hiển thị để phối hợp ghép lời thoại/phụ đề.

---

## 2. Quy chuẩn khi mở rộng các cảnh tiếp theo

1. **Khung hình:** 16:9 (`1920x1080` khi xuất bản cuối bằng `-qh`), dùng chung `apply_shared_theme()` từ `style.py`.
2. **Hình ảnh tự dựng:** Mọi sơ đồ, đồ thị đều vẽ bằng code Manim; **không** chèn ảnh cắt trực tiếp từ bài báo gốc.
3. **Nguồn trên màn hình:** Dùng `make_citation_badge(...)` ngay tại nhịp diễn giải nội dung từ bài báo.
4. **Font tiếng Việt:** Dùng `Text(..., font=FONT_SANS)` cho chữ tiếng Việt và chỉ đặt ký hiệu toán học ASCII/LaTeX trong `MathTex(...)`.

---

## 3. Lệnh render Cảnh 1 (TV1)

Chạy từ thư mục gốc `KAN-project/`:

```powershell
# Preview nhanh (480p, 15fps)
manim -ql Manim/scenes/01_problem_mlp.py ProblemAndMLPScene

# Xuất chất lượng cao 1080p60
manim -qh Manim/scenes/01_problem_mlp.py ProblemAndMLPScene
```
