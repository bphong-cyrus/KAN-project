# Ghi chú Cảnh 2: KAN và spline

- Phụ trách: TV2; class `KANSplineScene`, file `02_kan_spline.py`.
- Thời lượng thiết kế: **150 giây**, khung ngang 16:9, dùng theme chung `style.py`.
- Video render là hình ảnh không có lời đọc. TV4–TV5 ghép audio và hiệu chỉnh phụ đề theo lời đọc cuối.
- Phụ đề nháp: `tv2_scene02.srt`. Timecode dưới đây tính từ đầu Cảnh 2; khi ghép sau Cảnh 1 dài 110 giây, dịch phụ đề thêm 110 giây hoặc theo timeline thực tế.
- Đã render và kiểm tra bản 1080p60 bằng Manim 0.19.1: **1920×1080, 60 fps, 150.00 giây**, xác nhận qua metadata PyAV và các khung hình ở cả năm đoạn.

| Timecode | Hình ảnh | Lời thoại gợi ý | Nguồn trên màn hình |
|---|---|---|---|
| 00:00–00:35 | KAN `[2,1,1]`, ba hàm cạnh, tín hiệu đi vào nút cộng và ra đầu ra | Một cạnh KAN biến đổi đầu vào bằng một hàm một biến học được. Hai hàm ở lớp đầu nhận hai đầu vào riêng, rồi nút giữa cộng chúng thành h. Hàm trên cạnh tiếp theo biến đổi h thành dự đoán. Kiến trúc này có ba hàm cạnh; trong trường hợp minh họa, các nút thực hiện phép cộng. | Liu et al. (2025), Mục 2.2 |
| 00:35–01:05 | Ba đồ thị: nhánh cơ sở SiLU, spline có trọng số, tổng hai thành phần | Hàm trên cạnh gồm một nhánh cơ sở SiLU và một nhánh spline. Các hệ số điều chỉnh đóng góp của từng nhánh. Đường bên phải là tổng thực của hai đường bên trái trong ví dụ, không phải một checkpoint KAN đã huấn luyện. | (2.10)–(2.12) |
| 01:05–01:40 | Cơ sở B-spline bậc ba, chọn B5, thay c5 và quan sát tổng | Spline được tạo từ tổng có trọng số của các hàm cơ sở. G là số khoảng lưới, k là bậc spline. Mỗi hàm cơ sở có miền hỗ trợ cục bộ. Khi thay một hệ số, đường tổng chỉ thay đổi trong miền hỗ trợ tương ứng; các phần bên ngoài giữ nguyên. Đường vàng là B5; đường xanh đậm là tổng các cơ sở với hệ số minh họa. | Mục 2.2 |
| 01:40–02:15 | Hàm mục tiêu nét đứt, spline xanh, điểm lưới vàng; G=3→6→12 | Khi tăng lưới, ta chuyển spline cũ sang biểu diễn mới trước khi tiếp tục tối ưu. Ví dụ dùng lưới lồng nhau: chèn knot giữ nguyên đường spline. Sau đó fit lại spline với hàm mục tiêu để minh họa khả năng biểu diễn chi tiết hơn. Trong KAN, việc tối ưu tiếp là quá trình huấn luyện cả mạng. | Mục 2.4, (2.16) |
| 02:15–02:30 | Ba ý chính và câu hỏi dẫn sang Cảnh 3 | KAN học hàm trên cạnh, cộng tín hiệu tại nút, và điều chỉnh độ linh hoạt bằng lưới. Nhưng độ linh hoạt cao hơn không tự bảo đảm test RMSE tốt hơn. Phần kết quả tiếp theo sẽ đối chiếu độ chính xác với số tham số. | Liu et al., ICLR 2025, arXiv:2404.19756v5 |

## Phân biệt minh họa và thực nghiệm

- Cảnh này không import `build_kan`, không dùng checkpoint, không sinh số liệu panel.
- Đồ thị hàm cạnh dùng các hệ số cố định để giải thích công thức.
- Đoạn cơ sở dùng `G=8`, `k=3`, có `G+k=11` hàm cơ sở và `G+1=9` điểm lưới cơ bản.
- Đoạn grid dùng **3→6→12** để minh họa lưới lồng nhau bằng chèn knot chính xác. Đây không phải lịch thực nghiệm **3→5→10→20→50** của nhóm.
- Bước fit lại trong ví dụ spline 1D giải least squares trên các mẫu hàm xác định, không phải log LBFGS hoặc kết quả huấn luyện KAN. Không dùng ví dụ này để kết luận về RMSE hay thời gian của KAN.
- Với lịch lưới tổng quát, bài báo chuyển spline bằng cách xấp xỉ spline cũ trên lưới mới; không khẳng định mọi lần refine đều giữ nguyên hàm chính xác.
- Hình tự dựng bằng Manim và SciPy, không cắt hình từ bài báo.

## Render

Chạy từ thư mục gốc dự án, sau khi cài bộ `Manim/requirements.txt` chung:

```powershell
.\.venv\Scripts\python.exe -m manim -ql Manim/scenes/02_kan_spline.py KANSplineScene --media_dir Manim/media
.\.venv\Scripts\python.exe -m manim -qh Manim/scenes/02_kan_spline.py KANSplineScene --media_dir Manim/media
```

Đầu ra cuối: `Manim/media/videos/02_kan_spline/1080p60/KANSplineScene.mp4`.
Scene dùng `MathTex`, cần `latex` và `dvisvgm` trên PATH. Giữ `--media_dir` tương đối theo quy ước chung.

## Kiểm tra trước khi ghép

- Xem video ở tốc độ thường: chữ, công thức và nguồn không chồng/cắt nhau.
- Kiểm tra thời lượng file xuất; nếu chỉnh hoạt hình làm vượt thời lượng đoạn, code báo lỗi để cập nhật timecode.
- Đối chiếu màu đường SiLU/spline/tổng và ý nghĩa G, k với báo cáo.
- TV4–TV5 sửa lời thoại, chỉnh SRT theo audio và ghép vào timeline chung; SRT nháp chưa được nhúng vào video.
