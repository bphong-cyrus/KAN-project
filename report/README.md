# Template báo cáo ML — KAN (5 thành viên)

Bộ khung được chuyển từ source LaTeX trong `Project-1 - Copy.zip` sang đồ án KAN theo `Project-Plan.md` đính kèm. Giữ nguyên mã khung bìa gốc (viền xanh 4pt và họa tiết góc), logo trường, chữ serif, mục lục, công thức, hình/bảng và mã nguồn; làm lại cấu hình và nội dung cho ML. Không mang tên sinh viên, giảng viên, kết quả hay nội dung Hệ điều hành của mẫu cũ sang báo cáo mới.

**Đây là template để viết, chưa có kết quả thực nghiệm.** Mọi ô `[Điền]` và `[CẦN ĐIỀN: ...]` cần được thay bằng nội dung có kiểm chứng. Các công thức mở đầu và sơ đồ KAN chỉ là gợi ý ban đầu; nhóm phải đọc/đối chiếu nguồn trước khi nộp.

## 1. Bắt đầu trên Overleaf

1. Tạo **New Project → Upload Project**, tải ZIP này lên.
2. Trong **Settings/Menu**, chọn **Compiler: XeLaTeX** và **Main document: main.tex**.
3. Sửa tên nhóm, tên/MSSV của 5 người, lớp, giảng viên và ngày nộp tại `config/metadata.tex`.
4. Mỗi thành viên mở file `sections/tv*.tex` của mình theo bảng bên dưới.
5. Recompile; khi vừa đổi citation/mục lục, cho quá trình biên dịch hoàn tất. Nếu danh mục còn cũ, chọn **Recompile from scratch**.

Không chọn pdfLaTeX. File cần biên dịch luôn là `main.tex`; các file thành viên là nội dung được ghép vào, không phải tài liệu độc lập. Không cần upload source Hệ điều hành cũ.

## 2. Ai sửa file nào?

| Người | File chính — một file mỗi thành viên | Nội dung theo project plan | File phụ liên quan |
|---|---|---|---|
| TV1 | `sections/tv1_introduction_related.tex` | Giới thiệu; ví dụ thực tế; ≥2 hướng trước; phân loại đóng góp; dữ liệu và MLP | `frontmatter/abstract.tex`; `figures/tv1/` |
| TV2 | `sections/tv2_kan_theory.tex` | Ký hiệu; KAN; spline/SiLU; sơ đồ; định lý, giả định và phạm vi | `figures/tv2/` |
| TV3 | `sections/tv3_training_code.tex` | Loss/LBFGS; grid extension; regularization/pruning; độ phức tạp; công thức ↔ code | `code/`; cùng cập nhật phụ lục tái lập |
| TV4 | `sections/tv4_experiments_results.tex` | Thực nghiệm tác giả; setup nhóm; bảng/hình thật; khảo sát G; sai khác | `tables/core_results.tex`; `figures/tv4/` |
| TV5 | `sections/tv5_evaluation_conclusion.tex` | ≥2 điểm mạnh, ≥2 hạn chế; cải tiến; nhận định và kết luận | `appendices/tv5_contributions_ai.tex`; tổng hợp phụ lục |

Cả năm được thêm tài liệu vào `references.bib`, nhưng phải dùng khóa riêng; kiểm tra trùng trước khi ghép. Thông số chung ở `config/` do nhóm thống nhất; hạn chế sửa cùng lúc để tránh xung đột.

### Ánh xạ dàn 10 mục của plan sang 5 chương

| Mục trong plan | Vị trí trong template |
|---|---|
| 1. Giới thiệu; 2. Hướng trước | Chương 1 — TV1 |
| 3. KAN và định lý | Chương 2 — TV2 |
| 4. Huấn luyện/lưới/độ phức tạp; 7. Phân tích code | Chương 3 — TV3 |
| 5. Thực nghiệm tác giả; 6. Thiết lập nhóm; 8. Kết quả/sai khác | Chương 4 — TV4 |
| 9. Đánh giá; 10. Kết luận | Chương 5 — TV5 |
| Phân công, AI, tái lập và bảng đối chiếu yêu cầu | Phụ lục A–B |

Template gom thành 5 chương để mỗi người làm trong một file chính. Tất cả nội dung của 10 mục vẫn có vị trí; nhóm có thể thay tên chương nếu cần. Mục tiêu 22–28 trang **sau khi viết nội dung thật** theo plan, không phải số trang cố định của template. Đây là mục tiêu của kế hoạch; kiểm lại đề gốc trước khi nộp.

## 3. Cấu trúc dùng chung

| Đường dẫn | Vai trò |
|---|---|
| `main.tex` | Điểm vào, thứ tự ghép 5 chương, bibliography và phụ lục |
| `config/metadata.tex` | Thông tin bìa/nhóm ở một nơi |
| `config/preamble.tex` | Font, lề, heading, bảng, toán, code và liên kết |
| `config/commands.tex` | Lệnh TODO, hướng dẫn và khung hình tự thay |
| `frontmatter/cover.tex` | Bìa kế thừa tinh thần source gốc |
| `frontmatter/project_info.tex` | Thông tin đồ án và hướng dẫn chế độ nháp |
| `frontmatter/abstract.tex` | Tóm tắt — TV1 cập nhật cuối cùng |
| `frontmatter/contents.tex` | Mục lục + danh mục hình + danh mục bảng tự sinh |
| `sections/` | Năm file nội dung cho năm người |
| `tables/core_results.tex` | Khung ≥4 điểm KAN và ≥4 MLP, chưa có số liệu |
| `figures/` | Hình thật của từng người; chưa có ảnh thì hiển thị khung chờ |
| `assets/` | Hai logo từ source gốc; bìa đang dùng `hcmus_logo.png` |
| `references.bib` | Nguồn KAN v5 ban đầu; cần thêm các nguồn nhóm thật sự dùng |
| `appendices/` | Đóng góp thực tế, AI, lệnh tái lập, schema CSV, kiểm yêu cầu |
| `docs/Project-Plan.md` | Bản plan đầu vào để đối chiếu |
| `build.sh`, `latexmkrc` | Biên dịch cục bộ với XeLaTeX/BibTeX |

## 4. Viết và ghép nội dung

- Viết trực tiếp vào file của mình, thay các lệnh `\todo{...}`/`\blank` bằng nội dung hoàn chỉnh.
- Giữ nguyên `\chapter{...}` và các `\label` đang được tham chiếu nếu chưa thống nhất đổi.
- Không thêm `\documentclass`, `\begin{document}` hay `\end{document}` trong file thành viên.
- Không sửa font/lề/đánh số riêng trong từng chương. Cần package mới thì bổ sung vào `config/preamble.tex` sau khi thống nhất.
- Dùng nhãn riêng: `sec:tv1-...`, `fig:tv2-...`, `tab:tv4-...`, `eq:tv3-...` để tránh trùng.
- Dẫn chéo bằng `Chương~\ref{ch:theory}`, `Hình~\ref{fig:rmse-params}`, `Bảng~\ref{tab:core-results}`; không gõ tay số chương/trang.
- Citation mẫu: `\cite{liu2025kan}`. Thêm nguồn vào `.bib` bằng khóa khác nhau, không viết tay danh mục tài liệu.
- Dùng `\codepath{Source/src/data.py}` cho đường dẫn có dấu gạch dưới. Trong văn bản thường, viết `\_`, `\%`, `\&`, `\#` cho các ký tự đặc biệt.
- Khi dùng Git: mỗi người sửa file chính và thư mục hình tương ứng; ghép từ sớm rồi chạy build, kiểm PDF cùng nhau.

### Hình ảnh và code

Khung hình tự đổi sang ảnh thật khi file tồn tại, ví dụ:

```latex
\reportfigure{figures/tv4/rmse_params.pdf}
  {Nội dung hướng dẫn khi chưa có ảnh.}
  {Test RMSE theo số tham số. Nguồn: CSV/commit thực tế.}
  {fig:rmse-params}
```

Sau khi có ảnh, sửa caption từ “khung” sang mô tả kết quả thật; thêm nguồn, thang đo, seed và run ID thích hợp. Sơ đồ KAN `[2,1,1]` trong TV2 là TikZ có thể sửa trực tiếp; không phải ảnh bài báo.

TV3 có chỗ bật `\lstinputlisting` để đưa đoạn code Python thật từ `code/` vào báo cáo. Cập nhật file/dòng/commit, không trích mã giả như thể mã đã chạy. Liên kết và trích dẫn mã tác giả theo nguồn thực tế.

### Nháp và bản nộp

`\drafttrue` ở `main.tex` hiển thị hộp xanh (người viết, người kiểm, yêu cầu). Đổi thành `\draftfalse` sẽ ẩn hộp xanh và hướng dẫn nội bộ ở trang thông tin. **Nó không xóa TODO, ô trống, khung hình, caption nháp hay bảng chưa điền.** Nhóm phải thay/xóa các phần đó bằng tay trước khi xuất bản nộp.

## 5. Build trên máy

Cần TeX Live/MiKTeX có XeLaTeX, BibTeX, các package chuẩn trong preamble. Font ưu tiên TeX Gyre Pagella/Heros; nếu không có, dùng Noto Serif/Sans. Cần DejaVu Sans Mono cho code. Overleaf thường có đủ các package và font ưu tiên.

Linux/macOS/Git Bash:

```bash
bash build.sh
```

Windows PowerShell hoặc chạy tay từ thư mục có `main.tex`:

```text
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

`build.sh` tạo `main.pdf` và sao chép thành `Report.pdf`. Với lệnh chạy tay, đổi tên/copy `main.pdf` thành `Report.pdf` sau khi kiểm. File `latexmkrc` dùng XeLaTeX; tài liệu tham khảo dùng BibTeX `unsrt`, không cần Biber.

## 6. Checklist trước khi nộp

- [ ] Metadata là của nhóm ML hiện tại, không còn thông tin mẫu cũ.
- [ ] Không còn `[Điền]`, `[CẦN ĐIỀN`, khung ảnh thiếu, caption nháp hoặc mục bổ sung chưa chạy.
- [ ] Đổi `\draftfalse`, biên dịch và đọc toàn bộ PDF; không còn citation/ref `?`, lỗi font, bảng tràn hoặc caption sai.
- [ ] Tóm tắt và kết luận lấy số liệu từ bảng/hình đã khóa.
- [ ] Phân biệt số tác giả, số nhóm, giá trị đọc ước lượng từ hình và giả thuyết chưa thử.
- [ ] Cùng dữ liệu/split/độ đo, ≥4 điểm KAN + ≥4 MLP nếu tuyên bố đủ panel; ghi phạm vi thật nếu chưa đạt.
- [ ] Phân biệt lịch grid extension nối tiếp với mô hình fixed-G khởi tạo lại.
- [ ] Chỉ dùng error bar/TB/SD tại cấu hình đã chạy lặp seed.
- [ ] Mỗi bảng/hình truy về CSV/config/log/run ID và commit thật; PDF/video cùng phiên bản số liệu.
- [ ] Có hai điểm mạnh, hai hạn chế và ít nhất một cải tiến gắn bằng chứng/phép thử.
- [ ] Nguồn lý thuyết, code, hình và thư viện được dẫn đầy đủ; không chỉ có một nguồn KAN như bộ khung ban đầu.
- [ ] Phụ lục đóng góp ghi tỷ lệ thực tế có xác nhận; khai báo AI theo đúng sử dụng.
- [ ] Kiểm yêu cầu gốc về Source, Manim, video 5–10 phút, phụ đề, URL, hashtag và cách đặt tên gói nộp.

## 7. Ghi chú nguồn và thiết kế

Source gốc dùng `report/extreport`, bìa khung xanh/logo HCMUS và chia chapter. Template này chuẩn hóa tên file, bỏ package trùng, thay nội dung Hệ điều hành, nới lề để đọc báo cáo dài, dùng Unicode/XeLaTeX và BibTeX thay backend Biber của source cũ. Các danh mục được sinh tự động; không sửa tay file `.toc/.lof/.lot`.

Metadata tham khảo KAN được đối chiếu trang chính thức: https://arxiv.org/abs/2404.19756v5. Nhóm vẫn cần đọc toàn văn v5 để hoàn thiện định lý, năm panel của Hình 3.1 và cấu hình tác giả; bộ khung không tự xác nhận các kết quả đó.
