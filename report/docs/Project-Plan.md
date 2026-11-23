# Kế hoạch đồ án KAN: Kolmogorov–Arnold Networks

**Nhóm:** 5 thành viên  
**Thời gian:** 14 ngày, tính từ ngày nhóm bắt đầu  
**Định hướng:** Cả 5 người viết báo cáo; TV1–TV3 chạy thực nghiệm; TV1, TV2, TV4 làm Manim; TV4–TV5 sản xuất/ghép video.

## 1. Căn cứ và mục tiêu


Tỷ trọng đánh giá trong đề: **báo cáo 40%, video 40%, mã nguồn 20%**; mở rộng có ý nghĩa có thể được thưởng **tối đa 10%**. 

### Mục tiêu đầu ra

- Giải thích được KAN bằng ngôn ngữ của nhóm, đủ để người đọc hiểu mà không cần mở bài báo gốc.
- Tái lập **một panel so sánh chính** của Mục 3.1, Hình 3.1: nhiều điểm KAN và MLP trên cùng trục test RMSE – số tham số. Chỉ mở thêm hàm khi panel này đã đạt tiêu chí nghiệm thu.
- Phân tích định lượng sai khác với bài báo và lý giải các nguyên nhân có căn cứ.
- Có phân tích bổ sung về kích thước lưới spline.
- Hoàn thành video Manim 5–10 phút, mã nguồn chạy được và báo cáo đầy đủ tiêu chí; ghi rõ đóng góp theo đúng phần được phân công, không ép mọi người tham gia mọi sản phẩm.


## 2. Phạm vi thực hiện

| Hạng mục | Phạm vi đề xuất |
|---|---|
| Lý thuyết | Kiến trúc KAN, spline, huấn luyện, grid extension, regularization, pruning và giả định của kết quả lý thuyết |
| Tái lập cốt lõi bắt buộc | Một panel hoàn chỉnh cho hàm chính nêu ở Mục 2.1: quét ít nhất 4 mức G của KAN và ít nhất 4 cấu hình MLP với nhiều mức tham số; cùng dữ liệu và phép đo. Vẽ RMSE – số tham số, đối chiếu đúng panel tương ứng ở Hình 3.1 |
| Bổ sung trong phạm vi cốt lõi | Chạy hàm Bessel một biến và hàm nhân hai biến ở Mục 2.1 nếu khả thi; ghi là **panel bổ sung một phần** nếu chưa quét đủ đường so sánh |
| Phân tích bổ sung bắt buộc | Với hàm chính, khảo sát G và train/test RMSE, số tham số, thời gian; phân biệt lịch **mở rộng lưới liên tiếp** với các mô hình **khởi tạo lại ở từng lưới** |
| Mở rộng nếu đủ thời gian | Ablation có/không grid extension hoặc dữ liệu có nhiễu, dựa vào hạn chế đã nêu; chỉ mở sau khi ba sản phẩm chính có bản nháp |
| Triển khai | Chốt một đường chính: chạy mã chính thức của tác giả ở phiên bản cố định; tự cài đặt lớp KAN nhỏ để đối chiếu chỉ khi đủ thời gian, không thay thế đường chính giữa chừng |
| Video | Khoảng 7–8 phút, bằng Manim; có thuyết minh, phụ đề xuyên suốt và nguồn |
| Báo cáo | Khoảng 22–28 trang nội dung chính, chưa tính tài liệu tham khảo và phụ lục |

### 2.1. Thứ tự ưu tiên của ba hàm

#### Hàm chính — so sánh KAN–MLP

**Hàm:** `f(x, y) = exp(sin(πx) + y²)`.

Kiến trúc KAN bài báo nêu: `[2,1,1]`. Đây là panel cần nhiều mức số tham số để xét xu hướng RMSE; dùng thống nhất trong báo cáo và video.

#### Hàm bổ sung 1 — hàm một biến

**Hàm:** `f(x) = J₀(20x)`.

Kiến trúc KAN bài báo nêu: `[1,1]`. Ký hiệu J₀ là hàm Bessel loại một bậc 0.

#### Hàm bổ sung 2 — tương tác hai đầu vào

**Hàm:** `f(x, y) = xy`.

Kiến trúc KAN bài báo nêu: `[2,2,1]`. Hàm này kiểm tra việc biểu diễn tương tác giữa hai đầu vào.

Phạm vi chính tương ứng **một trong năm panel** của Hình 3.1, không phải toàn bộ hình hay toàn bộ bài báo. Nếu không đủ thời gian cho hai hàm bổ sung, vẫn giữ panel chính có đường so sánh và phân tích bổ sung. Hàm một biến có thể là bài chạy thử để kiểm tra pipeline, nhưng không thay panel chính. Không mở thêm PDE hay bài toán vật lý trong hai tuần.

Nhóm chọn mức **tái lập cốt lõi** theo Mục 1.2.1 của đề vì Hình 3.1 là phép so sánh KAN–MLP trực tiếp cho nhận định chính về độ chính xác theo số tham số. Phần còn lại của bài báo được **tóm tắt trong lý thuyết**; thực nghiệm khác được thay bằng khảo sát $G$ do nhóm thiết kế. Báo cáo cần ghi rõ lý do chọn, phần đã tái lập và phần không tái lập.

### 2.2. Nguyên tắc thực hiện

- Chốt giao diện Source, số liệu, báo cáo và cảnh Manim trước khi làm riêng; từng nhóm tự ghép phần mình từ sớm rồi mới mở rộng thí nghiệm.
- Không hứa trước rằng KAN sẽ thắng MLP; kết luận dựa trên số liệu thực tế.
- Không thay số liệu bài báo bằng số ước đoán không được ghi nhãn.
- Mọi thay đổi dữ liệu, kiến trúc, optimizer, số bước, grid hoặc thư viện đều phải ghi lại.
- Phân biệt **tái lập kết quả tác giả** (cùng bài toán và so sánh) với **bản thu nhỏ** (khác ngân sách/dữ liệu); không tuyên bố tái lập đường cong định lượng nếu chỉ có một điểm mỗi mô hình.
- Khóa giao diện ngày 2; các nhóm ghép thử Source, báo cáo, Manim và video ngày 5; dùng kết quả thật ngày 8–9. Khóa số liệu ngày 9; từ ngày 10 chỉ sửa lỗi ảnh hưởng kết luận.

## 3. Phân công

Thay TV1–TV5 bằng tên/MSSV thật. **Báo cáo: TV1–TV5 (5 người). Thực nghiệm: TV1–TV3 (3 người). Manim: TV1, TV2, TV4 (3 người). Video: TV4–TV5 (2 người).** Nhóm nào làm sản phẩm nào thì cùng ghép và kiểm chính sản phẩm đó.

| Người | Báo cáo — cả 5 | Thực nghiệm — TV1–TV3 | Manim — TV1/TV2/TV4 | Video — TV4–TV5 |
|---|---|---|---|---|
| **TV1 — dữ liệu, MLP** | Giới thiệu, ví dụ thực tế, ≥2 phương pháp trước, mô tả dữ liệu và baseline; cùng rà PDF | `Source/src/data.py`, `mlp_baseline.py`: split/seed, MLP nhiều mức tham số, log; cùng TV2–TV3 ghép Source | `Manim/scenes/01_problem_mlp.py`: bài toán, dữ liệu, MLP; cùng TV2/TV4 ghép ba cảnh | - |
| **TV2 — KAN** | Kiến trúc KAN, spline/SiLU, ký hiệu, định lý, sơ đồ; kiểm công thức báo cáo | `Source/src/kan_model.py`, `scripts/check_kan.py`: adapter mã tác giả, forward, tham số; cùng TV1/TV3 ghép Source | `Manim/scenes/02_kan_spline.py`: hàm trên cạnh, nút cộng, spline và trực giác grid; cùng TV1/TV4 ghép ba cảnh | - |
| **TV3 — huấn luyện/lưới** | LBFGS, grid extension, độ phức tạp, liên hệ môn học, công thức ↔ code | `Source/src/train_grid.py`, `metrics.py`, `scripts/run_core.py`, `analysis/evaluate.py`: lịch G, train, RMSE, CSV/bảng/hình; cùng TV1/TV2 ghép/chạy Source | cung cấp log, mô tả grid | - |
| **TV4 — kết quả** | Thiết lập và kết quả nhóm, sai khác bài báo, khảo sát G; kiểm CSV/hình. | Chạy lệnh phân tích do TV3 viết để kiểm hình/JSON | `Manim/scenes/03_results.py`: đồ thị từ CSV thật, hạn chế và kết luận trực quan. | Cùng TV5 chốt storyboard, viết lời thoại và phụ đề theo timecode, kiểm số liệu trên video |
| **TV5 — đánh giá/hậu kỳ** | Hai điểm mạnh/hạn chế, cải tiến, kết luận, đóng góp/AI/nguồn; cùng rà PDF | — | — | Cùng TV4 chốt kịch bản; thu/chỉnh audio, hậu kỳ, cân âm, chèn/soát phụ đề, xuất 1080p, đăng video, kiểm URL/hashtag |

### 3.1. Giao diện và tự ghép trong từng nhóm

1. **TV1–TV3 (thực nghiệm):** thống nhất `experiment_spec.md` (hàm, miền, split/seed, KAN/MLP, lịch G, optimizer, RMSE) và `results_schema.md`; tự ghép `data → MLP/KAN → train → metrics → CSV`, chạy thử rồi bàn giao `Source/`, README, config, log và `results/runs.csv`.
2. **CSV:** một dòng cho mỗi cấu hình/checkpoint/seed, gồm `run_id`, `experiment_id`, `task`, `model`, `architecture`, `G`, `k`, `seed`, `split_id`, `params`, `train_rmse`, `test_rmse`, `time_s`, `optimizer`, `steps`, `config_path`, `code_commit`. `run_id` duy nhất; MLP để trống `G,k`. TV3 viết lệnh tạo bảng/hình; TV4 chạy và kiểm đầu ra, tạo `results/summary.json` cho cảnh kết quả.
3. **TV1–TV5 (báo cáo):** chốt ký hiệu/citation/template, mỗi người viết mục riêng dưới `report/sections/`; cả năm ghép và rà PDF. TV1 chuẩn bị lệnh build, TV2 soát lý thuyết, TV3 soát code/setup, TV4 soát số liệu, TV5 soát kết luận/phụ lục. Ai có lỗi tự sửa mục của mình.
4. **TV1, TV2, TV4 (Manim):** chốt font, tỉ lệ, màu và nguồn trên màn hình. Mỗi người render cảnh riêng; cả ba ghép/render bộ cảnh. TV3 kiểm kỹ thuật grid, TV4 kiểm đồ thị và số liệu.
5. **TV4–TV5 (video):** TV4 soạn storyboard/lời/subtitle; TV5 thu/chỉnh tiếng và hậu kỳ. Hai người cùng xem/nghe bản nháp và xuất cuối. TV1–TV3 góp lỗi học thuật, không gánh sản xuất video.

### 3.2. Bàn giao

| Nhóm tự ghép | Đầu ra phải hoàn chỉnh | Nhóm nhận kiểm |
|---|---|---|
| **TV1–TV3: thực nghiệm** | Mã chạy từ đầu đến CSV, ≥4 điểm KAN + ≥4 MLP của hàm chính, seed/config/log/README, lệnh dựng bảng/hình | TV4 kiểm RMSE/split/run ID; TV5 lấy bằng chứng cho nhận định |
| **TV1–TV5: báo cáo** | `Report.pdf` có năm phần, hình/bảng, nguồn, phân công/AI | Cả năm đọc bản cuối và xác nhận |
| **TV1/TV2/TV4: Manim** | Ba scene có lệnh render chung, preview, nguồn đúng lúc; cảnh kết quả dùng JSON thật | TV4–TV5 thử đưa vào timeline; báo lỗi cho người viết cảnh |
| **TV4–TV5: video** | Storyboard, audio, phụ đề xuyên suốt, video 5–10 phút, URL công khai | TV1–TV3 xem bản nháp, rà khoa học |

Nếu CSV đổi, TV4 tạo lại hình/JSON; nhóm báo cáo xuất lại PDF, nhóm Manim render lại cảnh 3 và nhóm video xuất lại video. Mỗi nhóm giữ cùng commit số liệu. Tỷ lệ hoàn thành cá nhân ghi theo công việc thực tế, không tự điền 100% vì bảng dự kiến nhìn cân.

## 4. Lịch 14 ngày theo phân công

| Ngày | Báo cáo — cả 5 | Thực nghiệm — TV1–TV3 | Manim — TV1/TV2/TV4 | Video — TV4–TV5 | Mốc chung |
|---|---|---|---|---|---|
| **1** | Chia năm mục, dàn bài/nguồn | Chốt dữ liệu, MLP, KAN, grid | Chia ba cảnh | Chốt mạch video 7–8 phút | Thống nhất đầu ra, ngày bàn giao |
| **2** | Tạo file mục viết/template | Khóa đặc tả/API/CSV; smoke test | Khóa font, tỉ lệ, input cảnh 3 | Outline lời/phụ đề | Khóa giao diện giữa nhóm |
| **3** | Mỗi người viết bản đầu | TV1 data/MLP, TV2 KAN, TV3 train/metric; đo thời gian run | TV1 cảnh mở, TV2 KAN, TV4 cảnh kết quả demo; đo preview | TV4 lời thoại nháp, TV5 timeline/audio thử | Kiểm lại ước lượng tải theo run/preview thật |
| **4** | Hoàn thành phần nền, rà nguồn | Cả ba ghép Source, chạy một cấu hình mỗi mô hình | Ba preview độc lập | Thu thử lời, dựng timeline | Pipeline đầu-cuối chạy được |
| **5** | Cả năm ghép PDF nháp | Cả ba chạy thử ma trận, sửa API/log | Cả ba ghép/render cảnh nháp | Hai người ghép video nháp | Ghép thử bốn đầu việc |
| **6** | Viết setup/kết quả sơ bộ | Hoàn thành ≥4 KAN + ≥4 MLP; seed trọng tâm | TV1/TV2 chốt cảnh ý tưởng | TV4 chốt lời, TV5 thu/chỉnh audio | Có CSV và hình đầu |
| **7** | PDF v0.5, kiểm mục mình | Bàn giao CSV/log/config/README bản chính | Cảnh 1–2 xong, cảnh 3 dữ liệu đầu | Video thô ≥ nửa thời lượng | Panel chính nghiệm thu |
| **8** | Viết sai khác, hạn chế | Chạy G/hàm phụ nếu còn lực | TV4 cập nhật cảnh 3 từ JSON thật, cả ba render lại | Ghép audio/phụ đề từ cảnh thật | PDF/video có kết quả thật |
| **9** | Khóa nội dung số liệu, rà PDF | Khóa CSV/config/commit | Khóa ba cảnh theo JSON cuối | Khóa audio/subtitle | Cùng phiên bản số liệu |
| **10** | Năm người xuất Report.pdf v1 | Ba người đóng gói Source v1 | Ba người render Manim v1 | Hai người xuất video v1 | Ba sản phẩm đầy đủ lần đầu |
| **11** | Rà rubric, sửa mục mình | Chạy lại một run từ môi trường sạch | Render cả ba cảnh từ source | Nghe/xem, sửa chữ/nhịp | Lỗi trả đúng người/nhóm |
| **12** | Năm người chốt PDF, phân công/AI | Khóa mã/README/CSV | Khóa Manim/assets | Video 1080p cuối | PDF/video/CSV đồng bộ |
| **13** | Cả nhóm xác nhận đóng góp/nguồn | Kiểm lệnh Source | Kiểm lệnh Manim | Đăng video, kiểm URL/hashtag | Đóng gói ZIP/RAR tên MSSV |
| **14** | Ôn vấn đáp và kiểm bản nộp | Giải thích code/kết quả | Giải thích cảnh | Kiểm video công khai | Nộp trước hạn Moodle |

## 5. Khung báo cáo chi tiết — 5 người cùng viết

| Phần trong PDF | Nội dung bắt buộc | Người viết chính | Người kiểm tra |
|---|---|---|---|
| Tóm tắt, 1. Giới thiệu, 2. Hướng trước | Bài toán, ví dụ tự đặt, ít nhất hai cách trước, đóng góp phân loại; dữ liệu và MLP đối chứng | **TV1**; tóm tắt cập nhật sau kết quả | TV2, TV5 |
| 3. KAN và định lý | Ký hiệu, sơ đồ tự vẽ, lớp KAN, spline/SiLU, định lý và giả định | **TV2** | TV1, TV3 |
| 4. Huấn luyện/lưới/độ phức tạp; 7. Phân tích code | LBFGS, grid extension, gradient, số tham số/thời gian/bộ nhớ; công thức ↔ code, tác động sửa/bỏ | **TV3**, TV2 hỗ trợ phần lý thuyết | TV2, TV5 |
| 5. Thực nghiệm tác giả; 6. Thiết lập nhóm; 8. Kết quả và sai khác | Năm hàm tác giả, hàm nhóm chọn, dữ liệu/split/seed, bảng KAN–MLP, đối chiếu, khảo sát G | **TV4**, TV1–TV3 xác nhận setup | TV3, TV5 |
| 9. Đánh giá; 10. Kết luận; phụ lục | Hai điểm mạnh/hạn chế, cải tiến gắn hạn chế, nhận định, phân công và tỷ lệ, AI, nguồn | **TV5**, TV4 cung cấp chứng cứ | TV1–TV4 |

TV1 cung cấp mẫu và lệnh xuất PDF. **Cả năm người cùng ghép, đọc và sửa báo cáo**; TV1 hỗ trợ thống nhất văn phong, TV2 kiểm lý thuyết, TV3 kiểm code/thiết lập, TV4 kiểm số liệu, TV5 kiểm nhận định/phụ lục. Phụ lục ghi đóng góp đúng nhóm việc, không yêu cầu năm người đều có code/Manim.

### 5.1. Giải thích chính xác cơ chế KAN

Công thức lớp KAN:

$$
x^{(\ell+1)}_j=\sum_{i=1}^{n_\ell}\phi_{\ell,j,i}\left(x^{(\ell)}_i\right).
$$

Trong đó:

- $\ell$: chỉ số lớp; $n_\ell$: số nút đầu vào của lớp.
- $x_i^{(\ell)}$: tín hiệu tại nút i của lớp hiện tại.
- $\phi_{\ell,j,i}$: hàm một biến học được trên cạnh nối i đến j.
- Nút tiếp theo cộng các tín hiệu đã biến đổi từ các cạnh đi vào.

Tham số hóa hàm trên cạnh theo bài báo:

$$
\phi(x)=w_b\,\mathrm{SiLU}(x)+w_s\sum_r c_rB_r(x).
$$

Giải thích $w_b,w_s$ là hệ số điều chỉnh, $c_r$ là hệ số spline học được và $B_r$ là các hàm cơ sở B-spline. Nêu vai trò của nhánh cơ sở, lưới và bậc spline. Không viết rằng KAN “không có trọng số học được”; nó vẫn có tham số học được trong các hàm trên cạnh.

### 5.2. Hai phương pháp trước đó

1. **MLP:** học phép biến đổi tuyến tính và dùng activation cố định; nêu hạn chế liên quan đến biểu diễn/khả năng diễn giải trong bối cảnh bài báo.
2. **Xấp xỉ spline truyền thống:** linh hoạt với hàm một biến; mở rộng trực tiếp bằng lưới tensor sang nhiều chiều có thể tốn tài nguyên.

Giải thích KAN kết hợp cấu trúc hợp thành của mạng với biểu diễn hàm một biến bằng spline. Mỗi nhận định cần nguồn thích hợp; không chỉ mô tả theo cảm tính.

### 5.3. Lý thuyết và phạm vi áp dụng

Định lý 2.1 ở Mục 2.3 giả định $f$ **có một biểu diễn KAN** mà từng hàm trên cạnh có đạo hàm liên tục đến cấp $k+1$. Khi ấy tồn tại xấp xỉ spline bậc $k$ trên lưới cỡ $G$ với sai số chuẩn $C^m$ bị chặn bởi $C G^{-(k+1-m)}$ cho $0\le m\le k$; $C$ phụ thuộc vào hàm và biểu diễn. Với $m=0$, đây là một **chặn xấp xỉ**, không phải cam kết lỗi test của quá trình học từ mẫu hữu hạn sẽ giảm đúng $G^{-4}$. Cần phân biệt ba vấn đề:

- Tồn tại một biểu diễn tốt.
- Thuật toán huấn luyện có tìm được biểu diễn đó hay không.
- Mô hình có tổng quát hóa từ hữu hạn dữ liệu hay không.

Không suy diễn rằng định lý biểu diễn Kolmogorov–Arnold gốc tự động cho các hàm cạnh trơn: biểu diễn gốc có thể không trơn; định lý xấp xỉ cần giả định bổ sung. Không suy diễn KAN luôn thắng MLP hoặc đã loại bỏ mọi ảnh hưởng của số chiều: hằng số $C$ có thể phụ thuộc biểu diễn/số chiều. Bài báo ở Hình 2.3 còn báo cáo test RMSE theo $G$ gần $G^{-3}$ trong một ví dụ, khác mức $G^{-4}$ của chặn lý tưởng và thảo luận về hiệu ứng biên.

### 5.4. Độ phức tạp

Với L lớp, các lớp cùng độ rộng N, spline bậc k và G khoảng lưới, bài báo nêu số tham số khoảng:

$$
O(LN^2(G+k)),
$$

so với khoảng $O(LN^2)$ của MLP cùng độ sâu và độ rộng.

Đây là ước lượng **số tham số**, không tự động là thời gian chạy. Phân tích thời gian và bộ nhớ cần nêu batch size, cách đánh giá spline, dữ liệu trung gian được lưu và các giả định cài đặt. Bổ sung đo thực tế; ít tham số hơn không đồng nghĩa luôn chạy nhanh hơn.

Với batch $B$, $E=\sum_{\ell} n_{\ell}n_{\ell+1}$ cạnh và một lần forward, nếu cài đặt đánh giá **mọi** hàm cơ sở trên từng cạnh thì thời gian và bộ nhớ trung gian có thể ở cỡ $O(BE(G+k))$; nếu tận dụng giá đỡ cục bộ của B-spline bậc $k$ thì số cơ sở khác 0 tại một điểm chỉ cỡ $k+1$, nhưng chi phí định vị khoảng và chi phí tensor vẫn phụ thuộc cài đặt. Backward và LBFGS (nhiều lần đánh giá closure, lưu lịch sử) thêm chi phí. Báo cáo phải ghi rõ **đây là ước lượng theo giả định**, đo thời gian/bộ nhớ thực trên code đang chạy và so với MLP cùng điều kiện.

### 5.5. Ánh xạ công thức với mã nguồn

TV2 và TV3 cùng lập bảng dưới đây, điền tên file/hàm theo phiên bản mã nguồn thực sự sử dụng, không đoán tên API.

| Thành phần | Công thức/mục bài báo | File/hàm thực tế | Chi tiết cài đặt | Nếu sửa hoặc bỏ |
|---|---|---|---|---|
| Forward lớp KAN | Mục 2.2 | Điền sau kiểm tra | Chiều tensor, phép cộng đầu ra | Phân tích tác động |
| B-spline và nhánh cơ sở | Công thức 2.10–2.12 | Điền sau kiểm tra | Hệ số, khởi tạo, bậc spline | Phân tích tác động |
| Cập nhật/mở rộng lưới | Mục 2.2 và 2.4 | Điền sau kiểm tra | Lịch cập nhật, chuyển hệ số | Phân tích tác động |
| Regularization/pruning | Mục 2.5 | Điền sau kiểm tra | Penalty, ngưỡng, phạm vi có sử dụng | Phân tích tác động |
| Loss và optimizer | Thiết lập thực nghiệm | Điền sau kiểm tra | MSE/RMSE, LBFGS, closure | Phân tích tác động |

### 5.6. Quy chuẩn viết và kiểm tra báo cáo

- Mỗi kết luận thực nghiệm dẫn đến bảng/hình hoặc log tương ứng.
- Tách rõ kết quả tác giả, kết quả nhóm và giả thuyết giải thích.
- Mỗi hình có số thứ tự, chú thích, trục/đơn vị và nguồn; sơ đồ quy trình do nhóm tự vẽ.
- Dùng nhất quán ký hiệu G, k, L, N, seed và RMSE trong báo cáo lẫn video.
- Không chỉ liệt kê công thức: giải thích thành phần làm gì, vì sao cần và hoạt động ra sao.
- Không bịa số liệu, không điền sẵn kết luận hoặc tỷ lệ hoàn thành trước khi có bằng chứng.
- Một ví dụ ứng dụng thực tế do nhóm tự đặt phải nêu **đầu vào, đầu ra, vì sao cần dự đoán**; không nhầm ví dụ này với dữ liệu thực nghiệm tổng hợp của bài báo.
- Đóng góp phải được **phân loại**: lý thuyết (điều kiện/chặn xấp xỉ), kỹ thuật (kiến trúc và thủ thuật grid extension, đơn giản hóa) và thực nghiệm (so sánh độ chính xác, số tham số và khả năng diễn giải). Diễn đạt đúng phạm vi tác giả thực sự chứng minh.
- Hai điểm mạnh và hai hạn chế trong báo cáo phải dẫn đến **bằng chứng hoặc phép thử cụ thể**; nếu phép thử chưa thực hiện, ghi là giả thuyết, không biến thành kết luận.

## 6. Kế hoạch thực nghiệm

### 6.1. Cấu hình khởi đầu đề xuất

| Thành phần | Cách thực hiện |
|---|---|
| Dữ liệu | Hàm chính sinh dữ liệu theo mã/thiết lập tác giả nếu xác định được. Ghi miền, phương pháp lấy mẫu, số điểm, train/validation/test và seed; nếu bài báo không nêu rõ thì ghi **không xác định từ bài báo** rồi công bố quy tắc nhóm chọn, không tự nhận là thiết lập gốc |
| KAN | Kiến trúc `[2,1,1]` cho panel chính, spline bậc k = 3; ít nhất 4 mức G khả thi trong lịch tăng lưới, ưu tiên `{3,5,10,20}` rồi `{50}` nếu đủ tài nguyên. G là số khoảng lưới; số điểm lưới là G + 1 |
| MLP | Ít nhất 4 cấu hình độ sâu/độ rộng, cho số tham số trải quanh các điểm KAN; khóa activation, depth, width, init, optimizer và quy tắc chọn **trước khi xem test**; báo cáo số tham số thật |
| Optimizer/ngân sách | Dùng LBFGS cho cả hai như Mục 3.1 nếu được; ghi iterations, function evaluations, line search, batch và thời gian. Nếu giảm bước hoặc thay optimizer, so sánh thêm theo thời gian và đánh dấu khác tác giả |
| Lặp lại | Mục tiêu **3 seed** trên ít nhất một cấu hình KAN và một MLP gần mức tham số tương ứng; mọi điểm khác có ít nhất một run. Nếu ngân sách không đủ, ghi số lần chạy thực tế và giới hạn độ ổn định; không vẽ thanh sai số từ một run |
| Chọn cấu hình | Dùng validation để chọn cấu hình hoặc điểm dừng; giữ test cho đánh giá cuối. Nếu nhóm dùng test để vẽ đường scaling giống bài báo, ghi rõ đó là **đường đánh giá**, không dùng test để chọn một cấu hình rồi công bố cùng điểm đó như ước lượng không thiên lệch |
| Độ đo | RMSE trên cùng bộ test (công thức ngay dưới bảng); train RMSE, số tham số học được, thời gian trên một máy. Ghi cách đo thời gian và đồng bộ thiết bị khi dùng GPU |
| Theo dõi | `run_id`, commit, seed, function, split, model, architecture, G, k, optimizer, steps/evals, train/test RMSE, parameter count, wall time, hardware, phiên bản thư viện; cấu hình + CSV + log đầy đủ |

Định nghĩa RMSE trên tập gồm $n$ điểm:

$$
\operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2}.
$$

Bài báo Mục 3.1 dùng $G\in\{3,5,10,20,50,100,200,500,1000\}$, tăng lưới sau mỗi 200 bước và tổng 1.800 bước LBFGS. **Lịch tăng lưới là một quá trình huấn luyện nối tiếp**: giá trị tại $G=10$ tiếp tục từ mô hình trước đó, không tương đương một mô hình $G=10$ khởi tạo từ đầu. Kế hoạch thu nhỏ phải nêu lịch và số bước **tại mỗi $G$**, không chỉ tổng số bước. Đếm “bước” theo API thực sự dùng và đối chiếu mã tác giả để tránh nhầm LBFGS iteration với closure/function evaluation.

Để so sánh công bằng, KAN và MLP dùng cùng dữ liệu/split và quy tắc đánh giá. Báo cáo số tham số thực tế, ngân sách huấn luyện và thời gian; không coi cùng số bước LBFGS là chắc chắn cùng chi phí tính toán.

**Hai thí nghiệm riêng, không trộn số liệu:** (A) Panel chính dùng các checkpoint trên lịch grid extension cho KAN và các cấu hình MLP đã huấn luyện; (B) phép thử có/không grid extension (nếu làm) huấn luyện mô hình $G$ cố định từ đầu với cùng ngân sách đã định, ghi phương pháp khởi tạo. Đường theo $G$ từ (A) cho thấy hiệu ứng kết hợp **tăng tham số + huấn luyện tiếp + bước tối ưu**, không tự nó chứng minh tác động nhân quả riêng của $G$.

### 6.2. Ma trận chạy tối thiểu và quyết định thu gọn

| Ưu tiên | Run | Đủ để nghiệm thu |
|---|---|---|
| P0 | Smoke test 1 KAN + 1 MLP, dữ liệu nhỏ | Loss hữu hạn; chiều tensor/target đúng; seed có hiệu lực; CSV xuất được |
| P1 | Hàm chính: KAN tại ít nhất 4 checkpoint G trên một lịch; MLP ít nhất 4 mức tham số, cùng split | Một đồ thị test RMSE–tham số có cả hai đường; mỗi điểm truy được tới run ID, cấu hình và log |
| P1 | Cấu hình trọng tâm lặp seed, khảo sát train/test theo G | Báo cáo số lần chạy, trung bình/độ lệch chuẩn ở nơi có lặp; đồ thị G có ghi lịch mở rộng |
| P2 | Hai hàm bổ sung ở Mục 2.1 | Mỗi hàm có ít nhất một so sánh cùng dữ liệu; chỉ gọi là panel tái lập nếu đủ nhiều điểm |
| P3 | Ablation hoặc dữ liệu nhiễu | Giả thuyết, đối chứng, cùng ngân sách, kết quả; bỏ nếu đe dọa báo cáo/video |

**Quy tắc ngày 3:** Nếu một run chuẩn quá chậm, giảm lần lượt số điểm dữ liệu hoặc bước huấn luyện, rồi giảm mức $G$ cuối, ghi mọi thay đổi. Giữ 4 điểm mỗi mô hình và hàm chính trước; hủy P2/P3 trước khi cắt P1. Nếu thực tế không thể đạt P1, ghi trung thực là tái lập giới hạn và chỉ kết luận trong phạm vi số điểm đã chạy.

### 6.3. Bốn nhóm bảng/hình cần có

1. **RMSE theo số tham số:** KAN–MLP; chỉ hiển thị biến thiên seed tại những điểm thật sự được lặp.
2. **Train/test RMSE theo G:** ghi rõ đường grid extension nối tiếp, số bước và số tham số từng checkpoint; không khẳng định biến thiên chỉ do $G$.
3. **RMSE theo thời gian huấn luyện:** góc nhìn chi phí thực tế.
4. **Bảng đối chiếu tác giả–nhóm:** hàm, cấu hình, độ đo, kết quả, thay đổi thiết lập và mức độ so sánh được.

Khi có giá trị gốc phù hợp và khác 0:

$$
\text{Sai khác tương đối}=\frac{|\mathrm{RMSE}_{nhóm}-\mathrm{RMSE}_{gốc}|}{|\mathrm{RMSE}_{gốc}|}\times100\%.
$$

Nếu đọc giá trị từ Hình 3.1, ghi là **ước lượng từ biểu đồ**, không trình bày như số chính xác. Nếu thiết lập khác nhiều, nêu giới hạn so sánh. Nếu mẫu số bằng 0 hoặc quá nhỏ, ưu tiên sai khác tuyệt đối và giải thích vì sao tỷ lệ phần trăm không phù hợp.

Đối chiếu phải **ghép đúng hàm, độ đo, trục và mức tham số**: chọn điểm bài báo gần cùng số tham số hoặc so xu hướng trong khoảng chung. Không tính phần trăm giữa hai cấu hình lệch xa rồi gọi là sai khác tái lập. Bảng ghi rõ nguồn là panel Hình 3.1 hay bảng số gốc; nếu chỉ số hóa từ hình, lưu quy trình đọc và sai số xấp xỉ. Khi cách sinh dữ liệu/ngân sách chưa xác định hoặc khác bài báo, ưu tiên mô tả định tính về xu hướng và nêu giới hạn so sánh.

### 6.4. Mẫu bảng kết quả

| Hàm | Mô hình | Kiến trúc | G/k | Tham số | Số seed | Test RMSE (TB ± SD nếu chạy lặp) | Thời gian | Run ID |
|---|---|---|---|---|---|---|---|---|
| Hàm chính | KAN | `[2,1,1]` | Điền | Điền | Điền | Điền | Điền | Điền |
| Hàm chính | MLP | Điền | Không áp dụng | Điền | Điền | Điền | Điền | Điền |

### 6.5. Hạn chế và hướng cải tiến

| Hạn chế cần khảo sát | Thực nghiệm | Bằng chứng |
|---|---|---|
| Tăng độ linh hoạt spline có thể tăng overfitting và chi phí | Đường G theo lịch gốc để **quan sát**; nếu muốn quy nguyên nhân cho G, huấn luyện các mức G cố định từ đầu theo cùng ngân sách và seed | Train/test RMSE, khoảng cách train–test, thời gian; tách hai loại đồ thị |
| Lợi thế về số tham số chưa chắc là lợi thế tốc độ | So KAN–MLP cùng máy, split, điều kiện đo; báo cáo cả ngân sách optimizer | RMSE–thời gian cạnh RMSE–tham số, số closure evaluations |

Hướng cải tiến đề xuất: chọn G bằng validation hoặc dừng mở rộng lưới khi validation không cải thiện. Nếu chỉ đề xuất thì ghi rõ; nếu đã triển khai phải có đối chứng. Phân tích bổ sung không tự động bảo đảm điểm thưởng.

## 7. Manim và sản xuất video là hai nhóm việc

**Manim (TV1, TV2, TV4):** ba người cùng ghép/render ba cảnh Python. **Video (TV4–TV5):** TV4 viết lời/subtitle và kiểm số liệu, TV5 thu/chỉnh audio và hậu kỳ; hai người cùng duyệt bản xuất. TV3 cung cấp log và kiểm cách mô tả grid extension, không phải dựng cảnh.

| Cảnh Manim / người viết | Thời lượng gợi ý | Hình ảnh và ý chính | File bàn giao |
|---|---|---|---|
| **1 — TV1** | 0:00–1:50 | Bài toán xấp xỉ hàm, điểm dữ liệu, MLP với activation tại nút | `Manim/scenes/01_problem_mlp.py` + preview |
| **2 — TV2** | 1:50–4:50 | KAN với hàm trên cạnh, nút cộng, B-spline và trực giác lưới thưa/dày; TV3 duyệt phần lịch mở rộng | `Manim/scenes/02_kan_spline.py` + preview |
| **3 — TV4** | 4:50–8:00 | Đồ thị KAN–MLP từ CSV thật, so sánh tác giả–nhóm, giới hạn/kết luận và nguồn | `Manim/scenes/03_results.py` đọc `summary.json` + preview |

**TV4 bàn giao video:** `video/storyboard.md`, lời thoại, phụ đề theo timecode và nguồn cần hiện lúc dùng. **TV5 bàn giao video:** audio đã thu/chỉnh, bản hậu kỳ 5–10 phút, 1080p, URL công khai/hashtag. Hai người cùng nghe/xem và sửa bản cuối. Cảnh dùng dữ liệu demo phải thay bằng số thật; không chèn trực tiếp hình bài báo.

### Yêu cầu kiểm tra video

- Dài 5–10 phút và có mã Manim chạy được.
- Hình ảnh do nhóm tự dựng, không chèn trực tiếp hình từ bài báo.
- Ghi nguồn trên màn hình tại thời điểm sử dụng; danh mục tài liệu cuối video hoặc phần mô tả.
- Có thuyết minh rõ, tự nhiên; được phép dùng AI Voice nhưng phải khai báo.
- Có phụ đề xuyên suốt.
- Nên xuất 1080p, 16:9.
- Nếu dùng nhạc, bảo đảm quyền sử dụng và âm lượng không lấn lời đọc.
- Đăng công khai, kèm **#fithcmus #machinelearning #ai #ml**.

## 8. Checklist nghiệm thu

### Lý thuyết và báo cáo

- [ ] Có ví dụ thực tế nhóm tự đưa ra.
- [ ] Ít nhất hai phương pháp trước đó, đủ ý tưởng, hạn chế và hệ quả.
- [ ] Phân loại đóng góp lý thuyết, kỹ thuật và thực nghiệm.
- [ ] Có ký hiệu thống nhất, phát biểu bài toán và sơ đồ tự vẽ.
- [ ] Giải thích từng thành phần và biến trong công thức.
- [ ] Có định lý, giả định, trực giác/phác thảo chứng minh và phạm vi áp dụng.
- [ ] Phân tích số tham số, thời gian và bộ nhớ.
- [ ] Có liên hệ loss, gradient, regularization, bias–variance với môn học.
- [ ] Có tổng quan thực nghiệm và kết quả tác giả.
- [ ] Ít nhất hai điểm mạnh có bằng chứng.
- [ ] Ít nhất hai hạn chế có thể kiểm chứng.
- [ ] Ít nhất một hướng cải tiến gắn với hạn chế.
- [ ] Có nhận định riêng của nhóm và giới hạn kết luận.

### Thực nghiệm và mã nguồn

- [ ] Ghi rõ phạm vi tái lập cốt lõi có thu nhỏ.
- [ ] Có dữ liệu/cách sinh, miền lấy mẫu, split, seed và tiền xử lý.
- [ ] Có phần cứng, phiên bản phần mềm và cấu hình.
- [ ] Liệt kê đầy đủ thay đổi so với bài báo và lý do.
- [ ] Ánh xạ công thức với code, giải thích chi tiết cài đặt.
- [ ] Có kết quả đối chiếu bài báo và phân tích sai khác.
- [ ] Có phân tích bổ sung về G.
- [ ] README đủ để người khác cài và chạy lại.
- [ ] Đã thử chạy từ môi trường sạch với cấu hình kiểm tra phù hợp.
- [ ] Mọi bảng/hình truy được về log gốc.

### Video và bộ nộp

- [ ] Video 5–10 phút, Manim, hình tự dựng.
- [ ] Có nguồn, thuyết minh và phụ đề xuyên suốt.
- [ ] Video công khai, link truy cập được, đủ hashtag.
- [ ] Mã Manim và tài nguyên cần thiết chạy được.
- [ ] Báo cáo có phân công và tỷ lệ hoàn thành thực tế.
- [ ] Khai báo đầy đủ công cụ AI và mục đích sử dụng.
- [ ] Có tài liệu tham khảo và phụ lục cần thiết.
- [ ] Có Source, Manim, Report.pdf và url.txt.
- [ ] Tệp nén đặt theo MSSV; nếu trên 25 MB, thực hiện phương án Drive theo đề.
- [ ] Đã kiểm tra bài nộp và không chỉnh sửa sau hạn chót.

## 9. Cấu trúc làm việc và nhóm tự ghép

```text
KAN-project/
├── experiment_spec.md              # TV1–TV3 chốt
├── results_schema.md               # Giao diện CSV/JSON
├── Source/                         # TV1–TV3 làm/ghép/chạy
│   ├── README.md
│   ├── requirements.txt
│   ├── configs/
│   ├── src/
│   │   ├── data.py                 # TV1
│   │   ├── mlp_baseline.py         # TV1
│   │   ├── kan_model.py            # TV2
│   │   ├── train_grid.py           # TV3
│   │   └── metrics.py              # TV3
│   └── scripts/
│       ├── check_kan.py            # TV2
│       └── run_core.py             # TV3
├── results/
│   ├── runs.csv                    # TV1–TV3 chạy/khóa
│   └── summary.json                # TV4 kiểm/xuất từ CSV
├── analysis/
│   └── evaluate.py                 # TV3 viết, TV4 chạy/kiểm
├── report/
│   ├── sections/                   # Cả 5 viết/ghép
│   ├── figures/
│   └── build.sh                    # TV1 chuẩn bị, cả nhóm chạy/kiểm
├── Manim/                          # TV1, TV2, TV4 làm/ghép
│   ├── README.md
│   ├── scenes/
│   │   ├── 01_problem_mlp.py       # TV1
│   │   ├── 02_kan_spline.py        # TV2
│   │   └── 03_results.py           # TV4
│   └── assets/
├── video/                          # TV4–TV5 làm/ghép
└── fixtures/                       # Demo phát triển; không dùng trong kết quả cuối
```

**Chuỗi bàn giao:** TV1–TV3 ghép/chạy Source, chốt CSV → TV4 kiểm CSV, chạy lệnh tạo hình/JSON → TV1–TV5 ghép bài viết/hình thành `Report.pdf` → TV1/TV2/TV4 ghép ba cảnh Manim với JSON thật → TV4–TV5 ghép video/audio/phụ đề, xuất và đăng, ghi `url.txt` → cả nhóm đối chiếu số liệu PDF/video/CSV và đóng gói. Nhóm nào nhận đầu vào đổi thì tự xuất lại sản phẩm của nhóm mình.

```text
MSSV1_MSSV2_MSSV3_MSSV4_MSSV5/
├── Source/
├── Manim/
├── Report.pdf
└── url.txt
```

Đề cho phép link repo công khai thay `Source/` và `Manim/`; nếu dùng, dẫn đúng commit tạo kết quả. Nộp ZIP/RAR lên Moodle; nếu vượt 25 MB, dùng Google Drive công khai và nộp link theo đề. File demo không được xuất hiện như kết quả thật.

## 10. Mẫu bảng đóng góp và khai báo AI

### 10.1. Đóng góp theo nhóm việc

| Người / họ tên / MSSV | Báo cáo — cả 5 | Thực nghiệm — TV1–TV3 | Manim — TV1/TV2/TV4 | Video — TV4–TV5 | Minh chứng, tỷ lệ hoàn thành |
|---|---|---|---|---|---|
| TV1 / Điền | Giới thiệu/related/dữ liệu/MLP; rà PDF | Data + MLP, cùng ghép Source | Cảnh 1, cùng render bộ cảnh | — | Phần viết, commit, log, preview; điền cuối kỳ |
| TV2 / Điền | KAN/định lý; rà PDF | KAN adapter/check, cùng ghép Source | Cảnh 2, cùng render bộ cảnh | — | Phần viết, commit, preview; điền cuối kỳ |
| TV3 / Điền | Grid/huấn luyện/code; rà PDF | Train, metric, CSV, script hình, cùng ghép Source | —; duyệt kỹ thuật grid | — | Phần viết, CSV/log, lệnh tái chạy; điền cuối kỳ |
| TV4 / Điền | Setup/kết quả/sai khác; rà PDF | —; kiểm CSV/hình để viết | Cảnh 3, cùng render bộ cảnh | Storyboard, lời thoại, phụ đề; cùng ghép video | Phần viết, hình/JSON, preview, script/subtitle; điền cuối kỳ |
| TV5 / Điền | Đánh giá/kết luận/phụ lục; rà PDF | — | — | Thu/chỉnh audio, hậu kỳ, xuất/đăng; cùng ghép video | Phần viết, audio, video/URL; điền cuối kỳ |

Bảng chia dự kiến nhằm cân **độ nặng**, không phải cam kết mỗi người sẽ hoàn thành đúng bằng nhau. Tỷ lệ hoàn thành cuối kỳ dựa vào đầu ra và thời gian thực tế, có minh chứng, cả nhóm xác nhận.

### 10.2. Khai báo AI

| Công cụ | Mục đích sử dụng thực tế | Phần bị ảnh hưởng | Cách nhóm kiểm chứng |
|---|---|---|---|
| ChatGPT | Hỗ trợ lập kế hoạch và phân công ban đầu | Kế hoạch đồ án | Nhóm đối chiếu với đề và bài báo |
| Công cụ khác, nếu có | Điền đúng thực tế | Điền | Điền |

Không khai công cụ chưa sử dụng; cập nhật khi dùng AI để hỗ trợ code, giải thích, biên tập hoặc thuyết minh. Tất cả thành viên cần hiểu phần việc của mình và ý tưởng cốt lõi để sẵn sàng vấn đáp.

## 11. Mốc tự ghép và nghiệm thu giữa các nhóm

- **Ngày 2:** TV1–TV3 chốt đặc tả/CSV, cả năm chốt dàn báo cáo, TV1/TV2/TV4 chốt template ba cảnh, TV4–TV5 chốt storyboard/audio/subtitle.
- **Ngày 3:** Đo một run KAN/MLP và một preview cảnh; rà lại ước lượng tải ở Mục 3.1, giảm hàm phụ/điểm thưởng nếu quá sức.
- **Ngày 5:** Các nhóm tự ghép thử: Source chạy một cấu hình, PDF năm mục, Manim ba cảnh demo, video cảnh/audio nháp. Lỗi trả đúng người sở hữu.
- **Ngày 7:** TV1–TV3 bàn giao panel KAN–MLP/CSV/log; TV4 kiểm hình/JSON; nhóm báo cáo ghép v0.5, nhóm Manim/video cập nhật.
- **Ngày 9:** TV1–TV3 khóa CSV/config/commit, TV4 khóa hình/JSON; cả năm chốt báo cáo, TV1/TV2/TV4 render cảnh thật, TV4–TV5 khóa video theo cùng số liệu.
- **Ngày 10:** Nhóm báo cáo xuất Report.pdf v1, nhóm thực nghiệm Source v1, nhóm Manim ba cảnh v1, nhóm video video v1; mỗi nhóm chạy/kiểm lại.
- **Ngày 11–12:** Thử pipeline sạch, render cảnh, nghe/xem phụ đề; lỗi nhóm nào nhóm đó sửa, nhóm phụ thuộc xuất lại bản mới nhất.
- **Ngày 13–14:** Cả nhóm kiểm PDF–Source–Manim–video/URL, đóng gói và nộp trước hạn Moodle.

Không có người ghép toàn dự án. Mỗi nhóm cùng kiểm và chịu trách nhiệm sản phẩm của mình; người sở hữu sửa lỗi phần mình sau khi ghép.

## 12. Bảng truy vết yêu cầu → bằng chứng nộp

| Yêu cầu trong đề | Vị trí dự kiến | Người kiểm tra | Tiêu chí đạt |
|---|---|---|---|
| Bài báo đã qua bình duyệt | Mở đầu + tài liệu tham khảo | TV1 | Ghi đúng bản **arXiv v5, 09/02/2025**, trên PDF có dòng được nhận tại **ICLR 2025** |
| Bài toán, ví dụ tự đặt, ≥2 phương pháp trước, đóng góp phân loại | Báo cáo mục 1–2 | TV2 | Mỗi phương pháp có ý tưởng, hạn chế, hệ quả; ví dụ đủ đầu vào/đầu ra |
| Ký hiệu, sơ đồ tự vẽ, từng thành phần, định lý, độ phức tạp | Báo cáo mục 3–4 | TV1 | Có công thức giải nghĩa biến, giả định và phạm vi áp dụng; số tham số/tính toán/bộ nhớ tách biệt |
| Thực nghiệm tác giả và liên hệ môn học | Báo cáo mục 4–5 | TV2 | Nêu 5 bài toán ở Hình 3.1, thiết lập tác giả; liên hệ loss, gradient, regularization và bias–variance |
| Phạm vi tái lập, dữ liệu, cấu hình, seed, thay đổi | Báo cáo mục 6 + `Source/configs` | TV4 | Chỉ rõ một panel chính, danh mục sai khác, split, môi trường, phần cứng và lệnh chạy |
| Phân tích code và tác động khi bỏ/sửa thành phần | Báo cáo mục 7 + source cố định commit | TV2 | Bảng công thức ↔ file/hàm/đoạn code thật, chi tiết mã và hậu quả dự kiến hoặc đo được |
| Kết quả tác giả–nhóm, độ lệch và lý giải | Báo cáo mục 8 + `results/` | TV3 | Có đồ thị RMSE–tham số hai mô hình, run ID từng điểm, sai khác định lượng nơi so được |
| ≥2 điểm mạnh, ≥2 hạn chế kiểm chứng được, ≥1 cải tiến gắn hạn chế | Báo cáo mục 9 | TV1 | Mỗi nhận định gắn chứng cứ hoặc kế hoạch kiểm chứng; cải tiến có đối chứng nếu nhận là đã làm |
| Video Manim 5–10 phút, phụ đề, nguồn, công khai | `Manim/`, URL công khai | TV4 + TV5, TV2 duyệt khoa học | Render từ mã được, hình tự dựng, lời rõ, phụ đề suốt, nguồn lúc dùng, đủ 4 hashtag |
| Phân công, tỷ lệ hoàn thành, AI, tài liệu tham khảo | Cuối Report.pdf | Cả nhóm | Tỷ lệ thực tế theo nhiệm vụ; khai đúng công cụ/mục đích; dẫn nguồn cho code/dữ liệu/hình |
| Gói nộp trên Moodle | ZIP/RAR hoặc link Drive nếu >25 MB | Cả nhóm kiểm, một người thao tác nộp | Đúng tên MSSV, PDF + source/Manim hoặc link repo công khai + `url.txt`; truy cập được, nộp trước hạn |

**Ngưỡng ngày 7:** TV1–TV3 tạo CSV có ≥4 điểm KAN và ≥4 điểm MLP cùng test split, run ID đầy đủ và seed lặp ở ít nhất một cặp cấu hình; TV4 dựng đồ thị tham số–RMSE; nhóm báo cáo ghép bản nháp, nhóm Manim ghép ba cảnh. **Ngưỡng ngày 10:** PDF có bảng/hình thật từ TV4, video xem từ đầu đến cuối, Source và Manim chạy được. **Ngưỡng ngày 13:** PDF/video/CSV cùng phiên bản, URL công khai hoạt động; xác nhận đóng góp đúng nhóm việc và khai báo AI.

## 13. Nguồn dùng để triển khai kế hoạch

1. **Đề đồ án CSC14005**, `ML - Course Project(1).pdf`: Mục 1.1–1.3 (lý thuyết, tái lập và video), Mục 2 (nộp bài), Mục 3 (tỷ trọng).
2. **Liu và cộng sự, “KAN: Kolmogorov–Arnold Networks”**, `2404.19756v5(1).pdf`: Mục 2.2 (kiến trúc/cài đặt), Định lý 2.1 ở Mục 2.3, Mục 2.4 (grid extension), Mục 2.5 (diễn giải), Hình 3.1 và Mục 3.1 (năm bài toán và so sánh KAN–MLP).

Trước khi chạy mã, TV3 ghi commit/release thực tế của kho mã tác giả và nguồn của MLP, thư viện, dữ liệu. Tài liệu tham khảo của **báo cáo cuối** phải trích dẫn bibliographic đầy đủ theo định dạng nhóm chọn; danh sách hai PDF trên chỉ là nguồn của kế hoạch này.
