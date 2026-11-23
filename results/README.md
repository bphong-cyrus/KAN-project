# Thư mục kết quả thực nghiệm (`results/`)

- `runs.csv`: Bảng log thực nghiệm chính thức theo schema trong [../results_schema.md](../results_schema.md). Hiện tại **chỉ chứa dòng tiêu đề** vì các lượt chạy KAN và MLP chính thức sẽ do **TV1–TV3** thực hiện đồng bộ sau khi TV2 hoàn tất `kan_model.py` và TV3 hoàn tất `train_grid.py`.
- `summary.json`: Do **TV4** sinh ra từ `analysis/evaluate.py` sau khi kiểm tra tính hợp lệ của `runs.csv`.
- **Tuyệt đối không** chép kết quả smoke test (`fixtures/tv1_smoke_runs.csv`) hay dữ liệu mẫu (`fixtures/sample_summary.json`) vào thư mục này.
