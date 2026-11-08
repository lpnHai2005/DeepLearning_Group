# Kế hoạch hoàn thiện BTL01 (bàn giao giữa các phiên làm việc)

> Cập nhật: 08/11/2026, sau commit `0ae7bf6` trên `main`.
> Mục tiêu: đạt đủ rubric 100 điểm của `MLP_Exercise - Yeu Cau.pdf`. Mọi lựa chọn phải truy vết được về file log, CSV hoặc hình trong repo.

## 0. Trạng thái tổng quát

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 1 | Lấp lỗ hổng minh chứng (code, chạy, log, hình, bảng) | ✅ Xong, đã push lên `main` (`246400d`, `0ae7bf6`) |
| 2 | Nhật ký quyết định `report/NHAT_KY_QUYET_DINH.md` | ✅ Xong (D1–D17, số liệu đã đối chiếu CSV) |
| 3 | Báo cáo `.docx` + `.pdf` (6 chương, 10–20 trang) | 🟡 Có `report/BaoCao_Nhom10.docx` (nguồn nội dung `report/BAO_CAO.md`, nhóm tự dàn trang Word và xuất PDF). Còn: xuất PDF, điền họ tên/MSSV (nội dung chính ~24 trang, đề chỉ đề xuất 10–20 nên giữ nguyên) |
| 4 | Slide `.pptx` + `.pdf` (10–15 slide), chuẩn bị demo | ✅ `report/Slide_Nhom10.pptx/.pdf`, 14 slide, có ghi chú người trình bày (build `src/build_slides.py`). Còn: điền họ tên |
| 5 | QA tái lập, bảng % đóng góp, đóng gói zip | ❌ |

Rubric chưa có minh chứng: **7. Báo cáo (15đ)**, **8. Slide (10đ)**. Các mục 1–6 đã có code, số liệu và log.

## 1. Giai đoạn 1 đã làm những gì (để không làm lại)

- `src/posthoc_test.py`: nạp `best.pt` đã đóng băng, đo test cho từng run, ghi vào `summary.csv`, `metrics.json`, `posthoc_test.log`. Kết quả chỉ để trình bày, không dùng để chọn cấu hình.
- `src/analyze_stopping.py`: mô phỏng patience từ `results/grid_optimizer/adam_0.01` (chạy đủ 2000 epoch), thống kê run chạm trần epoch. Kết quả ở `results/stopping/` và `figures/stopping_patience.png`.
- `src/make_tables.py`: sinh `report/tables/*.md|csv` từ `summary.csv`. Khi dán bảng vào báo cáo, phải lấy từ đây, không gõ tay.
- `src/env_info.py` → `results/env.txt`.
- `run_experiment.py`: thêm khóa `cases` (đổi nhiều tham số cùng lúc).
- Thí nghiệm mới:
  - `expA_steps`: A với cùng ngân sách khoảng 64k bước cập nhật.
  - `expB_wide`: B = 16/32/64 ở A = 1000.
  - `expB_A200`: B ở A = 200.
  - `grid_batch`: batch 16/32/64/128.
- `tests/test_config_evidence.py`: kiểm tra `baseline.json` và `final.json` khớp với cấu hình có val thấp nhất trong log.
- Kết quả: 62/62 test pass.
- Các file `report/THUC_NGHIEM_A.md`, `THUC_NGHIEM_B.md`, `THUC_NGHIEM_C.md`, `MO_HINH_CUOI_TEST.md` (mục 4.2–4.5) đã được cập nhật số liệu mới. Đã sửa 10 ô sai so với CSV.

## 2. Các con số chính (để viết báo cáo và slide)

- Hàm Gaussian: μ = (0, 0), C = [[0.3, −0.5], [−0.5, 2]].
  - det(C) = 0.35, đỉnh y_max = 1/(2π√0.35) ≈ 0.2690.
  - Hệ số tương quan ρ ≈ −0.645.
  - 99.52% thể tích nằm trong miền [−2, 2] × [−4, 4].
  - 62.6% mẫu train có y < 0.01.
- Dữ liệu: train_pool 2000 / val 200 / test 100, seed 2026 / 2027 / 2028. x lấy đều trên miền.
- Baseline: `[10]`, sigmoid, Adam, lr 0.01, batch 32, 2000 epoch, patience 200, 3 seed.
- Mô hình hằng số (luôn đoán trung bình): val MSE 2.49e-3, test MSE 2.37e-3.
- **A** (`expA`):
  - Val MSE giảm từ 7.06e-5 (A = 200) xuống mức chững quanh 1.5–2.6e-5 khi A ≥ 600.
  - Với cùng ngân sách bước, A = 200 còn 4.04e-5 (giảm 43%). Như vậy một phần lỗi ở A nhỏ là do được train ít bước hơn.
  - Từ A ≥ 600, giới hạn chính là dung lượng của mạng `[10]`.
- **B** ở A = 1000:
  - Val MSE từ 2.29e-3 (B = 1) xuống 1.89e-5 (B = 10), rồi 2.09e-6 (B = 64), 1.46e-6 (B = 93).
  - Không có overfit.
- **B** ở A = 200:
  - Tỉ số Val/Train tăng từ 1.30 (B = 10) lên 1.74 (B = 64): có khoảng cách tổng quát hóa.
  - Val vẫn giảm theo B, nên chưa có đường chữ U.
  - 17/39 run chạm trần 2000 epoch.
- **C**:
  - Val MSE: C = 1 là 1.89e-5, C = 2 là 1.05e-7, C = 3 là 6.02e-8, C = 4 là 9.21e-8.
  - So với mạng 1 tầng cùng số tham số, mạng sâu tốt hơn 16–28 lần.
  - Có vanishing gradient lúc khởi tạo nhưng không cản trở việc huấn luyện, vì Adam bù lại.
- **Mô hình cuối**: C = 3 `[10,10,10]`, 261 tham số.
  - Test MSE 7.13e-8 ± 2.36e-8, R² = 0.99997, Grid MSE 7.34e-8.
  - Checkpoint `results/final/best.pt` lấy từ seed 2.
- Batch 32 có val tốt nhất (1.89e-5), nhưng chênh lệch với các batch khác nhỏ hơn std.
- patience 200 chỉ kém 0.7% so với không dừng sớm. patience 100 kém 31%.

## 3. Giai đoạn 2: Nhật ký quyết định (khoảng nửa ngày)

Tạo `report/NHAT_KY_QUYET_DINH.md`. Mỗi dòng gồm: quyết định, lý do, file minh chứng, thứ tự (chọn trước rồi kiểm chứng sau, hay khảo sát rồi mới chọn). Nội dung này đưa vào mục 4.1 và phụ lục.

| Quyết định | Minh chứng | Ghi chú cần trung thực |
|---|---|---|
| Chia train/val/test 2000/200/100, seed 2026/27/28 | `data/meta.json`, `results/eda/data_summary.csv`, hình `eda_*` | |
| Lấy x đều trên miền (không lấy theo Gaussian) | `src/data/gaussian.py::sample_uniform` | Khác hình scatter (h) trong slide của thầy |
| X_SCALE = (2, 4), Y_SCALE = 0.269 | `src/data/gaussian.py` | Nhóm tự chọn; lấy từ hằng số đề bài, không từ dữ liệu, nên không rò rỉ |
| sigmoid, lr 0.01 | `results/grid_baseline/` (27 run), `grid_baseline_val_mse.png` | Khảo sát rồi mới chọn |
| Adam | `results/grid_optimizer/` (126 run), `optimizer_loss.png` | **Chọn trước, kiểm chứng sau** (commit `3565482`). Chỉ giữ thư mục run của lr tốt nhất mỗi optimizer; đủ 126 dòng trong `summary.csv` |
| batch 32 | `results/grid_batch/`, `report/tables/grid_batch.md` | Chọn trước, kiểm chứng sau; chênh lệch nhỏ hơn std |
| patience 200 | `results/stopping/patience_sim.csv`, `stopping_patience.png` | Chọn trước, kiểm chứng sau |
| max_epochs 2000 | `results/stopping/epoch_cap.csv` | **Là hạn chế**: 11/30 run A, 17/39 run B_A200 chạm trần |
| Đối chứng A cùng ngân sách bước | `configs/expA_steps.json`, `results/expA_steps/`, `expA_steps.png` | Phát hiện từ log, không có trong kế hoạch ban đầu |
| B mở rộng 16/32/64 và A = 200 | `results/expB_wide/`, `results/expB_A200/`, `expB_A200.png` | |
| Đối chứng C cùng số tham số (B = 38/65/93) | `configs/expC_ctrl.json` (theo công thức 4B+1) | |
| Vanishing gradient | `results/expC/grad_norms.csv`, `expC_grad_norms.png` | |
| Chọn mô hình cuối theo argmin val trên 53 cấu hình | `results/summary_all.csv`, `configs/final.json`, `tests/test_config_evidence.py` | |
| Test chỉ dùng hậu kiểm và đánh giá cuối | `results/*/posthoc_test.log`, `results/final/run.log` | |
| Tìm siêu tham số theo từng tọa độ (không thử mọi tổ hợp) | — | Hạn chế: có thể bỏ sót tổ hợp tốt hơn, ví dụ relu với batch khác |

## 4. Giai đoạn 3: Báo cáo `.docx` + `.pdf` (khoảng 2 ngày, TV4 chính)

Bố cục theo mục 6 của đề, tham khảo cách trình bày của `Sample_Report.pdf` nhưng **không chép kết quả**. Hình và bảng phải đánh số kèm chú thích.

1. **Ch.1 Giới thiệu** *(viết mới)*: mục tiêu, phát biểu bài toán, đóng góp của từng thành viên.
2. **Ch.2 Dữ liệu**: `report/CHUONG_2_DU_LIEU.md`. Bổ sung phần giải thích Gaussian: ý nghĩa của C, ma trận C⁻¹, khai triển Q = 5.714x₁² + 2.857x₁x₂ + 0.857x₂², bảng tính tay vài điểm, trị riêng 2.136 / 0.164, đồi nghiêng khoảng 15°.
3. **Ch.3 Mô hình và cài đặt**: `report/CHUONG_3_MO_HINH.md`. Thêm:
   - sơ đồ pipeline train/val/test;
   - bảng tóm tắt mã nguồn (file nào làm gì);
   - trích đoạn `MLP` và vòng lặp train;
   - câu ghi rõ X_SCALE/Y_SCALE là lựa chọn của nhóm;
   - môi trường lấy từ `results/env.txt`.
4. **Ch.4 Thực nghiệm**:
   - 4.1 (viết mới): bảng siêu tham số, quy trình đánh giá công bằng, lấy từ giai đoạn 2.
   - 4.2 đến 4.5: dùng `THUC_NGHIEM_A/B/C.md` và `MO_HINH_CUOI_TEST.md`.
5. **Ch.5 Thảo luận** *(viết mới)*:
   - Vì sao không thấy đường chữ U: nhãn không nhiễu, có early stopping, hàm trơn.
   - Vì sao mạng sâu tốt hơn mạng rộng.
   - Kết quả ngược với Sample_Report.
   - Yếu tố lẫn giữa số mẫu và số bước cập nhật ở A.
   - A = 2000 lỗi tăng nhẹ (do phương sai giữa seed, không phải do ngân sách bước).
   - Hạn chế: trần 2000 epoch; chỉ 3 seed; tìm siêu tham số theo từng tọa độ; chỉ dùng CPU; thời gian của 90 run bổ sung đo khi 4 tiến trình chạy song song nên không so sánh được.
   - Hướng cải tiến.
6. **Ch.6 Kết luận**, **tài liệu tham khảo** (PyTorch docs; Goodfellow et al. *Deep Learning*; Kingma & Ba 2015; Glorot & Bengio 2010; Cybenko 1989 / Hornik 1989), **phụ lục** (lệnh tái lập lấy từ README, nhật ký quyết định).

Gợi ý công cụ: skill `docx` để tạo file Word từ các file `.md`, sau đó xuất PDF.

## 5. Giai đoạn 4: Slide (khoảng 1 ngày)

- Ghép từ `report/TV1_SLIDE_NOTES.md`, `TV2_SLIDE_NOTES.md`, `TV3_SLIDE_NOTES.md`.
- Khoảng 12–14 slide: bài toán và Gaussian, dữ liệu, sơ đồ MLP và code chính, baseline và optimizer, A (gồm phần cùng ngân sách bước), B (gồm A = 200), C (gồm đối chứng), mô hình cuối (hình y–ŷ, bề mặt), thảo luận, kết luận.
- Dùng skill `pptx`.
- Lệnh demo: `python src/run_experiment.py --config configs/demo.json` (khoảng 10 giây).

## 6. Giai đoạn 5: QA và nộp (khoảng nửa ngày)

1. Clone repo vào thư mục mới, tạo venv mới, `pip install -r requirements.txt`, chạy `pytest -q` và các lệnh trong README. Lưu output vào `results/qa_reproduce.log`.
2. Kiểm chéo: mọi con số trong báo cáo và slide phải có trong `report/tables/` hoặc CSV.
3. Lập bảng thành viên, nhiệm vụ và % đóng góp (tổng 100%). Tick checklist mục 7 trong `PHAN_CONG_NHIEM_VU.md`.
4. Nén file `BTL01_Nhom_HoTen_MSSV.zip`: code, data, results, figures, docx, pdf, pptx.

## 7. Việc lặt vặt còn treo

- Báo TV1 và TV2 rằng `summarize_expA.py`, `summarize_expC.py`, `run_experiment.py` đã bị sửa. Nhắc cả nhóm `git pull`.
- Nhánh `bo-sung-minh-chung-gd1` (local và remote) đã merge vào `main`, có thể xoá.
- TV1, TV2, TV3 nên đọc lại văn phong các mục 4.2–4.4 sau khi đã chèn nội dung mới.

## 8. Quy ước làm việc (cho phiên sau)

- Máy Windows: chạy Python với `PYTHONUTF8=1` để tránh lỗi in tiếng Việt.
- Script Python có chứa `\` (LaTeX) thì ghi ra file bằng tool Write rồi chạy. Không nhét vào heredoc của bash, vì dễ lỗi trích dẫn.
- Thứ tự chạy lại sau khi thay đổi thí nghiệm: `run_experiment` → `posthoc_test.py` → `analyze_stopping.py` → `summarize_expA/B/C` → `make_report_assets.py` → `make_tables.py` → `pytest`.
- Commit: message tiếng Việt, **không** thêm dòng tác giả AI hay Co-Authored-By. Nhóm commit thẳng lên `main`.
- Test chỉ được dùng hậu kiểm. Không bao giờ dùng test để chọn cấu hình.
