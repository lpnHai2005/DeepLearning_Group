# Biên bản QA Tái lập Toàn bộ Kho mã nguồn BTL02 (Nhiệm vụ T27)

> **Phụ trách chính (`R`)**: TV4 – Huỳnh Viễn Thông  
> **Phối hợp (`C1`, `C2`)**: TV1 – Đoàn Trung Kiên (clone sạch, venv mới, README & pytest trên máy riêng), TV2 – Lê Phan Nguyên Hải (chạy lại notebook trên máy riêng)  
> **Kiểm thử độc lập (`T`)**: TV3 – Võ Hoàng Bảo  
> **Ngày thực hiện**: `2026-11-27`  
> **Commit QA**: `208e0e3` (`origin/main`, bản clone sạch tại `%TEMP%\qa_t27_clean`, đã xóa sau QA)  
> **Môi trường máy QA**: `Administrator` | Windows | `Python 3.11.4` | `PyTorch 2.6.0+cu124` | `NVIDIA GeForce RTX 3060 Laptop GPU` (CUDA khả dụng)

---

## 1. Tóm tắt Kết quả QA (`R` — TV4)

| Hạng mục kiểm tra | Phạm vi / Lệnh chạy | Kết quả | Ghi chú |
| :--- | :--- | :---: | :--- |
| **1. Clone sạch từ GitHub** | `git clone --depth 1 https://github.com/lpnHai2005/DeepLearning_Group.git` | **PASS** | Commit `208e0e3`, 1,915 file repo gốc |
| **2. Cài đặt `.venv` mới theo `README.md`** | `python -m venv .venv` + `pip install --dry-run -r requirements.txt` | **PHÁT HIỆN LỖI (`QA-01`)** | `numpy==2.5.2` yêu cầu `Python >= 3.12`, lỗi trên Python `3.11.4`; thiếu `python-docx` |
| **3. Bộ kiểm thử tự động `pytest`** | `.\.venv\Scripts\python.exe -m pytest -q` (có `data/fer2013.csv` cục bộ) | **PASS (127/127)** | `127 passed, 0 failed, 0 skipped` trong `108.07s` |
| **4. Kiểm tra dữ liệu & EDA (`README.md`)** | `src.data.check_fer2013`, `src.data.eda --seed 0` | **PASS (2/2)** | `SHA-256 = 3b8d9617d101...`, đủ `28709 / 3589 / 3589`, sinh 5 hình + 5 CSV |
| **5. Script tổng hợp & trực quan hóa** | `baseline_summary.py`, `make_tables.py`, `e6_analysis.py`, `architecture_diagram`, `visualize_filters.py` | **PASS (5/5)** | Đều `exit = 0`; ghi nhận góp ý nhẹ `QA-03` ở `make_tables.py` sau `T19` |
| **6. Script theo dõi & slide tự động** | `tracking/build_tracking.py`, `task_status.py`, `build_slides.py` | **PASS (3/3)** | 28 nhiệm vụ (`7` lượt `T`/người), xuất `Slide_TienDo_BTL02.pptx` (13 slide) |
| **7. Notebook `Restart & Run All`** | `nbconvert --to notebook --execute notebooks/BTL02_reproducible.ipynb` | **PASS (5/5 cell)** | `execution_count = [1, 2, 3, 4, 5]`, `0` lỗi, cell cuối in `PASS`, tạo `_nb-004_s0` (`val_acc = 0.3770` sau 2 ep, không chạm `PrivateTest`) |
| **8. Kiểm toán tĩnh & vệ sinh kho mã (`BTL02/`)** | Header `Phụ trách`, cấm TF/Keras, đường dẫn tuyệt đối, file cấm Git, `figures_manifest.csv`, artifact `Completed` | **PASS (100%)** | `34/34` file `.py` đủ header; `0` TF/Keras; `0` đường dẫn tuyệt đối; `0` file cấm trong `BTL02/`; `116/116` hình trong `figures_manifest.csv` tồn tại kèm CSV/NPZ và khớp 100% SHA-256; `0` thiếu artifact lõi ở các run `Completed` |
| **9. Khóa bảo vệ `PrivateTest` (`T19`)** | `results/E7/TEST_DONE.json` & thử chạy `--eval-test` lần 2 | **PASS** | Từ chối ngay với `exit = 1`, 0 byte thay đổi (xác nhận tại commit `57eacf2`) |

---

## 2. Chi tiết Các Lệnh Đã Chạy trên Bản Clone Sạch (`%TEMP%\qa_t27_clean\BTL02`)

1. **Dữ liệu cục bộ chép vào bản clone sạch**:
   - `data/fer2013.csv`: SHA-256 `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b` (`301,072,766` bytes).
   - `results/E1/E1-014_s0/best.pt`: SHA-256 `8c79c23b35dd8e109ea57998844ae295279744a63964b233c0cee5e51933fbcf` (dùng kiểm chứng lệnh `visualize_filters.py` trên checkpoint thật, không commit).
2. **Kiểm thử cài đặt độc lập (`venv` thuần không thừa kế site-packages)**:
   - Lệnh: `python -m venv .venv` → `.\.venv\Scripts\python.exe -X utf8 -m pip install --dry-run -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`
   - Kết quả: `torch==2.6.0+cu124` và `torchvision==0.21.0+cu124` có bánh xe `cp311`, nhưng dừng lỗi ở dòng 5 (`ERROR: No matching distribution found for numpy==2.5.2`, vì `numpy>=2.5.0` yêu cầu `Python >= 3.12`).
3. **Kiểm thử chạy toàn repo trên môi trường `.venv` (`Python 3.11.4`, `torch 2.6.0+cu124`, `numpy 2.2.6`, `pandas 2.2.3`, `scikit-learn 1.5.2`)**:
   - `pytest -q`: **127 passed in 108.07s** (`0 failed`, `0 skipped`).
   - Toàn bộ 10 lệnh CLI trong `README.md` (`check_fer2013`, `eda`, `baseline_summary`, `make_tables`, `e6_analysis`, `architecture_diagram`, `visualize_filters`, `build_tracking`, `task_status`, `build_slides`): **10/10 exit code = 0**.
   - `nbconvert --to notebook --execute notebooks/BTL02_reproducible.ipynb` trên kernel mới `qa_t27_venv`: **exit code = 0**, `5/5` cell thực thi tuần tự `[1, 2, 3, 4, 5]`, `0` error output; cell 5 xác nhận `PASS: Toàn bộ kiểm chứng tái lập (run.log, history.csv, metrics.json, best.pt) đều khớp!` (`sha256(history) = 2442a21bd851d511...`).
4. **Dọn dẹp sau QA**: Đã gỡ kernelspec tạm `qa_t27_venv` và xóa hoàn toàn `%TEMP%\qa_t27_clean`.

---

## 3. Danh sách Lỗi & Góp ý Phát hiện qua QA (`T27`)

| Mã | Mức độ | File liên quan | Người phụ trách | Mô tả lỗi / vấn đề | Đề xuất khắc phục | Trạng thái |
| :--- | :---: | :--- | :---: | :--- | :--- | :---: |
| **`QA-01`** | **Trung bình** | `BTL02/requirements.txt` | TV1 | (1) Ghim cứng `numpy==2.5.2` (yêu cầu `Python >= 3.12`) khiến `pip install -r requirements.txt` thất bại trên máy chạy `Python 3.11` (như máy TV4 và mô tả phần cứng trong `NHAT_KY_THUC_NGHIEM.md`). (2) Thiếu gói `python-docx` dùng khi `make_tables.py` xuất `BTL02_tables_report.docx`. | Dùng environment marker theo phiên bản Python (hoặc ghi rõ bắt buộc `Python 3.12` đồng nhất giữa `README.md` và `NHAT_KY_THUC_NGHIEM.md`) và bổ sung `python-docx` vào `requirements.txt`. | Mở (đã báo TV1) |
| **`QA-02`** | **Nhẹ** | `BTL02/README.md` | TV1 | Nội dung hiện trạng còn dừng ở `2026-11-24`: (1) dòng 5 chưa liệt kê E1–E7, `eval_e7.py`, `e6_analysis.py`, `TEST_DONE.json`; (2) dòng 80 ghi tên thư mục mẫu `DeepLearning_Group-1/` thay vì `DeepLearning_Group/`; (3) dòng 95 vẫn ghi *"Chưa có kết quả cài đặt độc lập từ TV2/TV4"*. | Cập nhật mục Hiện trạng, sửa tên thư mục `DeepLearning_Group/`, cập nhật trạng thái cài đặt độc lập trước khi đóng gói `T28`. | Mở (đã báo TV1) |
| **`QA-03`** | **Nhẹ** | `BTL02/src/make_tables.py` | TV3 | Sau khi `T19` ghi dòng `E7-TEST` (chỉ có `test_acc`, không có `val_acc`) vào `tracking/experiments.csv`, chạy lại `python src/make_tables.py` tạo thêm 1 dòng `E7,T19,private_test_flip` với các cột validation bỏ trống trong `E7_summary.csv` và `summary_all.csv` (không lỗi thực thi, nhưng làm lệch cấu trúc bảng validation). | Bỏ qua dòng `task_id == "T19"` / `exp_id == "E7-TEST"` khi gom bảng validation `E7_summary.csv` (hoặc hiển thị rõ cột `test_acc`). | Mở (đã báo TV3) |

---

## 4. Xác nhận Phối hợp (`C1`, `C2`) trước khi Chốt `done(T27)`

- **Xác nhận `C1` — TV1 – Đoàn Trung Kiên**: *(chờ TV1 chạy clone sạch, venv mới theo `README.md` và `pytest` trên máy Python 3.12, đồng thời rà soát `QA-01`, `QA-02`)*
- **Xác nhận `C2` — TV2 – Lê Phan Nguyên Hải**: *(chờ TV2 chạy lại `notebooks/BTL02_reproducible.ipynb` trên máy riêng và xác nhận không lỗi)*
