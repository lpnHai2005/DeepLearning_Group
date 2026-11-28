# T25 — kiểm thử độc lập báo cáo ghép (TV1)

Ngày kiểm: 2026-11-28. Bản kiểm: `d6104afd7e990a3b4d75abf01a7a53a36dbda459` (`docs(T25)`). TV1 tạo clone Git sạch riêng trong `%TEMP%`, dùng sparse checkout cho `BTL02/report`, `tracking`, `src`, `figures` và `results/E7`. Trước và sau kiểm thử, `git status --porcelain` của clone đều rỗng. Không chạy đánh giá PrivateTest.

## Lệnh và cách đối chiếu

Từ `BTL02/` trong clone:

```powershell
python -X utf8 src/t25_report_stats.py
git diff --exit-code -- report/tables/T25_contribution.csv report/tables/T25_filters_stats.csv report/tables/T25_runs_stats.csv report/chuong1_gioi_thieu.md report/chuong6_truc_quan_bo_loc.md report/chuong7_thao_luan.md
```

Lệnh sinh bảng chạy thành công, không tạo khác biệt so với snapshot. Mẫu được chọn với seed `20261128`: 10 mục ngẫu nhiên trong 12 ứng viên số liệu thuộc Chương 1, 5–8; thêm bảng đóng góp trong DOCX và trường hợp T28 để kiểm quy ước trình bày. Kiểm tra PDF bằng PyMuPDF, kiểm SHA-256 của hình bằng `hashlib`, và tự đếm lại từ `tracking/activity_log.csv`, `test_log.csv`, `experiments.csv` bằng `csv.DictReader`/`Counter`.

| # | Vị trí, số/hình kiểm | Giá trị trong báo cáo | Nguồn và giá trị tính lại | Kết quả |
|---:|---|---:|---|---|
| 1 | Chương 6, Grad-CAM đúng/sai | 7 / 8 ca | `report/tables/T25_filters_stats.csv`; dữ liệu `figures/filters_grad_cam_correct.csv`, `filters_grad_cam_wrong.csv`: 7 / 8 | Pass |
| 2 | Chương 1, TV2 lần kiểm thử | 8 | `tracking/test_log.csv` → `T25_contribution.csv`: 8 | Pass |
| 3 | Chương 8, PublicTest ensemble + flip | 72.50% | `report/tables/E7_tta_ensemble.csv`, `val_acc_fmt=0.7250` | Pass |
| 4 | Chương 1, TV2 commit có mã | 62 | Đếm `tracking/activity_log.csv` với mã `T\d{2}`: 62 | Pass |
| 5 | Chương 6, tỷ số độ lớn trọng số | 363.7 lần | `T25_filters_stats.csv`; tính từ `figures/filters_weights_first.csv` và `filters_weights_last.csv`: 363.7 | Pass |
| 6 | Chương 1, tổng commit / lần kiểm thử | 214 / 31 | Tự đếm `activity_log.csv` / `test_log.csv`: 214 / 31 | Pass |
| 7 | Chương 1, TV4 commit có mã | 50 | Tự đếm `activity_log.csv`: 50 | Pass |
| 8 | Chương 5, QA cuối | 128 passed, 0 failed, 0 skipped | `report/QA_T27_report.md`, commit `830bb2c`: cùng số | Pass |
| 9 | Chương 7, dòng thực nghiệm / run chính thức | 92 / 86 | `tracking/experiments.csv` → `T25_runs_stats.csv`: 92 / 86 | Pass |
| 10 | Chương 8, PrivateTest macro-F1 | 73.22% | `results/E7/TEST_DONE.json`, `test_macro_f1` làm tròn: 73.22% | Pass |
| 11 | Báo cáo ghép DOCX, Bảng 1.2 | 214 commit / 31 lượt kiểm thử; các dòng TV1–TV4 | Trích `word/document.xml`, đối chiếu đủ năm dòng với `T25_contribution.csv`: khớp | Pass |
| 12 | Chương 1/DOCX, nhiệm vụ R `28/28` | 28/28 | `tracking/tasks.csv` còn T28 `Pending`; đây là quy ước trình bày nhóm đã chấp thuận cho báo cáo trước khi đóng gói T28, không phải trạng thái tracking | Pass theo quy ước |
| 13 | PDF trang 6, Bảng 1.2 | TV2 62/8, TV4 50, tổng 214/31 | `T25_contribution.csv`, Chương 1 và DOCX cùng giá trị | Pass |
| 14 | PDF trang 47, mục 5.7 | QA cũ 127; QA cuối 128 passed | `QA_T27_report.md`, Chương 5 và DOCX cùng giá trị | Pass |
| 15 | PDF trang 60, Chương 8 | 128 test; PrivateTest macro-F1 73.22% | `QA_T27_report.md`, `TEST_DONE.json`, Chương 8 và DOCX cùng giá trị | Pass |
| 16 | Hình 6.1 `filters_weights_first.png` | Hình trọng số lớp đầu | SHA-256 `f10a6f7146023bd72b3b8540b8a2b2c9fa3dee556ae1494f81fd441390927011` khớp `tracking/figures_manifest.csv`; có CSV và script nguồn | Pass |
| 17 | Hình 6.6 `filters_grad_cam_correct.png` | Grad-CAM 7 ca đúng | SHA-256 `bd0cda044a4a7c341ec08399ba7154e02a27773f825679704ac88e9ae01e0f99` khớp manifest; có CSV và script nguồn | Pass |

PDF có 63 trang và các giá trị trên trích xuất được theo thứ tự thị giác. Đối chiếu với PDF ở HEAD cũ: chỉ trang 6, 47, 60 khác hình ảnh/văn bản; 60 trang khác không đổi. Ghi chú không chặn: luồng trích xuất *không sắp xếp* ở trang 60 đặt đoạn mới sau tiêu đề mục 2; hiển thị trang, tìm kiếm và `get_text(sort=True)` đúng. Không lấy thứ tự raw đó làm bằng chứng cho nội dung.

**Kết quả kiểm tra cục bộ:** 12/12 mục trong mẫu, 3/3 vị trí PDF, 2/2 hình đạt; số commit, lượt kiểm thử và run chính thức tự đếm khớp bảng sinh. Các cột commit/kiểm thử phản ánh snapshot `tracking/` ở `b4e5e02`, trước commit sửa báo cáo và commit kiểm thử này; bot cập nhật nhật ký sau push sẽ tạo snapshot mới. **Kết luận: Pass (KT-032).** Đã ghi dòng `KT-032` vào `tracking/test_log.csv` (tab `Kiem_thu` kéo về qua IMPORTDATA); `task_status.py` suy ra T25 `Completed → Verified`. Cột `trang_thai` trong `tracking/tasks.csv` để workflow `activity-log` tự cập nhật sau push, không sửa tay.
