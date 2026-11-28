# Biên bản Kiểm tra Checklist Mục 8 — Phần `C2` (Nhiệm vụ `T28`)

> **Phụ trách phối hợp (`C2`)**: TV3 – Võ Hoàng Bảo (`hoangbao2908`, MSSV `3123411030`)  
> **Phụ trách chính (`R`)**: TV2 – Lê Phan Nguyên Hải (Nhóm trưởng)  
> **Kiểm thử độc lập (`T`)**: TV4 – Huỳnh Viễn Thông  
> **Thời điểm kiểm tra**: `2026-11-28 18:48 (+07:00)`  
> **Commit đối chiếu trên GitHub**: `c1f8607` (`origin/main`, bao gồm `93931aa` `done(T28)`)  
> **Gói nộp kiểm chứng**: `BTLON02_Emotion_LePhanNguyenHai_3123411081.zip` (`239.96 MB` / `251,614,391 bytes`, `788` file, SHA-256 `451d6e9eb1fc9f4980eb8886b2b970df5beed449601ff5ff7991ec73d8e30b8c`)

---

## 1. Phạm vi kiểm tra của TV3 (`C2` theo `PHAN_CONG_NHIEM_VU.md` mục 5.3 & mục 8)

Kiểm tra độc lập phần **báo cáo (`report/`)**, **slide & kịch bản**, **nhật ký thực nghiệm (`NHAT_KY_THUC_NGHIEM.md`)**, **bộ `tracking/*.csv`**, đối chiếu đồng bộ với **Google Sheets (`Tien_do`, `Phan_cong`)** & **Google Slides**, và kiểm tra vệ sinh file cấm / weights trong gói zip nộp bài.

---

## 2. Kết quả đối chiếu từng hạng mục (Checklist mục 8)

| Hạng mục | Bằng chứng kiểm tra thực tế (`HEAD = c1f8607` & Zip) | Kết quả |
|---|---|:-:|
| **1. Báo cáo Word & PDF** | `report/BaoCao_BTL02.docx` (`3,118,556 B`, OOXML hợp lệ `43` mục, SHA-256 `fcfb13444042`) và `report/BaoCao_BTL02.pdf` (`3,508,144 B`, header `%PDF-`, SHA-256 `0536b402638e`); đủ `chuong1..8`, `phu_luc.md`, `tai_lieu_tham_khao.md`, `QA_T25_TV1.md`, `QA_T27_report.md`, `report/tables/` (`35` file bảng); bảng đóng góp `T25_contribution.csv` = `25.0% × 4 = 100.0%` (`28/28` nhiệm vụ `R`). Khớp 100% SHA-256 trong gói zip. | **PASS** |
| **2. Slide & Kịch bản thuyết trình** | `report/Slide_BTL02.pptx` (`1,817,455 B`, đọc trực tiếp bằng `python-pptx`: đủ `15` slide tỷ lệ `13.333 × 7.500"`, SHA-256 `79af603451cb`), `report/Slide_BTL02.pdf` (`1,441,756 B`, `%PDF-`, SHA-256 `458bf32ce3b2`), `report/KICH_BAN_THUYET_TRINH_VA_DEMO.md` (`27,302 B`, SHA-256 `ebc593f8cca3`). **Google Slides** (`1rAOrL4kSXBp65r5EwoXaGcyesiBAYCSxy3HibfPiPT4`) đã cập nhật lên đủ **15 slides**. Khớp 100% SHA-256 trong gói zip. | **PASS** |
| **3. Nhật ký thực nghiệm** | `report/NHAT_KY_THUC_NGHIEM.md` (`168,980 B`, `948` dòng, `88` mục run `### [...]`, SHA-256 `9a3937693281`), đã đạt `Verified` (`T24`). Khớp 100% SHA-256 trong gói zip. | **PASS** |
| **4. Hệ thống `tracking/*.csv` & Google Sheets** | Đủ `6/6` CSV đọc hợp lệ không lỗi schema: `tasks.csv` (`28` dòng: **`27 Verified · 1 Completed (T28)`**), `members.csv` (`4` dòng), `test_log.csv` (`32` dòng, `KT-001`–`KT-032`), `experiments.csv` (`92` dòng: `86 KEEP`, `6 DISCARD`), `figures_manifest.csv` (`116` dòng, **`116/116` hình tồn tại và khớp 100% SHA-256**), `activity_log.csv` (`225` dòng tự sinh từ git). Đối chiếu trực tiếp **Google Sheets** (`1mgwx1TGTL0fQewJ3Qto-K_HcZEUbgN4_P7Cy2-Z-aFk`, tab `Tien_do` & `Phan_cong` lúc `18:48`): khớp **100%** `27 Verified · 1 Completed (T28)`, `92 runs`, `116 hình`, `TV1 7P/3F`, `TV2 7P/0F`, `TV3 7P/0F`, `TV4 7P/0F`. | **PASS** |
| **5. Notebook tái lập & Thông tin nhóm** | Có đủ `notebooks/BTL02_reproducible.ipynb` và `3` bản đã chạy lưu output tại `notebooks/executed/_nb-001..003_s0.ipynb`; file `THONG_TIN_NHOM.txt` ở gốc gói zip ghi đúng 4 thành viên, MSSV, phân công `R`, tỷ lệ `25.0% × 4 = 100.0%` và đủ 2 link Google Sheets/Slides. | **PASS** |
| **6. Toàn vẹn Zip, file cấm & weights cuối** | `zipfile.testzip()` đạt **PASS (`0` file lỗi CRC)** trên toàn bộ `788` file; thư mục gốc đúng tên `BTLON02_Emotion_LePhanNguyenHai_3123411081/`. **`0` file `data/fer2013.csv`**, **`0` file/thư mục `.venv/`, `__pycache__/`, `.git/`, `.pytest_cache/`**, **`0` checkpoint trung gian hoặc `DISCARD`**. Giữ đúng **3/3 `best.pt` của ensemble cuối** (`results/E1/E1-014_s0/_s1/_s2/best.pt`, `77.92 MB`/file) và **cả 3/3 khớp 100% SHA-256 với `results/E7/TEST_DONE.json`** (`8c79c23b...`, `48f04f12...`, `5e20695d...`). | **PASS** |

---

## 3. Kết luận xác nhận `C2` (TV3)

Toàn bộ hạng mục thuộc phạm vi `C2` của TV3 (**báo cáo `.docx/.pdf`, slide `.pptx/.pdf` + Google Slides 15 trang, kịch bản demo, `NHAT_KY_THUC_NGHIEM.md`, `tracking/*.csv`, đối chiếu Google Sheets và vệ sinh gói zip**) đều **ĐẠT 100% yêu cầu mục 8 của `PHAN_CONG_NHIEM_VU.md`**. Sẵn sàng bàn giao `T` (TV4 – Huỳnh Viễn Thông) thực hiện nghiệm thu độc lập cuối cùng (`KT-033`) cho `T28`.
