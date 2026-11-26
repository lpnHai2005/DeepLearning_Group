# NHẬT KÝ THỰC NGHIỆM (NHẬT KÝ HUẤN LUYỆN & PHÁT TRIỂN MÔ HÌNH)

> **Môn học**: Deep Learning (Học sâu) – Bài tập lớn 02 (FER-2013)  
> **Người lập mẫu & phụ trách chính**: TV4 – Huỳnh Viễn Thông (Nhiệm vụ T24)  
> **Các thành viên tham gia**: TV1 – Đoàn Trung Kiên, TV2 – Lê Phan Nguyên Hải, TV3 – Võ Hoàng Bảo, TV4 – Huỳnh Viễn Thông  
> **Cập nhật lần cuối**: 2026-11-26  

---

## 1. Môi trường Thực nghiệm & Thông số Chung

### 1.1. Dữ liệu gốc & Tính toàn vẹn
- **Tập dữ liệu**: FER-2013 (Facial Expression Recognition 2013)
- **File**: `data/fer2013.csv`
- **SHA-256**: `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b` (xác nhận độc lập bởi TV4 tại KT-005)
- **Phân chia tập**:
  - `Training`: 28,709 ảnh (79.99%) – Dùng để huấn luyện mô hình.
  - `PublicTest` (Validation): 3,589 ảnh (10.00%) – Dùng để chọn siêu tham số, điều chỉnh Learning Rate và Early Stopping.
  - `PrivateTest` (Test): 3,589 ảnh (10.00%) – **BỊ KHÓA HOÀN TOÀN**, chỉ đánh giá đúng 1 lần duy nhất ở mô hình cuối cùng (`T19: --eval-test`).

### 1.2. Môi trường Phần cứng & Phần mềm (4 Máy Trạm của Nhóm)
| Thành viên | Vai trò | Hostname | GPU | Môi trường Python / PyTorch |
| :--- | :--- | :--- | :--- | :--- |
| **TV1 – Đoàn Trung Kiên** | Thành viên | `LAPTOP-RIMELLAL` | NVIDIA GeForce RTX 3050 Laptop GPU | Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4 |
| **TV2 – Lê Phan Nguyên Hải** | Nhóm trưởng | `DESKTOP-IOD0676` | NVIDIA GeForce RTX 4060 Laptop GPU | Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4 |
| **TV3 – Võ Hoàng Bảo** | Thành viên | `Hbao` | NVIDIA GeForce RTX 3050 Laptop GPU | Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4 |
| **TV4 – Huỳnh Viễn Thông** | Thành viên | `Administrator` | NVIDIA GeForce RTX 3060 Laptop GPU | Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4 |

### 1.3. Quy ước Ghi chép & Truy vết
- **Mã thực nghiệm (`exp_id`)**: `<nhóm>-<số thứ tự 3 chữ số>_s<seed>` (ví dụ: `baseline-002_s2`, `E1-001_s0`).
- **Nguồn số liệu**: Mục `- Kết quả:` được tự động trích xuất và đồng bộ từ `tracking/experiments.csv` thông qua `src/make_tables.py --update-log`. Tuyệt đối không nhập số liệu thủ công hoặc ước lượng.
- **Quyết định trọng số (`weights`)**: Hệ thống tự động lưu `best.pt` và quyết định `KEEP` / `DISCARD` theo quy tắc dung lượng và hiệu năng.

---

## 2. Mô hình Baseline (Mốc Đối chuẩn Đóng băng – T11)

### 2.1. Cấu hình Chuẩn Đóng băng (`configs/baseline.json`)
- **Kiến trúc**: CNN kiểu VGG gồm 3 khối Conv (6 lớp Conv):
  - Khối 1: 2× [Conv3×3 (64) + BatchNorm2d + ReLU] + MaxPool2d(2×2) + Dropout(0.25)
  - Khối 2: 2× [Conv3×3 (128) + BatchNorm2d + ReLU] + MaxPool2d(2×2) + Dropout(0.25)
  - Khối 3: 2× [Conv3×3 (256) + BatchNorm2d + ReLU] + MaxPool2d(2×2) + Dropout(0.25)
  - Phân loại (Classifier): Flatten ($256 \times 6 \times 6 = 9,216$) $\to$ Linear(9216, 512) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear(512, 7)
- **Tổng tham số**: $5,867,847$ tham số.
- **Siêu tham số huấn luyện**:
  - Tối đa: 100 epochs
  - Batch size: 128 (nạp toàn bộ dữ liệu vào RAM)
  - Optimizer: SGD momentum 0.9 (Nesterov=True), learning rate ban đầu $\eta = 0.01$, weight decay $5 \times 10^{-4}$
  - Lịch học: `ReduceLROnPlateau` (factor=0.5, patience=5) theo validation accuracy
  - Early stopping: Dừng sớm nếu sau 15 epochs không cải thiện
  - Tăng cường ảnh: Random Horizontal Flip ($p=0.5$) + RandomAffine (xoay $\pm 10^\circ$, dịch $\pm 10\%$, tỉ lệ $0.9 - 1.1$)
  - Tăng tốc: PyTorch AMP (Automatic Mixed Precision - FP16)
- **SHA-256 (`configs/baseline.json`)**: `255c87e91696427bb3af9f1623623a30e489bd1693f37b6faae6bde7322968c5` (đã đóng băng).

### 2.2. Kết quả Tổng hợp 3 Seed Baseline (`results/baseline/baseline_summary.json`)
- **Validation Accuracy**: $\mathbf{68.15\% \pm 0.75\%}$ ($0.6815 \pm 0.0075$)
- **Validation Macro-F1**: $\mathbf{64.08\% \pm 1.95\%}$ ($0.6408 \pm 0.0195$)
- **Validation Loss**: $0.9109 \pm 0.0094$
- **Dung sai tái lập khoa học ($\sigma$)**: $\mathbf{0.0075}$ ($0.75\%$). Mọi thí nghiệm khảo sát $E1 \to E6$ nếu có độ chênh lệch nhỏ hơn dung sai này cần được chạy thêm 2 seed trước khi đưa ra kết luận.

---

## 3. Nhật ký Chi tiết Từng Lần Chạy (Run Log)

### [baseline-002_s0] Baseline CNN (Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: Mô hình Baseline CNN kiểu VGG 3 khối có BatchNorm huấn luyện 100 epoch với SGD lr=0.01 kèm Plateau scheduler sẽ bão hòa ở khoảng epoch 80-95 và đạt độ chính xác trên 67%.
- Thay đổi so với baseline: Mốc chuẩn (không đổi).
- Lệnh: `python src/run_experiment.py --config configs/baseline.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.915008519412615, val_acc=0.6815268877124547, val_macro_f1=0.6416322850865286, best_epoch=99, time_s=1548.225, run_dir=results/baseline/baseline-002_s0
- Nhận xét: Huấn luyện hoàn thành trọn vẹn 100 epochs trên RTX 4060 (25.8 phút). Val Acc đạt đỉnh 68.15% ở epoch 99, khớp hoàn toàn với run thăm dò `_probe-001_s0`. Learning rate giảm dần từ 0.01 xuống 0.00125.
- Quyết định: Giữ (`KEEP` weights, `best.pt` 22.4 MB). Đạt yêu cầu làm mốc chuẩn Seed 0.

### [baseline-002_s1] Baseline CNN (Seed 1) — TV1 – Đoàn Trung Kiên – 2026-11-25
- Giả thuyết: Khởi tạo hạt giống ngẫu nhiên số 1 (Seed 1) trên máy khác (RTX 3050) kiểm tra tính tái lập độc lập của cấu hình Baseline.
- Thay đổi so với baseline: Seed = 1.
- Lệnh: `python src/run_experiment.py --config configs/baseline.json --member TV1 --seeds 1`
- Kết quả: status=Completed, best_val_loss=0.9175978793709948, val_acc=0.6740039008080245, val_macro_f1=0.6209245313912722, best_epoch=93, time_s=3085.588, run_dir=results/baseline/baseline-002_s1
- Nhận xét: Chạy xong 100 epochs trên RTX 3050 (~51.4 phút). Val Acc đạt cực đại 67.40% tại epoch 93, Macro-F1 đạt 62.09%. Khoảng cách train/val không quá lớn, mô hình hội tụ ổn định.
- Quyết định: Giữ (`KEEP` weights, `best.pt` 22.4 MB). Đạt yêu cầu làm mốc chuẩn Seed 1.

### [baseline-002_s2] Baseline CNN (Seed 2) — TV4 – Huỳnh Viễn Thông – 2026-11-25
- Giả thuyết: Khởi tạo hạt giống ngẫu nhiên số 2 (Seed 2) trên máy trạm RTX 3060 để hoàn thiện bộ 3 seed độc lập cho Baseline.
- Thay đổi so với baseline: Seed = 2.
- Lệnh: `python src/run_experiment.py --config configs/baseline.json --member TV4 --seeds 2`
- Kết quả: status=Completed, best_val_loss=0.9001500216590624, val_acc=0.6890498746168849, val_macro_f1=0.6598913756852982, best_epoch=99, time_s=3173.867, run_dir=results/baseline/baseline-002_s2
- Nhận xét: Huấn luyện 100 epochs trên RTX 3060 (~52.5 phút). Mô hình đạt hiệu năng tốt nhất trong cả 3 seed với Val Acc đạt 68.90% và Macro-F1 đạt 65.99% ở epoch 99. Loss tiếp tục giảm đều đặn.
- Quyết định: Giữ (`KEEP` weights, `best.pt` 22.4 MB). Đạt yêu cầu làm mốc chuẩn Seed 2.

---

## 4. Khung Nhật ký Các Thí nghiệm Cải tiến Tiếp theo (E1 – E7)

> **Hướng dẫn cho các thành viên**: Mỗi khi chạy xong một run bằng lệnh `python src/run_experiment.py`, thành viên phụ trách thêm một mục theo cú pháp mẫu dưới đây. Sau đó chạy `python src/make_tables.py --update-log report/NHAT_KY_THUC_NGHIEM.md` để tự động điền dòng `- Kết quả:`.

### [E1-001_s0] Khảo sát kiến trúc SMALL_CNN — TV1 – <chưa thực hiện>
- Giả thuyết: Mô hình SMALL_CNN chỉ có 2 khối Conv và không có BatchNorm sẽ huấn luyện nhanh hơn nhưng độ chính xác thấp hơn Baseline đáng kể (dự kiến < 60%).
- Thay đổi so với baseline: `model = "small"` (không BN, 4 Conv, FC 1024).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only small`
- Kết quả: status=Completed, best_val_loss=0.9094449157799874, val_acc=0.6915575369183616, val_macro_f1=0.6731403436946511, best_epoch=96, time_s=2133.077, run_dir=results/E1/E1-001_s0
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-002_s0] Khảo sát kiến trúc MEDIUM_01 — TV1 – <chưa thực hiện>
- Giả thuyết: Tái hiện mô hình MEDIUM_01 từ notebook mẫu (12.1M tham số, không BN) xem mức độ quá khớp (overfitting) khi thiếu chuẩn hóa theo lô.
- Thay đổi so với baseline: `model = "medium_01"`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only medium_01`
- Kết quả: status=Completed, best_val_loss=0.9161594292020758, val_acc=0.6951797157982725, val_macro_f1=0.6788904667222425, best_epoch=100, time_s=1691.147, run_dir=results/E1/E1-002_s0
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-003_s0] Khảo sát kiến trúc LARGE_CNN — TV1 – <chưa thực hiện>
- Giả thuyết: Tái hiện mô hình LARGE_CNN (20.4M tham số, 16 lớp Conv) để đánh giá chi phí tính toán và hiện tượng biến mất gradient khi thiếu BatchNorm.
- Thay đổi so với baseline: `model = "large"`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only large`
- Kết quả: status=Completed, best_val_loss=1.8107485001915573, val_acc=0.2493730844246308, val_macro_f1=0.0570281636294125, best_epoch=1, time_s=326.687, run_dir=results/E1/E1-003_s0
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-004_s0] Khảo sát loại bỏ BatchNorm (No BN) — TV3 – <chưa thực hiện>
- Giả thuyết: Khi loại bỏ toàn bộ các lớp BatchNorm trong Baseline, gradient sẽ kém ổn định hơn, tốc độ hội tụ chậm hơn và validation accuracy giảm ít nhất 3-5%.
- Thay đổi so với baseline: `bn = False`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV3 --only no_bn`
- Kết quả: status=Completed, best_val_loss=1.8107463234013648, val_acc=0.2493730844246308, val_macro_f1=0.0570281636294125, best_epoch=1, time_s=397.254, run_dir=results/E1/E1-004_s0
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-005_s0] Khảo sát Global Average Pooling (GAP) — TV3 – <chưa thực hiện>
- Giả thuyết: Thay thế Flatten bằng Global Average Pooling (GAP) sẽ giảm mạnh số lượng tham số ở phần phân loại và giúp mô hình chống overfitting tốt hơn.
- Thay đổi so với baseline: `head = "gap"`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV3 --only gap`
- Kết quả: status=Completed, best_val_loss=0.9146933610075928, val_acc=0.6701030927835051, val_macro_f1=0.6329196107692682, best_epoch=90, time_s=1397.591, run_dir=results/E1/E1-005_s0
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-007_s1] Khảo sát loại bỏ BatchNorm (`baseline_no_bn`, Seed 1 — C2) — TV3 – Võ Hoàng Bảo – 2026-11-25
- Giả thuyết: Kiểm chứng độc lập trên máy khác (RTX 3050, seed 1) xem mức tăng bất ngờ của `baseline_no_bn` ở `E1-007_s0` (`0.6907` trên RTX 4060 của TV2) có tái lập ổn định hay chỉ là dao động 1 seed.
- Thay đổi so với baseline: `model_kwargs = {"bn": false}` (`5,866,951` tham số, giảm 896 tham số BN), `seed = 1`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV3 --only baseline_no_bn baseline_gap --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.9074415885561786, val_acc=0.6940651992198384, val_macro_f1=0.6680916497711471, best_epoch=100, time_s=1973.071, run_dir=results/E1/E1-007_s1
- Nhận xét: Hoàn tất trọn vẹn 100 epochs trên RTX 3050 (~32.9 phút, ~19.6s/ep). `val_acc` đạt **69.41%** (+1.25% so với trung bình baseline 68.15%) và `val_macro_f1` đạt **66.81%** (+2.73% so với baseline 64.08%), `best_val_loss = 0.9074` thấp hơn baseline (`0.9109`). Không xảy ra hiện tượng triệt tiêu gradient trên kiến trúc 6 lớp Conv này.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.39 MB). Xác nhận xu hướng vượt baseline ở seed 1.

### [E1-007_s2] Khảo sát loại bỏ BatchNorm (`baseline_no_bn`, Seed 2 — C2) — TV3 – Võ Hoàng Bảo – 2026-11-25
- Giả thuyết: Hoàn thiện bộ 3 seed độc lập (s0 TV2 + s1, s2 TV3) cho `baseline_no_bn` để kết luận có ý nghĩa thống kê so với dung sai baseline (`0.0075`).
- Thay đổi so với baseline: `model_kwargs = {"bn": false}`, `seed = 2`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV3 --only baseline_no_bn baseline_gap --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.90168928442376, val_acc=0.6974087489551407, val_macro_f1=0.6726705298117717, best_epoch=97, time_s=2775.095, run_dir=results/E1/E1-007_s2
- Nhận xét: Đạt **69.74%** `val_acc`, **67.27%** `val_macro_f1`, `best_val_loss = 0.9017` tại epoch 97/100 (~46.3 phút). Tổng hợp đủ 3 seed (`E1-007_s0/s1/s2`): `val_acc = 0.6941 ± 0.0033` (**+0.0125**, cao hơn rõ rệt ngưỡng `std = 0.0075` của baseline), `val_macro_f1 = 0.6682 ± 0.0044` (+0.0274 so với baseline), `best_val_loss = 0.9059`. Trái với giả thuyết ban đầu, mạng VGG 6 lớp Conv + Kaiming init đủ nông để hội tụ không cần BN (khác với `medium_01` 12 lớp và `large` 16 lớp bị sập ở `0.2494`), đồng thời bỏ BN tránh xung đột thống kê giữa Dropout và BatchNorm nên vừa tăng độ chính xác vừa giảm một nửa độ lệch chuẩn giữa các seed (`0.0033` vs `0.0075`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.39 MB). Chốt kết luận 3 seed cho `baseline_no_bn`: **vượt baseline có ý nghĩa thống kê**, là ứng viên kiến trúc mạnh cho các bước kế tiếp / `E7`.

### [E1-008_s1] Khảo sát Global Average Pooling (`baseline_gap`, Seed 1 — C2) — TV3 – Võ Hoàng Bảo – 2026-11-25
- Giả thuyết: Kiểm chứng độc lập trên RTX 3050 (seed 1) xem `baseline_gap` (`1.28M` tham số, giảm 78.2% so với baseline) có duy trì được hiệu năng tương đương baseline như `E1-008_s0` (`0.6829`) hay không.
- Thay đổi so với baseline: `model_kwargs = {"head": "gap"}` (`1,280,327` tham số thay vì `5,867,847`), `seed = 1`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV3 --only baseline_no_bn baseline_gap --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.9081022443795276, val_acc=0.6834772917247144, val_macro_f1=0.6460131237868115, best_epoch=100, time_s=3356.223, run_dir=results/E1/E1-008_s1
- Nhận xét: Hoàn tất 100 epochs trên RTX 3050 (~55.9 phút). `val_acc` đạt **68.35%** và `val_macro_f1` đạt **64.60%**, `best_val_loss = 0.9081`, khớp sát với `E1-008_s0` (`68.29%`, `64.49%`). Khoảng cách `train_acc_cuoi − val_acc = +0.0396` gần tương đương baseline (`+0.0397`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` chỉ 4.90 MB). Xác nhận độ ổn định tốt ở seed 1.

### [E1-008_s2] Khảo sát Global Average Pooling (`baseline_gap`, Seed 2 — C2) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: Hoàn thiện bộ 3 seed độc lập cho `baseline_gap` để đánh giá đánh đổi giữa số tham số (giảm 78.2%) và độ chính xác.
- Thay đổi so với baseline: `model_kwargs = {"head": "gap"}`, `seed = 2`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV3 --only baseline_no_bn baseline_gap --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.9059900776461062, val_acc=0.685427695736974, val_macro_f1=0.6523252167324272, best_epoch=98, time_s=2067.558, run_dir=results/E1/E1-008_s2
- Nhận xét: Hoàn tất 100 epochs (~34.5 phút, ~20.5s/ep), đạt **68.54%** `val_acc`, **65.23%** `val_macro_f1`, `best_val_loss = 0.9060` tại epoch 98. Tổng hợp đủ 3 seed (`E1-008_s0/s1/s2`): `val_acc = 0.6839 ± 0.0013` (**+0.0024**, trong ngưỡng `< std baseline = 0.0075`), `val_macro_f1 = 0.6478 ± 0.0040`, `best_val_loss = 0.9077`. Đúng với giả thuyết: thay `Flatten` bằng `GAP` cắt giảm **78.2%** tham số (`5.87M → 1.28M`, trọng số `22.39 MB → 4.90 MB`) mà hiệu năng vẫn tương đương (nhỉnh nhẹ) baseline, đặc biệt độ lệch chuẩn giữa 3 seed thu hẹp mạnh còn `0.0013` (~1/6 của baseline).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 4.90 MB). Chốt kết luận 3 seed cho `baseline_gap`: **hiệu quả tham số vượt trội** (tương đương baseline với 21.8% số tham số), đáng thử kết hợp với `no_bn` ở các thí nghiệm sau / `E7`.

### [E2-001_s0] Khảo sát hàm kích hoạt LeakyReLU — TV3 – <chưa thực hiện>
- Giả thuyết: LeakyReLU cho phép gradient nhỏ đi qua khi đầu vào âm, hạn chế hiện tượng Dying ReLU ở các lớp sâu.
- Thay đổi so với baseline: `activation = "leaky_relu"`.
- Lệnh: `python src/run_experiment.py --config configs/E2_activation.json --member TV3 --only leaky_relu`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E2-002_s0] Khảo sát hàm kích hoạt Sigmoid (Vanishing Gradient) — TV4 – <chưa thực hiện>
- Giả thuyết: Dùng Sigmoid thay cho ReLU ở mạng sâu 6 lớp Conv sẽ gặp hiện tượng triệt tiêu gradient nghiêm trọng (đặc biệt khi không có BN), mô hình gần như không học được.
- Thay đổi so với baseline: `activation = "sigmoid"`.
- Lệnh: `python src/run_experiment.py --config configs/E2_activation.json --member TV4 --only sigmoid`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E3-001_s0] Khảo sát AdamW lr=0.001 — TV4 – 2026-11-25
- Giả thuyết: AdamW kết hợp với lịch học Plateau sẽ hội tụ nhanh hơn SGD trong 30 epoch đầu nhưng độ chính xác cực đại có thể tương đương hoặc hơi thấp hơn một chút so với SGD Nesterov.
- Thay đổi so với baseline: `optimizer = "adamw"`, `lr = 0.001`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only adamw_1e-3`
- Kết quả: status=Completed, best_val_loss=0.9626304985720257, val_acc=0.6723321259403734, val_macro_f1=0.5877659010880202, best_epoch=99, time_s=3620.904, run_dir=results/E3/E3-001_s0
- Nhận xét: Val Acc đạt 67.23% (thấp hơn baseline 0.92%), Val Macro-F1 đạt 58.78% (giảm mạnh 5.30% so với baseline 64.08%). AdamW với lr=1e-3 hội tụ chậm hơn và tổng quát hóa kém hơn SGD Nesterov; phân bố dự đoán thiên lệch vào các lớp đông mẫu hơn. Best epoch rơi vào epoch 99/100.
- Quyết định: Giữ weights (KEEP) theo đúng luật kỹ thuật (OK). Kết luận AdamW 1e-3 chưa tối ưu bằng SGD Nesterov, tiếp tục khảo sát các biến thể tối ưu khác trong E3.

### [E3-002_s0] Khảo sát AdamW lr=0.0003 — TV4 – 2026-11-25
- Giả thuyết: Giảm tốc độ học của AdamW từ 1e-3 xuống 3e-4 sẽ giúp hàm mất mát hội tụ mượt mà hơn, tránh bước nhảy quá lớn và cải thiện khả năng tổng quát hóa so với adamw_1e-3.
- Thay đổi so với baseline: `optimizer = "adamw"`, `lr = 0.0003`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only adamw_3e-4`
- Kết quả: status=Completed, best_val_loss=0.917564547271176, val_acc=0.6703817219281136, val_macro_f1=0.5630497131569857, best_epoch=91, time_s=4999.794, run_dir=results/E3/E3-002_s0
- Nhận xét: Best Val Loss được cải thiện rõ rệt từ 0.9626 (adamw_1e-3) xuống 0.9176 (xấp xỉ mốc 0.9109 của Baseline), chứng minh giảm lr giúp ổn định loss. Tuy nhiên Val Acc đạt 67.04% và Macro-F1 đạt 56.30% (tiếp tục giảm 2.48% so với adamw_1e-3 và kém Baseline 7.78%). Hoàn tất 100 epochs, best epoch đạt tại epoch 91.
- Quyết định: Giữ weights (KEEP, mã OK). Hoàn tất nhóm khảo sát AdamW: cả hai mức lr 1e-3 và 3e-4 đều kém hơn Baseline SGD Nesterov về độ chính xác và khả năng nhận diện các lớp cảm xúc thiểu số.

### [E3-003_s0] Khảo sát Lịch học Cosine Annealing (sched_cosine) — TV4 – 2026-11-26
- Giả thuyết: Lịch học Cosine Annealing giảm learning rate mượt mà theo chu kỳ nửa sóng cosin, giúp mô hình thoát khỏi cực tiểu địa phương tốt hơn lịch học Plateau và đạt độ chính xác tương đương hoặc vượt Baseline.
- Thay đổi so với baseline: `scheduler = "cosine"`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only sched_cosine`
- Kết quả: status=Completed, best_val_loss=0.9126208526704862, val_acc=0.6770688213987183, val_macro_f1=0.6180704523759227, best_epoch=85, time_s=2709.745, run_dir=results/E3/E3-003_s0
- Nhận xét: Huấn luyện hoàn thành 100 epochs với tốc độ rất nhanh (26.7s/epoch, tổng thời gian 44.5 phút trên RTX 3060). Val Acc đạt 67.71% (chênh lệch -0.0045 so với baseline, nằm hoàn toàn trong ngưỡng dung sai tái lập < std baseline = 0.0075). Val Macro-F1 đạt 61.81% và Best Val Loss đạt 0.9126 (sát mốc Baseline 0.9109). Kết quả này vượt trội hơn rõ rệt so với nhóm AdamW (67.04% - 67.23%), cho thấy lịch học Cosine Annealing kết hợp SGD Nesterov hoạt động rất ổn định và hội tụ mượt mà.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Ghi nhận Cosine Annealing là một phương án lịch học xuất sắc có thể cân nhắc cho mô hình cuối E7.

### [E3-004_s0] Khảo sát Lịch học StepLR (sched_step) — TV4 – 2026-11-26
- Giả thuyết: StepLR giảm learning rate 50% sau mỗi 20 epoch theo chu kỳ bậc thang định trước; việc giảm bước cứng nhắc có thể khiến mô hình bị mất đà thoát khỏi cực tiểu địa phương so với lịch học thích ứng Plateau hoặc mượt mà Cosine.
- Thay đổi so với baseline: `scheduler = "step"`, `step_size = 20`, `gamma = 0.5`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only sched_step`
- Kết quả: status=Completed, best_val_loss=0.9208383716956324, val_acc=0.6648091390359432, val_macro_f1=0.5918614905456292, best_epoch=92, time_s=2564.179, run_dir=results/E3/E3-004_s0
- Nhận xét: Huấn luyện hoàn thành 100 epochs với tốc độ rất nhanh (25.4s/epoch, tổng thời gian 42.3 phút). Val Acc đạt 66.48% (thấp hơn baseline 1.67%) và Val Macro-F1 đạt 59.19% (thấp hơn baseline 4.89%). So sánh 3 lịch học trên cùng optimizer SGD Nesterov: Cosine (67.71%) và Plateau (68.15%) vượt trội hơn hẳn StepLR (66.48%). Việc giảm lr đột ngột ở epoch 20, 40, 60, 80 làm hạn chế khả năng tìm kiếm nghiệm tối ưu ở các thung lũng phẳng rộng.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Kết luận StepLR kém hiệu quả hơn Plateau và Cosine Annealing trên bài toán này.

### [E3-005_s0] Khảo sát Lịch học Constant LR (sched_constant) — TV4 – 2026-11-26
- Giả thuyết: Giữ nguyên learning rate không đổi (0.01) trong suốt 100 epoch sẽ giúp mô hình duy trì tốc độ cập nhật trọng số, tuy nhiên có nguy cơ dao động quanh cực tiểu ở các epoch cuối thay vì hội tụ sâu.
- Thay đổi so với baseline: `scheduler = "constant"`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only sched_constant`
- Kết quả: status=Completed, best_val_loss=0.9137963193743034, val_acc=0.6759543048202842, val_macro_f1=0.6376443348653521, best_epoch=95, time_s=2497.091, run_dir=results/E3/E3-005_s0
- Nhận xét: Hoàn thành 100 epochs (24.7s/epoch). Val Acc đạt 67.60% (chênh lệch -0.0056 so với baseline, nằm trong dung sai $\sigma=0.0075$), Val Macro-F1 đạt 63.76% và Best Val Loss đạt 0.9138 tại epoch 95. Việc giữ LR cố định 0.01 vẫn cho kết quả khá tốt nhờ momentum của SGD Nesterov, nhưng do không giảm learning rate ở cuối nên không thể hội tụ mịn bằng Plateau hay OneCycle.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB).

### [E3-006_s0] Khảo sát Lịch học OneCycleLR (sched_onecycle) — TV4 – 2026-11-26
- Giả thuyết: OneCycleLR tăng dần learning rate từ nhỏ lên đỉnh (warmup) rồi giảm dần theo cosine xuống rất nhỏ, giúp mô hình vượt qua các cực tiểu địa phương gồ ghề trong giai đoạn đầu và hội tụ vào thung lũng phẳng tối ưu ở cuối quá trình huấn luyện.
- Thay đổi so với baseline: `scheduler = "onecycle"`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only sched_onecycle`
- Kết quả: status=Completed, best_val_loss=0.8997333023866327, val_acc=0.6862635831707997, val_macro_f1=0.6584174598142835, best_epoch=100, time_s=2533.215, run_dir=results/E3/E3-006_s0
- Nhận xét: Xuất sắc vượt qua Baseline! Val Acc đạt 68.63% (+0.48% so với baseline 68.15%), Val Macro-F1 đạt 65.84% (+1.76% so với baseline 64.08%), và Best Val Loss giảm sâu xuống 0.8997 (phá mốc 0.90). Cơ chế super-convergence của OneCycleLR chứng minh hiệu quả vượt trội trên FER2013, đưa mô hình đạt phong độ cao nhất tại đúng epoch 100.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Đề xuất đưa cấu hình OneCycleLR làm ứng viên sáng giá cho mô hình cuối E7.

### [E3-007_s0] Khảo sát Tối ưu hóa RMSprop lr=0.001 (rmsprop_1e-3) — TV4 – 2026-11-26
- Giả thuyết: RMSprop điều chỉnh learning rate thích ứng theo căn bậc hai trung bình bình phương gradient (EMA gradient bình phương), giúp cân bằng bước cập nhật giữa các hướng dốc và phẳng, có thể hội tụ nhanh và đạt loss thấp.
- Thay đổi so với baseline: `optimizer = "rmsprop"`, `lr = 0.001`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only rmsprop_1e-3`
- Kết quả: status=Completed, best_val_loss=0.8809435215458693, val_acc=0.6857063248815826, val_macro_f1=0.6620334209448906, best_epoch=95, time_s=2526.565, run_dir=results/E3/E3-007_s0
- Nhận xét: Kỷ lục Val Loss thấp nhất trong toàn bộ E3! Best Val Loss giảm kỷ lục xuống 0.8809 (thấp hơn baseline 0.9109 tới -0.0300). Val Acc đạt 68.57% (+0.42% so với baseline) và Val Macro-F1 đạt tới 66.20% (+2.12% so với baseline). RMSprop với lr=1e-3 khắc phục hoàn toàn nhược điểm thiên lệch lớp của AdamW, mang lại khả năng tổng quát hóa xuất sắc trên các lớp cảm xúc thiểu số.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Ghi nhận RMSprop lr=1e-3 là thuật toán tối ưu tốt nhất về độ mượt của hàm mất mát.

### [E3-008_s0] Khảo sát Tối ưu hóa RMSprop lr=0.0003 (rmsprop_3e-4) — TV4 – 2026-11-26
- Giả thuyết: Giảm learning rate của RMSprop xuống 3e-4 để khảo sát tính nhạy cảm với bước học; lr nhỏ hơn có thể giúp hội tụ an toàn nhưng tốc độ tiếp cận nghiệm tối ưu có thể chậm hơn.
- Thay đổi so với baseline: `optimizer = "rmsprop"`, `lr = 0.0003`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only rmsprop_3e-4`
- Kết quả: status=Completed, best_val_loss=0.90139705034045, val_acc=0.6809696294232377, val_macro_f1=0.6332322205170513, best_epoch=87, time_s=2521.717, run_dir=results/E3/E3-008_s0
- Nhận xét: Hoàn thành 100 epochs (25.0s/epoch). Val Acc đạt 68.10% (tương đương baseline 68.15%, sai khác chỉ -0.05%), Best Val Loss đạt 0.9014 (tốt hơn baseline 0.9109) và Macro-F1 đạt 63.32%. So với rmsprop_1e-3, mức lr 3e-4 hội tụ an toàn nhưng không bứt phá mạnh bằng 1e-3.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Khẳng định họ thuật toán RMSprop rất phù hợp cho mạng CNN phân loại cảm xúc.

### [E3-009_s0] Khảo sát Kích thước Batch bs=64 (bs_64) — TV4 – 2026-11-26
- Giả thuyết: Giảm batch size từ 128 xuống 64 sẽ tăng nhiễu gradient ngẫu nhiên (stochastic noise), đóng vai trò như một cơ chế điều hòa tự nhiên (implicit regularization) giúp mô hình thoát khỏi cực tiểu sắc nhọn (sharp minima) và cải thiện khả năng tổng quát hóa trên tập Validation.
- Thay đổi so với baseline: `batch_size = 64`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only bs_64`
- Kết quả: status=Completed, best_val_loss=0.905830695789217, val_acc=0.6923934243521872, val_macro_f1=0.6642436768620258, best_epoch=82, time_s=2499.457, run_dir=results/E3/E3-009_s0
- Nhận xét: ĐỘT PHÁ KỶ LỤC CỦA TOÀN BỘ E3! Val Acc đạt 69.24% (+1.09% so với baseline 68.15%, cao hơn std baseline 0.0075 đáng kể), Val Macro-F1 đạt đỉnh 66.42% (+2.34% so với baseline 64.08%) và Best Val Loss đạt 0.9058 tại epoch 82. Giả thuyết về lợi thế của batch size nhỏ trong việc tăng khả năng tổng quát hóa hoàn toàn được xác thực bằng số liệu thực nghiệm.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Ứng viên số 1 về batch size cho mô hình cuối E7.

### [E3-010_s0] Khảo sát Kích thước Batch bs=256 (bs_256) — TV4 – 2026-11-26
- Giả thuyết: Tăng batch size lên 256 sẽ giúp ước lượng gradient ổn định hơn trong mỗi bước và tăng tốc độ xử lý phần cứng, tuy nhiên có thể dẫn đến hiện tượng "generalization gap" khiến Val Acc giảm sút so với batch size nhỏ.
- Thay đổi so với baseline: `batch_size = 256`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only bs_256`
- Kết quả: status=Completed, best_val_loss=0.91694409404169, val_acc=0.6726107550849819, val_macro_f1=0.6298344029609881, best_epoch=94, time_s=2477.546, run_dir=results/E3/E3-010_s0
- Nhận xét: Thời gian mỗi epoch nhanh nhất (24.4s/epoch, tổng 2477s). Tuy nhiên Val Acc giảm xuống 67.26% (thấp hơn baseline 0.89%), Val Macro-F1 đạt 62.98% và Val Loss đạt 0.9169. Kết quả đối sánh giữa bs=64 (69.24%), bs=128 (68.15%) và bs=256 (67.26%) thể hiện quy luật đơn điệu rõ ràng: batch size càng lớn thì độ chính xác tổng quát hóa càng giảm trên bài toán FER2013.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Kết luận chọn batch size 64 hoặc 128 là tối ưu, không nên dùng batch size 256.

### [E3-005_s0] Khảo sát Lịch học Constant LR (sched_constant) — TV4 – 2026-11-26
- Giả thuyết: Giữ nguyên learning rate không đổi (0.01) trong suốt 100 epoch sẽ giúp mô hình duy trì tốc độ cập nhật trọng số, tuy nhiên có nguy cơ dao động quanh cực tiểu ở các epoch cuối thay vì hội tụ sâu.
- Thay đổi so với baseline: `scheduler = "constant"`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only sched_constant`
- Kết quả: status=Completed, best_val_loss=0.9137963193743034, val_acc=0.6759543048202842, val_macro_f1=0.6376443348653521, best_epoch=95, time_s=2497.091, run_dir=results/E3/E3-005_s0
- Nhận xét: Hoàn thành 100 epochs (24.7s/epoch). Val Acc đạt 67.60% (chênh lệch -0.0056 so với baseline, nằm trong dung sai $\sigma=0.0075$), Val Macro-F1 đạt 63.76% và Best Val Loss đạt 0.9138 tại epoch 95. Việc giữ LR cố định 0.01 vẫn cho kết quả khá tốt nhờ momentum của SGD Nesterov, nhưng do không giảm learning rate ở cuối nên không thể hội tụ mịn bằng Plateau hay OneCycle.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB).

### [E3-006_s0] Khảo sát Lịch học OneCycleLR (sched_onecycle) — TV4 – 2026-11-26
- Giả thuyết: OneCycleLR tăng dần learning rate từ nhỏ lên đỉnh (warmup) rồi giảm dần theo cosine xuống rất nhỏ, giúp mô hình vượt qua các cực tiểu địa phương gồ ghề trong giai đoạn đầu và hội tụ vào thung lũng phẳng tối ưu ở cuối quá trình huấn luyện.
- Thay đổi so với baseline: `scheduler = "onecycle"`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only sched_onecycle`
- Kết quả: status=Completed, best_val_loss=0.8997333023866327, val_acc=0.6862635831707997, val_macro_f1=0.6584174598142835, best_epoch=100, time_s=2533.215, run_dir=results/E3/E3-006_s0
- Nhận xét: Xuất sắc vượt qua Baseline! Val Acc đạt 68.63% (+0.48% so với baseline 68.15%), Val Macro-F1 đạt 65.84% (+1.76% so với baseline 64.08%), và Best Val Loss giảm sâu xuống 0.8997 (phá mốc 0.90). Cơ chế super-convergence của OneCycleLR chứng minh hiệu quả vượt trội trên FER2013, đưa mô hình đạt phong độ cao nhất tại đúng epoch 100.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Đề xuất đưa cấu hình OneCycleLR làm ứng viên sáng giá cho mô hình cuối E7.

### [E3-007_s0] Khảo sát Tối ưu hóa RMSprop lr=0.001 (rmsprop_1e-3) — TV4 – 2026-11-26
- Giả thuyết: RMSprop điều chỉnh learning rate thích ứng theo căn bậc hai trung bình bình phương gradient (EMA gradient bình phương), giúp cân bằng bước cập nhật giữa các hướng dốc và phẳng, có thể hội tụ nhanh và đạt loss thấp.
- Thay đổi so với baseline: `optimizer = "rmsprop"`, `lr = 0.001`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only rmsprop_1e-3`
- Kết quả: status=Completed, best_val_loss=0.8809435215458693, val_acc=0.6857063248815826, val_macro_f1=0.6620334209448906, best_epoch=95, time_s=2526.565, run_dir=results/E3/E3-007_s0
- Nhận xét: Kỷ lục Val Loss thấp nhất trong toàn bộ E3! Best Val Loss giảm kỷ lục xuống 0.8809 (thấp hơn baseline 0.9109 tới -0.0300). Val Acc đạt 68.57% (+0.42% so với baseline) và Val Macro-F1 đạt tới 66.20% (+2.12% so với baseline). RMSprop với lr=1e-3 khắc phục hoàn toàn nhược điểm thiên lệch lớp của AdamW, mang lại khả năng tổng quát hóa xuất sắc trên các lớp cảm xúc thiểu số.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Ghi nhận RMSprop lr=1e-3 là thuật toán tối ưu tốt nhất về độ mượt của hàm mất mát.

### [E3-008_s0] Khảo sát Tối ưu hóa RMSprop lr=0.0003 (rmsprop_3e-4) — TV4 – 2026-11-26
- Giả thuyết: Giảm learning rate của RMSprop xuống 3e-4 để khảo sát tính nhạy cảm với bước học; lr nhỏ hơn có thể giúp hội tụ an toàn nhưng tốc độ tiếp cận nghiệm tối ưu có thể chậm hơn.
- Thay đổi so với baseline: `optimizer = "rmsprop"`, `lr = 0.0003`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only rmsprop_3e-4`
- Kết quả: status=Completed, best_val_loss=0.90139705034045, val_acc=0.6809696294232377, val_macro_f1=0.6332322205170513, best_epoch=87, time_s=2521.717, run_dir=results/E3/E3-008_s0
- Nhận xét: Hoàn thành 100 epochs (25.0s/epoch). Val Acc đạt 68.10% (tương đương baseline 68.15%, sai khác chỉ -0.05%), Best Val Loss đạt 0.9014 (tốt hơn baseline 0.9109) và Macro-F1 đạt 63.32%. So với rmsprop_1e-3, mức lr 3e-4 hội tụ an toàn nhưng không bứt phá mạnh bằng 1e-3.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Khẳng định họ thuật toán RMSprop rất phù hợp cho mạng CNN phân loại cảm xúc.

### [E3-009_s0] Khảo sát Kích thước Batch bs=64 (bs_64) — TV4 – 2026-11-26
- Giả thuyết: Giảm batch size từ 128 xuống 64 sẽ tăng nhiễu gradient ngẫu nhiên (stochastic noise), đóng vai trò như một cơ chế điều hòa tự nhiên (implicit regularization) giúp mô hình thoát khỏi cực tiểu sắc nhọn (sharp minima) và cải thiện khả năng tổng quát hóa trên tập Validation.
- Thay đổi so với baseline: `batch_size = 64`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only bs_64`
- Kết quả: status=Completed, best_val_loss=0.905830695789217, val_acc=0.6923934243521872, val_macro_f1=0.6642436768620258, best_epoch=82, time_s=2499.457, run_dir=results/E3/E3-009_s0
- Nhận xét: ĐỘT PHÁ KỶ LỤC CỦA TOÀN BỘ E3! Val Acc đạt 69.24% (+1.09% so với baseline 68.15%, cao hơn std baseline 0.0075 đáng kể), Val Macro-F1 đạt đỉnh 66.42% (+2.34% so với baseline 64.08%) và Best Val Loss đạt 0.9058 tại epoch 82. Giả thuyết về lợi thế của batch size nhỏ trong việc tăng khả năng tổng quát hóa hoàn toàn được xác thực bằng số liệu thực nghiệm.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Ứng viên số 1 về batch size cho mô hình cuối E7.

### [E3-010_s0] Khảo sát Kích thước Batch bs=256 (bs_256) — TV4 – 2026-11-26
- Giả thuyết: Tăng batch size lên 256 sẽ giúp ước lượng gradient ổn định hơn trong mỗi bước và tăng tốc độ xử lý phần cứng, tuy nhiên có thể dẫn đến hiện tượng "generalization gap" khiến Val Acc giảm sút so với batch size nhỏ.
- Thay đổi so với baseline: `batch_size = 256`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV4 --only bs_256`
- Kết quả: status=Completed, best_val_loss=0.91694409404169, val_acc=0.6726107550849819, val_macro_f1=0.6298344029609881, best_epoch=94, time_s=2477.546, run_dir=results/E3/E3-010_s0
- Nhận xét: Thời gian mỗi epoch nhanh nhất (24.4s/epoch, tổng 2477s). Tuy nhiên Val Acc giảm xuống 67.26% (thấp hơn baseline 0.89%), Val Macro-F1 đạt 62.98% và Val Loss đạt 0.9169. Kết quả đối sánh giữa bs=64 (69.24%), bs=128 (68.15%) và bs=256 (67.26%) thể hiện quy luật đơn điệu rõ ràng: batch size càng lớn thì độ chính xác tổng quát hóa càng giảm trên bài toán FER2013.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 22.4 MB). Kết luận chọn batch size 64 hoặc 128 là tối ưu, không nên dùng batch size 256.

### [E4-001_s0] Khảo sát không dùng Data Augmentation — TV3 – <chưa thực hiện>
- Giả thuyết: Huấn luyện mô hình không có Data Augmentation sẽ dẫn đến overfitting nghiêm trọng; chênh lệch giữa Train Acc và Val Acc sẽ rất lớn (> 20%).
- Thay đổi so với baseline: `augmentation = "none"`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV3 --only no_aug`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E5-001_s0] Khảo sát mất cân bằng lớp bằng WeightedRandomSampler — TV4 – <chưa thực hiện>
- Giả thuyết: Lớp Disgust chỉ có 436 ảnh trong khi Happy có hơn 7,000 ảnh. Dùng WeightedRandomSampler sẽ tăng đáng kể Recall của lớp Disgust và Fear mà không làm suy giảm quá nhiều tổng Accuracy.
- Thay đổi so với baseline: Kỹ thuật lấy mẫu có trọng số cân bằng lớp.
- Lệnh: `python src/run_experiment.py --config configs/E5_imbalance.json --member TV4 --only weighted_sampler`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E6-007_s0] Khảo sát ResNet-18 Fine-tuning toàn bộ (resnet18_96_ft) — TV4 – 2026-11-26
- Giả thuyết: Khởi tạo từ trọng số ImageNet và fine-tune toàn bộ các tầng trên ảnh 96x96 (3 kênh) sẽ giúp mạng thừa hưởng các bộ lọc trích xuất đặc trưng bậc thấp mạnh mẽ, từ đó đạt độ chính xác cao hơn và hội tụ nhanh hơn mô hình CNN huấn luyện từ đầu.
- Thay đổi so với baseline: `resize = 96`, `channels = 3`, `transfer = {"name": "resnet18", "pretrained": true, "freeze_backbone": false}`.
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV4 --only resnet18_96_ft`
- Kết quả: status=Completed, best_val_loss=0.9717674393546252, val_acc=0.6873780997492338, val_macro_f1=0.6698490430387746, best_epoch=42, time_s=4572.14, run_dir=results/E6/E6-007_s0
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E6-002_s0] Khảo sát ResNet-18 Đóng băng Backbone (resnet18_96_frozen) — TV4 – 2026-11-26
- Giả thuyết: Đóng băng toàn bộ backbone ResNet-18 tiền huấn luyện ImageNet và chỉ huấn luyện lớp phân loại (Linear Probing / Feature Extractor) sẽ giúp giảm tối đa số tham số cần cập nhật và chống overfitting, tuy nhiên do đặc trưng ImageNet (vật thể tự nhiên) khác biệt với ảnh khuôn mặt cảm xúc xám nên độ chính xác có thể bị giới hạn so với fine-tune toàn bộ.
- Thay đổi so với baseline: `resize = 96`, `channels = 3`, `transfer = {"name": "resnet18", "pretrained": true, "freeze_backbone": true}`.
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV4 --only resnet18_96_frozen`
- Kết quả: status=Completed, best_val_loss=1.0006212015011144, val_acc=0.6770688213987183, val_macro_f1=0.6671904234002443, best_epoch=60, time_s=4182.096, run_dir=results/E6/E6-002_s0
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E7-001_s0] Mô hình Tối ưu Cuối cùng (Final Best Model) — TV3 – <chưa thực hiện>
- Giả thuyết: Tổng hợp cấu hình tốt nhất từ các thực nghiệm E1–E6 (kiến trúc, activation, optimizer, augmentation, xử lý mất cân bằng) sẽ mang lại độ chính xác validation cao nhất nhóm, vượt ngưỡng 70%.
- Thay đổi so với baseline: Cấu hình tối ưu toàn diện.
- Lệnh: `python src/run_experiment.py --config configs/E7_final.json --member TV3 --seeds 0 1 2`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)
