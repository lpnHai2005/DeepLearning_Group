# NHẬT KÝ THỰC NGHIỆM (NHẬT KÝ HUẤN LUYỆN & PHÁT TRIỂN MÔ HÌNH)

> **Môn học**: Deep Learning (Học sâu) – Bài tập lớn 02 (FER-2013)  
> **Người lập mẫu & phụ trách chính**: TV4 – Huỳnh Viễn Thông (Nhiệm vụ T24)  
> **Các thành viên tham gia**: TV1 – Đoàn Trung Kiên, TV2 – Lê Phan Nguyên Hải, TV3 – Võ Hoàng Bảo, TV4 – Huỳnh Viễn Thông  
> **Cập nhật lần cuối**: 2026-11-27  

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

### [baseline-001_s1] Baseline CNN phiên bản 60 epoch trước đóng băng (Seed 1) — TV1 – Đoàn Trung Kiên – 2026-11-24
- Giả thuyết: Thử nghiệm cấu hình Baseline CNN 6 Conv trên máy TV1 (RTX 3050) với giới hạn 60 epoch ban đầu trước khi chốt ngân sách 100 epoch cho `configs/baseline.json`.
- Thay đổi so với baseline đóng băng: Chạy bản `epochs = 60` (trước khi nâng lên `100` epoch ở bộ `baseline-002_s0/_s1/_s2`).
- Lệnh: `python src/run_experiment.py --config configs/baseline.json --member TV1 --seeds 1`
- Kết quả: status=Completed, best_val_loss=0.9354210293170104, val_acc=0.6564502646976874, val_macro_f1=0.590580758179016, best_epoch=53, time_s=2955.496, run_dir=results/baseline/baseline-001_s1
- Nhận xét: Hoàn tất 60/60 epoch trên RTX 3050 (~49.3 phút), đạt `val_acc = 0.6565` và `val_macro_f1 = 0.5906` (đỉnh epoch 53, lr cuối 0.00125). Kết quả thấp hơn rõ rệt (~2.5 điểm %) so với bộ 100 epoch vì mô hình chưa đủ số vòng lặp để hội tụ hẳn.
- Quyết định: Giữ lưu vết lịch sử (`KEEP/OK`), nhưng **không dùng làm mốc đối chuẩn chính thức**; nhóm nâng ngân sách lên 100 epoch, đóng băng `configs/baseline.json` và chạy lại đủ 3 seed chuẩn `baseline-002_s0/_s1/_s2` (`68.15% ± 0.75%`).

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

> **Ghi chú đánh số E1**: `E1-001`…`E1-008` là 8 biến thể gốc theo thứ tự trong `configs/E1_arch.json` (small, medium, medium_01, large, baseline_w0.5, baseline_w2, baseline_no_bn, baseline_gap). Seed 0 cả 8 biến thể do TV2 chạy trên RTX 4060 (commit `2945b830`). `E1-009_s1` là **small seed 1** chạy lại sau khi `E1-001_s1` bị ngắt (Aborted): theo luật của `run_experiment.py`, số đã có run lỗi ở seed đó không được dùng lại nên run chạy lại nhận số mới, không phải biến thể mới. Mốc so sánh: baseline 3 seed `val_acc = 0.6815 ± 0.0075`.

### [E1-001_s0] Tái hiện SMALL_CNN của notebook mẫu (`small`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: SMALL_CNN của mẫu (2 khối, 6 lớp Conv, không BN, FC 1024) có lớp FC rất lớn sau Flatten nên nhiều tham số nhưng nông; dự kiến hội tụ được nhưng quá khớp mạnh hơn baseline.
- Thay đổi so với baseline: `model = "small"` (blocks `[[64,64],[128,128,256,256]]`, không BN, dropout 0.2/0.5, `38,901,191` tham số, gấp 6.6 lần baseline).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9094449157799874, val_acc=0.6915575369183616, val_macro_f1=0.6731403436946511, best_epoch=96, time_s=2133.077, run_dir=results/E1/E1-001_s0
- Nhận xét: Chạy đủ 100 epoch (~35.6 phút), đỉnh ở epoch 96. `val_acc = 0.6916`, cao hơn baseline 0.0101, nhưng `train_acc_cuoi = 0.8167` nên khoảng cách train–val khoảng 0.125, lớn gấp 3 lần baseline: phần lớn tham số nằm ở lớp FC đầu (Flatten 256×12×12 → 1024). Chênh lệch so với baseline xấp xỉ 1 điểm nên đã chạy thêm seed 1, 2 (`E1-009_s1`, `E1-001_s2`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 148.4 MB). Không chọn làm kiến trúc chính dù val cao, vì chi phí tham số và mức quá khớp; xem kết luận 3 seed ở `E1-001_s2`.

### [E1-002_s0] Tái hiện MEDIUM_CNN của notebook mẫu (`medium`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: MEDIUM_CNN (3 khối, 8 lớp Conv, không BN) sâu hơn SMALL, bản đồ đặc trưng nhỏ hơn trước Flatten nên ít tham số hơn SMALL; dự kiến tương đương hoặc tốt hơn SMALL.
- Thay đổi so với baseline: `model = "medium"` (blocks `[[64]*2,[128]*3,[256]*3]`, không BN, FC 1024, `11,327,303` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9161594292020758, val_acc=0.6951797157982725, val_macro_f1=0.6788904667222425, best_epoch=100, time_s=1691.147, run_dir=results/E1/E1-002_s0
- Nhận xét: Chạy đủ 100 epoch (~28.2 phút), đỉnh ở epoch 100 nên có thể chưa bão hòa. `val_acc = 0.6952` (+0.0137 so với baseline), `val_macro_f1 = 0.6789`. Khoảng cách train–val khoảng 0.110. Mạng 8 lớp Conv không BN vẫn học bình thường với SGD lr 0.01. Đã chạy thêm seed 1, 2 (`E1-002_s1`, `E1-002_s2`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 43.2 MB). Xem kết luận 3 seed ở `E1-002_s2`.

### [E1-003_s0] Tái hiện MEDIUM1_CNN của notebook mẫu (`medium_01`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: MEDIUM1_CNN (3 khối, 11 lớp Conv, không BN) sâu hơn MEDIUM; nếu độ sâu có ích thì val tăng, nếu không có BN thì có nguy cơ khó tối ưu.
- Thay đổi so với baseline: `model = "medium_01"` (blocks `[[64]*3,[128]*4,[256]*4]`, không BN, FC 1024, `12,101,895` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=1.8107485001915573, val_acc=0.2493730844246308, val_macro_f1=0.0570281636294125, best_epoch=1, time_s=326.687, run_dir=results/E1/E1-003_s0
- Nhận xét: **Mạng không học được.** `train_loss` đứng ở khoảng 1.81 từ epoch 1 đến 16 (`history.csv`), `train_acc = 0.2513`, `val_acc = 0.2494` đúng bằng tỷ lệ lớp Happy trong PublicTest (895/3589) và `val_macro_f1 = 0.0570`: mô hình luôn đoán một lớp. Early stopping dừng ở epoch 16 (best_epoch 1, ~5.4 phút). Không có NaN. Đây là lỗi tối ưu, chưa cho biết gì về năng lực của kiến trúc: MEDIUM 8 lớp Conv cùng cấu hình vẫn học được, còn notebook mẫu huấn luyện các mạng này bằng Adam lr=1e-4, không phải SGD lr 0.01.
- Quyết định: Giữ run làm failure case trong bảng, **không kết luận về độ sâu từ run này**. Weights (`best.pt` 46.2 MB, ở epoch 1) không có giá trị sử dụng. Đã thêm 2 biến thể để tách nguyên nhân: `medium_01_adam1e-4` (optimizer của mẫu, vẫn không BN) và `medium_01_bn` (thêm BN, giữ SGD 0.01).

### [E1-004_s0] Tái hiện LARGE_CNN của notebook mẫu (`large`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: LARGE_CNN (4 khối, 16 lớp Conv, không BN, FC 2048) là mạng sâu nhất; nếu MEDIUM1 đã khó tối ưu thì LARGE càng dễ bị kẹt.
- Thay đổi so với baseline: `model = "large"` (blocks `[[64]*4,[128]*4,[256]*4,[512]*4]`, không BN, FC 2048, `20,406,599` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=1.8107463234013648, val_acc=0.2493730844246308, val_macro_f1=0.0570281636294125, best_epoch=1, time_s=397.254, run_dir=results/E1/E1-004_s0
- Nhận xét: **Sập giống hệt `E1-003_s0`**: loss đứng ở khoảng 1.81, `val_acc = 0.2494`, `val_macro_f1 = 0.0570`, dừng sớm ở epoch 16 (~6.6 phút). Quy luật trong E1: không BN, SGD lr 0.01 thì 6–8 lớp Conv (small, medium, baseline_no_bn) học được, còn 11–16 lớp Conv (medium_01, large) không học được.
- Quyết định: Giữ run làm failure case, không kết luận về độ sâu. Weights (`best.pt` 77.9 MB, ở epoch 1) không có giá trị sử dụng. Đã thêm `large_adam1e-4` và `large_bn` để tách nguyên nhân như với `medium_01`.

### [E1-005_s0] Thu hẹp baseline còn ×0.5 số kênh (`baseline_w0.5`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: Giảm một nửa số kênh mỗi khối làm số tham số giảm còn khoảng 45%; dự kiến val giảm nhẹ nhưng ít quá khớp hơn.
- Thay đổi so với baseline: `model_kwargs.blocks = [[32,32],[64,64],[128,128]]` (`2,650,279` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9146933610075928, val_acc=0.6701030927835051, val_macro_f1=0.6329196107692682, best_epoch=90, time_s=1397.591, run_dir=results/E1/E1-005_s0
- Nhận xét: Chạy đủ 100 epoch (~23.3 phút), đỉnh ở epoch 90. `val_acc = 0.6701` (−0.0114 so với baseline), `val_macro_f1 = 0.6329`. Khoảng cách train–val chỉ khoảng 0.017, nhỏ nhất trong E1: mạng hẹp thiếu năng lực (underfit nhẹ) hơn là quá khớp. Mức giảm sát ngưỡng 1 điểm và chỉ có 1 seed.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 10.1 MB). Chạy thêm seed 1, 2 trước khi kết luận.

### [E1-006_s0] Mở rộng baseline ×2 số kênh (`baseline_w2`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: Gấp đôi số kênh làm số tham số tăng khoảng 2.4 lần; dự kiến val tăng nhẹ nếu baseline đang thiếu năng lực, giảm nếu quá khớp.
- Thay đổi so với baseline: `model_kwargs.blocks = [[128,128],[256,256],[512,512]]` (`14,017,159` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9472882772229382, val_acc=0.6701030927835051, val_macro_f1=0.5823856011107499, best_epoch=99, time_s=2603.728, run_dir=results/E1/E1-006_s0
- Nhận xét: Chạy đủ 100 epoch (~43.4 phút), đỉnh ở epoch 99. `val_acc = 0.6701`, bằng đúng `baseline_w0.5` (cùng 2405/3589 mẫu đúng, chỉ là trùng số), nhưng `val_macro_f1 = 0.5824` thấp hơn hẳn (w0.5 0.6329, baseline 0.6408) và `best_val_loss = 0.9473` cao nhất trong các run học được: mạng rộng đoán lệch về lớp lớn, các lớp nhỏ kém hơn. Mở rộng kênh không giúp ích ở cấu hình huấn luyện hiện tại.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 53.5 MB). Chạy thêm seed 1, 2 trước khi kết luận, vì mức chênh val_acc sát ngưỡng 1 điểm.

### [E1-007_s0] Bỏ BatchNorm khỏi baseline (`baseline_no_bn`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: Bỏ BN làm gradient kém ổn định và hội tụ chậm hơn; dự kiến val giảm.
- Thay đổi so với baseline: `model_kwargs = {"bn": false}` (`5,866,951` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9086250141491362, val_acc=0.6907216494845361, val_macro_f1=0.6639630435039692, best_epoch=91, time_s=1413.567, run_dir=results/E1/E1-007_s0
- Nhận xét: Chạy đủ 100 epoch (~23.6 phút), đỉnh ở epoch 91. `val_acc = 0.6907` (+0.0092), trái với giả thuyết. Mạng 6 lớp Conv đủ nông để học được không cần BN với SGD lr 0.01. Mức tăng dưới 1 điểm nên đã giao TV3 chạy seed 1, 2 (`E1-007_s1`, `E1-007_s2`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Xem kết luận 3 seed ở `E1-007_s2`.

### [E1-008_s0] Thay Flatten bằng Global Average Pooling (`baseline_gap`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: GAP bỏ gần hết tham số của phần phân loại; dự kiến val tương đương baseline và quá khớp ít hơn.
- Thay đổi so với baseline: `model_kwargs = {"head": "gap"}` (`1,280,327` tham số, 21.8% baseline).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9088872493426092, val_acc=0.6829200334354973, val_macro_f1=0.6449168262056626, best_epoch=74, time_s=1273.726, run_dir=results/E1/E1-008_s0
- Nhận xét: Early stopping dừng ở epoch 89 (best_epoch 74, ~21.2 phút). `val_acc = 0.6829` (+0.0014, trong dung sai), `val_macro_f1 = 0.6449`. Đã giao TV3 chạy seed 1, 2 (`E1-008_s1`, `E1-008_s2`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 4.9 MB). Xem kết luận 3 seed ở `E1-008_s2`.

### [E1-001_s1] SMALL_CNN (`small`, Seed 1 — C1) — TV1 – Đoàn Trung Kiên – 2026-11-25
- Giả thuyết: Kiểm tra độ ổn định của `small` ở seed 1 trên máy khác (RTX 3050).
- Thay đổi so với baseline: `model = "small"`, `seed = 1`.
- Lệnh: (TV1 điền đúng lệnh đã chạy, dạng `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only ... --seeds ...`)
- Kết quả: status=Aborted, best_val_loss=0.914619894338604, val_acc=0.6893285037614935, val_macro_f1=0.6668776907716686, best_epoch=77, time_s=, run_dir=results/E1/E1-001_s1
- Nhận xét: Run bị ngắt ở epoch 77/100 (`run.log`: `END | status=Aborted | reason=INTERRUPTED@epoch77`). Không dùng run này trong bảng; chạy lại thành `E1-009_s1`.
- Quyết định: Bỏ (`DISCARD`, đã xóa weights dở).

### [E1-009_s1] SMALL_CNN chạy lại (`small`, Seed 1 — C1) — TV1 – Đoàn Trung Kiên – 2026-11-25
- Giả thuyết: Chạy lại `small` seed 1 sau khi `E1-001_s1` bị ngắt.
- Thay đổi so với baseline: `model = "small"`, `seed = 1`.
- Lệnh: (TV1 điền đúng lệnh đã chạy, dạng `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only ... --seeds ...`)
- Kết quả: status=Completed, best_val_loss=0.914619894338604, val_acc=0.6921147952075787, val_macro_f1=0.6766273703235255, best_epoch=98, time_s=3261.997, run_dir=results/E1/E1-009_s1
- Nhận xét: (TV1 viết)
- Quyết định: (TV1 viết)

### [E1-001_s2] SMALL_CNN (`small`, Seed 2 — C1) — TV1 – Đoàn Trung Kiên – 2026-11-25
- Giả thuyết: Hoàn thiện 3 seed cho `small`.
- Thay đổi so với baseline: `model = "small"`, `seed = 2`.
- Lệnh: (TV1 điền đúng lệnh đã chạy, dạng `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only ... --seeds ...`)
- Kết quả: status=Completed, best_val_loss=0.918979596945528, val_acc=0.6965728615213151, val_macro_f1=0.6820279907366499, best_epoch=83, time_s=3215.931, run_dir=results/E1/E1-001_s2
- Nhận xét: (TV1 viết) Tổng hợp 3 seed (`E1-001_s0`, `E1-009_s1`, `E1-001_s2`) theo `report/tables/E1_summary.md`: `val_acc = 0.6934 ± 0.0027` (+0.0119), khoảng cách train–val trung bình 0.123.
- Quyết định: (TV1 viết)

### [E1-002_s1] MEDIUM_CNN (`medium`, Seed 1 — C1) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Kiểm tra độ ổn định của `medium` ở seed 1.
- Thay đổi so với baseline: `model = "medium"`, `seed = 1`.
- Lệnh: (TV1 điền đúng lệnh đã chạy, dạng `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only ... --seeds ...`)
- Kết quả: status=Completed, best_val_loss=0.9076775662179576, val_acc=0.6890498746168849, val_macro_f1=0.6730121936066201, best_epoch=100, time_s=2891.625, run_dir=results/E1/E1-002_s1
- Nhận xét: (TV1 viết)
- Quyết định: (TV1 viết)

### [E1-002_s2] MEDIUM_CNN (`medium`, Seed 2 — C1) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Hoàn thiện 3 seed cho `medium`.
- Thay đổi so với baseline: `model = "medium"`, `seed = 2`.
- Lệnh: (TV1 điền đúng lệnh đã chạy, dạng `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only ... --seeds ...`)
- Kết quả: status=Completed, best_val_loss=0.9119477483282252, val_acc=0.6915575369183616, val_macro_f1=0.6670449351732132, best_epoch=98, time_s=3099.882, run_dir=results/E1/E1-002_s2
- Nhận xét: (TV1 viết) Tổng hợp 3 seed (`E1-002_s0/s1/s2`) theo `report/tables/E1_summary.md`: `val_acc = 0.6919 ± 0.0031` (+0.0104), khoảng cách train–val trung bình 0.101.
- Quyết định: (TV1 viết)

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
- Nhận xét: Đạt **69.74%** `val_acc`, **67.27%** `val_macro_f1`, `best_val_loss = 0.9017` tại epoch 97/100 (~46.3 phút). Tổng hợp đủ 3 seed (`E1-007_s0/s1/s2`): `val_acc = 0.6941 ± 0.0033` (**+0.0125**, cao hơn rõ rệt ngưỡng `std = 0.0075` của baseline), `val_macro_f1 = 0.6682 ± 0.0044` (+0.0274 so với baseline), `best_val_loss = 0.9059`. Trái với giả thuyết ban đầu, mạng VGG 6 lớp Conv với khởi tạo mặc định của PyTorch (`kaiming_uniform a=√5`, phương sai `~1/(3·fan_in)`) đủ nông để hội tụ không cần BN (khác với `medium_01` 11 lớp và `large` 16 lớp bị sập ở `0.2494` do phương sai khởi tạo nhỏ qua nhiều tầng), đồng thời bỏ BN tránh xung đột thống kê giữa Dropout và BatchNorm nên vừa tăng độ chính xác vừa giảm một nửa độ lệch chuẩn giữa các seed (`0.0033` vs `0.0075`).
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

### [E1-010_s0] MEDIUM1_CNN với Adam lr=1e-4 như mẫu (`medium_01_adam1e-4`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: `E1-003_s0` sập vì cách tối ưu chứ không phải vì kiến trúc; đổi sang optimizer của notebook mẫu (Adam lr=1e-4), giữ không BN và init mặc định của PyTorch, thì mạng sẽ học được.
- Thay đổi so với baseline: `model = "medium_01"`, `optimizer = "adam"`, `lr = 1e-4` (so với `E1-003_s0` chỉ đổi optimizer và lr).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only medium_01_adam1e-4 large_adam1e-4 medium_01_bn large_bn --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9729039279090624, val_acc=0.6391752577319587, val_macro_f1=0.5992779215288392, best_epoch=99, time_s=2064.053, run_dir=results/E1/E1-010_s0
- Nhận xét: **Học được.** Epoch 1 vẫn đứng 0.2494 như bản SGD, nhưng loss giảm ngay từ epoch 2 và val tăng đều: 0.47 (epoch 10), 0.57 (30), 0.61 (50), 0.63 (90). `val_acc = 0.6392`, `val_macro_f1 = 0.5993`, thấp hơn baseline 0.042 và thấp hơn mẫu (0.65). Train và val gần nhau (`train_acc_cuoi = 0.6536`) và best_epoch = 99, tức là với lr nhỏ thì 100 epoch chưa đủ hội tụ chứ chưa quá khớp. Chạy đủ 100 epoch, khoảng 34 phút. `git_commit` ghi `35e59bbd-dirty` do lúc chạy thư mục có file chưa commit của E6; `src/` không đổi nên kết quả không bị ảnh hưởng.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 46.2 MB). Xác nhận phần "tối ưu" của nguyên nhân: Adam chia gradient cho độ lớn của chính nó nên bù được gradient rất nhỏ ở các lớp đầu (đo lúc khởi tạo: 7.6e-6 tới `conv1_1`). Chạy thêm `medium_01_glorot_adam1e-4` để tách phần "khởi tạo".

### [E1-011_s0] LARGE_CNN với Adam lr=1e-4 như mẫu (`large_adam1e-4`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-010_s0`, đổi sang Adam lr=1e-4 thì LARGE (16 lớp Conv, không BN) học được.
- Thay đổi so với baseline: `model = "large"`, `optimizer = "adam"`, `lr = 1e-4` (so với `E1-004_s0` chỉ đổi optimizer và lr).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only medium_01_adam1e-4 large_adam1e-4 medium_01_bn large_bn --seeds 0`
- Kết quả: status=Aborted, best_val_loss=1.8114331007070215, val_acc=0.24937308442463083, val_macro_f1=0.05702816362941251, best_epoch=1, time_s=, run_dir=results/E1/E1-011_s0
- Nhận xét: Dừng tay ở epoch 4 (TV2 tắt tiến trình để kiểm tra), nên không đủ để kết luận. Cả 4 epoch đều đứng `val_acc = 0.2494`, `train_loss` khoảng 1.812 không giảm. Đo lúc khởi tạo (ảnh nhiễu N(0,1), tắt dropout): chuẩn gradient tới `conv1_1` của LARGE không BN là 7.6e-8 với init mặc định của PyTorch, so với 1.0e-3 với Glorot (init mặc định của Keras, mẫu dùng) và 7.6e-6 ở MEDIUM1. Gradient từng phần tử cỡ hoặc nhỏ hơn `eps = 1e-8` trong mẫu số của Adam nên Adam không bù được như ở MEDIUM1.
- Quyết định: Run bị ngắt, `run_experiment.py` tự đánh dấu `Aborted` và xóa `best.pt` dở. Không chạy lại `large_adam1e-4`; thay bằng `large_glorot_adam1e-4` (Glorot + Adam 1e-4, đúng thiết lập mẫu, mẫu đạt 0.67) và `large_bn`.

### [E1-012_s0] MEDIUM1_CNN thêm BatchNorm (`medium_01_bn`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: BN chuẩn hóa đầu ra từng lớp nên tín hiệu và gradient không co lại qua 11 lớp Conv; giữ SGD lr=0.01 và init mặc định thì mạng sẽ học được.
- Thay đổi so với baseline: `model = "medium_01"`, `model_kwargs = {"bn": true}` (so với `E1-003_s0` chỉ thêm BN).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only medium_01_bn large_bn medium_01_glorot_adam1e-4 large_glorot_adam1e-4 --seeds 0`
- Kết quả: status=Aborted, best_val_loss=1.1380014236207858, val_acc=0.5742546670381722, val_macro_f1=0.45911900431262, best_epoch=8, time_s=, run_dir=results/E1/E1-012_s0
- Nhận xét: TV2 tắt tiến trình ở epoch 9 để tạm dừng máy. Run được code đánh dấu `Aborted` (`INTERRUPTED@epoch9`) và xóa `best.pt` dở. Trong 9 epoch val đã tăng từ 0.317 lên tối đa 0.574 (epoch 8), trong khi bản không BN đứng ở 0.2494. Chạy lại thành `E1-013_s0`, epoch 1 trùng khớp (val 0.3168), xác nhận cùng seed thì tái lập được.
- Quyết định: Bỏ (`DISCARD`). Dùng `E1-013_s0`.

### [E1-013_s0] MEDIUM1_CNN thêm BatchNorm, chạy lại (`medium_01_bn`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-012_s0`.
- Thay đổi so với baseline: `model = "medium_01"`, `model_kwargs = {"bn": true}`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only medium_01_bn large_bn medium_01_glorot_adam1e-4 large_glorot_adam1e-4 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.920362198841077, val_acc=0.6912789077737531, val_macro_f1=0.6686331095461697, best_epoch=76, time_s=1982.429, run_dir=results/E1/E1-013_s0
- Nhận xét: **BN cứu được MEDIUM1.** Val tăng ngay từ epoch 1: 0.317 (ep 1), 0.577 (ep 10), 0.640 (ep 30), đạt đỉnh `val_acc = 0.6913` ở epoch 76 (+0.0098 so với baseline), `val_macro_f1 = 0.6686`. Early stopping dừng ở epoch 91. Quá khớp rõ: `train_acc_cuoi = 0.8582`, val_loss thấp nhất 0.920 ở epoch 52 rồi tăng lên 1.146 ở epoch 91 trong khi val_acc vẫn lập kỷ lục.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 46.2 MB). Cùng SGD lr 0.01 và init mặc định: không BN ra 0.2494, có BN ra 0.6913, nên nguyên nhân sập của `E1-003_s0` là thiếu chuẩn hóa tín hiệu qua mạng sâu chứ không phải độ sâu.

### [E1-014_s0] LARGE_CNN thêm BatchNorm (`large_bn`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-013_s0`, BN cho phép LARGE (16 lớp Conv) học được với SGD lr=0.01 và init mặc định.
- Thay đổi so với baseline: `model = "large"`, `model_kwargs = {"bn": true}` (`20,410,439` tham số; so với `E1-004_s0` chỉ thêm BN).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only medium_01_bn large_bn medium_01_glorot_adam1e-4 large_glorot_adam1e-4 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9163508506896071, val_acc=0.7085539147394817, val_macro_f1=0.6952893066297167, best_epoch=90, time_s=2608.403, run_dir=results/E1/E1-014_s0
- Nhận xét: **Học được và cao nhất E1 tới thời điểm này:** `val_acc = 0.7086` ở epoch 90 (+0.0271 so với baseline, vượt kỷ lục public Kaggle 0.69), `val_macro_f1 = 0.6953`. Chạy đủ 100 epoch (~43.5 phút). Quá khớp mạnh nhất E1: `train_acc_cuoi = 0.8902`; val_loss thấp nhất 0.916 ở epoch 38 rồi tăng lên 1.246 ở epoch 100. Mới có 1 seed.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 77.9 MB). Ứng viên kiến trúc cho E7, nhưng cần chạy thêm seed 1, 2 trước khi kết luận và nên kết hợp với điều chuẩn của E4 (dropout, weight decay, label smoothing) vì khoảng cách train–val lớn.

### [E1-014_s1] LARGE_CNN thêm BatchNorm (`large_bn`, Seed 1) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-014_s0`; chạy thêm seed để xác nhận `large_bn` hơn baseline vượt std baseline chứ không phải do seed.
- Thay đổi so với baseline: `model = "large"`, `model_kwargs = {"bn": true}` (`20,410,439` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only large_bn --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.9088867051450613, val_acc=0.7046531067149624, val_macro_f1=0.6910078526038917, best_epoch=84, time_s=2552.002, run_dir=results/E1/E1-014_s1
- Nhận xét: Dừng sớm ở epoch 99 (đỉnh epoch 84, ~42.5 phút): `val_acc = 0.7047`, `val_macro_f1 = 0.6910`, gần seed 0. Quá khớp giống seed 0: `train_acc_cuoi = 0.9022` (khoảng train–val 0.198); val_loss thấp nhất 0.909 ở epoch 40 rồi tăng lên 1.375 ở epoch cuối, trong khi val_acc vẫn nhích lên tới epoch 84.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 77.9 MB).

### [E1-014_s2] LARGE_CNN thêm BatchNorm (`large_bn`, Seed 2) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-014_s1`.
- Thay đổi so với baseline: `model = "large"`, `model_kwargs = {"bn": true}` (`20,410,439` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only large_bn --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.8999231593062135, val_acc=0.7046531067149624, val_macro_f1=0.6903086854238897, best_epoch=85, time_s=2547.809, run_dir=results/E1/E1-014_s2
- Nhận xét: Dừng sớm ở epoch 100 (đỉnh epoch 85, ~42.5 phút): `val_acc = 0.7047` (trùng số ảnh đúng với seed 1, 2529/3589, nhưng loss và F1 khác), `val_macro_f1 = 0.6903`. `train_acc_cuoi = 0.9110` (khoảng train–val 0.206); val_loss thấp nhất 0.900 ở epoch 42 rồi tăng lên 1.350. Ba seed của `large_bn`: 0.7086 / 0.7047 / 0.7047, thấp nhất vẫn cao hơn seed tốt nhất của baseline (0.6890).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 77.9 MB). Đủ 3 seed, tổng hợp ở Kết luận E1.

### [E1-015_s0] MEDIUM1_CNN init Glorot + Adam 1e-4, đúng thiết lập mẫu (`medium_01_glorot_adam1e-4`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Khác biệt còn lại giữa `E1-010_s0` (0.6392) và mẫu (0.65) là khởi tạo trọng số: Keras mặc định Glorot uniform, PyTorch mặc định kaiming_uniform a=√5 có phương sai nhỏ hơn khoảng 3 lần. Đổi sang Glorot thì gradient ở các lớp đầu lớn hơn nhiều bậc (đo lúc khởi tạo: 1.1e-2 so với 7.6e-6 tới `conv1_1`) nên mạng học nhanh và tốt hơn.
- Thay đổi so với baseline: `model = "medium_01"`, `model_kwargs = {"init": "glorot"}`, `optimizer = "adam"`, `lr = 1e-4` (so với `E1-010_s0` chỉ đổi init).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only medium_01_bn large_bn medium_01_glorot_adam1e-4 large_glorot_adam1e-4 --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9063716961766857, val_acc=0.6720534967957649, val_macro_f1=0.6383022535970843, best_epoch=90, time_s=2080.734, run_dir=results/E1/E1-015_s0
- Nhận xét: **Tái hiện được và vượt mẫu:** `val_acc = 0.6721` ở epoch 90 (mẫu 0.65), `val_macro_f1 = 0.6383`, val_loss thấp nhất 0.906 ở epoch 80 (mẫu 0.94–0.97). So với `E1-010_s0` (init mặc định) cao hơn 0.033 và học nhanh hơn rõ: ep 2 / 5 / 10 / 20 lần lượt 0.362 / 0.492 / 0.552 / 0.588, so với log notebook mẫu (MEDIUM1, Adam 1e-4) là 0.39 / 0.52 / 0.57 (ep 9) / 0.63 (ep 19). Mẫu vẫn nhanh hơn một chút ở đầu; khác biệt còn lại là mẫu đưa ảnh thô 0–255 vào mạng còn nhóm chuẩn hóa mean/std. Nhóm cao hơn mẫu ở cuối vì chạy 100 epoch có giảm lr (plateau) còn mẫu dừng sớm theo val_loss ở khoảng epoch 32. Quá khớp nhẹ (`train_acc_cuoi = 0.7266`). Thấp hơn baseline 0.0094 (trong khoảng ±1 điểm).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 46.2 MB). Xác nhận phần "khởi tạo" của nguyên nhân. Thứ tự với MEDIUM1 (1 seed): init mặc định + SGD 0.2494 < init mặc định + Adam 0.6392 < Glorot + Adam 0.6721 < init mặc định + SGD + BN 0.6913.

### [E1-016_s0] LARGE_CNN init Glorot + Adam 1e-4, đúng thiết lập mẫu (`large_glorot_adam1e-4`, Seed 0) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-015_s0` với MEDIUM1: đổi sang init Glorot thì gradient tới `conv1_1` của LARGE không BN tăng từ 7.6e-8 lên 1.0e-3 (đo lúc khởi tạo), lớn hơn hẳn `eps = 1e-8` của Adam, nên LARGE thoát được trạng thái đoán toàn Happy mà `E1-011_s0` (init mặc định) bị kẹt. Mẫu báo cáo LARGE 0.67.
- Thay đổi so với baseline: `model = "large"`, `model_kwargs = {"init": "glorot"}`, `optimizer = "adam"`, `lr = 1e-4` (so với `E1-011_s0` chỉ đổi init).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only large_glorot_adam1e-4 --seeds 0`
- Kết quả: status=Completed, best_val_loss=1.0055855756086305, val_acc=0.6235720256338813, val_macro_f1=0.5544728307543171, best_epoch=88, time_s=2463.596, run_dir=results/E1/E1-016_s0
- Nhận xét: **Thoát kẹt nhưng thiếu khớp (underfit).** Bốn epoch đầu val vẫn 0.2494 nhưng train_loss giảm dần (1.821 → 1.786, bản init mặc định đứng yên 1.812), tới epoch 5 val vượt 0.26 rồi học tiếp đủ 100 epoch (~41 phút). Đỉnh `val_acc = 0.6236` ở epoch 88, `val_macro_f1 = 0.5545`, thấp hơn mẫu (0.67) 0.046 và thấp hơn `medium_01_glorot_adam1e-4` (0.6721) 0.049. Không quá khớp: `train_acc_cuoi = 0.6548`, khoảng train–val 0.035, val_loss thấp nhất 1.005 ở epoch 98 và gần như không tăng. Nguyên nhân dự đoán: lr 1e-4 đã nhỏ, lịch plateau còn giảm thêm 5 lần xuống 3.1e-6 trước khi mạng 16 lớp học xong; mẫu thì dùng lr cố định và ảnh thô 0–255. Chưa có run kiểm chứng giả thuyết này.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 77.9 MB). Đủ để xác nhận phần "khởi tạo" của nguyên nhân sập ở LARGE; không chạy thêm vì `large_bn` (0.7086) đã là lời giải tốt hơn cho mạng sâu.

### [E1-005_s1] Thu hẹp baseline còn ×0.5 số kênh (`baseline_w0.5`, Seed 1) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-005_s0`; chạy thêm seed vì chênh lệch seed 0 so với baseline (−0.0114) sát ngưỡng 1 điểm.
- Thay đổi so với baseline: `model_kwargs.blocks = [[32,32],[64,64],[128,128]]` (`2,650,279` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only baseline_w0.5 baseline_w2 --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.894616552965659, val_acc=0.6798551128448036, val_macro_f1=0.6456993756644813, best_epoch=100, time_s=1382.73, run_dir=results/E1/E1-005_s1
- Nhận xét: Chạy đủ 100 epoch (~23 phút), đỉnh ở epoch cuối: `val_acc = 0.6799`, `val_macro_f1 = 0.6457`, `best_val_loss = 0.8946` (thấp nhất trong mọi run E1 và baseline). Khoảng train–val 0.014, val_loss còn giảm tới epoch 100: mạng hẹp vẫn chưa hội tụ hẳn, không có dấu hiệu quá khớp.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 10.1 MB).

### [E1-005_s2] Thu hẹp baseline còn ×0.5 số kênh (`baseline_w0.5`, Seed 2) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-005_s1`.
- Thay đổi so với baseline: `model_kwargs.blocks = [[32,32],[64,64],[128,128]]` (`2,650,279` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only baseline_w0.5 baseline_w2 --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.9033515358343202, val_acc=0.674282529952633, val_macro_f1=0.6253687426428443, best_epoch=82, time_s=1337.356, run_dir=results/E1/E1-005_s2
- Nhận xét: Dừng sớm ở epoch 97 (đỉnh epoch 82, ~22 phút): `val_acc = 0.6743`, `val_macro_f1 = 0.6254`. Khoảng train–val 0.021. Ba seed của w0.5: 0.6701 / 0.6799 / 0.6743.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 10.1 MB). Đủ 3 seed, tổng hợp ở Kết luận E1.

### [E1-006_s1] Mở rộng baseline ×2 số kênh (`baseline_w2`, Seed 1) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-006_s0`; chạy thêm seed để xác nhận w2 không hơn baseline và macro-F1 thấp không phải do seed.
- Thay đổi so với baseline: `model_kwargs.blocks = [[128,128],[256,256],[512,512]]` (`14,017,159` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only baseline_w0.5 baseline_w2 --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.9616181550832404, val_acc=0.6687099470604625, val_macro_f1=0.5836231151850474, best_epoch=99, time_s=2721.67, run_dir=results/E1/E1-006_s1
- Nhận xét: Chạy đủ 100 epoch (~45 phút). Hai epoch đầu đứng ở 0.2494 rồi mới học (epoch 3), giống `E1-006_s0`. Đỉnh `val_acc = 0.6687` ở epoch 99, `val_macro_f1 = 0.5836`, lặp lại đúng mẫu hình của seed 0: accuracy kém baseline khoảng 1 điểm, macro-F1 kém khoảng 6 điểm. lr ở epoch 100 còn 0.0025 (mới giảm 2 lần).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 53.5 MB).

### [E1-006_s2] Mở rộng baseline ×2 số kênh (`baseline_w2`, Seed 2) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
- Giả thuyết: Như `E1-006_s1`.
- Thay đổi so với baseline: `model_kwargs.blocks = [[128,128],[256,256],[512,512]]` (`14,017,159` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only baseline_w0.5 baseline_w2 --seeds 1 2`
- Kết quả: status=Completed, best_val_loss=0.9900542456115909, val_acc=0.657286152131513, val_macro_f1=0.5491253925533953, best_epoch=100, time_s=2632.556, run_dir=results/E1/E1-006_s2
- Nhận xét: Chạy đủ 100 epoch (~44 phút), đỉnh ở epoch cuối: `val_acc = 0.6573`, `val_macro_f1 = 0.5491`, thấp nhất trong 3 seed. `train_acc_cuoi = 0.6469` còn thấp hơn val: mạng chưa học xong tập train sau 100 epoch (lr cuối 0.00125, val_loss vẫn giảm tới epoch 99). Ba seed của w2: 0.6701 / 0.6687 / 0.6573.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 53.5 MB). Đủ 3 seed, tổng hợp ở Kết luận E1.

### Kết luận E1 (T12) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
Bảng số liệu sinh bằng `python src/make_tables.py` → `report/tables/E1_summary.md` (Bảng 4.1). Mốc so sánh: baseline 3 seed `val_acc = 0.6815 ± 0.0075`, `val_macro_f1 = 0.6408 ± 0.0195`, khoảng train–val `0.0397`. Chênh lệch dưới std baseline coi là chưa phân biệt được; biến thể chỉ có 1 seed chỉ dùng làm định hướng.

**1. Tái hiện 4 CNN của notebook mẫu (không BN).**
- `small` `0.6934 ± 0.0027` và `medium` `0.6919 ± 0.0031` (3 seed) cao hơn baseline khoảng 1.0–1.2 điểm, vượt std baseline, nhưng quá khớp mạnh hơn nhiều (khoảng train–val `0.123` và `0.101` so với `0.040`) và nặng hơn (38.9 M và 11.3 M tham số so với 5.9 M).
- `medium_01` (11 lớp conv) và `large` (16 lớp conv) với SGD lr 0.01 và init mặc định **sập**: loss đứng 1.81 từ epoch 1, đoán toàn lớp Happy (`0.2494`).
- Tách nguyên nhân bằng các biến thể đổi một yếu tố (1 seed mỗi biến thể):

| Mô hình | Init | Optimizer | BN | val_acc | Run |
|---|---|---|---|---|---|
| MEDIUM1 | mặc định | SGD 0.01 | không | 0.2494 | `E1-003_s0` |
| MEDIUM1 | mặc định | Adam 1e-4 | không | 0.6392 | `E1-010_s0` |
| MEDIUM1 | Glorot | Adam 1e-4 | không | 0.6721 (mẫu 0.65) | `E1-015_s0` |
| MEDIUM1 | mặc định | SGD 0.01 | có | 0.6913 | `E1-013_s0` |
| LARGE | mặc định | SGD 0.01 | không | 0.2494 | `E1-004_s0` |
| LARGE | mặc định | Adam 1e-4 | không | kẹt 0.2494, dừng tay epoch 4 | `E1-011_s0` (Aborted) |
| LARGE | Glorot | Adam 1e-4 | không | 0.6236 (mẫu 0.67) | `E1-016_s0` |
| LARGE | mặc định | SGD 0.01 | có | **0.7060 ± 0.0023** (3 seed) | `E1-014_s0`–`_s2` |

- Kết luận: mạng sâu không BN sập do **khởi tạo**. Init mặc định của PyTorch (kaiming_uniform a=√5) có phương sai nhỏ hơn Glorot của Keras khoảng 3 lần; qua 11–16 lớp, gradient tới lớp đầu chỉ còn cỡ 1e-7 (LARGE, 7.6e-8), nhỏ ngang `eps` của Adam, nên đổi optimizer chỉ cứu được MEDIUM1. Đổi sang Glorot tái hiện được mẫu ở MEDIUM1 nhưng LARGE vẫn thiếu khớp sau 100 epoch. **BatchNorm loại bỏ hoàn toàn sự phụ thuộc vào init** và cho kết quả tốt nhất ở cả hai độ sâu.

**2. Độ rộng (số kênh), giữ 6 lớp conv của baseline.**
- `baseline_w0.5` `0.6747 ± 0.0049` (−0.0068, dưới std baseline) với 45% tham số; macro-F1 `0.6347 ± 0.0103` gần baseline. Thu hẹp một nửa gần như không mất gì: baseline dư sức chứa.
- `baseline_w2` `0.6654 ± 0.0070` (−0.0162, vượt std baseline) với 2.4× tham số; macro-F1 `0.5717 ± 0.0196` thấp hơn baseline khoảng 7 điểm, cả 3 seed đều thấp (0.582 / 0.584 / 0.549): mạng rộng đoán lệch về các lớp lớn. Nguyên nhân không phải quá khớp (khoảng train–val `0.023`, nhỏ hơn baseline) mà là **học chậm**: cả 3 seed đứng ở 0.2494 trong 2–3 epoch đầu, đỉnh ở epoch 99–100 và lr cuối còn 0.005 / 0.0025 / 0.00125, tức chưa hội tụ trong 100 epoch với lr 0.01. Không kết luận w2 kém về bản chất, chỉ kết luận: **với cấu hình huấn luyện cố định của nhóm, mở rộng kênh không có lợi**.

**3. BatchNorm và phần phân loại ở độ sâu của baseline.**
- `baseline_no_bn` `0.6941 ± 0.0033` (+0.0125, vượt std baseline): ở 6 lớp conv, bỏ BN không gây sập mà còn cao hơn baseline. BN chỉ cần thiết khi mạng sâu (mục 1).
- `baseline_gap` `0.6839 ± 0.0013` (+0.0024, dưới std baseline) với **1.28 M tham số (22% baseline)**: lớp FC sau Flatten chiếm 80% tham số của baseline nhưng không đóng góp vào độ chính xác. GAP là cách rẻ nhất để giảm kích thước mô hình.

**4. Độ sâu khi có BN.** `large_bn` **`0.7060 ± 0.0023`** (3 seed: 0.7086 / 0.7047 / 0.7047), cao nhất E1, **+0.0244** so với baseline (hơn 3 lần std baseline; seed thấp nhất vẫn cao hơn seed tốt nhất của baseline 0.6890), vượt kỷ lục public Kaggle 0.69. `val_macro_f1 = 0.6922 ± 0.0027`, hơn baseline `+0.051`, và dao động giữa các seed nhỏ hơn baseline khoảng 3 lần. `medium_01_bn` `0.6913` (1 seed). Đổi lại quá khớp mạnh nhất E1: khoảng train–val của `large_bn` `0.182 / 0.198 / 0.206` (baseline `0.040`), val_loss thấp nhất ở epoch 38–42 (0.900–0.916) rồi tăng lên 1.25–1.37 dù val_acc vẫn nhích tới epoch 84–90. Chi phí: 20.4 M tham số (3.5× baseline), ~25.5 s/epoch. **Tăng sâu có lợi hơn tăng rộng** cho FER2013 ở cùng cấu hình huấn luyện.

**5. Quyết định cho E7 (T18).**
- Baseline vẫn là mốc so sánh của E2–E6 (mỗi thí nghiệm đổi một yếu tố so với baseline); `large_bn` chỉ là ứng viên kiến trúc cho mô hình cuối.
- Ứng viên kiến trúc chính: `large_bn` (`0.7060 ± 0.0023`, đủ 3 seed). Nên kết hợp với điều chuẩn của E4 (label smoothing, dropout, weight decay) vì khoảng train–val lớn.
- Ứng viên dự phòng đã đủ 3 seed: `baseline_no_bn` (0.6941) và `small` (0.6934). Nếu cần mô hình nhỏ, dùng head GAP.
- Kết quả E2–E5 đo trên baseline nên chưa chắc còn đúng khi đổi sang `large_bn`; E7 nên so cả "baseline + lựa chọn tốt nhất" lẫn "large_bn + lựa chọn tốt nhất", mỗi bên 3 seed, thay vì cộng dồn mức cải thiện.

### [E2-001_s0] Khảo sát hàm kích hoạt ReLU kèm ghi chuẩn gradient (`relu`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: Chạy lại mốc đối chứng `ReLU` (giống Baseline) với cờ `log_grad_norms=true` để ghi chuẩn L2 gradient từng lớp (`grad_norms.csv`) làm mốc so sánh vanishing gradient với các hàm kích hoạt khác; `val_acc` kỳ vọng nằm trong khoảng dung sai `0.6815 ± 0.0075` của Baseline.
- Thay đổi so với baseline: `activation = "relu"`, bật `log_grad_norms = true` (không đổi bước cập nhật trọng số).
- Lệnh: `python src/run_experiment.py --config configs/E2_activation.json --member TV3 --only relu leaky_relu elu`
- Kết quả: status=Completed, best_val_loss=0.9032580699054402, val_acc=0.6862635831707997, val_macro_f1=0.6542281383200632, best_epoch=98, time_s=3417.117, run_dir=results/E2/E2-001_s0
- Nhận xét: Hoàn tất 100 epochs trên RTX 3050 (~57.0 phút). `val_acc` đạt **68.63%** (`+0.0047` so với trung bình baseline `68.15%`, nằm trong ngưỡng `< std baseline = 0.0075`), `val_macro_f1 = 65.42%`, `best_val_loss = 0.9033`. File `grad_norms.csv` ghi đủ **800 dòng** (100 epoch × 8 lớp Conv/Linear, 0 giá trị NaN): chuẩn L2 gradient lớp đầu `features.conv1_1` tăng ổn định từ `0.0594` (ep 1) lên `0.7241` (ep 100), trong khi lớp cuối `classifier.4` ở mức `1.0644 → 0.3592` — xác nhận dòng gradient truyền thông suốt về lớp đầu, không bị triệt tiêu.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Đạt chuẩn làm mốc đối chứng gradient cho toàn nhóm E2.

### [E2-002_s0] Khảo sát hàm kích hoạt LeakyReLU (`leaky_relu`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: LeakyReLU cho phép gradient nhỏ đi qua khi đầu vào âm, hạn chế hiện tượng Dying ReLU ở các lớp Conv/FC và có thể cải thiện nhẹ `val_acc` / `val_macro_f1` so với ReLU.
- Thay đổi so với baseline: `activation = "leaky_relu"`, bật `log_grad_norms = true`.
- Lệnh: `python src/run_experiment.py --config configs/E2_activation.json --member TV3 --only relu leaky_relu elu`
- Kết quả: status=Completed, best_val_loss=0.9016122124460156, val_acc=0.6935079409306213, val_macro_f1=0.666391284577099, best_epoch=98, time_s=3725.805, run_dir=results/E2/E2-002_s0
- Nhận xét: Hoàn tất 100 epochs trên RTX 3050 (~62.1 phút). `val_acc` đạt **69.35%** (**`+0.0120` cao hơn baseline `68.15%`**, vượt dung sai `0.0075`, và nhỉnh hơn `E2-001_s0` `relu` **+0.72%** trên cùng máy và cùng seed 0); `val_macro_f1` tăng mạnh lên **66.64%** (`+2.56%` so với baseline `64.08%`, `+1.22%` so với `relu`); `best_val_loss = 0.9016` thấp hơn cả Baseline và ReLU. File `grad_norms.csv` đủ **800 dòng** (0 NaN), chuẩn gradient `features.conv1_1` (`0.0577 → 0.6187`) và `classifier.4` (`1.2091 → 0.3694`) cân bằng tốt suốt 100 epochs.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Kết quả rất tích cực, ủng hộ giả thuyết giảm Dying ReLU; `leaky_relu` là ứng viên hàm kích hoạt mạnh cho `E7` (nên cân nhắc chạy thêm 2 seed nếu tiếp tục dẫn đầu sau khi hoàn tất `elu`, `gelu`, `relu_no_bn`).

### [E2-003_s0] Khảo sát hàm kích hoạt ELU (`elu`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: ELU có nhánh âm mượt tiến về `-1`, giúp đầu ra trung bình gần 0 hơn ReLU và có thể ổn định hội tụ.
- Thay đổi so với baseline: `activation = "elu"`, bật `log_grad_norms = true`.
- Lệnh: `python src/run_experiment.py --config configs/E2_activation.json --member TV3 --only elu gelu relu_no_bn`
- Kết quả: status=Completed, best_val_loss=0.9435755574759682, val_acc=0.6553357481192533, val_macro_f1=0.6168795023384526, best_epoch=95, time_s=1905.26, run_dir=results/E2/E2-003_s0
- Nhận xét: Hoàn tất 100 epochs trên RTX 3050 (~31.8 phút, ~18.9s/ep). Tuy nhiên `val_acc` chỉ đạt **65.53%** (**`-0.0262` thấp hơn baseline `68.15%`**), `val_macro_f1 = 61.69%`, `best_val_loss = 0.9436`. Đáng chú ý `train_acc_cuoi = 64.40%` còn thấp hơn `val_acc` (dấu hiệu underfitting dưới cùng `SGD lr=0.01 + BN + Dropout`), đồng thời trong `grad_norms.csv` (800 dòng, 0 NaN) chuẩn gradient ở lớp cuối `classifier.4` rất cao (`3.5450 → 1.1056`) so với `features.conv1_1` (`0.0561 → 0.5629`) cho thấy phân bố gradient giữa đầu/cuối mạng kém cân bằng hơn nhóm ReLU/LeakyReLU/GELU.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Kết luận `ELU` không phù hợp với cấu hình SGD+BN+Dropout của Baseline.

### [E2-004_s0] Khảo sát hàm kích hoạt GELU (`gelu`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: GELU làm mượt vùng chuyển tiếp quanh 0 theo xác suất chuẩn, cho gradient liên tục và có thể cải thiện khả năng tổng quát hóa so với ReLU.
- Thay đổi so với baseline: `activation = "gelu"`, bật `log_grad_norms = true`.
- Lệnh: `python src/run_experiment.py --config configs/E2_activation.json --member TV3 --only elu gelu relu_no_bn`
- Kết quả: status=Completed, best_val_loss=0.8806721030187726, val_acc=0.6918361660629702, val_macro_f1=0.6614454696521527, best_epoch=100, time_s=1905.97, run_dir=results/E2/E2-004_s0
- Nhận xét: Hoàn tất 100 epochs (~31.8 phút, ~18.9s/ep). `val_acc` đạt **69.18%** (**`+0.0103` cao hơn baseline**, vượt dung sai `0.0075`), `val_macro_f1 = 66.14%` (`+2.06%` so với baseline), và đặc biệt **`best_val_loss = 0.8807`** — thấp nhất trong cả 5 biến thể phần TV3 (giảm `0.0302` so với baseline `0.9109`)! File `grad_norms.csv` đủ 800 dòng (0 NaN), chuẩn gradient lớp đầu `features.conv1_1` khởi đầu mạnh nhất (`0.0922 → 0.5980`) và lớp cuối `classifier.4` rất cân bằng (`1.1106 → 0.3742`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Cùng với `leaky_relu` (`0.6935`), `gelu` (`0.6918`, loss `0.8807`) là ứng viên hàm kích hoạt hàng đầu cho `E7`.

### [E2-005_s0] Khảo sát đối chứng ReLU không BatchNorm (`relu_no_bn`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: Chạy đối chứng `ReLU` bỏ toàn bộ BatchNorm kèm `log_grad_norms=true` để so sánh trực tiếp chuẩn gradient từng lớp với `tanh_no_bn` / `sigmoid_no_bn` của TV4 và kiểm chứng tái lập độc lập với `E1-007_s0` (`baseline_no_bn = 0.6907` của TV2).
- Thay đổi so với baseline: `activation = "relu"`, `bn = false`, bật `log_grad_norms = true` (`5,866,951` tham số).
- Lệnh: `python src/run_experiment.py --config configs/E2_activation.json --member TV3 --only elu gelu relu_no_bn`
- Kết quả: status=Completed, best_val_loss=0.8968940197043048, val_acc=0.6912789077737531, val_macro_f1=0.6675158378052052, best_epoch=100, time_s=1821.821, run_dir=results/E2/E2-005_s0
- Nhận xét: Hoàn tất 100 epochs (~30.4 phút, nhanh nhất ở ~18.1s/ep do bỏ BN). `val_acc` đạt **69.13%** (**`+0.0098` cao hơn baseline**, khớp sát `E1-007_s0 = 0.6907` của TV2 trên RTX 4060 — chỉ lệch `0.0006`, xác nhận tái lập chéo cực tốt!), `val_macro_f1` đạt **66.75%** (cao nhất trong 5 biến thể E2 phần TV3), `best_val_loss = 0.8969`. Về `grad_norms.csv` (800 dòng, 0 NaN): ở epoch 1, chuẩn gradient lớp đầu `features.conv1_1` là `0.0132` (nhỏ hơn ~4.5× so với bản có BN `0.0594`), nhưng vẫn tăng khỏe lên `0.4189` ở epoch 100 (`classifier.4`: `0.3127 → 0.4839`) — khẳng định trên mạng VGG 6 lớp Conv + ReLU, bỏ BN **không** gây vanishing gradient (tạo mốc đối chứng lý tưởng cho `sigmoid_no_bn` của TV4).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.39 MB). Hoàn tất trọn vẹn 5/5 biến thể phần `C1` của TV3 trong `T13`.

### [E2-006_s0] Khảo sát hàm kích hoạt Tanh (`tanh`, Seed 0 — C2) — TV4 – Huỳnh Viễn Thông – 2026-11-26
- Giả thuyết: Tanh bão hòa ở hai đầu nhưng đạo hàm tại 0 bằng 1 và đầu ra đối xứng quanh 0, nên có BN thì gradient vẫn tới được lớp đầu; độ chính xác có thể thấp hơn ReLU vì vùng bão hòa.
- Thay đổi so với baseline: `activation = "tanh"`, bật `log_grad_norms = true`.
- Lệnh (tái lập): `python src/run_experiment.py --config configs/E2_activation.json --member TV4 --only tanh`
- Kết quả: status=Completed, best_val_loss=0.9687542175309974, val_acc=0.641404290888827, val_macro_f1=0.5951930862888543, best_epoch=92, time_s=3590.856, run_dir=results/E2/E2-006_s0
- Nhận xét (TV2 viết khi tổng hợp T13): Chạy đủ 100 epoch trên RTX 3060 (~60 phút). `val_acc = 0.6414` (−0.0401 so với baseline), `val_macro_f1 = 0.5952`; `train_acc_cuoi = 0.6388` thấp hơn val, tức thiếu khớp chứ không quá khớp. Gradient không bị triệt tiêu: ở epoch 1 chuẩn gradient `conv1_1` là `9.7e-2`, cao hơn cả ReLU (`5.9e-2`), tỉ lệ lớp đầu/lớp cuối `0.046` (ReLU `0.056`). Kém ReLU là do vùng bão hòa làm mạng học chậm, không phải do vanishing gradient. Mục này trước ghi nhầm là Sigmoid; dòng Kết quả tự điền cho thấy đây là run `tanh`.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Loại tanh khỏi ứng viên E7.

### [E2-007_s0] Khảo sát hàm kích hoạt Sigmoid có BN (`sigmoid`, Seed 0 — C2) — TV4 – Huỳnh Viễn Thông – 2026-11-26
- Giả thuyết: Đạo hàm sigmoid tối đa 0.25, đi qua 6 lớp Conv thì gradient tới lớp đầu co lại mạnh (vanishing gradient); BN giữ đầu vào mỗi lớp quanh 0 (vùng đạo hàm lớn nhất) nên có thể cứu được một phần.
- Thay đổi so với baseline: `activation = "sigmoid"`, bật `log_grad_norms = true`.
- Lệnh (tái lập): `python src/run_experiment.py --config configs/E2_activation.json --member TV4 --only sigmoid`
- Kết quả: status=Completed, best_val_loss=1.4288181444169685, val_acc=0.45110058512120366, val_macro_f1=0.3302818115004095, best_epoch=29, time_s=1322.763, run_dir=results/E2/E2-007_s0
- Nhận xét (TV2 viết khi tổng hợp T13): Dừng sớm ở epoch 44 (đỉnh epoch 29, ~21.5 phút): `val_acc = 0.4511` (−0.2304), `val_macro_f1 = 0.3303`, `train_acc_cuoi = 0.4152`, train_loss cuối 1.51. Epoch 1 có dấu hiệu vanishing gradient: chuẩn gradient `conv1_1` chỉ `1.8e-3` (ReLU `5.9e-2`, nhỏ hơn 33 lần), tỉ lệ lớp đầu/lớp cuối `2.5e-3`. Nhưng gradient hồi phục nhanh nhờ BN: `1.3e-2` ở epoch 3, `0.28` ở epoch 10, ngang ReLU từ đó tới cuối (Hình `figures/e2_grad_norms.png` (b)). val_acc dao động mạnh (0.23–0.45), lr bị cắt 3 lần, rồi dừng sớm khi mạng vẫn thiếu khớp nặng.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Loại sigmoid khỏi ứng viên E7.

### [E2-007_s1] Khảo sát hàm kích hoạt Sigmoid có BN (`sigmoid`, Seed 1) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: Sigmoid có BN dao động mạnh giữa các epoch nên một seed không đủ; chạy thêm seed 1, 2 để có mean ± std (bổ sung sau KT-017).
- Thay đổi so với baseline: `activation = "sigmoid"`, bật `log_grad_norms = true` (cùng cấu hình `E2-007_s0`, khác seed).
- Lệnh (tái lập): `python src/run_experiment.py --config configs/E2_activation.json --member TV2 --only sigmoid --seeds 0 1 2`
- Kết quả: status=Completed, best_val_loss=1.3039109300989133, val_acc=0.49986068542769574, val_macro_f1=0.3894467131185843, best_epoch=69, time_s=1230.553, run_dir=results/E2/E2-007_s1
- Nhận xét: Dừng sớm ở epoch 84 (đỉnh epoch 69, lr bị cắt 7 lần): `val_acc = 0.4999`, `val_macro_f1 = 0.3894`, `train_acc_cuoi = 0.4422`, vẫn thiếu khớp nặng. Gradient lặp lại đúng mẫu của seed 0: `conv1_1` ở epoch 1 là `2.1e-3`, tỉ lệ lớp đầu/lớp cuối `2.8e-3`, hồi phục lên `0.23` ở epoch 10.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB).

### [E2-007_s2] Khảo sát hàm kích hoạt Sigmoid có BN (`sigmoid`, Seed 2) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: Như `E2-007_s1`.
- Thay đổi so với baseline: `activation = "sigmoid"`, bật `log_grad_norms = true` (cùng cấu hình `E2-007_s0`, khác seed).
- Lệnh (tái lập): `python src/run_experiment.py --config configs/E2_activation.json --member TV2 --only sigmoid --seeds 0 1 2`
- Kết quả: status=Completed, best_val_loss=1.3958264401643912, val_acc=0.46614655893006407, val_macro_f1=0.3487409219043109, best_epoch=62, time_s=1129.014, run_dir=results/E2/E2-007_s2
- Nhận xét: Dừng sớm ở epoch 77 (đỉnh epoch 62, lr bị cắt 8 lần): `val_acc = 0.4661`, `val_macro_f1 = 0.3487`, `train_acc_cuoi = 0.4131`. Gradient `conv1_1` ở epoch 1 là `1.9e-3`, tỉ lệ `2.5e-3`, hồi phục lên `0.31` ở epoch 10.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB).

### [E2-010_s0] Chạy lại Sigmoid có BN seed 0 trên main mới nhất (`sigmoid`, Seed 0, cho KT-017) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: KT-017 Fail vì run của TV1 (`0.4790`, đỉnh epoch 85) lệch run gốc `E2-007_s0` (`0.4511`, RTX 3060) +0.0279. Code không đổi: giữa commit `286d85e8` (run gốc) và `99e0e02c` (run này) chỉ thêm tuỳ chọn `sampler`, mặc định `none` giữ nguyên DataLoader `shuffle=True`; `config.json` của hai run trùng nhau trừ mã run, máy và thời gian. Nếu nguyên nhân là sai khác giữa GPU cộng với early stopping trên đường val dao động mạnh, run này sẽ cũng lệch run gốc.
- Thay đổi so với baseline: `activation = "sigmoid"`, bật `log_grad_norms = true` (trùng cấu hình và seed của `E2-007_s0`, máy RTX 4060). Mã mới vì `E2-007_s0` đã có.
- Lệnh (tái lập): `python src/run_experiment.py --config configs/E2_activation.json --member TV2 --only sigmoid --seeds 0 1 2`
- Kết quả: status=Completed, best_val_loss=1.3347707404569518, val_acc=0.4889941487879632, val_macro_f1=0.38680280564429353, best_epoch=57, time_s=1073.093, run_dir=results/E2/E2-010_s0
- Nhận xét: Dừng sớm ở epoch 72 (đỉnh epoch 57, lr bị cắt 5 lần): `val_acc = 0.4890`, `val_macro_f1 = 0.3868`, `train_acc_cuoi = 0.4314`. So với run gốc: +0.0379; so với run của TV1: +0.0100. Ba lần chạy cùng seed 0 trên ba GPU (RTX 3060, GPU của TV1, RTX 4060) cho `0.4511 / 0.4790 / 0.4890`, chênh nhau tới 0.038. Epoch 1 đã khác ở chữ số thứ 3 (train_loss `1.8274` so với `1.8285`), sau đó tách xa dần. Run gốc là run dừng sớm nhất (epoch 44, đỉnh 29) nên thấp nhất. Phần định tính thì tái lập đúng: gradient `conv1_1` ở epoch 1 là `1.8e-3` (gốc `1.8e-3`), tỉ lệ lớp đầu/lớp cuối `2.3e-3` (gốc `2.5e-3`), hồi phục lên `0.29` ở epoch 10 (gốc `0.28`). Mức dao động val_acc giữa hai epoch liền nhau trung bình 0.027 (gốc 0.031), lớn nhất 0.135. Vì vậy dung sai 0.0075 (std baseline) không áp dụng được cho sigmoid. Xem Kết luận E2 mục 6.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB).

### [E2-008_s0] Khảo sát Tanh không BatchNorm (`tanh_no_bn`, Seed 0 — C2) — TV4 – Huỳnh Viễn Thông – 2026-11-26
- Giả thuyết: Bỏ BN để xem tanh có bị triệt tiêu gradient như sigmoid không; nếu không thì vanishing gradient là đặc tính riêng của sigmoid (đạo hàm ≤ 0.25, đầu ra không đối xứng quanh 0).
- Thay đổi so với baseline: `activation = "tanh"`, `bn = false`, bật `log_grad_norms = true` (`5,866,951` tham số).
- Lệnh (tái lập): `python src/run_experiment.py --config configs/E2_activation.json --member TV4 --only tanh_no_bn`
- Kết quả: status=Completed, best_val_loss=0.966715449441871, val_acc=0.6522708275285595, val_macro_f1=0.6201139639943545, best_epoch=99, time_s=2501.554, run_dir=results/E2/E2-008_s0
- Nhận xét (TV2 viết khi tổng hợp T13): Chạy đủ 100 epoch (~41 phút). `val_acc = 0.6523` (−0.0293), cao hơn bản có BN (`E2-006_s0`, 0.6414) khoảng 1 điểm (1 seed). `train_acc_cuoi = 0.6361`, vẫn thiếu khớp. Không vanishing gradient: chuẩn gradient `conv1_1` ở epoch 1 là `1.5e-2`, tỉ lệ lớp đầu/lớp cuối `0.035`, ngang `relu_no_bn` (`1.3e-2`, `0.042`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Dùng làm đối chứng cho `sigmoid_no_bn`.

### [E2-009_s0] Khảo sát Sigmoid không BatchNorm (`sigmoid_no_bn`, Seed 0 — C2) — TV4 – Huỳnh Viễn Thông – 2026-11-26
- Giả thuyết: Không có BN, sigmoid sẽ triệt tiêu gradient ở các lớp Conv đầu và mạng không học được.
- Thay đổi so với baseline: `activation = "sigmoid"`, `bn = false`, bật `log_grad_norms = true` (`5,866,951` tham số).
- Lệnh (tái lập): `python src/run_experiment.py --config configs/E2_activation.json --member TV4 --only sigmoid_no_bn`
- Kết quả: status=Completed, best_val_loss=1.8108197900703538, val_acc=0.24937308442463083, val_macro_f1=0.05702816362941251, best_epoch=1, time_s=474.724, run_dir=results/E2/E2-009_s0
- Nhận xét (TV2 viết khi tổng hợp T13): **Không học được.** Dừng sớm ở epoch 16, đỉnh ở epoch 1: `val_acc = 0.2494` (đoán toàn lớp Happy, 895/3589 ảnh PublicTest), `val_macro_f1 = 0.0570`. train_loss đứng ở 1.81 suốt 16 epoch, bằng entropy của phân bố lớp trong tập train (1.8096): mạng chỉ học được tần suất lớp qua bias của lớp FC cuối, không học gì từ ảnh. Chuẩn gradient ở epoch 1 giảm đều từ lớp ra về lớp vào: `fc2 8.1e-1 → fc1 6.5e-1 → conv3_2 4.6e-2 → conv3_1 4.9e-3 → conv2_2 6.9e-4 → conv2_1 7.9e-5 → conv1_2 1.1e-5 → conv1_1 3.2e-7`. Mỗi lớp Conv nhân gradient thêm khoảng 0.07–0.14 lần (lớp `conv1_1` 0.03), lớp đầu nhỏ hơn lớp cuối khoảng 2.5 triệu lần. Gradient lớp đầu còn giảm tiếp tới `1.0e-7` ở epoch 16 chứ không hồi phục. Đây là **vanishing gradient điển hình**.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB) làm bằng chứng cho báo cáo (mục E2).

### Kết luận E2 (T13) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
Bảng số liệu sinh bằng `python src/make_tables.py` → `report/tables/E2_summary.md` (Bảng 4.2). Hình chuẩn gradient sinh bằng `python src/visual/grad_norms.py --group E2 --epoch 1` → `figures/e2_grad_norms.png` (dữ liệu `figures/e2_grad_norms.csv`). Mỗi biến thể 1 seed, chạy trên 2 máy (TV3 RTX 3050, TV4 RTX 3060); riêng `sigmoid` có 4 run (seed 0, 1, 2 và một lần chạy lại seed 0, máy RTX 3060 và RTX 4060), Bảng 4.2 tính mean ± std trên cả 4 run. Mốc so sánh là baseline 3 seed `0.6815 ± 0.0075`; chênh lệch dưới std coi là chưa phân biệt được.

**1. Họ ReLU (có BN).**
- `relu` `0.6863` (+0.0047, dưới std). Cấu hình trùng baseline, chỉ bật ghi gradient và chạy máy khác; lệch `baseline-002_s0` 0.0048 là sai khác giữa các GPU. Cùng máy thì tái lập đúng tới từng chữ số (xem `E5-003_s0`).
- `leaky_relu` `0.6935` (+0.0120) và `gelu` `0.6918` (+0.0103, best_val_loss thấp nhất E2 0.8807) vượt std baseline, nhưng mới 1 seed và chỉ nhỉnh hơn ngưỡng.
- `elu` `0.6553` (−0.0262): thiếu khớp (`train_acc_cuoi` 0.644 thấp hơn val), gradient không có vấn đề (tỉ lệ lớp đầu/lớp cuối ở epoch 1 là 0.016).

**2. Hàm bão hòa.** `tanh` `0.6414` và `tanh_no_bn` `0.6523` kém ReLU 3–4 điểm và đều thiếu khớp. `sigmoid` `0.4765 ± 0.0220` (4 run, khoảng 0.451–0.500) thiếu khớp nặng. `sigmoid_no_bn` `0.2494` không học được.

**3. Vanishing gradient của Sigmoid (Hình e2_grad_norms).** Chuẩn gradient lớp Conv đầu ở epoch 1:

| Biến thể | conv1_1 | conv1_1 / fc2 | Diễn biến |
|---|---|---|---|
| `relu` | 5.9e-2 | 0.056 | tăng lên 0.72 ở epoch 100 |
| `relu_no_bn` | 1.3e-2 | 0.042 | tăng lên 0.42 |
| `tanh_no_bn` | 1.5e-2 | 0.035 | tăng lên 0.34 |
| `sigmoid` (có BN) | 1.8e-3 | 2.5e-3 | hồi phục: 0.28 ở epoch 10 |
| `sigmoid_no_bn` | **3.2e-7** | **4.0e-7** | giảm còn 1.0e-7 ở epoch 16, không hồi phục |

- **Xác nhận giả thuyết:** không có BN, sigmoid triệt tiêu gradient. Mỗi lớp Conv nhân gradient thêm khoảng 0.1 lần, khớp với đạo hàm sigmoid ≤ 0.25. Qua 6 lớp Conv, lớp đầu nhận gradient nhỏ hơn lớp ra khoảng 2.5 triệu lần, nên mạng chỉ học được tần suất lớp (train_loss đứng ở 1.81 = entropy phân bố lớp).
- **Vanishing gradient là đặc tính của sigmoid, không phải do bỏ BN hay do bão hòa nói chung:** `tanh_no_bn` (cũng bão hòa) và `relu_no_bn` có tỉ lệ lớp đầu/lớp cuối ngang nhau (0.035 và 0.042), lớn hơn `sigmoid_no_bn` khoảng 100.000 lần. Khác biệt: đạo hàm tanh tại 0 là 1 và đầu ra đối xứng quanh 0, còn sigmoid là 0.25 và đầu ra luôn dương.
- **BN sửa được gradient nhưng không sửa được độ chính xác:** với BN, gradient lớp đầu của sigmoid về ngang ReLU sau 10 epoch, nhưng val_acc vẫn chỉ `0.4765 ± 0.0220` và `train_acc_cuoi` 0.41–0.44 ở cả 4 run. Ở bản có BN, nút thắt không còn là vanishing gradient mà là sigmoid học rất chậm với cấu hình SGD lr 0.01 cố định, khiến plateau cắt lr 3–8 lần và early stopping dừng ở epoch 44–84. Chưa tách được nguyên nhân cụ thể (đầu ra không đối xứng quanh 0 hay vùng bão hòa); E2 không chạy thêm vì sigmoid đã bị loại.

**4. BatchNorm ở độ sâu baseline.** `relu_no_bn` `0.6913` khớp `baseline_no_bn` của E1 (`0.6941 ± 0.0033`, 3 seed, RTX 4060) trên một máy khác, xác nhận chéo kết luận E1: với 6 lớp Conv và ReLU, bỏ BN không gây vanishing gradient và không làm giảm độ chính xác. Với hàm bão hòa: `tanh_no_bn` hơn `tanh` khoảng 1 điểm (1 seed), còn sigmoid thì BN là điều kiện để học được.

**5. Quyết định cho E7 (T18).**
- Giữ họ ReLU. `leaky_relu` và `gelu` là ứng viên thay ReLU, nhưng cần chạy thêm seed 1, 2 trước khi chọn vì mức hơn (≈ +0.01) chỉ vừa vượt std baseline với 1 seed.
- Loại `sigmoid`, `tanh`, `elu`.
- Báo cáo (T22, mục E2): dùng `E2-009_s0` cùng Hình e2_grad_norms làm bằng chứng vanishing gradient của Sigmoid mà đề yêu cầu so với ReLU.

**6. Tái lập Sigmoid có BN (KT-017 Fail, chạy lại 2026-11-27).**

| Run | Máy | Seed | Dừng / đỉnh (epoch) | Số lần cắt lr | val_acc | conv1_1 epoch 1 |
|---|---|---|---|---|---|---|
| `E2-007_s0` (gốc) | RTX 3060 | 0 | 44 / 29 | 3 | 0.4511 | 1.8e-3 |
| KT-017 (TV1, không push) | GPU TV1 | 0 | 100 / 85 | — | 0.4790 | đạt (theo KT-017) |
| `E2-010_s0` | RTX 4060 | 0 | 72 / 57 | 5 | 0.4890 | 1.8e-3 |
| `E2-007_s1` | RTX 4060 | 1 | 84 / 69 | 7 | 0.4999 | 2.1e-3 |
| `E2-007_s2` | RTX 4060 | 2 | 77 / 62 | 8 | 0.4661 | 1.9e-3 |

- Code không phải nguyên nhân: cấu hình trùng nhau, phần code đổi giữa hai commit (tuỳ chọn `sampler`) không tác động khi `sampler = none`. Cùng máy thì run tái lập tới từng chữ số (`E5-003_s0` trùng `baseline-002_s0`).
- Nguyên nhân: khác GPU làm số học dấu phẩy động lệch ngay từ epoch 1, và với sigmoid độ lệch đó bị khuếch đại. Đường val_acc dao động trung bình 0.02–0.03 mỗi epoch, nhảy lớn nhất 0.10–0.18, nên epoch đỉnh, số lần cắt lr và thời điểm early stopping thay đổi theo máy. Run gốc dừng sớm nhất nên thấp nhất. Cùng seed 0, ba GPU cho kết quả chênh tới 0.038; ba seed trên cùng một máy chênh 0.034 (std `0.0172`).
- Dung sai 0.0075 là std của baseline (cấu hình ổn định), không áp dụng được cho một biến thể có std khoảng 0.02. Đề xuất kiểm lại T13 theo một trong hai cách: (a) với sigmoid, so run chạy lại với mean ± std 4 run (`0.4765 ± 0.0220`); run của TV1 lệch +0.0025, nằm trong 1 std; (b) dùng dung sai 0.0075 cho một biến thể ổn định của E2 (`relu`, `gelu` hoặc `leaky_relu`).
- Kết luận định tính của E2 không đổi: sigmoid có BN thấp hơn ReLU khoảng 0.21 ở cả 4 run, và mẫu gradient (lớp đầu nhỏ ở epoch 1, hồi phục ở epoch 10) tái lập ở mọi run.

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

### [E3-011_s0] Khảo sát SGD không Nesterov lr=0.01 (`sgd_no_nesterov_1e-2`) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Tắt Nesterov momentum (giữ SGD + momentum 0.9, lr=0.01) có thể làm hội tụ chậm hơn nhẹ hoặc tương đương Baseline trong dung sai 1 seed.
- Thay đổi so với baseline: `nesterov = false`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV1 --only sgd_no_nesterov_1e-2`
- Kết quả: status=Completed, best_val_loss=0.913428645905893, val_acc=0.68431317915854, val_macro_f1=0.6442004565407555, best_epoch=100, time_s=3751.397, run_dir=results/E3/E3-011_s0
- Nhận xét: Hoàn tất 100 epochs (`best_epoch = 100`), `val_acc = 0.6843` (`+0.0028`, nằm trong ngưỡng `< std baseline = 0.0075`), `val_macro_f1 = 0.6442`, `best_val_loss = 0.9134`. Sự khác biệt giữa SGD có và không Nesterov ở `lr = 0.01` là không đáng kể về mặt thống kê trên 1 seed.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Tương đương Baseline.

### [E3-012_s0] Khảo sát SGD Nesterov lr=0.03 (`sgd_nesterov_3e-2`) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Tăng bước học lên gấp 3 lần (`lr = 0.03`) có thể đẩy nhanh hội tụ ban đầu nhưng có rủi ro vượt quá giới hạn ổn định dẫn tới phân kỳ hoặc sập về lớp đa số.
- Thay đổi so với baseline: `lr = 0.03`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV1 --only sgd_nesterov_3e-2`
- Kết quả: status=Completed, best_val_loss=1.805903781519922, val_acc=0.24937308442463083, val_macro_f1=0.05702816362941251, best_epoch=2, time_s=409.531, run_dir=results/E3/E3-012_s0
- Nhận xét: Đúng rủi ro dự báo: mô hình sập hoàn toàn về dự đoán duy nhất lớp đa số `Happy` (`val_acc = 0.2494`, `val_macro_f1 = 0.0570`, `best_val_loss = 1.8059`) ngay từ epoch 2 và dừng sớm tại epoch 17. Chứng tỏ `lr = 0.03` vượt ngưỡng ổn định của kiến trúc Baseline + SGD Nesterov.
- Quyết định: Giữ weights (`KEEP` theo luật kỹ thuật của runner). Loại bỏ `lr = 0.03` cho SGD.

### [E3-013_s0] Khảo sát SGD Nesterov lr=0.003 (`sgd_nesterov_3e-3`) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Giảm bước học SGD xuống `0.003` giúp cập nhật êm hơn nhưng có thể hội tụ chậm trong 100 epochs và đạt val acc thấp hơn mốc `0.01`.
- Thay đổi so với baseline: `lr = 0.003`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV1 --only sgd_nesterov_3e-3`
- Kết quả: status=Completed, best_val_loss=0.9113988570763095, val_acc=0.674282529952633, val_macro_f1=0.6375821065412292, best_epoch=96, time_s=2267.82, run_dir=results/E3/E3-013_s0
- Nhận xét: Hoàn tất 100 epochs, `val_acc = 0.6743` (`-0.0072`, sát mép dưới dung sai `0.0075`), `val_macro_f1 = 0.6376`, `best_val_loss = 0.9114`. Bước học nhỏ khiến `train_acc_cuoi − val_acc = +0.0186` thấp nhưng không đủ đà đạt hiệu năng như `lr = 0.01`.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Xác nhận `lr = 0.01` là điểm ngọt tối ưu cho SGD Nesterov.

### [E3-014_s0] Khảo sát Adam lr=0.001 (`adam_1e-3`) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Adam chuẩn (coupled weight decay `5e-4`, `lr = 1e-3`) có thể đạt hiệu năng tốt hơn AdamW và cạnh tranh trực tiếp với SGD Nesterov trên tập validation.
- Thay đổi so với baseline: `optimizer = "adam"`, `lr = 0.001`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV1 --only adam_1e-3`
- Kết quả: status=Completed, best_val_loss=0.8921920168396489, val_acc=0.6865422123154082, val_macro_f1=0.6591569191377215, best_epoch=99, time_s=2483.627, run_dir=results/E3/E3-014_s0
- Nhận xét: Đạt kết quả rất khả quan: `val_acc = 0.6865` (`+0.0050`, nằm trong `< std baseline`), `val_macro_f1 = 0.6592` (`+1.84%` so với baseline), `best_val_loss = 0.8922` (phá mốc 0.90). Vượt trội rõ rệt so với `AdamW` cùng lr (`0.6723`, macro-F1 `0.5878`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Là phương án thay thế đáng cân nhắc bên cạnh SGD Nesterov.

### [E3-015_s0] Khảo sát Adam lr=0.0003 (`adam_3e-4`, bị ngắt giữa chừng) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Giảm bước học Adam từ `1e-3` xuống `3e-4` giúp tối ưu mượt hơn và cải thiện độ ổn định trên các lớp cảm xúc khó.
- Thay đổi so với baseline: `optimizer = "adam"`, `lr = 0.0003`.
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV1 --only adam_3e-4`
- Kết quả: status=Aborted, best_val_loss=0.9523329204687936, val_acc=0.6497631652270828, val_macro_f1=0.578402721067604, best_epoch=44, time_s=, run_dir=results/E3/E3-015_s0
- Nhận xét: Phiên chạy bị ngắt giữa chừng tại epoch 46 (`stopped_by = killed`, `INTERRUPTED@epoch46`, `train_time_s = 1287.6s` trên RTX 3050) khi mô hình mới đạt `val_acc = 0.6498` (epoch 44) và vẫn đang tiếp tục hội tụ.
- Quyết định: Hệ thống tự động đánh dấu `Aborted`, `DISCARD` và xóa `best.pt` dở, loại khỏi bảng thống kê `E3_summary.csv`; chạy lại hoàn chỉnh thành `E3-016_s0` ngay bên dưới.

### [E3-016_s0] Khảo sát Adam lr=0.0003 (`adam_3e-4`, chạy lại sau `E3-015_s0` bị Aborted) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Giảm bước học Adam từ `1e-3` xuống `3e-4` giúp tối ưu mượt hơn và cải thiện độ ổn định trên các lớp cảm xúc khó.
- Thay đổi so với baseline: `optimizer = "adam"`, `lr = 0.0003`. *(Ghi chú: lần chạy trước `E3-015_s0` bị ngắt phiên tại epoch 46, đã được `sync_orphans()` đánh dấu `Aborted`, `DISCARD` và xóa checkpoint dở; không đưa vào thống kê.)*
- Lệnh: `python src/run_experiment.py --config configs/E3_optim.json --member TV1 --only adam_3e-4`
- Kết quả: status=Completed, best_val_loss=0.9012737215711201, val_acc=0.6879353580384509, val_macro_f1=0.6586960077389981, best_epoch=100, time_s=2890.139, run_dir=results/E3/E3-016_s0
- Nhận xét: Hoàn tất 100 epochs (`best_epoch = 100`), đạt `val_acc = 0.6879` (**cao nhất nhóm optimizer thay thế**, `+0.0064`, sát ngưỡng `std baseline = 0.0075`), `val_macro_f1 = 0.6587`, `best_val_loss = 0.9013`. Cùng với `adam_1e-3` (`0.6865`) và `rmsprop_1e-3` (`0.6857`), cho thấy bộ tối ưu thích ứng hoạt động tốt khi không tách weight decay quá mạnh như `AdamW`.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Hoàn tất đủ 15/15 biến thể E3.

#### Tổng kết & Kết luận E3 (`T14` — Phụ trách R: TV3 – Võ Hoàng Bảo – 2026-11-26)
- **Nguồn số liệu & biểu đồ**: Tổng hợp đầy đủ **15/15 biến thể `Completed`** trong `report/tables/E3_report.md` và biểu đồ so sánh `figures/e3_comparison.png` (`figures/e3_comparison.csv`, đã đăng ký `tracking/figures_manifest.csv`). Toàn bộ 15 biến thể đã được kiểm chứng chỉ thay đổi đúng yếu tố khảo sát so với `configs/baseline.json` (mốc đối chuẩn `0.6815 ± 0.0075`, macro-F1 `0.6408 ± 0.0195`).
- **So sánh theo 3 trục khảo sát**:
  1. **Optimizer × Learning rate**:
     - `SGD Nesterov` rất nhạy với LR: `lr=0.03` phân kỳ/sập về lớp đa số (`0.2494`), `lr=0.003` hội tụ kém hơn (`0.6743`), **`lr=0.01`** là điểm tối ưu (`0.6815 ± 0.0075`; bỏ Nesterov `0.6843`, khác biệt `< std`).
     - Các bộ tối ưu thích ứng không tách weight decay mạnh đều nhỉnh nhẹ baseline trong ngưỡng `< std`: `adam_3e-4` (`0.6879`, F1 `0.6587`), `adam_1e-3` (`0.6865`, F1 `0.6592`, loss `0.8922`), `rmsprop_1e-3` (`0.6857`, F1 `0.6620`, **loss thấp nhất E3 `0.8809`**). Ngược lại `AdamW` (`1e-3`: `0.6723`, `3e-4`: `0.6704`) tụt rõ rệt cả accuracy lẫn macro-F1 (`0.5631–0.5878`).
  2. **Lịch học (Scheduler)**: **`OneCycleLR`** (`0.6863`, F1 `0.6584`, loss `0.8997`) nhỉnh nhẹ `ReduceLROnPlateau` (`0.6815`, trong ngưỡng `< std`), vượt `Cosine` (`0.6771`), `Constant` (`0.6760`) và vượt xa `StepLR` (`0.6648`, giảm bước cứng nhắc mỗi 20 epoch).
  3. **Batch size (ảnh hưởng mạnh nhất toàn E3)**: Quy luật đơn điệu rõ rệt `bs=64` (**`0.6924`**, F1 **`0.6642`**, **`+0.0109 > std baseline 0.0075`**) > `bs=128` (`0.6815`) > `bs=256` (`0.6726`, `-0.0089`). Batch nhỏ tăng nhiễu gradient giúp tổng quát hóa tốt hơn trên tập FER-2013 nhiều nhãn nhiễu.
- **Quyết định chọn cho `E7`**: Biến thể thắng rõ rệt nhất của E3 theo validation là **`SGD Nesterov lr=0.01 + ReduceLROnPlateau + batch_size=64`** (`val_acc = 0.6924`, vượt duy nhất ngưỡng `+std baseline` trong cả 15 cấu hình). Đề xuất đưa `batch_size = 64` vào cấu hình ứng viên `E7` (có thể phối hợp thử thêm `OneCycleLR` hoặc đối chứng `Adam 3e-4` ở bước tổng hợp E7). Chốt `done(T14)` chuyển TV2 (`T`) kiểm thử độc lập.

### [E4-001_s0] Khảo sát tắt Weight Decay (`weight_decay_0`, Seed 0 — C2) — TV4 – Huỳnh Viễn Thông – 2026-11-26
- Giả thuyết: Tắt hẳn điều chuẩn L2 (`weight_decay = 0.0`, thay vì `5e-4` của Baseline) trong khi vẫn giữ `flip_affine` và Dropout (`0.25/0.5`) sẽ làm mô hình dễ quá khớp hơn nhẹ hoặc ít thay đổi nếu Augmentation và Dropout đã gánh phần lớn vai trò điều chuẩn.
- Thay đổi so với baseline: `weight_decay = 0.0`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV4 --only weight_decay_0`
- Kết quả: status=Completed, best_val_loss=0.9103259716103023, val_acc=0.6818055168570633, val_macro_f1=0.6380659375439424, best_epoch=100, time_s=2507.098, run_dir=results/E4/E4-001_s0
- Nhận xét: Hoàn tất 100 epochs trên RTX 3060 (41.8 phút, đỉnh tại epoch 100). Đạt `val_acc = 68.18%` (`+0.0003` so với trung bình baseline `68.15%`, trong ngưỡng `< std = 0.0075`), `val_macro_f1 = 63.81%`, `best_val_loss = 0.9103`. Theo bảng tổng hợp khoảng cách quá khớp `figures/e4_overfit_summary.csv`, `acc_gap@best = +2.97%` và `final_loss_gap = +0.1565` (thấp hơn nhẹ baseline `+4.72% / +0.1759`), cho thấy trên kiến trúc baseline đã có `flip_affine` + Dropout, việc bỏ L2 weight decay không gây bùng phát quá khớp trong 100 epochs.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Làm mốc không weight decay cho nhóm Regularization của E4.

### [E4-002_s0] Khảo sát tăng Weight Decay lên 1e-3 (`weight_decay_1e-3`, Seed 0 — C2) — TV4 – Huỳnh Viễn Thông – 2026-11-26
- Giả thuyết: Tăng gấp đôi hệ số điều chuẩn L2 (`weight_decay = 0.001` so với `0.0005` của Baseline) giúp phạt trọng số lớn mạnh hơn, có thể cải thiện nhẹ khả năng tổng quát hóa trên PublicTest.
- Thay đổi so với baseline: `weight_decay = 0.001`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV4 --only weight_decay_1e-3`
- Kết quả: status=Completed, best_val_loss=0.899992476468898, val_acc=0.6840345500139314, val_macro_f1=0.6510023115710255, best_epoch=95, time_s=2472.968, run_dir=results/E4/E4-002_s0
- Nhận xét: Hoàn tất 100 epochs trên RTX 3060 (41.2 phút, đỉnh ở epoch 95). Đạt **`val_acc = 68.40%`** (`+0.0025` so với baseline, trong `< std = 0.0075`), **`val_macro_f1 = 65.10%`** (`+1.02%` so với baseline `64.08%`, cao nhất trong 3 biến thể regularization phần TV4) và hạ `best_val_loss` xuống **`0.9000`**. Theo `figures/e4_overfit_summary.csv`, khoảng cách quá khớp ở mức vừa phải (`acc_gap@best = +5.24%`, `final_acc_gap = +6.00%`, `final_loss_gap = +0.2142`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Ứng viên regularization cân bằng tốt cả `val_acc`, `macro-F1` và `val_loss`.

### [E4-003_s0] Khảo sát Label Smoothing 0.1 (`label_smoothing_0_1`, Seed 0 — C2) — TV4 – Huỳnh Viễn Thông – 2026-11-26
- Giả thuyết: FER-2013 có tỷ lệ nhiễu nhãn cao; áp dụng `label_smoothing = 0.1` sẽ ngăn mô hình đẩy logit quá tự tin vào nhãn nhiễu, qua đó thu hẹp khoảng cách quá khớp `Train − Val` và cải thiện hiệu chỉnh xác suất (`val_loss`).
- Thay đổi so với baseline: `label_smoothing = 0.1`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV4 --only label_smoothing_0_1`
- Kết quả: status=Completed, best_val_loss=0.8896598656485093, val_acc=0.6829200334354973, val_macro_f1=0.627969412229088, best_epoch=94, time_s=2460.578, run_dir=results/E4/E4-003_s0
- Nhận xét: Hoàn tất 100 epochs trên RTX 3060 (41.0 phút, đỉnh ở epoch 94). Xác nhận rõ giả thuyết về hiệu chỉnh xác suất: **`best_val_loss = 0.8897`** (thấp nhất toàn bộ 10 biến thể E4 và thấp hơn rõ rệt Baseline `0.9109`), đồng thời kéo `acc_gap@best` giảm từ `+4.72%` (Baseline) xuống **`+3.32%`** (`final_loss_gap = -0.1601` do label smoothing đặt cận dưới cho train loss trong khi val loss đánh giá bằng CE chuẩn, xem `figures/e4_overfit.png/.csv`). Tuy nhiên `val_acc = 68.29%` chỉ nhỉnh nhẹ trong dung sai (`+0.0014`) và `val_macro_f1 = 62.80%` giảm nhẹ `-1.28%` so với Baseline.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Đã vẽ trọn bộ biểu đồ đường cong Train/Val & Overfit đủ 10/10 biến thể tại `figures/e4_overfit.png` (kèm `figures/e4_overfit.csv` và `figures/e4_overfit_summary.csv` bằng `src/visual/e4_overfit.py`) để bàn giao TV1 (`R` của T15) tổng hợp kết luận E4.

### [E4-004_s0] Khảo sát không dùng Data Augmentation (`aug_none`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: Tắt toàn bộ Data Augmentation sẽ khiến mô hình ghi nhớ tập huấn luyện rất nhanh, gây quá khớp nghiêm trọng (`Train − Val > 20%`) và kéo tụt `val_acc` so với Baseline (`0.6815 ± 0.0075`).
- Thay đổi so với baseline: `augmentation = "none"`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV3 --only aug_none aug_flip aug_flip_affine aug_random_erasing`
- Kết quả: status=Completed, best_val_loss=1.0511874254449358, val_acc=0.6656450264697688, val_macro_f1=0.6495204410982652, best_epoch=71, time_s=1172.345, run_dir=results/E4/E4-004_s0
- Nhận xét: Xác nhận hoàn toàn giả thuyết! Không có augmentation, mô hình đạt `train_acc_cuoi = 96.18%` nhưng `val_acc` chỉ đạt **66.56%** (`-0.0159` thấp hơn rõ rệt mốc baseline `68.15%`, vượt xa `std = 0.0075`), `best_val_loss` vọt lên `1.0512`, khoảng cách quá khớp **`Train − Val = +0.2962` (+29.62%)** và dừng sớm tại epoch `86/100` (`best_epoch = 71`, 19.5 phút).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Làm mốc đối chứng quá khớp cho nhóm Augmentation trong E4.

### [E4-005_s0] Khảo sát chỉ lật ngang (`aug_flip`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: Chỉ bật `RandomHorizontalFlip(p=0.5)` giúp tăng gấp đôi biến thiên đối xứng trái–phải, giảm bớt quá khớp so với `aug_none` nhưng vẫn chưa đủ chống overfit bằng `flip_affine`.
- Thay đổi so với baseline: `augmentation = "flip"`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV3 --only aug_none aug_flip aug_flip_affine aug_random_erasing`
- Kết quả: status=Completed, best_val_loss=1.001622456964858, val_acc=0.6692672053496795, val_macro_f1=0.6470134243853707, best_epoch=93, time_s=1361.514, run_dir=results/E4/E4-005_s0
- Nhận xét: Hoàn tất 100 epochs (22.7 phút). Lật ngang giúp giảm khoảng cách quá khớp từ `+0.2962` (`aug_none`) xuống **`+0.2341`** (`train_acc_cuoi = 90.34%`) và giảm `best_val_loss` từ `1.0512` xuống `1.0016`, tuy nhiên `val_acc = 66.93%` (`-0.0123` so với baseline) vẫn thấp hơn rõ rệt Baseline và vẫn quá khớp mạnh.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Chứng minh chỉ dùng Flip là chưa đủ cho FER-2013.

### [E4-006_s0] Khảo sát đối chứng Flip + Affine (`aug_flip_affine`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: Bật đầy đủ `Flip + RandomAffine` (xoay ±10°, dịch ±10%, tỉ lệ 0.9–1.1 — giống Baseline) sẽ cắt giảm mạnh quá khớp và đẩy `val_acc` trở lại quanh mốc Baseline (`0.6815 ± 0.0075`).
- Thay đổi so với baseline: `augmentation = "flip_affine"` (mốc đối chứng trong nhóm E4, seed 0 trên `Hbao`).
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV3 --only aug_none aug_flip aug_flip_affine aug_random_erasing`
- Kết quả: status=Completed, best_val_loss=0.9032580699054402, val_acc=0.6862635831707997, val_macro_f1=0.6542281383200632, best_epoch=98, time_s=2069.332, run_dir=results/E4/E4-006_s0
- Nhận xét: Bước nhảy lớn nhất trong cả chuỗi Augmentation! Thêm `RandomAffine` kéo khoảng cách `Train − Val` giảm mạnh gần **5×** từ `+0.2341` (`aug_flip`) xuống chỉ còn **`+0.0500`** (`train_acc_cuoi = 73.62%`), `best_val_loss` hạ sâu từ `1.0016` xuống **`0.9033`**, `val_acc` vọt lên **`68.63%`** (`+0.0047`, trong ngưỡng `< std baseline`; tái lập trùng khớp tuyệt đối tới chữ số thứ 16 với `E2-001_s0` trên cùng máy `Hbao` seed 0!) và `val_macro_f1 = 65.42%`.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Khẳng định `flip_affine` là yếu tố chống quá khớp cốt lõi của Baseline.

### [E4-007_s0] Khảo sát Flip + Affine + RandomErasing (`aug_random_erasing`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-26
- Giả thuyết: Cộng thêm `RandomErasing(p=0.5)` che ngẫu nhiên vùng nhỏ trên khuôn mặt có thể ép mô hình học đặc trưng cục bộ đa dạng hơn và tiếp tục thu hẹp khoảng cách `Train − Val`, nhưng cũng có rủi ro che mất vùng mắt/miệng quan trọng trên ảnh nhỏ 48×48.
- Thay đổi so với baseline: `augmentation = "random_erasing"` (`flip + affine + RandomErasing(p=0.5)`).
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV3 --only aug_none aug_flip aug_flip_affine aug_random_erasing`
- Kết quả: status=Completed, best_val_loss=0.9010044118095222, val_acc=0.6798551128448036, val_macro_f1=0.6405506586565944, best_epoch=99, time_s=2230.471, run_dir=results/E4/E4-007_s0
- Nhận xét: Hoàn tất 100 epochs (37.2 phút). `RandomErasing(p=0.5)` kéo khoảng cách `Train − Val` xuống gần bằng 0 (**`+0.0069`**, `train_acc_cuoi = 68.67%`) và đưa `best_val_loss` xuống thấp nhất nhóm (`0.9010`), nhưng mô hình học chậm hơn đáng kể (đến tận epoch 92 mới giảm lr lần đầu) và đạt `val_acc = 67.99%` (`-0.0017`, trong `< std baseline`, thấp hơn `aug_flip_affine` `0.6863` là `-0.64%`), `val_macro_f1 = 64.06%`. Với ảnh độ phân giải nhỏ `48×48` và `p = 0.5`, che ngẫu nhiên quá mạnh kèm `Dropout(0.25/0.5)` làm tăng độ khó bài toán huấn luyện (hướng nhẹ về underfit trong 100 epochs).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Kết luận trong nhóm Augmentation của E4: **`aug_flip_affine` (`0.6863`)** cho điểm cân bằng tối ưu giữa chống quá khớp và độ chính xác; `aug_random_erasing` (`p=0.5`) chống quá khớp mạnh nhất nhưng chưa vượt `flip_affine` trong 100 epochs.

### [E4-008_s0] Khảo sát tắt toàn bộ Dropout (`dropout_0`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-27
- Giả thuyết: Tắt cả `dropout_conv` và `dropout_fc` (`0.0 / 0.0`) dù vẫn giữ `flip_affine` augmentation sẽ làm mô hình hội tụ nhanh hơn nhưng gia tăng rõ rệt khoảng cách quá khớp `Train − Val` và đẩy `val_loss` tăng sớm hơn so với Baseline (`dropout_conv = 0.25, dropout_fc = 0.5`).
- Thay đổi so với baseline: `model_kwargs = {"dropout_conv": 0.0, "dropout_fc": 0.0}`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV3 --only dropout_0 dropout_0_25 dropout_0_5`
- Kết quả: status=Completed, best_val_loss=0.9481581089788241, val_acc=0.689607132906102, val_macro_f1=0.6745816000367967, best_epoch=52, time_s=1350.048, run_dir=results/E4/E4-008_s0
- Nhận xét: Kết quả rất thú vị về đánh đổi độ chính xác và hiệu chỉnh xác suất (calibration)! Không có Dropout, mô hình học rất nhanh và đạt đỉnh sớm tại `best_epoch = 52`, dừng sớm ở epoch `67/100` (21.9 phút). Đạt **`val_acc = 68.96%`** (`+0.0081` so với mốc baseline `0.6815`, hơi nhỉnh ngưỡng `std = 0.0075` nhưng mới 1 seed) và **`val_macro_f1 = 67.46%`** (**cao nhất trong cả 10 biến thể E4**, `+0.0204` so với `aug_flip_affine`). Tuy nhiên đúng như giả thuyết về quá khớp: `train_acc_cuoi` vọt lên `84.98%` (`Train − Val = +0.1602`, cao gấp 3.2× `aug_flip_affine` `+0.0500`) và **`best_val_loss = 0.9482`** xấu hơn hẳn `aug_flip_affine` (`0.9033`) và `dropout_0_25` (`0.8991`), cho thấy dự đoán bắt đầu quá tự tin (overconfident) khi thiếu Dropout.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Đánh dấu là ứng viên đáng chú ý nhưng cần kiểm tra thận trọng cùng `dropout_0_25` (vì quá khớp mạnh hơn và mới 1 seed).

### [E4-009_s0] Khảo sát Dropout nhẹ đồng đều (`dropout_0_25`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-27
- Giả thuyết: Giảm `dropout_fc` từ `0.5` (Baseline) xuống `0.25` (giữ `dropout_conv = 0.25`) có thể giải phóng thêm sức biểu diễn ở tầng phân loại FC khi đã có `flip_affine` hỗ trợ chống quá khớp, từ đó cải thiện đồng thời `val_acc`, `macro-F1` và giữ ổn định `val_loss`.
- Thay đổi so với baseline: `model_kwargs = {"dropout_conv": 0.25, "dropout_fc": 0.25}`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV3 --only dropout_0 dropout_0_25 dropout_0_5`
- Kết quả: status=Completed, best_val_loss=0.899058769525808, val_acc=0.6887712454722764, val_macro_f1=0.6642002723751766, best_epoch=93, time_s=1909.206, run_dir=results/E4/E4-009_s0
- Nhận xét: Hoàn tất 100 epochs (31.6 phút, `best_epoch = 93`). Xác nhận trọn vẹn giả thuyết! **`dropout_0_25`** đạt điểm cân bằng xuất sắc nhất trong nhóm Dropout: `val_acc = 68.88%` (`+0.0072`, sát ngưỡng `std baseline = 0.0075` và cao hơn đối chứng `aug_flip_affine` `68.63%` trên cùng máy `Hbao` seed 0), `val_macro_f1 = 66.42%` (`+0.0100` so với `aug_flip_affine`), đồng thời hạ **`best_val_loss` xuống `0.8991`** (thấp nhất trong toàn bộ 7 run phần TV3 và thấp hơn Baseline `0.9109`), trong khi khoảng cách quá khớp vẫn rất chặt (**`Train − Val = +0.0598`**, `train_acc_cuoi = 74.86%`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB). Khuyến nghị ưu tiên **`dropout_conv = 0.25, dropout_fc = 0.25`** (hơn là `dropout_fc = 0.5` của Baseline hoặc tắt hẳn `0.0`) khi đề xuất cho TV1 (`R` của T15) tổng hợp và cân nhắc đưa vào `E7`.

### [E4-010_s0] Khảo sát Dropout mạnh 0.5 toàn mạng (`dropout_0_5`, Seed 0 — C1) — TV3 – Võ Hoàng Bảo – 2026-11-27
- Giả thuyết: Tăng `dropout_conv` lên `0.5` (cùng `dropout_fc = 0.5`) trên ảnh nhỏ `48×48` kết hợp `flip_affine` sẽ điều chuẩn quá tay, làm mất quá nhiều thông tin đặc trưng không gian và dẫn tới hiện tượng **underfitting** (giảm mạnh `val_acc` và `macro-F1`).
- Thay đổi so với baseline: `model_kwargs = {"dropout_conv": 0.5, "dropout_fc": 0.5}`.
- Lệnh: `python src/run_experiment.py --config configs/E4_aug_reg.json --member TV3 --only dropout_0 dropout_0_25 dropout_0_5`
- Kết quả: status=Completed, best_val_loss=0.9196882782024938, val_acc=0.6662022847589858, val_macro_f1=0.6165225055837354, best_epoch=98, time_s=1910.222, run_dir=results/E4/E4-010_s0
- Nhận xét: Hoàn tất 100 epochs (31.7 phút, `best_epoch = 98`). Xác nhận rõ rệt hiện tượng **underfitting**! `train_acc_cuoi = 65.25%` thậm chí **thấp hơn `val_acc = 66.62%`** (`Train − Val = -0.0137`), `val_acc` tụt mạnh `-0.0153` so với Baseline (vượt quá `2 × std baseline`) và `val_macro_f1` giảm sâu xuống **`61.65%`** (thấp nhất toàn bộ E4). Việc loại bỏ 50% neuron ngay tại các tầng tích chập làm vỡ cấu trúc biểu cảm vi mô trên khuôn mặt `48×48`, đặc biệt gây hại cho các lớp cảm xúc thiểu số.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB) làm bằng chứng phản chứng về giới hạn trên của Dropout. TV3 (`C1`) hoàn tất đủ **7/7 biến thể** được giao trong `T15` (bảng `E4_report.md` đã đủ **10/10 biến thể**); bàn giao TV4 (`C2`) vẽ lại hình `e4_overfit.png` đủ 10 biến thể và TV1 (`R`) tổng hợp kết luận + chốt `done(T15)`.

### T15 – Tổng hợp kết quả E4 (TV1 – R)

- Đã gom đủ **10/10 biến thể E4**, đều có trạng thái `Completed` trong `tracking/experiments.csv` và có `history.csv`/`run.log`. Bảng kết quả nằm ở `report/tables/E4_report.md` và `report/tables/E4_summary.csv`; biểu đồ đường cong cùng số liệu theo epoch và bảng khoảng cách overfit được sinh từ history tại `figures/e4_overfit.png`, `figures/e4_overfit.csv` và `figures/e4_overfit_summary.csv`.
- **Augmentation:** tại epoch có `best_val_acc`, khoảng cách train–val của baseline là 4.72 điểm %. Không augmentation là 28.09 điểm %, chỉ flip là 22.04 điểm %, flip+affine là 4.72 điểm %, còn random erasing là 0.49 điểm %. Random erasing giảm gap mạnh nhất nhưng `val_acc` 67.99%, thấp hơn trung bình baseline 0.17 điểm %; flip+affine đạt 68.63% và gap gần baseline, nên là lựa chọn cân bằng hơn giữa validation accuracy và overfit.
- **Dropout:** tắt dropout cho gap 10.97 điểm %; dropout 0.25 giảm gap còn 5.55 điểm % và đạt `val_acc` 68.88%. Dropout 0.5 làm gap âm (-1.42 điểm %) nhưng `val_acc` giảm còn 66.62%, phù hợp với dấu hiệu underfitting; không nên xem gap nhỏ đơn lẻ là kết quả tốt hơn.
- **Weight decay và label smoothing:** bỏ weight decay cho gap 2.97 điểm %, còn tăng lên `1e-3` cho gap 5.24 điểm % (baseline 4.72 điểm %), nên tác động của weight decay không đơn điệu trong các run này. Label smoothing cho gap 3.32 điểm % và `best_val_loss` thấp nhất trong bảng E4 (0.8897); kết quả validation có triển vọng, nhưng cần thêm seed trước khi khẳng định.
- **Kết luận:** random erasing giảm gap đo được mạnh nhất; flip+affine và dropout 0.25 cho cân bằng tốt hơn giữa gap và validation score; dropout 0.5 có dấu hiệu điều chuẩn quá mức. Kết quả weight decay chưa cho xu hướng nhất quán. Mỗi biến thể mới có một seed và các run dùng RTX 3050/3060, baseline tham chiếu dùng RTX 4060, vì vậy đây là kết luận mô tả để chọn hướng chạy tiếp, chưa phải bằng chứng về khác biệt có ý nghĩa thống kê.

### [E5-001_s0] Khảo sát mất cân bằng lớp bằng WeightedRandomSampler (`weighted_random_sampler`, Seed 0 — C2) — TV4 – Huỳnh Viễn Thông – 2026-11-26
- Giả thuyết: Lớp `Disgust` chỉ có 436 ảnh trong Training trong khi `Happy` có 7,215 ảnh (~16.5×). Dùng `WeightedRandomSampler` (trọng số nghịch đảo tần suất lớp trên riêng tập Training) để cân bằng tần suất xuất hiện các lớp trong mỗi epoch sẽ tăng mạnh Recall của `Disgust` và `Fear` mà không làm sụt giảm tổng Accuracy như `class_weight`.
- Thay đổi so với baseline: `sampler = "weighted_random"` (`class_weights = null`).
- Lệnh: `python src/run_experiment.py --config configs/E5_imbalance.json --member TV4 --only weighted_random_sampler`
- Kết quả: status=Completed, best_val_loss=0.9361300506974436, val_acc=0.6823627751462803, val_macro_f1=0.6716885337089461, best_epoch=100, time_s=2471.275, run_dir=results/E5/E5-001_s0
- Nhận xét: Hoàn tất 100 epochs trên RTX 3060 (41.2 phút, đỉnh tại epoch 100). Xác nhận xuất sắc giả thuyết! Đạt **`val_acc = 68.24%`** (vượt nhẹ mốc CE chuẩn `E5-003_s0` `68.15%`, không bị đánh đổi accuracy như `class_weight` `66.12%`) và bứt phá **`val_macro_f1 = 67.17%`** (**`+3.01%`** so với CE chuẩn `64.16%`, **`+3.09%`** so với trung bình baseline `64.08%`). Đánh giá độc lập từng lớp trên `PublicTest` bằng `src/visual/e5_recall.py` (lưu `results/E5/E5-001_s0/val_predictions.npz`, `per_class_recall.csv` và `figures/e5_wrs_recall.png/.csv`, khớp `metrics.json` sai số `0.00e+00`): Angry `65.31%` (prec `58.99%`), **Disgust `67.86%` (`38/56`, +30.36% so với CE `37.50%`, prec `74.51%`, F1 `71.03%`)**, **Fear `44.96%` (`223/496`, +2.86% so với CE `42.10%` và vượt hẳn `class_weight` `35.5%`)**, Happy `85.14%` (prec `87.29%`), Sad `55.90%`, Surprise `85.54%`, Neutral `66.06%`.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.40 MB) kèm file dự đoán `val_predictions.npz` phục vụ Tester (`TV3`) tái tính recall độc lập. Là phương pháp xử lý mất cân bằng tốt nhất trong E5 (vừa giữ nguyên Accuracy vừa tăng mạnh Macro-F1 và Recall cho cả `Disgust` lẫn `Fear`); bàn giao TV1 (`R` của T16) tổng hợp kết luận E5.

### [E5-003_s0] CE thường, mốc đối chứng của E5 (`ce`, Seed 0 — C1) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: Biến thể `ce` có cấu hình trùng baseline (`sampler = "none"`, `class_weights = null`), nên phải cho lại đúng kết quả `baseline-002_s0` khi cùng seed 0 và cùng máy; đây là mốc để đọc recall từng lớp của `class_weight` và `weighted_random_sampler`.
- Thay đổi so với baseline: không đổi gì.
- Lệnh: `python src/run_experiment.py --config configs/E5_imbalance.json --member TV2 --only ce class_weight --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.915008519412615, val_acc=0.6815268877124547, val_macro_f1=0.6416322850865286, best_epoch=99, time_s=1560.584, run_dir=results/E5/E5-003_s0
- Nhận xét: Chạy đủ 100 epoch trên RTX 4060 (~26 phút). **Tái lập tất định:** `val_acc = 0.6815`, `val_macro_f1 = 0.6416`, best epoch 99, cả 100 dòng `history.csv` (trừ cột thời gian) và sha256 `best.pt` (`021c02e6…`) trùng `baseline-002_s0`. Recall/precision từng lớp trên PublicTest: Angry 0.647/0.574, **Disgust 0.375/0.778 (21/56 ảnh)**, **Fear 0.421/0.585**, Happy 0.883/0.853, Sad 0.574/0.585, Surprise 0.836/0.787, Neutral 0.662/0.599. Hai lớp yếu nhất là Disgust và Fear; Disgust ít được đoán (precision cao, recall thấp). Run được hàng đợi cấp mã `E5-001_s0` trước khi biết TV4 đã dùng mã đó cho `weighted_random_sampler` (máy khác, chạy cùng lúc), nên đổi sang `E5-003_s0` khi gộp (thư mục, `config.json`, `metrics.json`, `run.log`, dòng CSV; `curves.png` vẽ lại; nội dung run không đổi, ghi chú ở dòng đầu `run.log`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Dùng làm mốc CE của E5.

### [E5-002_s0] CE có trọng số lớp (`class_weight`, Seed 0 — C1) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: Trọng số lớp n/(7·số ảnh) (Disgust 9.41, Fear 1.29, Happy 0.57, …) làm lỗi trên lớp hiếm nặng hơn, nên recall Disgust và Fear tăng, đổi lại accuracy có thể giảm vì mô hình bớt đoán lớp đông.
- Thay đổi so với baseline: `class_weights = [1.026605, 9.406619, 1.001046, 0.568439, 0.849127, 1.293373, 0.826039]` cho loss train (loss val vẫn là CE thường nên so được với các run khác).
- Lệnh: `python src/run_experiment.py --config configs/E5_imbalance.json --member TV2 --only ce class_weight --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9414065220987741, val_acc=0.6611869601560323, val_macro_f1=0.639682417264928, best_epoch=97, time_s=1509.854, run_dir=results/E5/E5-002_s0
- Nhận xét: Chạy đủ 100 epoch (~25 phút), đỉnh ở epoch 97, lr cuối 0.000625. `val_acc = 0.6612`, **−0.0203 so với baseline** (vượt std 0.0075); `val_macro_f1 = 0.6397`, gần như không đổi so với `E5-003_s0` (0.6416). Recall/precision từng lớp: Angry 0.651/0.587, **Disgust 0.732/0.547 (41/56 ảnh, +0.357 recall)**, **Fear 0.355/0.515 (giảm 0.066)**, Happy 0.837/0.891, Sad 0.531/0.548, Surprise 0.851/0.759, Neutral 0.664/0.564. Trọng số 9.4 kéo mạnh Disgust nhưng Fear (trọng số chỉ 1.29) lại giảm recall, Sad và Happy cũng giảm; chỉ Disgust được lợi rõ (chưa xem ma trận nhầm lẫn nên chưa rõ Fear bị nhầm sang lớp nào). `train_acc_cuoi = 0.6653` gần val (0.6539 ở epoch cuối), khoảng train–val nhỏ hơn baseline nhiều vì loss có trọng số không tối ưu trực tiếp accuracy. So với `weighted_random_sampler` của TV4 (`E5-001_s0`, val 0.6824, macro-F1 0.6717, recall Disgust 0.679, Fear 0.450 theo `figures/e5_wrs_recall.csv`): sampler giữ accuracy và tăng cả hai lớp, class weight thì không.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 22.4 MB). Một seed: không chọn class weight cho E7 nếu mục tiêu là accuracy; nếu cần recall Disgust thì `weighted_random_sampler` tốt hơn. Kết luận nhóm E5 do TV1 (R của T16) tổng hợp.

### T16 – Tổng hợp E5: recall lớp hiếm và đánh đổi accuracy (TV1 – R)

Đã đối chiếu đủ ba run `Completed` và bảng `report/tables/E5_report.md`/`E5_summary.csv` với `tracking/experiments.csv`. Accuracy và macro-F1 trong bảng dưới lấy từ các run; recall của `E5-001_s0` được tính lại từ `val_predictions.npz`. Recall CE và class-weight được ghi trong nhận xét của TV2 cho các run tương ứng; checkpoint và prediction file của hai run này không có trong workspace hiện tại, nên các số đó vẫn chờ TV3 truy xuất độc lập.

| Biến thể | Val accuracy | Macro-F1 | Recall Disgust | Recall Fear | So với CE |
|---|---:|---:|---:|---:|---|
| CE (`E5-003_s0`) | 68.15% | 64.16% | 37.50% (21/56) | 42.1% | Mốc |
| Class-weight (`E5-002_s0`) | 66.12% | 63.97% | 73.2% (41/56) | 35.5% | Accuracy −2.03 điểm % |
| WeightedRandomSampler (`E5-001_s0`) | 68.24% | 67.17% | 67.86% (38/56) | 44.96% (223/496) | Accuracy +0.08 điểm % |

**Kết luận:** Class-weight tăng recall Disgust khoảng 35.7 điểm % nhưng làm recall Fear giảm khoảng 6.6 điểm % và accuracy giảm 2.03 điểm %. WeightedRandomSampler tăng Disgust khoảng 30.4 điểm % và Fear khoảng 2.9 điểm % so với CE, trong khi giữ accuracy gần như nguyên vẹn và tăng macro-F1 khoảng 3.01 điểm %. Trong ba run seed 0, sampler là lựa chọn cân bằng tốt nhất cho cả hai lớp hiếm; class-weight chỉ phù hợp nếu ưu tiên Disgust và chấp nhận suy giảm Fear/accuracy.

Đây là kết luận mô tả từ một seed mỗi biến thể; CE/class-weight chạy trên RTX 4060, sampler trên RTX 3060. Recall CE/class-weight hiện chỉ có trong nhật ký run, chưa có prediction artifact tại checkout để TV3 tự tính lại. T16 R đã bàn giao phần tổng hợp; trạng thái `Verified` vẫn thuộc TV3 sau khi có thể kiểm tra dự đoán độc lập.

### [E6-005_s0] Thử chạy VGG16 pretrained 48 px (`vgg16_48_ft` — Failed lỗi quyền cache) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Tái hiện "fail case" VGG16 pretrained tại đầu vào `48×48` (`resize = 48`, `channels = 3`).
- Thay đổi so với baseline: `resize = 48`, `channels = 3`, `transfer = {"name": "vgg16", "pretrained": true, "freeze_backbone": false}`.
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV1 --only vgg16_48_ft`
- Kết quả: status=Failed, best_val_loss=, val_acc=, val_macro_f1=, best_epoch=0, time_s=16.356, run_dir=results/E6/E6-005_s0
- Nhận xét: Lỗi môi trường ngay lúc nạp trọng số pretrained (`PermissionError: [WinError 5] Access is denied: 'C:\\Users\\Kien/.cache\\torch'`), chưa vào vòng huấn luyện (`best_epoch = 0`, `16.4s`). Không phải lỗi của mô hình hay mã nguồn.
- Quyết định: Hệ thống tự ghi `Failed`, `DISCARD` (`EXCEPTION:PermissionError`), loại khỏi thống kê E6; cấp lại quyền thư mục cache và chạy lại.

### [E6-003_s0] Thử chạy VGG16 pretrained 48 px (`vgg16_48_ft` — Failed lỗi mạng tải weights) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Chạy lại `vgg16_48_ft` sau khi khắc phục quyền ghi cache Torch.
- Thay đổi so với baseline: `resize = 48`, `channels = 3`, `transfer = {"name": "vgg16", "pretrained": true, "freeze_backbone": false}`.
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV1 --only vgg16_48_ft`
- Kết quả: status=Failed, best_val_loss=, val_acc=, val_macro_f1=, best_epoch=0, time_s=17.202, run_dir=results/E6/E6-003_s0
- Nhận xét: Lỗi socket khi tải checkpoint ImageNet (`URLError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`, `17.2s`, `best_epoch = 0`). Là sự cố môi trường/mạng cục bộ trên máy trạm.
- Quyết định: Hệ thống tự ghi `Failed`, `DISCARD` (`EXCEPTION:URLError`), loại khỏi thống kê E6; sau khi tải xong trọng số ImageNet vào cache, chạy lại thành công ở `E6-004_s0` ngay bên dưới.

### [E6-004_s0] Fine-tune VGG16 pretrained ở 48 px, tái hiện "fail case" (`vgg16_48_ft`) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: Ảnh 48 px quá nhỏ cho 5 lần MaxPool của VGG16 (bản đồ đặc trưng còn 1×1 ở pool5) nên mô hình sẽ kém hoặc không tốt hơn baseline dù có trọng số ImageNet ("fail case" trong `demo_report.pdf` mục 5.1).
- Thay đổi so với baseline: `resize = 48`, `channels = 3`, `transfer = {"name": "vgg16", "pretrained": true, "freeze_backbone": false}` (134,289,223 tham số).
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV1 --only vgg16_48_ft`
- Kết quả: status=Completed, best_val_loss=0.9373211630807676, val_acc=0.6937865700752298, val_macro_f1=0.6812115174655821, best_epoch=38, time_s=3268.183, run_dir=results/E6/E6-004_s0
- Nhận xét: Giả thuyết **không đúng về độ chính xác**: `val_acc = 0.6938` (`+0.0123` so với baseline, lớn hơn std `0.0075` nhưng mới 1 seed), `val_macro_f1 = 0.6812`; early stopping dừng ở epoch 53, tốt nhất epoch 38. Hội tụ nhanh nhờ pretrained (val ≥ 0.60 từ epoch 4, ≥ 0.65 từ epoch 11). Cái giá là quá khớp và lãng phí: `train_acc_cuoi = 0.9509`, khoảng cách train − val `+0.2571` (baseline `+0.0397`), 134.3 M tham số (22.9× baseline) và `best.pt` 512.3 MB. Hai lần thử trước (`E6-005_s0` PermissionError khi ghi cache, `E6-003_s0` URLError khi tải checkpoint) là lỗi môi trường, không phải lỗi mô hình. Giải thích bằng kích thước bản đồ đặc trưng ở mục "Kết luận E6" bên dưới.
- Quyết định: Giữ weights (`KEEP`, 512.3 MB, không đẩy lên git). Không đưa VGG16 48 px vào E7: lợi ích nhỏ so với chi phí và chưa đủ 3 seed.

### [E6-001_s0] Fine-tune VGG16 pretrained ở 96 px, 30 epoch (`vgg16_96_ft`) — TV2 – Lê Phan Nguyên Hải – 2026-11-25
- Giả thuyết: Phóng ảnh lên 96 px thì pool5 còn 3×3 thay vì 1×1, giữ được thông tin không gian nên VGG16 sẽ tốt hơn bản 48 px (đối chứng trực tiếp cho "fail case").
- Thay đổi so với baseline: `resize = 96`, `channels = 3`, `epochs = 30` (VGG16 ở 96 px đắt), `transfer = {"name": "vgg16", "pretrained": true, "freeze_backbone": false}`.
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV2 --only vgg16_96_ft`
- Kết quả: status=Completed, best_val_loss=0.9220232938318472, val_acc=0.6976873780997492, val_macro_f1=0.681773112029825, best_epoch=30, time_s=2214.106, run_dir=results/E6/E6-001_s0
- Nhận xét: Cao nhất E6: `val_acc = 0.6977` (`+0.0162` so với baseline), `val_macro_f1 = 0.6818`, `best_val_loss = 0.9220` thấp nhất E6. Tốt nhất ở chính epoch 30 (chạy hết 30 epoch, chưa early stop) nên còn dư địa. So với 48 px chỉ hơn `+0.0039` (< std baseline 0.0075, 1 seed mỗi bên) → **chưa đủ bằng chứng** 96 px tốt hơn 48 px về độ chính xác; khác biệt rõ hơn ở `best_val_loss` (0.9220 so với 0.9373) và khoảng cách train − val (`+0.1876` so với `+0.2571`, nhưng run 96 px ngắn hơn nên chưa so được hoàn toàn). Val acc dao động mạnh cuối run (epoch 28–30: 0.6782 → 0.6425 → 0.6977). 71.3 s/epoch trên RTX 4060.
- Quyết định: Giữ weights (`KEEP`, 512.3 MB, không đẩy lên git). Không chạy thêm epoch/seed cho VGG16 vì chi phí; ghi nhận là cận trên của nhóm transfer.

### [E6-002_s0] ResNet18 huấn luyện từ đầu ở 96 px (`resnet18_96_scratch`) — TV1 – Đoàn Trung Kiên – 2026-11-26
- Giả thuyết: ResNet18 từ đầu (khởi tạo ngẫu nhiên) là mốc để đo đóng góp riêng của trọng số ImageNet; kỳ vọng hội tụ chậm hơn và kém hơn bản pretrained.
- Thay đổi so với baseline: `resize = 96`, `channels = 3`, `transfer = {"name": "resnet18", "pretrained": false, "freeze_backbone": false}` (11,180,103 tham số).
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV1 --only resnet18_96_scratch`
- Kết quả: status=Completed, best_val_loss=1.0006212015011144, val_acc=0.6770688213987183, val_macro_f1=0.6671904234002443, best_epoch=60, time_s=4182.096, run_dir=results/E6/E6-002_s0
- Nhận xét: `val_acc = 0.6771` (`-0.0045`, trong std baseline), `val_macro_f1 = 0.6672`, `best_val_loss = 1.0006` (cao nhất trong các bản fine-tune). Đúng giả thuyết về tốc độ: epoch 1 chỉ `0.2862` (pretrained `0.5612`), val ≥ 0.65 ở epoch 45 (pretrained epoch 14), tốt nhất ở epoch 60 (pretrained 42). Quá khớp `+0.2437` dù có augmentation. 55.5 s/epoch trên RTX 3050.
- Quyết định: Giữ weights (`KEEP`, 42.7 MB). Dùng làm mốc "từ đầu" trong Bảng E6.2.

### [E6-007_s0] Khảo sát ResNet-18 Fine-tuning toàn bộ (resnet18_96_ft) — TV4 – 2026-11-26
- Giả thuyết: Khởi tạo từ trọng số ImageNet và fine-tune toàn bộ các tầng trên ảnh 96x96 (3 kênh) sẽ giúp mạng thừa hưởng các bộ lọc trích xuất đặc trưng bậc thấp mạnh mẽ, từ đó đạt độ chính xác cao hơn và hội tụ nhanh hơn mô hình CNN huấn luyện từ đầu.
- Thay đổi so với baseline: `resize = 96`, `channels = 3`, `transfer = {"name": "resnet18", "pretrained": true, "freeze_backbone": false}`.
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV4 --only resnet18_96_ft`
- Kết quả: status=Completed, best_val_loss=0.9717674393546252, val_acc=0.6873780997492338, val_macro_f1=0.6698490430387746, best_epoch=42, time_s=4572.14, run_dir=results/E6/E6-007_s0
- Nhận xét: Fine-tune toàn bộ ResNet-18 trên ảnh 96x96 đạt Val Acc 68.74% (vượt mốc Baseline 68.15%) và Val Macro-F1 đạt đỉnh 66.98% (tăng vượt bậc +2.90% so với Baseline 64.08%). Các đặc trưng cấp thấp (cạnh, góc, texture) thừa hưởng từ ImageNet kết hợp với việc tinh chỉnh toàn bộ trọng số giúp mô hình học rất nhạy các đặc trưng biểu cảm vi mô trên khuôn mặt, phân loại đồng đều trên các lớp cảm xúc thiểu số. Early stopping dừng tại epoch 57 giúp ngăn chặn hiện tượng quá khớp (train acc cuối đạt 98.57%).
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 42.7 MB). Khẳng định ResNet-18 fine-tune toàn bộ là một cấu hình xuất sắc cho Báo cáo.

### [E6-006_s0] Khảo sát ResNet-18 Đóng băng Backbone (resnet18_96_frozen) — TV4 – 2026-11-26
- Giả thuyết: Đóng băng toàn bộ backbone ResNet-18 tiền huấn luyện ImageNet và chỉ huấn luyện lớp phân loại (Linear Probing / Feature Extractor) sẽ giúp giảm tối đa số tham số cần cập nhật và chống overfitting, tuy nhiên do đặc trưng ImageNet (vật thể tự nhiên) khác biệt với ảnh khuôn mặt cảm xúc xám nên độ chính xác có thể bị giới hạn so với fine-tune toàn bộ.
- Thay đổi so với baseline: `resize = 96`, `channels = 3`, `transfer = {"name": "resnet18", "pretrained": true, "freeze_backbone": true}`.
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV4 --only resnet18_96_frozen`
- Kết quả: status=Completed, best_val_loss=1.4507161095534968, val_acc=0.45695179715798273, val_macro_f1=0.368315020130604, best_epoch=50, time_s=4918.338, run_dir=results/E6/E6-006_s0
- Nhận xét: Đóng băng backbone ResNet-18 và chỉ huấn luyện classifier đạt Val Acc 45.70% (kém Baseline tới 22.46%) và Macro-F1 36.83%. Mặc dù số tham số huấn luyện giảm mạnh, nhưng vì đặc trưng ImageNet được tối ưu hóa cho phân loại 1,000 lớp vật thể tự nhiên đa dạng (động vật, đồ vật...) nên khi không được tinh chỉnh (fine-tune), chúng không thể biểu diễn tốt các nếp nhăn và biến đổi vi mô trên khuôn mặt người. Kết quả đối sánh giữa Fine-tuning (68.74%) và Frozen (45.70%) mang giá trị khoa học đắt giá: chứng minh việc fine-tune backbone là điều kiện tiên quyết khi áp dụng Transfer Learning cho bài toán FER2013.
- Quyết định: Giữ weights (`KEEP`, mã `OK`, `best.pt` 42.7 MB) làm dữ liệu đối chứng phản biện cho Chương 4 và Chương 5 của Báo cáo.

### Kết luận E6 (T17) — TV2 – Lê Phan Nguyên Hải – 2026-11-26
Bảng số liệu sinh bằng `python src/e6_analysis.py` → `report/tables/E6_analysis.md` (Bảng E6.1 toàn bộ biến thể, E6.2 ResNet18 từ đầu so với pretrained, E6.3 kích thước bản đồ đặc trưng VGG16 đo bằng forward thật); bảng chung E6 của `make_tables.py` ở `report/tables/E6_report.md`. Mỗi biến thể 1 seed nên mọi chênh lệch dưới std baseline `0.0075` coi là chưa phân biệt được.

**1. Vì sao VGG16 ở 48 px bị coi là "fail case".** VGG16 có 5 khối conv, mỗi khối kết thúc bằng MaxPool 2×2 stride 2. Ở 48 px cạnh ảnh đi `48 → 24 → 12 → 6 → 3 → 1` (Bảng E6.3, khớp `results/transfer_summary.csv`):
- Khối conv5 chỉ chạy trên bản đồ 3×3, và pool5 (không `ceil_mode`) chỉ lấy cửa sổ 2×2 góc trên trái: **4/9 ô** của conv5 được dùng, hàng và cột thứ ba bị bỏ. Vùng tiếp nhận của conv5_3 là 196 px, lớn hơn cả ảnh 48 px, nên mỗi ô đã nhìn cả khuôn mặt (phần lớn là vùng đệm 0); ô còn lại sau pool5 là một vector 512 chiều toàn cục, **mất hết thông tin vị trí** (mắt, miệng nằm đâu).
- torchvision không báo lỗi vì có `AdaptiveAvgPool2d(7, 7)`: bản đồ 1×1 được lặp thành 7×7. Đo thật cho thấy mỗi kênh trong 49 ô **chỉ có 1 giá trị**, nên FC1 nhận 25,088 đầu vào nhưng chỉ 512 số thật: 102,764,544 tham số FC1 tương đương một lớp `Linear(512, 4096)` với 2,101,248 tham số hiệu dụng, tức ~100.7 M tham số dư thừa vẫn phải lưu, cập nhật và chịu weight decay. Ở 96 px pool5 còn 3×3 (36/36 ô conv5 được dùng, 9 ô thông tin thật, 18.9 M tham số FC1 hiệu dụng).
- `demo_report.pdf` chỉ nêu lý do "mạng quá sâu với 5 MaxPool, ảnh phải lớn" mà không có số liệu; `code/valid_imagenet_vgg16.html` sửa bằng cách phóng ảnh lên 128 px.

**2. Kết quả thực tế: "fail" về hiệu quả, không phải về độ chính xác.** Với PyTorch, trọng số ImageNet và fine-tune toàn bộ, VGG16 48 px vẫn đạt `0.6938` (`+0.0123` so với baseline), chỉ kém VGG16 96 px `0.0039` (< std). Đặc trưng ImageNet tóm thành vector toàn cục vẫn đủ phân biệt cảm xúc ở mức này. Điểm "fail" thật: 134.3 M tham số (22.9× baseline 5.87 M), `best.pt` 512.3 MB, quá khớp mạnh (train − val `+0.2571` so với baseline `+0.0397`), và ~100.7 M tham số FC1 không mang thêm thông tin. Cách sửa nếu vẫn dùng VGG16 ở 48 px: bỏ pool5 hoặc thay classifier bằng GAP + `Linear(512, 7)`; hoặc phóng ảnh ≥ 96 px như `vgg16_96_ft`.

**3. ResNet18 từ đầu so với pretrained (Bảng E6.2).** Fine-tune từ ImageNet (`0.6874`) hơn từ đầu (`0.6771`) `+0.0103`, vượt std baseline nhưng mới 1 seed; lợi ích rõ nhất là **tốc độ hội tụ** (epoch 1: 0.5612 so với 0.2862; val ≥ 0.65 ở epoch 14 so với 45; best epoch 42 so với 60). Cả hai đều quá khớp (train − val `+0.2983` và `+0.2437`). Đóng băng backbone chỉ đạt `0.4570`, train acc cuối `0.4436` còn thấp hơn val → **underfit**: một lớp `Linear(512, 7)` trên đặc trưng ImageNet cố định không tách được biểu cảm khuôn mặt xám, nên phải fine-tune backbone.

**4. Quyết định cho E7 (T18).** Không biến thể transfer nào vừa hơn baseline rõ ràng vừa rẻ: VGG16 hơn `+0.012–0.016` với 22.9× tham số; ResNet18 fine-tune chỉ `+0.0059` (< std). Đề xuất E7 giữ họ CNN của baseline làm mô hình chính; nếu cần đa dạng cho ensemble, `resnet18_96_ft` là ứng viên rẻ nhất trong E6 nhưng phải chạy đủ 3 seed trước khi chọn.

### [E7-001_s0] Sàng lọc E7 Bước 1: `large_bn` + `batch_size = 64` (`large_bn_bs64`, Seed 0 — C2) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: Trong E3 (`T14`), `batch_size = 64` là yếu tố duy nhất vượt `+std baseline` (`+0.0109`). Kiểm tra trên cùng GPU RTX 4060 xem hạ `batch_size` từ 128 xuống 64 có tiếp tục cải thiện `large_bn` (`E1-014_s0 = 0.7086`, `F1 = 0.6953`) hay không.
- Thay đổi so với `large_bn` (`E1-014_s0`): `batch_size = 64` (giữ nguyên `model = "large"`, `bn = true`, `dropout_conv = 0.2`, `dropout_fc = 0.5`, `SGD Nesterov lr = 0.01`).
- Lệnh: `python -u src/run_experiment.py --config configs/E7_final.json --member TV2 --only large_bn_bs64 large_bn_bs64_leaky large_bn_bs64_leaky_wrs baseline_best --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9025232671661675, val_acc=0.6971301198105322, val_macro_f1=0.677833402782655, best_epoch=69, time_s=2216.511, run_dir=results/E7/E7-001_s0
- Nhận xét: Hoàn tất 84 epoch (dừng sớm, `best_epoch = 69`, 36.9 phút trên RTX 4060). Dù vẫn vượt Baseline 6 Conv (`+0.0156`), nhưng so trực tiếp với mốc cùng GPU **`E1-014_s0` (`0.7086` / F1 `0.6953`)** lại **giảm `-0.0114` accuracy** (vượt quá `std baseline = 0.0075`) và giảm `-0.0175` macro-F1; khoảng quá khớp tương đương (`Train − Val = +0.1868`). Với mạng sâu 16 Conv có BatchNorm, hạ batch size xuống 64 làm thống kê batch của BN kém ổn định hơn `batch_size = 128` ở `lr = 0.01`.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 81.71 MB). Không chọn làm cấu hình cuối theo luật tiền đăng ký.

### [E7-002_s0] Sàng lọc E7 Bước 1: `large_bn` + `bs_64` + `leaky_relu` (`large_bn_bs64_leaky`, Seed 0 — C2) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: Cộng thêm `activation = "leaky_relu"` (tốt nhất E2 trên Baseline: `+0.0120`) vào `large_bn_bs64` để kiểm tra khả năng cải thiện trên mạng sâu 16 lớp có BN.
- Thay đổi so với `large_bn`: `batch_size = 64`, `model_kwargs = {"bn": true, "activation": "leaky_relu"}`.
- Lệnh: `python -u src/run_experiment.py --config configs/E7_final.json --member TV2 --only large_bn_bs64 large_bn_bs64_leaky large_bn_bs64_leaky_wrs baseline_best --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9030571929855112, val_acc=0.6910002786291446, val_macro_f1=0.672465413778312, best_epoch=64, time_s=2081.334, run_dir=results/E7/E7-002_s0
- Nhận xét: Dừng sớm tại epoch 79 (`best_epoch = 64`, 34.7 phút). Đạt `val_acc = 0.6910` (`-0.0176` so với `E1-014_s0`), `val_macro_f1 = 0.6725` (`-0.0228`), dù khoảng cách quá khớp giảm nhẹ xuống `+0.1589`. Đổi sang `LeakyReLU(0.1)` trên `large_bn` kết hợp `bs=64` làm giảm thêm hiệu năng so với `E7-001_s0` (`0.6971`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 81.71 MB). Không chọn làm cấu hình cuối.

### [E7-003_s0] Sàng lọc E7 Bước 1: `large_bn` + `bs_64` + `leaky_relu` + `WeightedRandomSampler` (`large_bn_bs64_leaky_wrs`, Seed 0 — C2) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: Bổ sung `sampler = "weighted_random"` (E5 nâng mạnh macro-F1 `+0.0309` trên Baseline) nhằm kéo tăng macro-F1 trên lớp thiểu số (`Disgust`/`Fear`) cho nhánh `large_bn_bs64_leaky`.
- Thay đổi so với `large_bn`: `batch_size = 64`, `model_kwargs = {"bn": true, "activation": "leaky_relu"}`, `sampler = "weighted_random"`.
- Lệnh: `python -u src/run_experiment.py --config configs/E7_final.json --member TV2 --only large_bn_bs64 large_bn_bs64_leaky large_bn_bs64_leaky_wrs baseline_best --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.9367389397290331, val_acc=0.6935079409306213, val_macro_f1=0.6873613364184453, best_epoch=93, time_s=2633.831, run_dir=results/E7/E7-003_s0
- Nhận xét: Chạy đủ 100 epoch (`best_epoch = 93`, 43.9 phút). Đúng như xu hướng E5, thêm `WeightedRandomSampler` giúp kéo `val_macro_f1` tăng rõ rệt `+0.0149` so với `E7-002_s0` (`0.6725 → 0.6874`) và nhích `val_acc` từ `0.6910` lên `0.6935`, nhưng vẫn thấp hơn rõ rệt mốc `E1-014_s0` (`0.7086` / F1 `0.6953`) ở cả Accuracy (`-0.0150`) lẫn Macro-F1 (`-0.0079`), đồng thời `Train − Val` tăng lên `+0.2053`.
- Quyết định: Giữ weights (`KEEP`, `best.pt` 81.71 MB). Không chọn làm cấu hình cuối.

### [E7-004_s0] Đối chứng E7 Bước 1: Baseline 6 Conv + tổ hợp tốt nhất E2–E4 (`baseline_best`, Seed 0 — C2) — TV2 – Lê Phan Nguyên Hải – 2026-11-27
- Giả thuyết: Gộp các yếu tố tốt nhất đo trên Baseline (`leaky_relu`, `bs=64`, `dropout_fc=0.25`, `wd=1e-3`, `label_smoothing=0.1`) để đối chứng xem phần tăng hiệu năng đến từ độ sâu kiến trúc (`large_bn`) hay từ bộ siêu tham số huấn luyện.
- Thay đổi so với `configs/baseline.json`: `model_kwargs = {"activation": "leaky_relu", "dropout_fc": 0.25}`, `batch_size = 64`, `weight_decay = 0.001`, `label_smoothing = 0.1` (`5,867,847` tham số).
- Lệnh: `python -u src/run_experiment.py --config configs/E7_final.json --member TV2 --only large_bn_bs64 large_bn_bs64_leaky large_bn_bs64_leaky_wrs baseline_best --seeds 0`
- Kết quả: status=Completed, best_val_loss=0.869294905004876, val_acc=0.70130955697966, val_macro_f1=0.6783639705148158, best_epoch=91, time_s=1504.033, run_dir=results/E7/E7-004_s0
- Nhận xét: Kết quả đối chứng rất xuất sắc! Chỉ với `5.87M` tham số (nhỏ hơn `large_bn` 3.5×) và 25.1 phút train, `baseline_best` bứt phá từ Baseline gốc (`0.6815 ± 0.0075`) lên **`0.7013`** (**`+0.0198`**, vượt ngưỡng 70% lần đầu tiên cho mạng 6 Conv!), đạt **`best_val_loss = 0.8693` thấp nhất toàn bộ dự án BTL02** và kiểm soát quá khớp rất gọn (`Train − Val = +0.0931`, bằng một nửa `large_bn`). Điều này khẳng định các lựa chọn E2–E4 thực sự cộng hưởng mạnh trên chính kiến trúc Baseline mà chúng được tìm ra, nhưng **không chuyển tuyến tính sang `large_bn`**. Tuy nhiên theo luật chọn tiền đăng ký, `baseline_best` vẫn thấp hơn `E1-014_s0` ở cả `val_acc` (`-0.0072`) lẫn `val_macro_f1` (`0.6784` vs `0.6953`, `-0.0169`).
- Quyết định: Giữ weights (`KEEP`, `best.pt` 23.49 MB) làm bằng chứng đối chứng đắt giá cho Chương 4. Không chọn làm mô hình cuối vì thua `large_bn` ở cả hai chỉ số.

### Kết luận E7 (T18 — Phụ trách R: TV3 – Võ Hoàng Bảo – 2026-11-27)
- **Nguồn số liệu & biểu đồ (100% trên PublicTest, chưa chạm PrivateTest):**
  - Bảng sàng lọc 4 biến thể E7: `report/tables/E7_report.md` / `E7_summary.csv` (sinh từ `python src/make_tables.py`).
  - Bảng tổng hợp 3 seed + TTA + Ensemble của mô hình cuối: `report/tables/E7_tta_ensemble.md` / `E7_tta_ensemble.csv`, `results/E7/e7_val_summary.json`, `results/E7/e7_val_per_class.csv`, các file dự đoán `results/E7/val_predictions_ens_{none,flip,ten_crop}.npz` và `results/E7/e7_val_best_predictions.npz` (sinh bằng `python src/eval_e7.py --run-dirs results/E1/E1-014_s0 results/E1/E1-014_s1 results/E1/E1-014_s2 --tta none flip ten_crop`).
  - 5 biểu đồ kèm CSV đã đăng ký vào `tracking/figures_manifest.csv`: `figures/e7_val_tta_ensemble.png`, `e7_val_confusion.png`, `e7_val_error_hist.png`, `e7_val_correct_grid.png`, `e7_val_wrong_grid.png`, và danh sách ca đúng/sai điển hình `results/E7/e7_gradcam_cases.csv` bàn giao TV4 (`T20`).
- **1. Kết quả Bước 1 sàng lọc và quyết định chọn cấu hình cuối:**
  - Trên cùng RTX 4060 (`seed 0`), cả 4 ứng viên mới (`E7-001_s0` `0.6971`, `E7-002_s0` `0.6910`, `E7-003_s0` `0.6935`, `E7-004_s0` `0.7013`) đều thấp hơn mốc `E1-014_s0` (`large_bn`: `0.7086`, macro-F1 `0.6953`) ở cả Accuracy lẫn Macro-F1.
  - Áp dụng đúng luật tiền đăng ký trong `configs/E7_final.json`: **chốt bộ 3 seed `large_bn` (`E1-014_s0`, `E1-014_s1`, `E1-014_s2` — đã `Verified` ở `T12`, xác minh khớp 100% SHA-256 3 file `best.pt`: `8c79c23b35dd`, `48f04f121fc4`, `5e20695d5006`) làm mô hình cuối**, không lặp train thừa.
- **2. Hiệu quả của TTA và Ensemble 3 seed trên PublicTest (`E7_tta_ensemble.md`):**
  - **Đơn lẻ 3 seed (`none`):** `val_acc = 0.7060 ± 0.0023`, `val_macro_f1 = 0.6922 ± 0.0027`, `Val NLL = 1.2175`.
  - **TTA `flip` (đơn lẻ):** tăng đều trên cả 3 seed (`0.7166 / 0.7122 / 0.7161` → **`0.7150 ± 0.0024`**, `+0.90%`, `F1 = 0.7015 ± 0.0014`, `NLL = 1.1157`).
  - **Ensemble 3 seed (`none`):** nhảy vọt lên **`0.7236`** (`+1.76%` so với đơn lẻ), `F1 = 0.7134` (`+2.12%`), đồng thời hạ mạnh `Val NLL` từ `1.2175` xuống `0.9780` (`-0.2395`), khắc phục rõ rệt hiện tượng dự đoán quá tự tin của từng seed `large_bn` đơn lẻ.
  - **Ensemble 3 seed + TTA `flip` (CAO NHẤT TOÀN BẢNG):** đạt **`val_acc = 0.7250` (`72.50%`, vượt Baseline `68.15%` tới `+4.35%` và vượt mốc kỷ lục PublicTest `69%` của đề bài tới `+3.50%`!)**, **`val_macro_f1 = 0.7106`** (`+6.98%` trên Baseline `64.08%`), và **`Val NLL = 0.9402`** (thấp nhất toàn bộ 6 tổ hợp!).
  - **TTA `ten_crop` (cắt `44×44` rồi phóng về `48×48`):** giảm hiệu năng rõ rệt (`0.6836 ± 0.0043` đơn lẻ, `0.6991` ensemble) do ảnh FER-2013 đã cắt sát mặt ở kích thước nhỏ `48×48`, cắt góc kèm nội suy bilinear làm mất vùng lông mày/miệng ở rìa ảnh và làm mờ chi tiết biểu cảm vi mô.
- **3. Hiệu năng từng lớp trên PublicTest của tổ hợp thắng (`Ensemble 3 seed + TTA=flip`):**
  - Angry `rec 65.31% / F1 64.41%`, **Disgust `rec 60.71% (34/56) / prec 87.18% / F1 71.58%`** (`+23.21%` recall so với CE baseline `37.50%`), **Fear `rec 51.21% (254/496) / F1 57.99%`** (vượt mốc 50% lần đầu tiên), Happy `rec 89.61% / F1 88.91%`, Sad `rec 65.08% / F1 63.58%`, Surprise `rec 86.02% / F1 84.00%`, Neutral `rec 70.02% / F1 66.98%`.
- **4. Quyết định tiền đăng ký cho `T19` (PrivateTest):**
  - Theo đúng luật trong `configs/E7_final.json` (chọn tổ hợp có `val_acc` cao nhất trên PublicTest): **chốt duy nhất `Ensemble 3 seed large_bn (E1-014_s0, E1-014_s1, E1-014_s2) + TTA = flip`** để chạy lệnh `--eval-test` **đúng 1 lần duy nhất** trên `PrivateTest` ở `T19` (có TV2 chứng kiến và TV4 xác minh khóa lần 2). Chốt `done(T18)` chuyển TV4 (`T`) kiểm thử độc lập (đồng thời TV1 `C1` kiểm độc lập ensemble 3 seed và TV2 `C2` kiểm độc lập TTA `flip`/`ten_crop`).

### [E7-TEST] Đánh giá cuối cùng trên PrivateTest ĐÚNG 1 LẦN DUY NHẤT (`T19` — R: TV3 – Võ Hoàng Bảo – 2026-11-27)
- **Điều kiện cổng & chứng kiến:** Thực hiện sau khi `T18` đạt **`Verified`** (`KT-021`, commit `eb33f9a`), xác nhận `results/E7/TEST_DONE.json` chưa tồn tại, dưới sự chứng kiến trực tiếp của **TV2 – Lê Phan Nguyên Hải (`C1`)** lúc `21:07:28 → 21:07:49 (+07:00)` (`21.8 s` trên `Hbao | RTX 3050`, commit `08be17a0`).
- **Tổ hợp mô hình khóa tiền đăng ký:** Ensemble 3 seed `large_bn` (`E1-014_s0` sha `8c79c23b35dd`, `E1-014_s1` sha `48f04f121fc4`, `E1-014_s2` sha `5e20695d5006`) + `TTA = flip`.
- **Lệnh duy nhất đã chạy:** `python src/run_experiment.py --eval-test --member TV3 --config configs/E7_final.json --run-dirs results/E1/E1-014_s0 results/E1/E1-014_s1 results/E1/E1-014_s2 --tta flip`
- **Kết quả (`results/E7/TEST_DONE.json` & `results/E7/test_run.log`):**
  - Từng seed đơn lẻ (`TTA = flip`) trên `PrivateTest` (`3,589` ảnh): `E1-014_s0 = 0.7247` (`F1 0.7215`), `E1-014_s1 = 0.7197` (`F1 0.7174`), `E1-014_s2 = 0.7222` (`F1 0.7103`) — trung bình đơn lẻ `0.7222`.
  - **Ensemble 3 seed + TTA `flip` (Kết quả chính thức):** **`test_acc = 0.736417` (`73.64%`, `2,643 / 3,589` ảnh đúng)**, **`test_macro_f1 = 0.732162` (`73.22%`)**!
  - **So sánh mốc chuẩn đề bài & báo cáo mẫu:** Vượt mốc kỷ lục cuộc thi Kaggle 2013 **`71% Private`** tới **`+2.64 điểm %`** (`+0.026417`), vượt mốc **`69% Public`** tới **`+4.64 điểm %`** (`+0.046417`), và vượt nhóm CNN báo cáo mẫu `demo_report.pdf` (`64%–67%`) tới `+6.6% – +9.6%`.
  - **Chỉ số từng lớp trên `PrivateTest` (`results/E7/test_per_class.csv`):** Angry `prec 67.36% / rec 66.40% / F1 66.87%` (`491` ảnh), **Disgust `prec 87.23% / rec 74.55% (41/55) / F1 80.39%`**, **Fear `prec 65.38% / rec 51.52% (272/528) / F1 57.63%`**, Happy `prec 90.32% / rec 91.24% / F1 90.78%` (`879`), Sad `prec 58.69% / rec 63.13% / F1 60.83%` (`594`), Surprise `prec 82.94% / rec 85.34% / F1 84.12%` (`416`), Neutral `prec 68.70% / rec 75.40% / F1 71.90%` (`626`).
- **Bằng chứng & khóa bảo vệ:** Đã tạo nguyên tử file khóa `results/E7/TEST_DONE.json` (`locked = true`), lưu `results/E7/test_predictions.npz`, `results/E7/test_run.log`, nối dòng `E7-TEST` vào `tracking/experiments.csv` và sinh 4 hình kèm CSV đăng ký vào `tracking/figures_manifest.csv` (`figures/e7_test_confusion.png`, `e7_test_error_hist.png`, `e7_test_correct_grid.png`, `e7_test_wrong_grid.png`).
- **Xác nhận phối hợp `T19`:** Chờ **TV2 (`C1`)** ghi xác nhận chứng kiến (giờ, lệnh) và **TV4 (`C2`)** xác nhận chạy lần 2 bị `TEST_DONE.json` từ chối ngay bên dưới trước khi TV3 chốt `done(T19)` cho TV1 (`T`) kiểm thử độc lập.
  - **Xác nhận C1 — TV2 – Lê Phan Nguyên Hải – 2026-11-27:** Tôi đã chứng kiến trực tiếp TV3 chạy đúng **1 lần** lệnh `python src/run_experiment.py --eval-test --member TV3 --config configs/E7_final.json --run-dirs results/E1/E1-014_s0 results/E1/E1-014_s1 results/E1/E1-014_s2 --tta flip` lúc `21:07:28 → 21:07:49 (+07:00)` (`21.8 s`, máy `Hbao`, RTX 3050, commit `08be17a0`), đúng tổ hợp tiền đăng ký ở `T18` (ensemble 3 seed `large_bn` + `TTA = flip`). Đối chiếu sau khi pull (không chạy lại mô hình, không gọi `--eval-test`): (1) `results/E7/test_run.log` ghi đúng lệnh, giờ và commit trên; (2) tính lại từ `results/E7/test_predictions.npz`: `y_true` trùng 100% nhãn PrivateTest đọc thẳng từ `data/fer2013.csv`, `argmax(y_prob) = y_pred`, `test_acc = 0.736417` (**73.64%**, `2,643 / 3,589`) và `test_macro_f1 = 0.732162`, khớp `results/E7/TEST_DONE.json`; (3) SHA-256 của 3 `best.pt` trên máy TV2 (bản gốc) khớp 3 giá trị `weights_sha256` trong `TEST_DONE.json`; (4) `tracking/experiments.csv` có đúng 1 dòng `E7-TEST` và `TEST_DONE.json` chỉ xuất hiện ở đúng 1 commit (`bb6d96a`) → không có lần đánh giá PrivateTest nào khác.
  - **Xác nhận C2 — TV4 – Huỳnh Viễn Thông – 2026-11-27:** Đã kiểm chứng độc lập trên máy `Administrator` (RTX 3060): (1) File khóa `results/E7/TEST_DONE.json` tồn tại hợp lệ (`locked = true`, `task_id = T19`, `split = PrivateTest`, `n_samples = 3589`, `n_models = 3`, `tta = flip`, SHA-256 `6b8dec995bd950c4`) và chỉ xuất hiện duy nhất ở 1 commit (`bb6d96a`); (2) Chạy thử lệnh `--eval-test` lần thứ 2 (`python src/run_experiment.py --eval-test --member TV4 --config configs/E7_final.json --run-dirs results/E1/E1-014_s0 results/E1/E1-014_s1 results/E1/E1-014_s2 --tta flip`) bị từ chối ngay ở đầu hàm trước khi nạp mô hình/dữ liệu với **`exit code = 1`** và thông báo: `TỪ CHỐI CHẠY --eval-test: PrivateTest đã được đánh giá và khóa tại 'results/E7/TEST_DONE.json'. Quy tắc BTL02 chỉ cho phép đánh giá PrivateTest đúng 1 lần duy nhất!`; (3) SHA-256 của cả 4 file bằng chứng (`TEST_DONE.json`, `test_run.log`, `test_predictions.npz`, `tracking/experiments.csv`) trước và sau khi thử chạy lần 2 **không đổi 1 byte**; (4) `tracking/experiments.csv` có đúng **1 dòng** `E7-TEST` có `test_acc`, và tái tính độc lập từ `results/E7/test_predictions.npz` cho `test_acc = 0.736416829200` (`2,643/3,589`), `test_macro_f1 = 0.732161842225` (sai số `0.00e+00` so với `TEST_DONE.json`). Đạt 100% yêu cầu `C2` của `T19`.
- **Chốt hoàn thành `done(T19)` — TV3 – Võ Hoàng Bảo – 2026-11-27:** Đã đủ 100% bằng chứng thực thi (`bb6d96a`, `1140d32`), xác nhận chứng kiến trực tiếp của `C1` TV2 (`5c2ae78`) và xác nhận khóa lần 2 bị từ chối của `C2` TV4 (`57eacf2`). Chuyển **TV1 (`T`)** kiểm thử độc lập `T19` trên bản clone sạch (tính lại chỉ số từ `results/E7/test_predictions.npz` và đối chiếu `TEST_DONE.json` / `test_run.log`, không chạy lại `--eval-test`).

### [T20] Xác nhận phối hợp C2 trực quan hóa bộ lọc và Grad-CAM (TV3 – Võ Hoàng Bảo – 2026-11-27)
- **Đối chiếu ca đúng/sai Grad-CAM từ `results/E7/e7_gradcam_cases.csv` (do TV3 xuất ở `6f5670a`) với sản phẩm `T20` của TV4 (`eac8bb4`, `af6cd2a`):**
  - **`figures/filters_grad_cam_correct.csv` (7/7 lớp cảm xúc, khớp 100%):** dùng đúng 7 ca dự đoán đúng độ tự tin cao nhất đại diện đủ 7 lớp từ `e7_gradcam_cases.csv` (`#248 Angry`, `#560 Disgust`, `#1535 Fear`, `#1623 Happy`, `#3283 Sad`, `#2338 Surprise`, `#1953 Neutral`), đều trích xuất Grad-CAM từ lớp conv cuối `conv4_4` (`cam_max = 1.0`).
  - **`figures/filters_grad_cam_wrong.csv` (8/8 cặp nhầm tiêu biểu, khớp 100%):** dùng đúng 8 ca dự đoán sai độ tự tin cao (`conf 0.9957–1.0000`) từ `e7_gradcam_cases.csv` đại diện các cặp nhầm chính (`#2930 Fear→Surprise`, `#441 Sad→Angry`, `#3148 Sad→Happy`, `#1202 Fear→Sad`, `#1774 Neutral→Happy`, `#606 Fear→Angry`, `#1892 Disgust→Angry`, `#432 Sad→Fear`), đều hook tại `conv4_4`.
  - **Kết luận vai `C2` (`T20`):** Xác nhận danh sách ca đúng/sai dùng cho Grad-CAM khớp 100% với dự đoán đã lưu (`PublicTest`, không đụng `PrivateTest`), đã đăng ký đủ 8 hình `filters_*.png` vào `tracking/figures_manifest.csv` — sẵn sàng cho TV2 (`T`) kiểm thử độc lập.

---

## 5. Sơ đồ Kiến trúc/Pipeline (`T24`), Trực quan hóa Bộ lọc (`T20`) & Kiểm toán Toàn vẹn Nhật ký

### 5.1. Bộ 3 sơ đồ kiến trúc & pipeline sinh từ mô hình thật (`T24` — TV4)
Toàn bộ sơ đồ được sinh tự động bằng lệnh `python -m src.visual.architecture_diagram` từ forward thật của `src/models/cnn.py` (`describe()`) và đã đăng ký vào `tracking/figures_manifest.csv`:
- `figures/arch_cnn_baseline.png` (+ `figures/arch_cnn_baseline.csv`, SHA-256 `0cff89968d62`): Kiến trúc Baseline CNN 6 Conv (`5,867,847` tham số, `64×24×24 → 128×12×12 → 256×6×6 → FC 512 → 7`).
- `figures/arch_cnn_comparison.png` (+ `figures/arch_cnn_comparison.csv`, SHA-256 `e3f418705f5b`): So sánh 5 kiến trúc `baseline`, `small`, `medium`, `medium_01`, `large` về số lớp Conv và tham số.
- `figures/pipeline_overview.png` (+ `figures/pipeline_overview.csv`, SHA-256 `46558c732b28`): Quy trình dữ liệu (`Training` / `PublicTest` / khóa `PrivateTest`), huấn luyện AMP, đánh giá TTA/Ensemble và kiểm thử.

### 5.2. Trực quan hóa bộ lọc & Grad-CAM trên mô hình cuối `E1-014_s0` (`T20` — TV4)
Thực hiện trên checkpoint `results/E1/E1-014_s0/best.pt` (`large_bn`, SHA-256 `8c79c23b35dd8e10...`, chỉ dùng `PublicTest`, tuyệt đối không chạm `PrivateTest`) bằng `python src/visualize_filters.py --run-dir results/E1/E1-014_s0 --data-csv data/fer2013.csv --sample-idx 1623 --cases-csv results/E7/e7_gradcam_cases.csv --steps 100 --register-manifest`:
- Xác nhận lớp Conv đầu `features.conv1_1` `(64, 1, 3, 3)` và lớp Conv cuối `conv4_4` `(512, 512, 3, 3)` (output `6×6×512`).
- Đã xuất và đăng ký đủ **8 hình** kèm file dữ liệu `.csv`/`.npz` vào `tracking/figures_manifest.csv`: `filters_weights_first`, `filters_weights_last`, `filters_feature_maps`, `filters_actmax_filter`, `filters_actmax_class`, `filters_grad_cam`, `filters_grad_cam_correct` (7 ca đúng đủ 7 lớp) và `filters_grad_cam_wrong` (8 ca sai tự tin cao minh họa trực quan nhiễu nhãn FER-2013 và vùng chú ý khuôn mặt).

### 5.3. Kiểm toán toàn vẹn `T24` (`NHAT_KY_THUC_NGHIEM.md` vs `tracking/experiments.csv`)
- **Độ phủ run thực nghiệm**: Đủ **87 / 87** run chính thức trong `tracking/experiments.csv` (Baseline + E1–E7 + `E7-TEST`; không tính 5 run thăm dò/hạ tầng nội bộ `_probe-001_s0`, `_nb-001..003_s0`, `_qa-001_s0`).
- **Độ phủ run lỗi / bị ngắt / metric thấp**: Đầy đủ 100% cả **6 run `Aborted`/`Failed`** (`E1-001_s1`, `E1-011_s0`, `E1-012_s0`, `E3-015_s0`, `E6-003_s0`, `E6-005_s0`) và toàn bộ các run sập/metric thấp (`E1-003_s0`, `E1-004_s0`, `E2-007_s0/_s1/_s2`, `E2-009_s0`, `E2-010_s0`, `E3-012_s0`, `E6-006_s0`) đều có mục riêng với giả thuyết, nhận xét nguyên nhân và quyết định rõ ràng.
- **Độ khớp số liệu**: Chạy đồng bộ `python src/make_tables.py --update-log report/NHAT_KY_THUC_NGHIEM.md` và kiểm toán tự động xác nhận **0/87 dòng lệch** giữa `- Kết quả:` trong nhật ký và `tracking/experiments.csv`. Sẵn sàng bàn giao TV3 kiểm thử độc lập `T24`.
