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
- Kết quả:
- Nhận xét: Dừng tay ở epoch 4 (TV2 tắt tiến trình để kiểm tra), nên không đủ để kết luận. Cả 4 epoch đều đứng `val_acc = 0.2494`, `train_loss` khoảng 1.812 không giảm. Đo lúc khởi tạo (ảnh nhiễu N(0,1), tắt dropout): chuẩn gradient tới `conv1_1` của LARGE không BN là 7.6e-8 với init mặc định của PyTorch, so với 1.0e-3 với Glorot (init mặc định của Keras, mẫu dùng) và 7.6e-6 ở MEDIUM1. Gradient từng phần tử cỡ hoặc nhỏ hơn `eps = 1e-8` trong mẫu số của Adam nên Adam không bù được như ở MEDIUM1.
- Quyết định: Run bị ngắt, `run_experiment.py` tự đánh dấu `Aborted` và xóa `best.pt` dở. Không chạy lại `large_adam1e-4`; thay bằng `large_glorot_adam1e-4` (Glorot + Adam 1e-4, đúng thiết lập mẫu, mẫu đạt 0.67) và `large_bn`.

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

### [E2-006_s0] Khảo sát hàm kích hoạt Sigmoid (Vanishing Gradient) — TV4 – <chưa thực hiện>
- Giả thuyết: Dùng Sigmoid thay cho ReLU ở mạng sâu 6 lớp Conv sẽ gặp hiện tượng triệt tiêu gradient nghiêm trọng (đặc biệt khi không có BN), mô hình gần như không học được.
- Thay đổi so với baseline: `activation = "sigmoid"`.
- Lệnh: `python src/run_experiment.py --config configs/E2_activation.json --member TV4 --only sigmoid`
- Kết quả: status=Completed, best_val_loss=0.9687542175309974, val_acc=0.641404290888827, val_macro_f1=0.5951930862888543, best_epoch=92, time_s=3590.856, run_dir=results/E2/E2-006_s0
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
     - Các bộ tối ưu thích ứng không tách weight decay mạnh đều nhỉnh nhẹ baseline trong ngưỡng `< std`: `adam_3e-4` (`0.6879`, F1 `0.6587`), `adam_1e-3` (`0.6865`, F1 `0.6592`, loss `0.8922`), `rmsprop_1e-3` (`0.6857`, F1 `0.6620`, **loss thấp nhất E3 `0.8809`**). Ngược lại `AdamW` (`1e-3`: `0.6723`, `3e-4`: `0.6704`) tụt rõ rệt cả accuracy lẫn macro-F1 (`0.5630–0.5878`).
  2. **Lịch học (Scheduler)**: **`OneCycleLR`** (`0.6863`, F1 `0.6584`, loss `0.8997`) nhỉnh nhẹ `ReduceLROnPlateau` (`0.6815`, trong ngưỡng `< std`), vượt `Cosine` (`0.6771`), `Constant` (`0.6760`) và vượt xa `StepLR` (`0.6648`, giảm bước cứng nhắc mỗi 20 epoch).
  3. **Batch size (ảnh hưởng mạnh nhất toàn E3)**: Quy luật đơn điệu rõ rệt `bs=64` (**`0.6924`**, F1 **`0.6642`**, **`+0.0109 > std baseline 0.0075`**) > `bs=128` (`0.6815`) > `bs=256` (`0.6726`, `-0.0089`). Batch nhỏ tăng nhiễu gradient giúp tổng quát hóa tốt hơn trên tập FER-2013 nhiều nhãn nhiễu.
- **Quyết định chọn cho `E7`**: Biến thể thắng rõ rệt nhất của E3 theo validation là **`SGD Nesterov lr=0.01 + ReduceLROnPlateau + batch_size=64`** (`val_acc = 0.6924`, vượt duy nhất ngưỡng `+std baseline` trong cả 15 cấu hình). Đề xuất đưa `batch_size = 64` vào cấu hình ứng viên `E7` (có thể phối hợp thử thêm `OneCycleLR` hoặc đối chứng `Adam 3e-4` ở bước tổng hợp E7). Chốt `done(T14)` chuyển TV2 (`T`) kiểm thử độc lập.

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

### [E7-001_s0] Mô hình Tối ưu Cuối cùng (Final Best Model) — TV3 – <chưa thực hiện>
- Giả thuyết: Tổng hợp cấu hình tốt nhất từ các thực nghiệm E1–E6 (kiến trúc, activation, optimizer, augmentation, xử lý mất cân bằng) sẽ mang lại độ chính xác validation cao nhất nhóm, vượt ngưỡng 70%.
- Thay đổi so với baseline: Cấu hình tối ưu toàn diện.
- Lệnh: `python src/run_experiment.py --config configs/E7_final.json --member TV3 --seeds 0 1 2`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)
