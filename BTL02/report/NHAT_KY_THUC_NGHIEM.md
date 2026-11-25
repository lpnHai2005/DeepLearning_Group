# NHẬT KÝ THỰC NGHIỆM (NHẬT KÝ HUẤN LUYỆN & PHÁT TRIỂN MÔ HÌNH)

> **Môn học**: Deep Learning (Học sâu) – Bài tập lớn 02 (FER-2013)  
> **Người lập mẫu & phụ trách chính**: TV4 – Huỳnh Viễn Thông (Nhiệm vụ T24)  
> **Các thành viên tham gia**: TV1 – Đoàn Trung Kiên, TV2 – Lê Phan Nguyên Hải, TV3 – Võ Hoàng Bảo, TV4 – Huỳnh Viễn Thông  
> **Cập nhật lần cuối**: 2026-11-25  

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
- Kết quả: status=Completed, best_val_loss=0.9175978793709947, val_acc=0.6740039008080245, val_macro_f1=0.6209245313912722, best_epoch=93, time_s=3085.588, run_dir=results/baseline/baseline-002_s1
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
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-002_s0] Khảo sát kiến trúc MEDIUM_01 — TV1 – <chưa thực hiện>
- Giả thuyết: Tái hiện mô hình MEDIUM_01 từ notebook mẫu (12.1M tham số, không BN) xem mức độ quá khớp (overfitting) khi thiếu chuẩn hóa theo lô.
- Thay đổi so với baseline: `model = "medium_01"`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only medium_01`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-003_s0] Khảo sát kiến trúc LARGE_CNN — TV1 – <chưa thực hiện>
- Giả thuyết: Tái hiện mô hình LARGE_CNN (20.4M tham số, 16 lớp Conv) để đánh giá chi phí tính toán và hiện tượng biến mất gradient khi thiếu BatchNorm.
- Thay đổi so với baseline: `model = "large"`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV1 --only large`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-004_s0] Khảo sát loại bỏ BatchNorm (No BN) — TV3 – <chưa thực hiện>
- Giả thuyết: Khi loại bỏ toàn bộ các lớp BatchNorm trong Baseline, gradient sẽ kém ổn định hơn, tốc độ hội tụ chậm hơn và validation accuracy giảm ít nhất 3-5%.
- Thay đổi so với baseline: `bn = False`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV3 --only no_bn`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E1-005_s0] Khảo sát Global Average Pooling (GAP) — TV3 – <chưa thực hiện>
- Giả thuyết: Thay thế Flatten bằng Global Average Pooling (GAP) sẽ giảm mạnh số lượng tham số ở phần phân loại và giúp mô hình chống overfitting tốt hơn.
- Thay đổi so với baseline: `head = "gap"`.
- Lệnh: `python src/run_experiment.py --config configs/E1_arch.json --member TV3 --only gap`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

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
- Kết quả: status=Completed, best_val_loss=0.9175645472711759, val_acc=0.6703817219281136, val_macro_f1=0.5630497131569857, best_epoch=91, time_s=4999.794, run_dir=results/E3/E3-002_s0
- Nhận xét: Best Val Loss được cải thiện rõ rệt từ 0.9626 (adamw_1e-3) xuống 0.9176 (xấp xỉ mốc 0.9109 của Baseline), chứng minh giảm lr giúp ổn định loss. Tuy nhiên Val Acc đạt 67.04% và Macro-F1 đạt 56.30% (tiếp tục giảm 2.48% so với adamw_1e-3 và kém Baseline 7.78%). Hoàn tất 100 epochs, best epoch đạt tại epoch 91.
- Quyết định: Giữ weights (KEEP, mã OK). Hoàn tất nhóm khảo sát AdamW: cả hai mức lr 1e-3 và 3e-4 đều kém hơn Baseline SGD Nesterov về độ chính xác và khả năng nhận diện các lớp cảm xúc thiểu số.

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

### [E6-001_s0] Khảo sát Transfer Learning ResNet-18 Pretrained — TV4 – <chưa thực hiện>
- Giả thuyết: ResNet-18 tiền huấn luyện trên ImageNet (được điều chỉnh đầu vào grayscale và fine-tune) sẽ giúp trích xuất đặc trưng khuôn mặt tốt hơn và vượt mốc Accuracy của mô hình CNN huấn luyện từ đầu.
- Thay đổi so với baseline: Backbone ResNet-18 pretrained.
- Lệnh: `python src/run_experiment.py --config configs/E6_transfer.json --member TV4 --only resnet18_pretrained`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)

### [E7-001_s0] Mô hình Tối ưu Cuối cùng (Final Best Model) — TV3 – <chưa thực hiện>
- Giả thuyết: Tổng hợp cấu hình tốt nhất từ các thực nghiệm E1–E6 (kiến trúc, activation, optimizer, augmentation, xử lý mất cân bằng) sẽ mang lại độ chính xác validation cao nhất nhóm, vượt ngưỡng 70%.
- Thay đổi so với baseline: Cấu hình tối ưu toàn diện.
- Lệnh: `python src/run_experiment.py --config configs/E7_final.json --member TV3 --seeds 0 1 2`
- Kết quả: (chưa thực hiện)
- Nhận xét: (chưa thực hiện)
- Quyết định: (chưa thực hiện)
