# BTL02 – Nhận dạng cảm xúc khuôn mặt (FER2013) bằng CNN (PyTorch) — Phân tích & Phân công

Nguồn đề: `Emotion_Recognition.pdf` (5 trang, "Homework – CNN for FER") + hướng dẫn nộp bài của giảng viên.
Tài liệu mẫu: `demo_report.pdf` (báo cáo mẫu của TS. Đỗ Như Tài, Keras), `demo_code.html` (notebook Colab, Keras),
`code/train.html`, `code/valid_vgg16.html`, `code/valid_imagenet_vgg16.html` (notebook Keras/TF 1.14, VGG16 ở 128 px).

## 1. Tóm tắt yêu cầu

| Hạng mục | Nội dung bắt buộc |
|---|---|
| Dữ liệu | Kaggle **FER2013** (Facial Expression Recognition Challenge), ảnh xám **48×48**, **7 lớp**: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral |
| Đầu vào / đầu ra | Input ảnh 48×48 → output vector 7 chiều (xác suất 7 cảm xúc) |
| Mốc so sánh | Kỷ lục cuộc thi: **69% (public), 71% (private)** |
| Hướng cải thiện (đề gợi ý) | Thiết kế CNN (Conv, MaxPooling, Dropout, FC, …); kỹ thuật huấn luyện (thuật toán tối ưu, data augmentation, …); hàm kích hoạt (ReLU, sigmoid, …) |
| Code | **Python + PyTorch** (code mẫu là Keras → chỉ tham khảo ý tưởng, **không dùng Keras/TensorFlow**) |
| Báo cáo phải có | (1) **Tóm tắt ngắn mã nguồn**; (2) **Kết quả nhiều thực nghiệm với nhiều bộ tham số**; (3) **Trực quan hóa các bộ lọc tích chập cuối cùng**; (4) **Phân tích, giải thích kết quả theo cách hiểu của nhóm** |
| Nộp | Report + slide + **nhật ký thực nghiệm** + code (**ghi rõ phân công từng thành viên**) |
| Đóng gói | Nén tất cả thành `BTLON02_Emotion_HoTen_MaSV.zip` (họ tên, mã SV của **nhóm trưởng**), **nhóm trưởng nộp** |

Đề không có rubric điểm; bảng dưới quy mỗi yêu cầu về một người chịu trách nhiệm chính.

| Yêu cầu của đề | Phụ trách chính |
|---|---|
| Dữ liệu, tiền xử lý, augmentation | TV1 |
| Mô hình CNN, vòng huấn luyện PyTorch | TV2 |
| Nhiều thực nghiệm với nhiều bộ tham số | TV1 (E4, E5), TV2 (E1, E2, E6), TV3 (E3, E7) |
| Đánh giá, ma trận nhầm lẫn, ca đúng/sai | TV3 |
| Trực quan hóa bộ lọc conv cuối | TV4 |
| Tóm tắt mã nguồn trong báo cáo | Mỗi người viết phần code mình, TV4 ghép |
| Phân tích & giải thích | Mỗi người phân tích nhóm thực nghiệm mình, TV4 viết phần tổng hợp |
| Report, slide, nhật ký thực nghiệm | TV4 (T24, T25, T26) |
| Đóng gói và nộp zip | TV2 – nhóm trưởng (T28) |

## 2. Phân tích bài toán

### 2.1. Dữ liệu FER2013

Nguồn tải (theo đề, trang 3):

- Kaggle (cần đăng nhập và bấm chấp nhận luật cuộc thi ở tab *Data*/*Rules*):
  https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
  - Tải bằng CLI (sau khi cài `pip install kaggle` và đặt `kaggle.json`):
    `kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge`
- Google Drive (link giảng viên cung cấp): https://drive.google.com/open?id=0BxINLo5jshCRbXpRcnl2SjJkdlk

Đặt file đã giải nén vào `BTL02/data/fer2013.csv` (thư mục `data/` không commit, không đưa vào zip).
**Không dùng** các bản mirror dạng thư mục ảnh chỉ có `train/` và `test/` (ví dụ `msambare/fer2013` trên Kaggle): bản đó gộp PublicTest và PrivateTest thành một tập test 7 178 ảnh, nên không tách được val/test như đề so sánh 69%/71%.

File gốc `fer2013.csv` (~290 MB): 35 887 dòng, 3 cột `emotion` (0–6), `pixels` (2304 số cách nhau bởi dấu cách), `Usage`.

| Usage | Số ảnh | Vai trò trong BTL |
|---|---|---|
| Training | 28 709 | Huấn luyện |
| PublicTest | 3 589 | **Validation**: early stopping, chọn cấu hình (tương ứng "public 69%") |
| PrivateTest | 3 589 | **Test**: chỉ đánh giá **một lần** ở mô hình cuối (tương ứng "private 71%") |

Phân bố lớp tập Training (số liệu công bố của bộ dữ liệu, **TV1 xác nhận lại bằng EDA**):

| Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
|---|---|---|---|---|---|---|
| 3 995 | **436** | 4 097 | **7 215** | 4 830 | 3 171 | 4 965 |

Đặc điểm cần lưu ý khi thiết kế và phân tích:

- **Mất cân bằng lớp**: Happy chiếm ~25%, Disgust chỉ ~1.5% (ít hơn Happy ~16.5 lần). Accuracy tổng có thể cao trong khi Disgust gần như không được nhận ra → báo cáo thêm **macro-F1, recall từng lớp, ma trận nhầm lẫn chuẩn hóa theo hàng**.
- **Ảnh nhỏ, xám, nhiễu nhãn**: 48×48, nhiều ảnh lệch tư thế, bị che, có cả ảnh trống/không phải mặt; nhãn có nhiễu (độ chính xác của người ~65 ± 5%). Các cặp dễ nhầm dự kiến: Fear↔Sad, Fear↔Surprise, Angry↔Sad, Sad↔Neutral. Phần thảo luận phải dựa trên ma trận nhầm lẫn thật của nhóm.
- **Thứ tự nhãn**: dùng đúng thứ tự Kaggle ở trên. Báo cáo mẫu dùng `LabelEncoder` nên thứ tự bị đổi theo bảng chữ cái (`Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`) → dễ gán nhầm tên lớp khi vẽ hình. Nhóm khai báo **một hằng `EMOTIONS` duy nhất** trong code.
- **Chuẩn hóa**: mean/std tính **chỉ trên Training** (không rò rỉ từ PublicTest/PrivateTest).

### 2.2. Mốc kết quả để đặt kỳ vọng

| Nguồn | Mô hình | Kết quả |
|---|---|---|
| Đề bài | Kỷ lục cuộc thi Kaggle 2013 | 69% public / 71% private |
| `demo_report.pdf` | SMALL / MEDIUM / MEDIUM_01 / LARGE CNN (48×48×1) | val acc 0.64 / 0.67 / 0.65 / 0.67 |
| `demo_report.pdf` | Fine-tune VGG16 ở 48×48 | "Fail case" (5 lần MaxPool → bản đồ đặc trưng còn 1×1) |
| `code/valid_vgg16.html` | VGG16 huấn luyện từ đầu, ảnh phóng lên **128×128×3** | val acc 0.699 (epoch 56) |
| `code/valid_imagenet_vgg16.html` | VGG16 pretrained ImageNet, 128×128×3, SGD lr 4e-4 | val acc **0.707** (epoch 32) |

Mục tiêu hợp lý: CNN tự thiết kế **≥ 65–68%** trên PublicTest; transfer learning + TTA hướng tới **~69–71%**.
"Fail case" VGG16 của mẫu thực chất do **ảnh quá nhỏ so với số lần pooling**, không phải do VGG16 không dùng được: notebook `code/` đã sửa bằng cách phóng ảnh lên 128 px. Nhóm nên tái hiện cả hai trường hợp (E6) và giải thích bằng kích thước bản đồ đặc trưng.

### 2.3. Code mẫu: học gì, tránh gì

| Học từ mẫu | Áp dụng bằng PyTorch |
|---|---|
| Họ kiến trúc VGG-style: khối [Conv3×3 ×k → MaxPool → Dropout 0.2], classifier Flatten → FC 1024 → Dropout 0.5 → 7 | `build_model("vgg_small"/"vgg_medium"/"vgg_medium01"/"vgg_large")` tái hiện đúng 4 cấu hình mẫu làm điểm xuất phát E1 |
| Augmentation: flip ngang, xoay ±10°, dịch 10%, zoom 10% (fill nearest) | `torchvision.transforms.v2`: `RandomHorizontalFlip`, `RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1))` |
| Callback: ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, CyclicLR | Tự viết trong `fit()`: lưu `best.pt` theo val acc, `history.csv` mỗi epoch, early stopping, scheduler của `torch.optim.lr_scheduler` |
| Mỗi lần chạy có `train_id`, thư mục `logs/`, `checkpoints/`, hình loss/acc theo epoch | `results/<exp>/<run_id>/{config.json, history.csv, metrics.json, best.pt, curves.png}` |
| Đánh giá: ma trận nhầm lẫn, lưới ảnh đoán sai/đúng, histogram ca sai | Giữ nguyên các hình này trong `evaluate.py` / `visual/` |
| Heatmap lớp conv đầu / giữa / cuối (`show_heatmap`) | Grad-CAM ở lớp conv cuối (phần trực quan hóa của TV4) |

Lỗi/hạn chế trong mẫu **không được lặp lại**:

- Dùng Keras/TF (đề yêu cầu PyTorch).
- `plot_images(x_train[valid_idx], y_valid_labels, …)`: vẽ ảnh **train** với nhãn **valid** → hình "Validating Images" của mẫu sai.
- Thứ tự nhãn theo `LabelEncoder` (mục 2.1).
- Mỗi cấu hình chạy 1 lần, không seed → chênh 1–2% giữa các mô hình (0.64 vs 0.67) chưa chắc là khác biệt thật. Nhóm chạy **≥ 3 seed** cho baseline và mô hình cuối, báo mean ± std.
- Chỉ báo accuracy; không báo per-class/macro-F1 dù dữ liệu lệch.
- Kết luận "fail case" không kèm phân tích kích thước bản đồ đặc trưng.

### 2.4. "Visualize the final convolution filters" làm thế nào

Bộ lọc của lớp conv cuối có kích thước 3×3×C_in (ví dụ 3×3×256), vẽ trọng số trực tiếp gần như không đọc được.
Báo cáo trình bày 4 cách, từ "đúng chữ" đến "dễ hiểu", và giải thích vì sao:

1. **Trọng số**: lưới bộ lọc lớp conv **đầu** (3×3×1, so sánh trước/sau huấn luyện) và lớp conv **cuối** (trung bình theo kênh vào hoặc chọn vài kênh có chuẩn lớn nhất).
2. **Feature maps**: đầu ra của lớp conv cuối cho 1 ảnh mỗi cảm xúc (7 ảnh × 16 kênh có kích hoạt mạnh nhất).
3. **Activation maximization**: tối ưu ảnh đầu vào (gradient ascent + chuẩn hóa L2/làm mờ) để cực đại kích hoạt trung bình của từng bộ lọc lớp conv cuối → "bộ lọc nhìn thấy gì". Thêm phiên bản cực đại logit từng lớp (7 ảnh).
4. **Grad-CAM** ở lớp conv cuối cho các ca đúng và sai → mô hình nhìn vào mắt/miệng/lông mày hay nhìn vào nền.

### 2.5. Ngân sách tính toán (đo thật trên máy nhóm trưởng)

> **Chưa xác minh**: bảng dưới được ghi vào kế hoạch từ phiên phân tích trước và **không có file log/script đo trong repo**. Chỉ dùng để ước lượng. T06/T07 phải đo lại bằng script, lưu `results/_bench/bench.csv`, rồi mới được trích dẫn trong báo cáo. Kiểm tra 2026-11-20: `torch 2.6.0+cu124` và CUDA khả dụng (xác nhận bằng lệnh); thông số GPU và thời gian/epoch chưa được đo lại.

Máy: RTX 4060 Laptop 8 GB, `torch 2.6.0+cu124`, `torchvision 0.21.0`. Đo 1 epoch train (28 709 ảnh) với **dữ liệu ngẫu nhiên**, AMP fp16, batch 128, chưa tính thời gian nạp dữ liệu và validation:

| Mô hình | Input | Tham số | Thời gian / epoch | VRAM |
|---|---|---|---|---|
| MEDIUM_01 của mẫu (+BN) | 48×48×1 | 12.1 M | 36.8 s | 1.2 GB |
| ResNet18 | 96×96×3 | 11.2 M | 17.7 s | 2.0 GB |
| VGG16-BN (batch 64) | 96×96×3 | 134.3 M | 133.7 s | 3.1 GB |

Hệ quả:

- 1 run 60 epoch của CNN cỡ MEDIUM_01 ≈ **40 phút**; mô hình 2 conv/khối ≈ một nửa. Toàn bộ kế hoạch (~50 run, mục 3.3 + baseline) ≈ **15–20 giờ GPU** → chia cho 4 máy/Colab, mỗi người ~4–5 giờ, chạy song song theo nhóm thực nghiệm.
- Nạp dữ liệu: 28 709 ảnh 48×48 chỉ ~66 MB (uint8) → **nạp hết vào RAM/GPU một lần**, augmentation theo batch, tránh `DataLoader` đọc file từng ảnh (Windows: `num_workers>0` rất chậm khởi động).
- Khảo sát (E1–E6) chạy **1 seed**; nếu chênh lệch giữa 2 cấu hình < 1 điểm % thì chạy thêm 2 seed trước khi kết luận. Baseline và mô hình cuối luôn 3 seed.
- VGG16 ở 96–128 px quá đắt (≥ 2 phút/epoch) → chỉ 1 run, ít epoch; dùng ResNet18 làm mô hình transfer chính.

## 3. Thiết kế thực nghiệm

### 3.1. Quy trình chung (áp dụng cho mọi run)

- Train trên Training, chọn epoch/cấu hình theo **val accuracy trên PublicTest**, PrivateTest chỉ chạy ở bước `--eval-test` cuối cùng.
- Mỗi lần chỉ đổi **một yếu tố** so với baseline; mọi thứ khác giữ nguyên (cùng số epoch, cùng seed).
- Chỉ số: accuracy, macro-F1, loss (train/val); thêm recall từng lớp, ma trận nhầm lẫn cho các run quan trọng; thời gian/epoch, số tham số.
- Mỗi run tự ghi 1 dòng vào `tracking/experiments.csv` (file duy nhất, nguồn số liệu cho nhật ký và bảng báo cáo).

### 3.2. Baseline đề xuất (chốt ở G2 chỉ dựa trên val, sau đó **đóng băng**)

| Tham số | Giá trị đề xuất |
|---|---|
| Input | 48×48×1, chuẩn hóa mean/std của Training |
| Kiến trúc | VGG-style 3 khối (64, 128, 256), 2 Conv3×3 + BN + ReLU mỗi khối, MaxPool 2×2, Dropout 0.25; Flatten → FC 512 → Dropout 0.5 → 7 |
| Loss | CrossEntropy |
| Optimizer | SGD momentum 0.9 (Nesterov), lr 0.01, weight decay 5e-4, batch 128 |
| Lịch lr | ReduceLROnPlateau (factor 0.5, patience 5) theo val acc |
| Augmentation | HFlip + RandomAffine(±10°, dịch 10%, scale 0.9–1.1) |
| Epoch / dừng | tối đa 60, early stopping patience 15 theo val acc, lưu `best.pt` theo val acc |
| Kỹ thuật | AMP fp16, seed 0/1/2 cho baseline |

### 3.3. Các nhóm thực nghiệm

| Nhóm | Người | Câu hỏi | Các giá trị | Số run (ước tính) |
|---|---|---|---|---|
| **E1 Kiến trúc** | TV2 | Độ sâu/độ rộng/BN ảnh hưởng ra sao? | Tái hiện SMALL, MEDIUM, MEDIUM_01, LARGE của mẫu; baseline ×0.5 / ×2 kênh; có/không BN; Flatten vs Global Average Pooling | 8 |
| **E2 Activation** | TV2 | ReLU so với các hàm khác (đề nêu ReLU, sigmoid) | ReLU, LeakyReLU, ELU, GELU, Tanh, Sigmoid (có BN và không BN) | 7 |
| **E3 Tối ưu** | TV3 | Thuật toán tối ưu, lr, lịch lr, batch size | SGD, SGD+Nesterov, Adam, AdamW, RMSprop (mỗi cái 2–3 lr); constant / StepLR / Plateau / Cosine / OneCycle; batch 64/128/256 | 12 |
| **E4 Augmentation & điều chuẩn** | TV1 | Augmentation và regularization giảm overfit bao nhiêu? | Không aug / flip / flip+affine / +RandomErasing; dropout 0 / 0.25 / 0.5; weight decay 0 / 5e-4 / 1e-3; label smoothing 0 / 0.1 | 10 |
| **E5 Mất cân bằng lớp** | TV1 | Có cải thiện Disgust/Fear không, đánh đổi accuracy thế nào? | CE thường / CE có trọng số lớp / WeightedRandomSampler | 3 |
| **E6 Transfer learning** | TV2 | Vì sao VGG16 "fail" ở 48 px; pretrained có giúp không? | VGG16 ở 48 px (tái hiện fail); ResNet18 từ đầu vs pretrained ImageNet ở 96 px; fine-tune toàn bộ vs đóng băng backbone; (tùy thời gian) VGG16 pretrained 96 px | 5 |
| **E7 Mô hình cuối** | TV3 | Kết hợp các lựa chọn tốt nhất, đánh giá test | Cấu hình tốt nhất theo val × 3 seed; TTA (flip, ten-crop); ensemble 3 seed; **test PrivateTest 1 lần** | 4 |

Kỳ vọng cần kiểm chứng (không được viết vào báo cáo nếu số liệu không ủng hộ): sigmoid không BN học rất chậm (vanishing gradient, đo chuẩn gradient từng lớp để chứng minh); bỏ augmentation → train acc ~95%+ nhưng val acc giảm rõ (overfit); BN cho phép lr lớn hơn; class weight tăng recall Disgust nhưng giảm nhẹ accuracy tổng.

## 4. Cấu trúc thư mục & hợp đồng giao diện

```
BTL02/
├── README.md, requirements.txt, PHAN_CONG_NHIEM_VU.md
├── configs/        baseline.json, E1_arch.json … E7_final.json
├── src/
│   ├── data/       fer2013.py (đọc CSV, EMOTIONS), loader.py, augment.py, eda.py
│   ├── models/     cnn.py (VGG-style, 4 cấu hình mẫu), transfer.py (ResNet18/VGG16)
│   ├── engine/     trainer.py, evaluate.py
│   ├── visual/     plots.py (curves, confusion, ca đúng/sai), filters.py (trọng số, feature map, activation max, Grad-CAM)
│   ├── utils/      seed.py, io.py
│   ├── run_experiment.py, make_tables.py, visualize_filters.py
├── tests/          test nhỏ: shape đầu ra, thứ tự nhãn, không rò rỉ test, 1 bước train giảm loss
├── data/           fer2013.csv (KHÔNG commit, KHÔNG đưa vào zip)
├── results/        <exp>/<run_id>/{config.json, history.csv, metrics.json, best.pt, curves.png}
├── figures/        hình cho báo cáo/slide
└── report/         BaoCao_BTL02.docx/.pdf, Slide_BTL02.pptx/.pdf, NHAT_KY_THUC_NGHIEM.md, KICH_BAN_THUYET_TRINH.md
```

Chốt ngay ở G1; ai đổi phải báo cả nhóm:

```
src/data/fer2013.py    EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]   # thứ tự Kaggle
                       load_fer2013(csv_path) -> {"train"|"val"|"test": (x uint8 (N,48,48), y int64 (N,))}
src/data/loader.py     make_loaders(cfg) -> (train_loader, val_loader);  make_test_loader(cfg)   # chỉ gọi khi --eval-test
src/models/__init__.py build_model(name, **kwargs) -> nn.Module        # logits (N,7), không softmax trong model
src/engine/trainer.py  fit(model, train_loader, val_loader, cfg, run_dir) -> dict(best_epoch, best_val_acc, train_time_s)
src/engine/evaluate.py predict(model, loader, tta=False) -> (y_true, y_prob);  metrics(y_true, y_pred) -> dict(acc, macro_f1, per_class, cm)
src/visual/filters.py  plot_weights(model, layer), feature_maps(...), activation_max(...), grad_cam(...)
src/run_experiment.py  --config configs/E1_arch.json [--eval-test]
```

Quy tắc bắt buộc:

- `fit()` không nhận PrivateTest. Không chọn cấu hình theo test.
- Không import `tensorflow`/`keras`. Đường dẫn tính từ vị trí file, không viết đường dẫn tuyệt đối/cá nhân.
- Mỗi file code mở đầu bằng docstring ghi **người phụ trách**, ví dụ `"""Vòng huấn luyện CNN. Phụ trách: TV2 – <Họ tên>."""` (đề yêu cầu ghi rõ phân công trong code).
- Commit theo quy ước nhóm (tiếng Việt, `feat:`/`fix:`/`docs:`), mỗi người commit phần mình để lịch sử git cũng là minh chứng phân công.

## 5. Phân công theo chiều ngang và chiều dọc

> Cập nhật 2026-11-20. Thay cho bản phân công 4 vai cố định ban đầu. Mô tả kỹ thuật của từng vai giữ ở mục 5.4.
> Nguồn của ma trận: `tracking/tasks.csv` (tab `Phan_cong` trên Google Sheets). Bảng dưới sinh từ file đó.

### 5.1. Hai chiều phân công

- **Chiều ngang (Horizontal)**: dự án chia thành 7 module gắn với giai đoạn G1–G5 (M0 Quản trị & truy vết, M1 Dữ liệu, M2 Mô hình & hạ tầng huấn luyện, M3 Baseline, M4 Thực nghiệm E1–E6, M5 Mô hình cuối & trực quan hóa, M6 Báo cáo/slide/QA/nộp). Trong một module nhiều thành viên cùng làm.
- **Chiều dọc (Vertical)**: mỗi nhiệm vụ có đúng 4 vai cho 4 người khác nhau.
  - **R**: chịu trách nhiệm chính (thiết kế, làm phần cốt lõi, ký trạng thái *Completed*). Việc cụ thể ghi ở cột `R – việc chính` của bảng 5.3 (`viec_cua_R` trong `tasks.csv`).
  - **C1, C2**: phối hợp. Mỗi người một phần việc riêng, ghi rõ tên người và việc ở bảng 5.3 (cột `viec_cua_C` của `tasks.csv`) (chạy seed/cấu hình trên máy riêng, viết test, review chéo). Việc của C thể hiện qua commit của chính người đó (mã nhiệm vụ trong thông điệp) hoặc run của người đó.
  - **T**: kiểm thử độc lập, không tham gia làm nhiệm vụ đó.
- Mỗi nhiệm vụ có R và T. Không nhiệm vụ nào thiếu người chịu trách nhiệm hoặc người kiểm tra (`build_tracking.py` kiểm điều này khi sinh bảng).
- Số nhiệm vụ làm R: TV1 7, TV2 8, TV3 7, TV4 6. Số lượt làm T (xoay vòng, chênh tối đa 1): TV1 7, TV2 7, TV3 7, TV4 7.

### 5.2. Quy tắc của người kiểm thử (T)

1. T **tự chạy** trên bản clone sạch và **tự tính lại** số liệu từ file gốc (`history.csv`, `experiments.csv`, dự đoán đã lưu). Không xác nhận dựa trên báo cáo hay lời của R.
2. Với nhiệm vụ có huấn luyện, T tự chọn ít nhất 1 cấu hình (không do R chỉ định) để chạy lại và so với `metrics.json`. **Dung sai tái lập = độ lệch chuẩn 3 seed của baseline**; con số này chưa có cho đến khi T11 xong, không được đặt trước.
3. T ghi 1 dòng `Kiem_thu`: lệnh đã chạy, số liệu gốc, số tái tính, `Pass`/`Fail`, bằng chứng. T khác `nguoi_thuc_hien`.
4. Chỉ T mới đưa nhiệm vụ lên **Verified**, và chỉ khi có dòng `Pass`. R không tự đặt Verified.
5. Fail: nhiệm vụ về *Failed*, R sửa rồi quay lại *In Progress*. Lỗi và cách sửa nằm trong dòng `Kiem_thu` và commit sửa lỗi.
6. Nhiệm vụ nền T04, T06, T07, T08, T11 là cổng: chưa Verified thì chưa chạy hàng loạt E1–E6.
7. Độc lập tính theo nhiệm vụ. Một người làm T ở nhiệm vụ này vẫn có thể làm R/C ở nhiệm vụ khác.
8. Đổi người giữa chừng: sửa `Phan_cong` và nêu lý do trong commit; ma trận vẫn phải đủ 4 người khác nhau cho R/C1/C2/T.

Vòng đời trạng thái: `Pending → In Progress → Completed (R, có bằng chứng) → Verified (T Pass)`; `Failed` có thể xảy ra ở bất kỳ bước nào. *Completed* không có bằng chứng thì hạ về *In Progress*.

### 5.3. Ma trận phân công (28 nhiệm vụ)

| ID | Module (ngang) | GĐ | Nhiệm vụ | R (chính) | R – việc chính | C1 – việc cụ thể | C2 – việc cụ thể | T (kiểm thử) | Phụ thuộc |
|---|---|---|---|---|---|---|---|---|---|
| T01 | M0 Quản trị & truy vết | G1 | Phân tích đề, lập kế hoạch và dựng khung theo dõi (Sheets, schema CSV, quy ước log) | TV4 | **TV4**: đọc đề, viết phân tích và kế hoạch (PHAN_CONG_NHIEM_VU.md), dựng schema các CSV và Google Sheets, soạn quy ước log | **TV2**: soát schema log huấn luyện và weights (tab Thuc_nghiem) | **TV3**: soát schema đánh giá và hình (tab Kiem_thu, Hinh_ve), mỗi người ghi góp ý vào commit mã T01 | **TV1** | – |
| T02 | M0 Quản trị & truy vết | G1 | Khung repo: cây thư mục, requirements.txt, README (cài đặt, tải dữ liệu, lệnh chạy, bảng phân công theo file), quy ước docstring Phụ trách và commit | TV1 | **TV1**: dựng cây thư mục, requirements.txt, README (cài đặt, tải dữ liệu, lệnh chạy, bảng phân công theo file), viết quy ước docstring Phụ trách và commit | **TV2**: cài đặt và chạy theo README trên máy riêng, báo chỗ sai | **TV4**: cài trên máy thứ hai và kiểm bảng phân công theo file khớp tab Phan_cong | **TV3** | T01 |
| T03 | M1 Dữ liệu | G1 | Tải fer2013.csv và kiểm tra toàn vẹn (số dòng, 3 giá trị Usage, sha256) | TV1 | **TV1**: tải fer2013.csv, viết script kiểm tra (số dòng theo Usage, sha256) và ghi results/data_check.json | **TV2**: tải fer2013.csv từ Kaggle và tính sha256 | **TV3**: tải từ Google Drive của giảng viên và tính sha256, hai người so với R | **TV4** | T02 |
| T04 | M1 Dữ liệu | G1 | fer2013.py, loader.py, augment.py và tests (nhãn, không rò rỉ test) | TV1 | **TV1**: viết fer2013.py (EMOTIONS, load_fer2013), loader.py (mean/std chỉ từ Training, nạp RAM), augment.py (HFlip, RandomAffine), sửa theo test và góp ý của hai C | **TV3**: viết test nhãn, split, chuẩn hóa trong tests/ | **TV4**: dùng thử loader trên máy riêng và báo lỗi | **TV2** | T03 |
| T05 | M1 Dữ liệu | G1 | EDA: phân bố lớp 3 tập, lưới ảnh mẫu, ảnh trung bình, ảnh lỗi, minh họa augmentation | TV1 | **TV1**: viết eda.py: phân bố lớp 3 tập, lưới ảnh mẫu, ảnh trung bình, thống kê độ sáng, đếm ảnh lỗi, minh họa augmentation, mỗi hình kèm CSV | **TV2**: soát nhãn của lưới ảnh khớp nhãn từng tập | **TV4**: soát kiểu hình và ghi vào Hinh_ve | **TV3** | T04 |
| T06 | M2 Mô hình & hạ tầng huấn luyện | G1 | cnn.py và transfer.py (in số tham số, kích thước bản đồ đặc trưng từng khối) | TV2 | **TV2**: viết cnn.py (VGG-style cấu hình được, 4 cấu hình mẫu) và transfer.py (ResNet18/VGG16 thay lớp cuối 7 lớp), in số tham số và kích thước feature map từng khối, ghi model_summary.csv | **TV1**: viết test shape đầu ra và số tham số | **TV3**: kiểm tay kích thước feature map 48→24→12→6→3→1 | **TV4** | T04 |
| T07 | M2 Mô hình & hạ tầng huấn luyện | G1-G2 | trainer.py và run_experiment.py (history.csv, run.log, ghi experiments.csv, tự quyết giữ/xóa weights theo luật) | TV2 | **TV2**: viết trainer.py (train/eval, AMP, scheduler, early stopping, best.pt, history.csv, run.log) và run_experiment.py (đọc config JSON, lặp giá trị × seed, ghi experiments.csv và config.json) | **TV3**: nối loader và chạy thử 1 epoch trên máy riêng | **TV4**: nối evaluate/plots vào run_dir | **TV1** | T04, T06 |
| T08 | M2 Mô hình & hạ tầng huấn luyện | G2 | evaluate.py và plots.py (Train/Val loss theo epoch, acc, confusion matrix, ca đúng/sai) | TV3 | **TV3**: viết evaluate.py (accuracy, macro-F1, recall từng lớp, ma trận nhầm lẫn, TTA) và plots.py (đường cong loss/acc, confusion, lưới ca đúng/sai, histogram ca sai), mỗi hình kèm CSV | **TV1**: viết test metric (dữ liệu giả lập chỉ nằm trong unit test) | **TV4**: thống nhất style hình cho mọi người dùng | **TV2** | T07 |
| T09 | M2 Mô hình & hạ tầng huấn luyện | G2 | make_tables.py: experiments.csv → bảng báo cáo E1–E7, mean ± std | TV3 | **TV3**: viết make_tables.py đọc experiments.csv và sinh bảng E1–E7 dạng markdown/Word với mean ± std | **TV1**: kiểm công thức mean/std | **TV2**: soát định dạng bảng đưa vào báo cáo | **TV4** | T07 |
| T10 | M2 Mô hình & hạ tầng huấn luyện | G2 | Notebook tái lập (Restart & Run All, log, output lưu lại) | TV3 | **TV3**: viết notebook tái lập gọi code trong src/, chạy Restart & Run All và lưu bản có output vào notebooks/executed/ | **TV2**: chạy notebook Restart & Run All trên máy mình | **TV4**: soát lời giải thích từng bước | **TV1** | T07, T08 |
| T11 | M3 Baseline | G2 | Baseline 3 seed (0/1/2) và đóng băng configs/baseline.json | TV2 | **TV2**: chạy baseline seed 0 trên máy mình, tổng hợp 3 seed (mean ± std), cùng nhóm chốt cấu hình theo val và đóng băng configs/baseline.json | **TV1**: chạy seed 1 trên máy riêng với đúng config | **TV4**: chạy seed 2 trên máy riêng với đúng config (R chạy seed 0) | **TV3** | T04, T06, T07, T08 |
| T12 | M4 Thực nghiệm E1–E6 | G3 | E1 Kiến trúc (SMALL/MEDIUM/MEDIUM_01/LARGE, ×0.5/×2, BN, GAP) | TV2 | **TV2**: viết configs/E1_arch.json, chia cấu hình cho hai C, gom run của cả ba người vào experiments.csv, tổng hợp bảng và kết luận E1 | **TV1**: chạy SMALL, MEDIUM, MEDIUM_01, LARGE | **TV3**: chạy ×0.5/×2 kênh, có/không BN, Flatten vs GAP, mỗi người viết nhận xét độc lập cho ít nhất 1 kết quả của mình | **TV4** | T11 |
| T13 | M4 Thực nghiệm E1–E6 | G3 | E2 Activation (ReLU, LeakyReLU, ELU, GELU, Tanh, Sigmoid) và chuẩn gradient từng lớp | TV2 | **TV2**: viết configs/E2_activation.json (có và không BN), gom run, tổng hợp bảng, kết luận về vanishing gradient của Sigmoid | **TV3**: chạy ReLU, LeakyReLU, ELU, GELU | **TV4**: chạy Tanh, Sigmoid và đo chuẩn gradient từng lớp | **TV1** | T11 |
| T14 | M4 Thực nghiệm E1–E6 | G3 | E3 Tối ưu (optimizer × lr, lịch lr, batch size) | TV3 | **TV3**: viết configs/E3_optim.json (optimizer × lr, lịch lr, batch size), gom run, tổng hợp bảng, chọn optimizer và lr tốt nhất theo val | **TV1**: chạy lưới optimizer × lr của SGD, SGD+Nesterov, Adam | **TV4**: chạy AdamW, RMSprop, lịch lr và batch size | **TV2** | T11 |
| T15 | M4 Thực nghiệm E1–E6 | G3 | E4 Augmentation và điều chuẩn (dropout, weight decay, label smoothing) | TV1 | **TV1**: viết configs/E4_aug_reg.json, gom run, tổng hợp bảng và kết luận mức giảm overfit của augmentation, dropout, weight decay, label smoothing | **TV3**: chạy nhóm augmentation (không aug, flip, flip+affine, +RandomErasing) và dropout | **TV4**: chạy weight decay, label smoothing và vẽ đường cong train/val so overfit từ history.csv | **TV2** | T11 |
| T16 | M4 Thực nghiệm E1–E6 | G3 | E5 Mất cân bằng lớp (CE thường / class weight / WeightedRandomSampler) | TV1 | **TV1**: viết configs/E5_imbalance.json, gom run, tổng hợp bảng, kết luận về recall Disgust/Fear và đánh đổi accuracy | **TV2**: chạy CE thường và CE có trọng số lớp | **TV4**: chạy WeightedRandomSampler và tính recall từng lớp độc lập | **TV3** | T11 |
| T17 | M4 Thực nghiệm E1–E6 | G3 | E6 Transfer learning (VGG16 ở 48 px, ResNet18 từ đầu/pretrained, đóng băng/fine-tune) | TV2 | **TV2**: viết configs/E6_transfer.json, giải thích "fail case" VGG16 bằng kích thước feature map, tổng hợp bảng ResNet18 từ đầu so với pretrained | **TV1**: chạy VGG16 ở 48 px và ResNet18 từ đầu | **TV4**: chạy ResNet18 pretrained (đóng băng, fine-tune) và kiểm giải thích kích thước feature map của VGG16 | **TV3** | T11 |
| T18 | M5 Mô hình cuối & trực quan hóa | G4 | E7 Mô hình cuối: cấu hình tốt nhất theo val × 3 seed, TTA, ensemble | TV3 | **TV3**: chọn cấu hình tốt nhất theo val từ E1–E6, chạy seed 0, ghép 3 seed và ensemble, tính TTA, ghi mean ± std | **TV1**: chạy seed 1 và kiểm ensemble 3 seed độc lập | **TV2**: chạy seed 2 và kiểm TTA (flip, ten-crop) độc lập (R chạy seed 0) | **TV4** | T12, T13, T14, T15, T16, T17 |
| T19 | M5 Mô hình cuối & trực quan hóa | G4 | Đánh giá PrivateTest đúng 1 lần với mô hình cuối | TV3 | **TV3**: chạy python src/run_experiment.py --eval-test đúng 1 lần, lưu TEST_DONE.json, so kết quả với mốc 69%/71% | **TV2**: chứng kiến lệnh --eval-test và ghi nhật ký (giờ, lệnh) | **TV4**: kiểm TEST_DONE.json được tạo và lệnh chạy lần hai bị từ chối | **TV1** | T18 |
| T20 | M5 Mô hình cuối & trực quan hóa | G4 | Trực quan hóa bộ lọc conv cuối (trọng số, feature map, activation maximization, Grad-CAM) | TV4 | **TV4**: viết filters.py và visualize_filters.py (trọng số conv đầu/cuối, feature map, activation maximization, Grad-CAM) và chạy trên best.pt | **TV1**: cung cấp hook/lớp conv cuối | **TV3**: chọn ca đúng/sai cho Grad-CAM từ dự đoán đã lưu | **TV2** | T18 |
| T21 | M6 Báo cáo, slide, QA, nộp | G5 | Báo cáo Ch.2 Dữ liệu và mục E4, E5 | TV1 | **TV1**: viết Ch.2 Dữ liệu và mục E4, E5 bằng bảng và hình sinh từ code | **TV2**: review chéo Ch.2 và dò số liệu về nguồn | **TV3**: review chéo mục E4, E5 và dò số liệu về nguồn | **TV4** | T05, T15, T16 |
| T22 | M6 Báo cáo, slide, QA, nộp | G5 | Báo cáo Ch.3 Mô hình & huấn luyện và mục E1, E2, E6 | TV2 | **TV2**: viết Ch.3 Mô hình & huấn luyện và mục E1, E2, E6 | **TV3**: review chéo Ch.3 và dò số liệu về nguồn | **TV4**: review chéo mục E1, E2, E6 và dò số liệu về nguồn | **TV1** | T12, T13, T17 |
| T23 | M6 Báo cáo, slide, QA, nộp | G5 | Báo cáo Ch.4 Đánh giá, mục E3, E7 và phân tích lỗi | TV3 | **TV3**: viết Ch.4 Đánh giá, mục E3, E7 và phân tích lỗi (cặp lớp hay nhầm, ca sai điển hình) | **TV1**: review chéo Ch.4 và mục E3, dò số liệu về nguồn | **TV4**: review chéo mục E7 và phân tích lỗi, dò số liệu về nguồn | **TV2** | T14, T18, T19 |
| T24 | M6 Báo cáo, slide, QA, nộp | G5 | Nhật ký thực nghiệm NHAT_KY_THUC_NGHIEM.md (giả thuyết, nhận xét, quyết định; kết quả tự điền từ experiments.csv) và sơ đồ kiến trúc/pipeline vẽ từ model thật | TV4 | **TV4**: lập mẫu NHAT_KY_THUC_NGHIEM.md, viết giả thuyết, nhận xét, quyết định cho run E7, vẽ sơ đồ kiến trúc và pipeline từ model thật, kiểm khớp với experiments.csv | **TV1**: viết nhận xét và quyết định cho các run E1, E2, E3 | **TV2**: viết nhận xét và quyết định cho các run E4, E5, E6, mỗi người đối chiếu số với experiments.csv | **TV3** | T12, T13, T14, T15, T16, T17, T19, T20 |
| T25 | M6 Báo cáo, slide, QA, nộp | G5 | Báo cáo Ch.1, 5, 6, 7, 8 và ghép toàn bộ | TV4 | **TV4**: viết Ch.1, 5, 6, 7, 8, ghép toàn bộ vào BaoCao_BTL02, xuất .docx/.pdf, điền bảng % đóng góp từ Tien_do | **TV2**: review chéo Ch.2–4 và dò từng số về nguồn | **TV3**: review chéo Ch.5–8 và dò từng số về nguồn | **TV1** | T20, T21, T22, T23, T24 |
| T26 | M6 Báo cáo, slide, QA, nộp | G5 | Slide .pptx/.pdf (Google Slides), kịch bản thuyết trình và chuẩn bị demo chạy code | TV4 | **TV4**: làm slide 12–15 trang, kịch bản thuyết trình, bộ câu hỏi Q&A dự kiến, chuẩn bị demo chạy code | **TV1**: viết 2–3 slide phần mình phụ trách và dò số liệu về nguồn | **TV3**: viết 2–3 slide phần mình phụ trách, cùng R tập demo chạy code | **TV2** | T25 |
| T27 | M6 Báo cáo, slide, QA, nộp | G5 | QA tái lập toàn repo (clone sạch, venv mới, theo README, pytest, chạy notebook) | TV4 | **TV4**: clone sạch, tạo venv mới, chạy theo README, pytest, chạy notebook, ghi biên bản QA và danh sách lỗi | **TV1**: chạy clone sạch, venv mới, theo README và pytest trên máy riêng | **TV2**: chạy các notebook trên máy riêng, mỗi người báo lỗi cho người phụ trách | **TV3** | T19, T20 |
| T28 | M6 Báo cáo, slide, QA, nộp | G5 | Đóng gói BTLON02_Emotion_HoTen_MaSV.zip (họ tên, MSSV nhóm trưởng) và nộp bài | TV2 | **TV2**: gom file theo checklist mục 8, tạo BTLON02_Emotion_HoTen_MaSV.zip (họ tên, MSSV của nhóm trưởng) và nộp bài | **TV1**: kiểm checklist mục 8 phần code, README, requirements, không có fer2013.csv và checkpoint trung gian | **TV3**: kiểm phần report, slide, nhật ký thực nghiệm, tracking/*.csv | **TV4** | T25, T26, T27 |

Tỷ lệ đóng góp ghi trong báo cáo (thay cho 25% dự kiến) lấy từ số liệu thật của tab `Tien_do` (số R/C/T, số commit gắn mã nhiệm vụ (từ `git_activity.py`), số lần kiểm thử Pass); nhóm thống nhất công thức ở G5.

### 5.4. Mô tả kỹ thuật theo vai (nội dung phần R chính; phần phối hợp và kiểm thử xem 5.3)

Thành viên (nhóm xác nhận 2026-11-20): **TV1 Đoàn Trung Kiên, TV2 Lê Phan Nguyên Hải, TV3 Võ Hoàng Bảo, TV4 Huỳnh Viễn Thông**. **Nhóm trưởng: TV2 Lê Phan Nguyên Hải** (đứng tên và nộp zip; xác nhận 2026-11-20). Cả 4 thành viên đều có GPU (xác nhận 2026-11-20; tên GPU và thời gian/epoch từng máy chưa đo, ghi ở `Thanh_vien` và `Thuc_nghiem`). MSSV, máy và GPU điền ở tab `Thanh_vien` (`tracking/members.csv`); họ tên cũng ghi trong docstring từng file code.

### TV1 – Dữ liệu, tiền xử lý, augmentation; E4, E5
- Tải `fer2013.csv` theo link ở mục 2.1 (Kaggle hoặc Drive), chép hướng dẫn tải + link vào README; kiểm tra file có đủ cột `Usage` với 3 giá trị Training/PublicTest/PrivateTest.
- `src/data/fer2013.py`: đọc CSV → mảng uint8, tách theo `Usage`, hằng `EMOTIONS`; test kiểm tra số ảnh 28 709 / 3 589 / 3 589 và thứ tự nhãn.
- `src/data/loader.py`, `augment.py`: chuẩn hóa theo mean/std Training, nạp toàn bộ vào bộ nhớ, augmentation theo batch, seed cho thứ tự batch; tùy chọn resize + 3 kênh cho E6.
- `src/data/eda.py`: histogram phân bố lớp 3 tập, lưới ảnh mẫu mỗi lớp (đúng nhãn của từng tập), ảnh trung bình mỗi lớp, thống kê độ sáng, đếm ảnh lỗi (toàn đen/trắng); hình minh họa augmentation.
- **E4** (augmentation, dropout, weight decay, label smoothing) và **E5** (mất cân bằng lớp): bảng + đường cong train/val để chỉ ra overfit.
- Khung repo, `requirements.txt`, `README.md` (cài đặt, tải dữ liệu, lệnh chạy từng nhóm thực nghiệm).
- Báo cáo: **Ch.2 Dữ liệu & tiền xử lý**, mục **E4, E5**; slide: dữ liệu + augmentation + kết quả E4/E5.

### TV2 – Mô hình CNN, vòng huấn luyện; E1, E2, E6; nhóm trưởng (đóng gói và nộp zip)
- `src/models/cnn.py`: VGG-style cấu hình được (số khối, số conv/khối, số kênh, BN, activation, dropout, Flatten/GAP); 4 cấu hình tái hiện mẫu; in số tham số và kích thước bản đồ đặc trưng từng khối.
- `src/models/transfer.py`: ResNet18/VGG16 torchvision, thay lớp cuối 7 lớp, tùy chọn đóng băng backbone.
- `src/engine/trainer.py`: `train()/eval()`, `torch.no_grad()`, AMP, scheduler, early stopping, `best.pt`/`last.pt`, `history.csv`, đo thời gian, ghi `experiments.csv`. **Ưu tiên xong sớm nhất** vì mọi người phụ thuộc.
- `src/run_experiment.py`: đọc config JSON, lặp các giá trị khảo sát × seed.
- Chạy baseline 3 seed, chốt baseline cùng nhóm (theo val).
- **E1** (kiến trúc), **E2** (activation, kèm chuẩn gradient từng lớp để chứng minh/phủ định vanishing gradient), **E6** (transfer learning, giải thích "fail case" VGG16 bằng kích thước feature map 48→24→12→6→3→1).
- Báo cáo: **Ch.3 Mô hình CNN & huấn luyện** (Conv, Pooling, BN, Dropout, ReLU, CE, SGD/Adam), mục **E1, E2, E6**; slide: kiến trúc + code PyTorch chính. Chuẩn bị **demo chạy code** trước lớp.
- **Nhóm trưởng**: đóng gói và nộp `BTLON02_Emotion_HoTen_MaSV.zip` (họ tên, MSSV của TV2) ở T28; điền MSSV ở tab `Thanh_vien`.

### TV3 – Đánh giá, hình kết quả, tối ưu; E3, E7
- `src/engine/evaluate.py`: accuracy, macro-F1, precision/recall từng lớp, ma trận nhầm lẫn (số lượng + chuẩn hóa), TTA (flip, ten-crop).
- `src/visual/plots.py`: đường cong loss/acc/lr theo epoch, ma trận nhầm lẫn, lưới ca đúng/sai (nhãn đoán/nhãn thật), histogram ca sai theo lớp, đồ thị so sánh các run (thống nhất style cho mọi người dùng).
- `src/make_tables.py`: đọc `tracking/experiments.csv` → bảng markdown/Word cho từng nhóm E1–E7 (số liệu báo cáo truy vết được).
- **E3** (optimizer × lr, lịch lr, batch size).
- **E7**: chọn mô hình cuối theo val, 3 seed, TTA, ensemble; **chạy test PrivateTest đúng 1 lần**; so sánh với mốc 69%/71% và với báo cáo mẫu.
- Báo cáo: **Ch.4 Phương pháp đánh giá**, mục **E3, E7**, phần **phân tích lỗi** (cặp lớp hay nhầm, ca sai điển hình); slide: kết quả tổng + ma trận nhầm lẫn.

### TV4 – Trực quan hóa bộ lọc; report, slide, nhật ký thực nghiệm, QA
- `src/visual/filters.py` + `src/visualize_filters.py`: 4 cách ở mục 2.4 (trọng số conv đầu/cuối, feature maps, activation maximization, Grad-CAM) trên mô hình cuối và so sánh với baseline/mô hình chưa huấn luyện.
- Template Word (bìa, mục lục tự động, đánh số hình/bảng) theo bố cục `demo_report.pdf`; gửi nhóm ngay G1.
- Sơ đồ kiến trúc CNN của nhóm và pipeline train/val/test (vẽ từ model thật).
- **Nhật ký thực nghiệm** `report/NHAT_KY_THUC_NGHIEM.md`: lập mẫu (mục 6), nhắc mọi người ghi sau mỗi run, kiểm tra khớp với `experiments.csv`.
- **QA tái lập**: clone repo sang thư mục mới, venv mới, chạy theo README; chạy `pytest`; báo lỗi cho người phụ trách.
- Báo cáo: **Ch.1 Giới thiệu** (bài toán, mục tiêu, bảng phân công % đóng góp), **Ch.5 Tóm tắt mã nguồn** (ghép từ phần mỗi người), **Ch.6 Trực quan hóa bộ lọc**, **Ch.7 Thảo luận tổng hợp** (ý kiến riêng của nhóm, trường hợp không như kỳ vọng, hạn chế, hướng cải tiến), **Ch.8 Kết luận**, TLTK, phụ lục lệnh chạy; xuất .docx/.pdf.
- Slide .pptx 12–15 trang + kịch bản thuyết trình, phân vai, bộ câu hỏi Q&A dự kiến.

## 6. Nhật ký làm việc và hệ thống theo dõi

### 6.1. Google Sheets (nơi nhập hằng ngày) và CSV trong git (bản lưu vết)

| Tab | Nội dung | CSV tương ứng |
|---|---|---|
| `Huong_dan` | Quy tắc đọc/ghi các tab | – |
| `Phan_cong` | Bảng nhiệm vụ: module, R/C1/C2/T, bằng chứng cần có, cách T kiểm, phụ thuộc, trạng thái | `tracking/tasks.csv` |
| `Ma_tran` | Ngang × dọc: từng người giữ vai R/C/T ở từng nhiệm vụ (công thức từ `Phan_cong`) | – |
| `Tien_do` | Đếm theo module và theo thành viên bằng công thức, không nhập tay | – |
| `Kiem_thu` | Nhật ký kiểm thử và đánh giá chéo | `tracking/test_log.csv` |
| `Hoat_dong` | Nhật ký git (bản sao chỉ đọc, IMPORTDATA) | `tracking/activity_log.csv` |
| `Thuc_nghiem` | Mỗi lần train một dòng (kể cả run lỗi); bản sao chỉ đọc, IMPORTDATA | `tracking/experiments.csv` |
| `Hinh_ve` | Mỗi hình: script, dữ liệu đầu vào, lệnh, experiment liên quan | `tracking/figures_manifest.csv` |

Liên kết giữa các bảng: `task_id` (T01–T28) nối Phan_cong ↔ Kiem_thu ↔ commit git (mã nhiệm vụ trong thông điệp); `exp_id` nối Thuc_nghiem ↔ Hinh_ve (weights nằm ngay trong dòng Thuc_nghiem); `weights_id`, `fig_id` là khóa của từng bảng.

Chuỗi truy vết: **nhiệm vụ → thành viên → source code (commit) → log (`run.log`) → kết quả (`experiments.csv`, `history.csv`) → kiểm thử (`Kiem_thu`) → báo cáo/slide**.

Đồng bộ lên Sheets (tự động một chiều, không nhập tay): tab `Hoat_dong` và `Thuc_nghiem` là **bản sao chỉ đọc** của `tracking/activity_log.csv` và `tracking/experiments.csv` trên GitHub, kéo về bằng `=IMPORTDATA(<đường dẫn raw>)`. Dữ liệu chỉ lên Sheets sau khi file được **commit và push**; Google làm mới bộ nhớ đệm sau khoảng vài chục phút đến vài giờ, nên có độ trễ. Không sửa tay hai tab này (sẽ bị công thức ghi đè). `Phan_cong` và `Kiem_thu` vẫn nhập trên Sheets; cuối mỗi buổi xuất CSV ghi đè `tracking/*.csv` và commit nếu muốn giữ bản lưu vết. `build_tracking.py` chỉ **khởi tạo** bảng (ghi đè `tasks.csv`, `tracking.xlsx`; không ghi đè `members.csv` và các CSV nhật ký đã có).

Liên kết Google Sheets/Slides: xem mục 10 (quyền chia sẻ cho 3 thành viên còn lại do nhóm trưởng cấp).

### 6.2. Nhật ký hoạt động: tự sinh, không nhập tay

Không có tab nhật ký để điền. Nhật ký hoạt động lấy từ ba nguồn mà công việc thật đã tạo ra:

| Nguồn | Có gì | Ai/cái gì ghi |
|---|---|---|
| Lịch sử git | Giờ, tác giả, file sửa, mã nhiệm vụ | Tự có khi commit; `git_activity.py` gom thành `tracking/activity_log.csv`; workflow `.github/workflows/activity-log.yml` tự chạy sau mỗi push lên `main` và commit lại file này (bỏ qua commit của bot) |
| `run.log`, `history.csv`, `config.json`, dòng `Thuc_nghiem` | Ai chạy, giờ, máy, tổng thời gian, loss từng epoch | `run_experiment.py` tự ghi (mục 9.1) |
| Tab `Kiem_thu` | Pass/Fail, lệnh, số tái tính | Người kiểm thử (một dòng khi kiểm xong) |

Quy ước duy nhất cần nhớ: **thông điệp commit mang mã nhiệm vụ**, ví dụ `feat(T07): thêm đường cong Train/Val loss`. Commit không có mã vẫn được ghi nhưng hiện là "(không có mã nhiệm vụ)". Tên git của từng người điền ở cột `git_user` của `members.csv` để quy ra TV1–TV4; chưa điền thì hiện tên git thô.

Quy tắc trạng thái: không đặt *Completed* khi chưa có commit hoặc output làm bằng chứng; không đặt *Verified* khi chưa có dòng `Pass`. Phần đóng góp thực tế trong báo cáo lấy từ số commit, số file và số lần kiểm thử mà script tổng hợp ra. Đo bằng số commit chỉ là thô (một commit lớn bằng nhiều commit nhỏ); công thức tỷ lệ nhóm thống nhất ở G5.

### 6.3. Nhật ký thực nghiệm dạng văn bản (`report/NHAT_KY_THUC_NGHIEM.md`)

Đề yêu cầu nộp "nhật ký thực nghiệm". File này **không nhập số tay**: dòng *Kết quả* do `make_tables.py` điền từ `experiments.csv`; người làm chỉ viết giả thuyết, nhận xét và quyết định.

```
### [E3-007] AdamW lr=1e-3, cosine — TV3 – <ngày chạy thật>
- Giả thuyết: …
- Thay đổi so với baseline: …
- Lệnh: python src/run_experiment.py --config configs/E3_optim.json --only adamw_1e-3
- Kết quả: (tự điền từ experiments.csv: best val loss/acc, epoch, thời gian, run_dir)
- Nhận xét: … (phân biệt: chạy thành công về kỹ thuật ≠ mô hình tốt)
- Quyết định: giữ / bỏ / chạy thêm seed. Weights: tự có trong dòng thực nghiệm (giữ/xóa + mã lý do); chỉ cần bình luận nếu muốn.
```

Mục đầu ghi môi trường (GPU, phiên bản torch, sha256 `fer2013.csv`), quy trình chia tập và baseline đóng băng. Run lỗi cũng ghi kèm nguyên nhân.

## 7. Tiến độ & phụ thuộc

| Giai đoạn | Việc chính | Mốc |
|---|---|---|
| G1 – Khởi động | Tải dữ liệu, khung repo, chốt interface (mục 4), template báo cáo, mẫu nhật ký | TV1: `load_fer2013` + loader; TV2: `fit()` chạy được 1 epoch; TV4: template |
| G2 – Baseline | Baseline 3 seed, chốt và **đóng băng** `configs/baseline.json`; xong evaluate/plots | Mọi người dùng chung baseline |
| G3 – Thực nghiệm | E1–E6 song song trên máy từng người, ghi nhật ký sau mỗi run | `tracking/experiments.csv` đủ các nhóm |
| G4 – Mô hình cuối | E7 + test 1 lần; TV4 trực quan hóa bộ lọc trên mô hình cuối | `results/E7/…/best.pt`, hình filters |
| G5 – Viết & nộp | Mỗi người viết phần mình + slide; TV4 ghép; review chéo (TV1↔TV3, TV2↔TV4); QA tái lập; tập thuyết trình; zip | Nộp bài |

Cổng chất lượng: không chuyển giai đoạn khi nhiệm vụ nền (T04, T06, T07, T08, T11) chưa **Verified**. Phụ thuộc từng nhiệm vụ ở cột `Phụ thuộc` mục 5.3.

Phụ thuộc chính: mọi nhóm thực nghiệm cần `fit()` của TV2 và loader của TV1 → hai phần này xong trước tiên. Đổi baseline sau G2 thì phải chạy lại các nhóm đã chạy.

## 8. Checklist nộp bài

- [ ] Code PyTorch chạy lại được theo README, không TF/Keras, không đường dẫn tuyệt đối
- [ ] Mỗi file code có dòng **Phụ trách: TVx – Họ tên**; README có bảng phân công theo file
- [ ] PublicTest chỉ để chọn mô hình; PrivateTest đánh giá đúng 1 lần ở mô hình cuối
- [ ] Bảng kết quả E1–E7 sinh từ `tracking/experiments.csv`; baseline và mô hình cuối có mean ± std 3 seed
- [ ] Báo cáo có đủ: tóm tắt mã nguồn, nhiều thực nghiệm, **trực quan hóa bộ lọc conv cuối**, phân tích của nhóm
- [ ] Báo cáo `.docx` + `.pdf`, slide `.pptx` + `.pdf`, `NHAT_KY_THUC_NGHIEM.md`
- [ ] Bảng thành viên – nhiệm vụ – % đóng góp (tổng 100%) trong báo cáo, slide và README
- [ ] Zip `BTLON02_Emotion_<HoTen>_<MaSV>.zip` (nhóm trưởng TV2): code, configs, report, slide, `NHAT_KY_THUC_NGHIEM.md`, `tracking/*.csv` (phân công, kiểm thử, thực nghiệm) và `activity_log.csv` đã sinh từ git, `experiments.csv` + log; **không** kèm `fer2013.csv` và checkpoint trung gian (chỉ giữ `best.pt` mô hình cuối nếu dung lượng cho phép)
- [ ] Mọi nhiệm vụ trong `Phan_cong` ở trạng thái Verified, hoặc ghi rõ lý do nếu không
- [ ] Mọi số liệu/hình trong báo cáo và slide truy vết được về `experiments.csv` hoặc `Hinh_ve`; không có hình/số minh họa
- [ ] Mỗi hình có file CSV dữ liệu vẽ và script sinh; có hình Train/Val loss theo epoch cho các run chính
- [ ] Notebook đã chạy từ đầu đến cuối (Restart & Run All), bản có output lưu ở `notebooks/executed/`
- [ ] Mỗi run có `weights_quyet_dinh` + `weights_ma_ly_do` do code ghi; không còn file `.pt` của run DISCARD (T kiểm bằng lệnh quét thư mục)
- [ ] PrivateTest chỉ chạy 1 lần (`results/E7/TEST_DONE.json`)

## 9. Quy chuẩn huấn luyện, log, biểu đồ, notebook và weights

Áp dụng cho mọi run từ G1. Phần này **chưa có code hiện thực**; T06/T07/T08/T10 là nhiệm vụ viết và kiểm các quy chuẩn này.

### 9.1. Mỗi lần train (run)

- `exp_id`: `<nhóm>-<số thứ tự 3 chữ số>`, ví dụ `E3-007`; mỗi seed là một run (`E3-007_s1`). Chạy lại sau khi lỗi dùng `exp_id` mới, không ghi đè.
- Thư mục `results/<nhóm>/<exp_id>/` gồm:
  - `config.json`: hyperparameters đầy đủ, seed, `member` (bắt buộc truyền `--member TVx`, chỉ nhận mã trong `members.csv`), `hostname`, tên user hệ điều hành, `git user.name`, tên GPU, `started_at` và `finished_at` (ISO có múi giờ), `duration_s`, `git_commit`, sha256 của `fer2013.csv`, phiên bản torch/CUDA, tên GPU.
  - `run.log`: mỗi epoch một dòng, ghi và flush ngay: `thoi_gian | exp_id | epoch/N | train_loss | val_loss | val_acc | val_macro_f1 | lr | giây/epoch | trạng thái`; có dòng START, END hoặc ERROR (kèm traceback).
  - `history.csv`: cùng số liệu, cột `epoch,train_loss,val_loss,train_acc,val_acc,val_macro_f1,lr,epoch_time_s`. Ghi từng epoch để run bị ngắt vẫn còn dữ liệu.
  - `metrics.json`: kết quả cuối, `best_epoch`, thời gian tổng, trạng thái.
  - `curves.png` + `curves.csv`: Train/Val loss (và acc) theo epoch, sinh bởi `plots.py` từ `history.csv`.
  - `best.pt` chỉ khi run hợp lệ (mục 9.2).
- Ai chạy và chạy lúc nào do **code tự ghi** (không gõ tay): `run_experiment.py` tự thêm dòng vào `tracking/experiments.csv` (file duy nhất; không còn `results/experiments.csv`) với `thanh_vien`, `bat_dau`, `ket_thuc`, `tong_thoi_gian_s`, `may_chay` (hostname + GPU), lấy từ `config.json`/`metrics.json`. Người kiểm thử đối chiếu `hostname` với tab `Thanh_vien` và `git log` (tác giả commit).
- Định nghĩa cột `experiments.csv` (tránh hiểu khác nhau): `val_acc`, `val_macro_f1`, `best_epoch` lấy tại epoch chọn theo val acc (cùng epoch với `best.pt`); `train_loss_cuoi`, `train_acc_cuoi`, `val_loss_cuoi` lấy ở epoch cuối đã chạy; `best_val_loss` là loss val thấp nhất qua các epoch (chỉ tham khảo, có thể khác `best_epoch`); `n_params` đếm bằng `sum(p.numel())`; `epoch_time_s_tb` là trung bình `epoch_time_s` trong `history.csv`; mọi đường dẫn (`config_path`, `weights_path`, `run_dir`, …) là **đường dẫn tương đối từ `BTL02/`**. File **không có** cột người kiểm thử hay nhận xét: người kiểm ghi ở `Kiem_thu`, nhận xét ở `NHAT_KY_THUC_NGHIEM.md`, vì tab `Thuc_nghiem` là bản sao chỉ đọc.
- Nhiều người cùng push: `experiments.csv` chỉ được nối thêm dòng, `BTL02/.gitattributes` đặt `merge=union` để git tự gộp, không conflict. Sau khi gộp kiểm lại không có `exp_id` trùng.
- Giới hạn: thông tin này khai báo qua `--member` và dấu vết máy, không chống giả mạo; độ tin cậy dựa vào việc T đối chiếu với commit và hostname.
- Seed cố định; PublicTest chỉ để chọn mô hình; PrivateTest chỉ ở `--eval-test` (ghi `TEST_DONE.json`, lệnh từ chối nếu file đã tồn tại).

### 9.2. Thành công kỹ thuật khác chất lượng mô hình

- `trang_thai_ky_thuat`: Completed / Failed / Aborted. Completed khi vòng lặp kết thúc (đủ epoch hoặc early stopping), mọi loss hữu hạn, `metrics.json` đầy đủ, `best.pt` nạp lại và val_acc tính lại khớp `history.csv` (dung sai số học do nhóm đặt trong test, chưa chốt).
- `so_voi_baseline_tu_tinh`: do `make_tables.py` tính từ số liệu (cao hơn / thấp hơn / chưa phân biệt được vì chênh lệch nhỏ hơn std 3 seed của baseline; khoảng cách train–val). Chỉ là con số, không phải diễn giải. Nhận xét bằng lời (vì sao, overfit hay không) do người viết ở `report/NHAT_KY_THUC_NGHIEM.md`; code không tự bịa nhận xét.
- **Không loại run chỉ vì metric thấp.** Kết quả không tốt vẫn ghi đầy đủ để phân tích.

### 9.3. Quản lý weights (tự động, không bảng nhập tay)

Cuối mỗi run (khối `finally` của `fit`), hàm `finalize_weights(run_dir)` quyết định giữ hay xóa **theo luật cố định**, rồi tự ghi kết quả vào chính dòng của run trong `experiments.csv`. Không có tab `Weights`, không ai điền.

| Điều kiện (kiểm theo thứ tự) | Quyết định | `weights_ma_ly_do` |
|---|---|---|
| Có exception khi train | DISCARD | `EXCEPTION:<tên lỗi>` |
| Loss không hữu hạn (NaN/Inf) | DISCARD | `NAN_LOSS@epoch<N>` |
| Bị ngắt (Ctrl+C, kernel dừng) | DISCARD | `INTERRUPTED@epoch<N>` |
| Không có `best.pt` | DISCARD | `NO_CHECKPOINT` |
| Nạp lại `best.pt`, tính lại val_acc không khớp `history.csv` | DISCARD | `RELOAD_MISMATCH` |
| Còn lại (kết thúc đủ epoch hoặc early stopping, loss hữu hạn, tính lại khớp) | **KEEP** | `OK` |

- **Luật không xét metric**: run có val_acc thấp mà chạy đúng vẫn KEEP và vẫn nằm trong bảng để phân tích.
- DISCARD: xóa mọi `*.pt` của run đó, **giữ** `config.json`, `run.log`, `history.csv`.
- Cột tự ghi: `weights_path`, `weights_sha256`, `weights_size_mb`, `weights_quyet_dinh`, `weights_ma_ly_do`. Run bị xóa weights thì `weights_path`, `weights_sha256`, `weights_size_mb` trống.
- Ghi weights qua file tạm rồi đổi tên (`best.pt.tmp` → `best.pt`) để không để lại file dở.
- **Không bao giờ xóa weights của run khác**: `finalize_weights` chỉ đụng thư mục của chính run đang kết thúc và không có đường xóa nào khác, nên baseline, E7 và mô hình dùng cho hình bộ lọc không thể bị xóa nhầm khi chạy các run sau.
- Process bị kill cứng (mất điện, `taskkill`) thì khối `finally` không chạy. Khi chạy `make_tables.py` hoặc run kế tiếp, mọi `run_dir` có dòng START mà không có END được quét, đánh `INTERRUPTED@epoch<N cuối trong history.csv>` và xóa `*.pt` dở. Việc này cũng tự động.
- Giới hạn: lý do chỉ là mã phân loại, không giải thích nguyên nhân sâu; nguyên nhân xem traceback trong `run.log`. Nếu luật cần đổi (ví dụ ngưỡng khớp val_acc) thì đổi trong code và commit, không sửa tay từng dòng.
- Weights không commit lên git (xem `.gitignore`); chỉ nộp `best.pt` mô hình cuối nếu dung lượng cho phép.
- Kiểm thử (T07): chạy một run lỗi cố ý, một run bị ngắt giữa chừng và một run bình thường; kiểm đúng 3 mã lý do và đúng file `.pt` còn/mất.

### 9.4. Biểu đồ và số liệu

- Mọi hình sinh từ code; mọi số trong báo cáo/slide lấy từ CSV bằng code (`make_tables.py`, `plots.py`). Cấm gõ tay, ước lượng hoặc dùng số minh họa. Dữ liệu giả chỉ được phép bên trong unit test, không bao giờ ra hình hay bảng.
- Mỗi hình `figures/<tên>.png` kèm `figures/<tên>.csv` (dữ liệu đã vẽ) và một dòng `Hinh_ve` (script, lệnh, `exp_ids`, sha256).
- Hình bắt buộc (khi có dữ liệu): Train/Val loss theo epoch cho mỗi run chính và đồ thị so sánh run; val acc theo epoch; phân bố lớp 3 tập; ma trận nhầm lẫn chuẩn hóa; recall từng lớp; hình bộ lọc (T20).
- Chưa có kết quả thì ghi **"chưa thực hiện"**, không điền chỗ trống.

### 9.5. Jupyter Notebook

Log được lưu **ở cả hai nơi**: file `run.log`/`history.csv` (bản gốc, không mất khi kernel chết) và output của notebook (để notebook tự chứng minh quá trình chạy).

- `notebooks/BTL02_<nhóm hoặc exp_id>.ipynb`: có giải thích từng bước, gọi hàm trong `src/` (không chép lại logic để tránh hai bản code lệch nhau), không phụ thuộc biến của lần chạy trước.
- Huấn luyện chạy **trong chính kernel của notebook** (gọi hàm `fit`, không chạy qua terminal hay subprocess), và `fit` ghi log qua `logging` với hai handler: file `run.log` và stdout. Nhờ vậy mỗi dòng epoch (thời gian, epoch, train/val loss, metric, lr, trạng thái) vừa hiện trong output ô chạy vừa nằm trong `run.log`.
- Ô đầu in: mã thành viên, hostname, GPU, torch, `git_commit`, sha256 dữ liệu, seed, giờ bắt đầu.
- Ô cuối: đọc `history.csv` vẽ Train/Val loss, in `metrics.json`, in `tong_thoi_gian_s`, in sha256 của `run.log` và `history.csv`, và **kiểm tra bằng `assert`** số dòng epoch trong `run.log` bằng số dòng `history.csv`.
- Bản bằng chứng: `jupyter nbconvert --to notebook --execute --output executed/<exp_id>.ipynb`, giữ nguyên output, commit. Đường dẫn bản này ghi vào cột `notebook_path` của `Thuc_nghiem`. Không sửa tay output; chạy lại thì lưu bản mới.
- Nếu kernel chết giữa chừng: `run.log` vẫn còn, run ghi Aborted (mục 9.3); notebook không có output đầy đủ thì không dùng làm bằng chứng, chạy lại với `exp_id` mới.
- Lệnh `python src/run_experiment.py` ghi cùng định dạng log; notebook chỉ là cách chạy có kèm output.
- Windows: `num_workers=0` (dữ liệu nằm trong RAM/GPU).

## 10. Hiện trạng kiểm kê (2026-11-20) và việc ưu tiên

Kiểm tra trực tiếp thư mục `BTL02/` (lệnh `find`, `git status`):

| Hạng mục | Hiện trạng |
|---|---|
| Source code `.py`, `src/`, `tests/`, `configs/` | **Chưa có** |
| Notebook `.ipynb` của nhóm | **Chưa có** |
| `data/fer2013.csv` | **Chưa có** (thư mục `data/` chưa tạo) |
| `results/`, `figures/`, log, weights | **Chưa có** |
| Git | `BTL02/` chưa được commit (`??`); không có commit nào của BTL02 |
| Tài liệu có | Đề (`Emotion_Recognition.pdf`), báo cáo mẫu, 4 notebook mẫu Keras (chỉ tham khảo), kế hoạch này |
| Môi trường | Python 3.12.10, torch 2.6.0+cu124, CUDA khả dụng, matplotlib/openpyxl/python-pptx có sẵn |

Do đó **chưa có số liệu thực nghiệm nào**; mọi nhiệm vụ trừ T01 ở trạng thái Pending, không nhiệm vụ nào Verified.

Chưa xác minh được: bảng thời gian/VRAM ở 2.5; phân bố lớp ở 2.1 (số công bố của bộ dữ liệu, TV1 xác nhận bằng EDA ở T05); độ chính xác của báo cáo mẫu ở 2.2 (lấy từ tài liệu mẫu, chưa tái hiện).

Liên kết (tạo 2026-11-20, nguồn khởi tạo là `tracking/build_tracking.py` và `tracking/build_slides.py`; sau đó nhập trực tiếp trên Google):

- Google Sheets: https://docs.google.com/spreadsheets/d/1mgwx1TGTL0fQewJ3Qto-K_HcZEUbgN4_P7Cy2-Z-aFk/edit
- Google Slides: https://docs.google.com/presentation/d/1rAOrL4kSXBp65r5EwoXaGcyesiBAYCSxy3HibfPiPT4/edit
- Bản v1, v2 (cùng ngày) thiếu nhóm trưởng và 2 nhiệm vụ mới, không dùng nữa (nhóm tự xóa trên Drive).

Việc ưu tiên:

1. Nhóm xem lại ma trận 5.3; mỗi người điền MSSV, hostname và GPU ở tab `Thanh_vien`.
2. Commit khung hiện tại (`PHAN_CONG_NHIEM_VU.md`, `tracking/`, `.gitignore`) theo quy ước commit nhóm.
3. T02: khung repo, `requirements.txt`, README có bảng phân công theo file.
4. T03: tải `fer2013.csv`, chạy kiểm toàn vẹn, ghi sha256.
5. T04, T06, T07: loader, model, trainer có log/history/weights an toàn; đo lại ngân sách tính toán (2.5).
6. T08, T10: evaluate/plots (hình Train/Val loss từ CSV) và notebook tái lập.
7. T11: baseline 3 seed, chốt dung sai tái lập; T xác nhận trước khi chạy E1–E6.
8. E1–E6 song song, rồi E7 và test PrivateTest 1 lần; trực quan hóa bộ lọc; báo cáo, slide, QA, zip.
