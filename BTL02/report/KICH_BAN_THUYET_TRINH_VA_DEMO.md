# Kịch bản Thuyết trình, Bộ câu hỏi Q&A và Quy trình Demo BTL02 (Nhiệm vụ T26)

> **Phụ trách chính (`R`)**: TV4 – Huỳnh Viễn Thông  
> **Phối hợp (`C1`, `C2`)**: TV1 – Đoàn Trung Kiên (slide dữ liệu, E4, E5), TV3 – Võ Hoàng Bảo (slide E3, E7, PrivateTest & tập demo)  
> **Kiểm thử độc lập (`T`)**: TV2 – Lê Phan Nguyên Hải  
> **Tệp slide**: `report/Slide_BTL02.pptx` và `report/Slide_BTL02.pdf` (15 trang, tỷ lệ 16:9)  
> **Google Slides tiến độ**: https://docs.google.com/presentation/d/1rAOrL4kSXBp65r5EwoXaGcyesiBAYCSxy3HibfPiPT4/edit  
> **Google Sheets theo dõi**: https://docs.google.com/spreadsheets/d/1mgwx1TGTL0fQewJ3Qto-K_HcZEUbgN4_P7Cy2-Z-aFk/edit

---

## 1. Phân vai Thuyết trình & Phân bổ Thời lượng (Tổng ~14 phút + 4 phút Demo)

| Phần | Slide | Người trình bày | Thời lượng | Trọng tâm nội dung |
| :--- | :---: | :--- | :---: | :--- |
| **Mở đầu & Bài toán** | 1 – 2 | **TV4 – Huỳnh Viễn Thông** | 1.5 phút | Giới thiệu đề tài, 3 thách thức FER2013, 4 mục tiêu, quy trình chống rò rỉ |
| **Dữ liệu, Augmentation & Cân bằng lớp** | 3, 8, 9 | **TV1 – Đoàn Trung Kiên** | 3.0 phút | EDA `35,887` ảnh (`16.55×`), E4 giảm gap `28.09 → 4.72` điểm %, E5 WRS tăng Macro-F1 `+3.01%` |
| **Kiến trúc, Kích hoạt & Transfer Learning** | 4, 5, 6, 10 | **TV2 – Lê Phan Nguyên Hải** | 4.0 phút | Baseline 3 seed (`68.15% ± 0.75%`), E1 `large_bn` & giải mã sập `24.94%`, E2 vanishing gradient `2.5M×`, E6 fail case VGG16 48 px |
| **Tối ưu, Mô hình cuối, PrivateTest & Lỗi** | 7, 11, 12, 13 | **TV3 – Võ Hoàng Bảo** | 3.5 phút | E3 (`bs_64 69.24%`), E7 Ensemble + flip (`72.50%`), **PrivateTest `73.64%`** (`TEST_DONE.json`), phân tích 946 lỗi |
| **Trực quan hóa, Kết luận & Dẫn Demo** | 14 – 15 | **TV4 – Huỳnh Viễn Thông** | 2.0 phút | Trọng số `conv1_1` vs `conv4_4`, Feature map, Grad-CAM 15 ca (`PublicTest`), 3 kết luận cốt lõi, phân công 25%/người |
| **Demo trực tiếp từ bản clone sạch** | — | **TV4 + TV3** | 3 – 4 phút | Khóa `TEST_DONE.json`, `pytest`, chạy `visualize_filters.py` trực tiếp trên `PublicTest #1623`, kiểm tra CSV/Notebook |

*(Lưu ý điều phối chuyển slide: TV1 trình bày Slide 3 xong chuyển TV2 trình bày 4–6; TV3 trình bày Slide 7 rồi trả TV1 trình bày 8–9; TV2 chốt Slide 10 rồi TV3 trình bày 11–13; TV4 chốt 14–15 — hoặc gộp trình tự theo mạch slide từ 1→15 và đổi người ngắn gọn theo từng chủ đề trên.)*

---

## 2. Kịch bản Lời dẫn Chi tiết Từng Slide (Slide 1 → Slide 15)

### Slide 1 — Trang bìa *(TV4, ~30s)*
- Chào giảng viên và lớp; giới thiệu nhóm 4 thành viên (TV1 Đoàn Trung Kiên, TV2 Lê Phan Nguyên Hải – nhóm trưởng, TV3 Võ Hoàng Bảo, TV4 Huỳnh Viễn Thông).
- Nêu ngay 3 con số đinh trên trang bìa: **73.64% PrivateTest** (Macro-F1 **73.22%**, vượt mốc 71% của cuộc thi Kaggle tới `+2.64%`), **86 run chính thức** (`51.8` giờ GPU) và **128/128 kiểm thử tự động `pytest`**, hoàn toàn thuần PyTorch và không dùng dữ liệu ngoài.

### Slide 2 — Bài toán, Mục tiêu & Quy trình *(TV4, ~60s)*
- Tóm tắt 3 thách thức FER2013: ảnh xám `48×48`, nhãn nhiễu (con người đạt `65 ± 5%`), lệch lớp `16.55×`.
- Nhấn mạnh 2 kỷ luật nghiên cứu: (1) mọi thực nghiệm E1–E6 đổi đúng 1 yếu tố so với Baseline 3 seed (`σ = 0.75%`); (2) tập `PrivateTest` bị khóa cứng chỉ chạy 1 lần duy nhất qua `TEST_DONE.json`, toàn bộ biểu đồ và bảng sinh trực tiếp từ code.

### Slide 3 — Dữ liệu FER2013 & EDA *(TV1, ~60s)*
- Trình bày 3 split gốc theo cột `Usage`: `28,709` Training (duy nhất dùng tính mean/std chuẩn hóa), `3,589` PublicTest (validation), `3,589` PrivateTest (khóa); kiểm tra toàn vẹn `PASS` với `SHA-256 = 3b8d9617…f871b`.
- Chỉ vào biểu đồ phân bố: `Happy` chiếm `25.13%` (`7,215` ảnh), `Disgust` chỉ `1.52%` (`436` ảnh), và ở `PublicTest` `Disgust` chỉ có `56/3,589` ảnh (`1.56%`). Vì vậy không thể chỉ nhìn Accuracy mà phải báo cáo thêm Macro-F1 và recall từng lớp.

### Slide 4 — Kiến trúc Baseline & Mốc 3 seed *(TV2, ~60s)*
- Giới thiệu CNN Baseline 6 lớp Conv (3 khối `64 → 128 → 256`), có BatchNorm, ReLU, MaxPool và Dropout, tổng `5.87M` tham số (trong đó `FC1` chiếm `80.4%`).
- 3 seed chạy độc lập trên 3 máy (RTX 4060 / 3050 / 3060) đạt **`68.15% ± 0.75%`** Val Acc và **`64.08% ± 1.95%`** Macro-F1. Chốt độ lệch chuẩn `σ = 0.75%` làm ngưỡng để phân biệt cải thiện thật với nhiễu ngẫu nhiên.

### Slide 5 — Thực nghiệm E1: Kiến trúc CNN *(TV2, ~75s)*
- Đọc nhanh bảng E1 (30 run): `large_bn` (16 Conv + BN, `20.41M` tham số) dẫn đầu với **`70.60% ± 0.23%`** (`+2.44%` vượt baseline, F1 `69.22%`).
- Nhấn mạnh phát hiện giải mã ca sập `24.94%`: `large` và `medium_01` bỏ BN đều đoán 100% lớp `Happy` do khởi tạo mặc định PyTorch có phương sai nhỏ hơn Glorot của Keras ~3× khiến tín hiệu tắt; thử đổi Glorot + Adam `1e-4` kéo `medium_01` lên `67.21%`, còn thêm BN đưa `large` lên `70.60%` hoàn toàn ổn định.
- Điểm thêm 2 bất ngờ: `baseline_gap` cắt 78% tham số (còn `1.28M`) vẫn giữ `68.39%`; `baseline_no_bn` (6 Conv nông) đạt `69.41%` cho thấy BN là điều kiện cần với mạng sâu chứ không phải luôn tối ưu ở mạng nông.

### Slide 6 — Thực nghiệm E2: Hàm kích hoạt & Vanishing Gradient *(TV2, ~60s)*
- Họ ReLU dẫn đầu (`leaky_relu 69.35%`, `gelu 69.18%`, 1 seed).
- Chỉ trực tiếp vào đồ thị trục log `figures/e2_grad_norms.png`: với `sigmoid_no_bn` ở epoch 1, chuẩn gradient `conv1_1` chỉ còn `3.21e-7` so với lớp đầu ra `classifier.4` (`0.807`), suy giảm **~2.52 triệu lần**, khiến mô hình kẹt tại `24.94%`. Thêm BN phục hồi gradient nhưng Sigmoid vẫn học rất chậm và chỉ dừng ở `47.65% ± 2.20%`.

### Slide 7 — Thực nghiệm E3: Tối ưu, Lịch LR & Batch Size *(TV3, ~55s)*
- Khảo sát 15 biến thể trên cùng Baseline: chỉ duy nhất **`bs_64`** đạt **`69.24%`** (`+1.09%`, vượt ngưỡng `σ = 0.75%`).
- `rmsprop_1e-3` cho Val Loss thấp nhất E3 (`0.8809`). Ngược lại, `sgd_nesterov_3e-2` phân kỳ về `24.94%`, còn `AdamW` kéo tụt Macro-F1 còn `56.30–58.78%`.

### Slide 8 — Thực nghiệm E4: Augmentation & Điều chuẩn *(TV1, ~60s)*
- Chỉ vào cột ảnh minh họa `eda_augmentation.png` và biểu đồ khoảng cách `Train − Val` tại best epoch: không augmentation gap lên tới **`28.09`** điểm %, chỉ flip còn `22.04`, còn `flip + affine` kéo mạnh xuống **`4.72`** điểm % (`68.63%`).
- Nhấn mạnh bài học: `aug_random_erasing` giảm gap thấp nhất (`0.49` điểm %) nhưng Macro-F1 chỉ `64.06%` — giảm quá khớp không tự động đồng nghĩa dự đoán tốt hơn.

### Slide 9 — Thực nghiệm E5: Mất cân bằng lớp *(TV1, ~60s)*
- Đối chiếu bảng 3 phương pháp (đã tính đầy đủ recall từ `best.pt` cho cả 3 run):
  - `CE thường`: Acc `68.15%`, F1 `64.16%`, Disgust `37.50%`, Fear `42.14%`.
  - `Class weight`: đẩy Disgust lên cao nhất `73.21%` nhưng giảm Acc `−2.03%` (`66.12%`) và tụt Fear xuống `35.48%`.
  - **`WeightedRandomSampler`**: giữ Acc (`68.24%`, `+0.08%`), tăng Macro-F1 **`+3.01`** điểm % (`67.17%`), kéo đồng thời Disgust lên `67.86%` (`38/56`) và Fear lên `44.96%` (`223/496`).

### Slide 10 — Thực nghiệm E6: Transfer Learning & Fail Case VGG16 48px *(TV2, ~60s)*
- Giải thích cấu trúc thất bại hiệu năng của `VGG16` ở `48×48`: 5 tầng MaxPool thu gọn bản đồ đặc trưng về `1×1×512`, nhưng `AdaptiveAvgPool2d(7,7)` sao chép ra 49 bản giống hệt nhau → riêng `FC1` chiếm `102.8M / 134.3M` tham số (nặng `22.9×` baseline, tốn `60–71s/epoch`) mà vẫn thua `large_bn`.
- Với `ResNet18`: trọng số ImageNet giúp nhảy vọt `56.12%` ngay epoch 1 (vs `28.62%` từ đầu), nhưng đóng băng backbone tụt còn `45.70%` do khoảng cách miền ảnh màu tự nhiên vs ảnh mặt xám 48 px.

### Slide 11 — Thực nghiệm E7: Sàng lọc, TTA & Ensemble trên PublicTest *(TV3, ~60s)*
- Bước 1: thử ghép `bs_64`, `leaky_relu`, `WRS` vào `large_bn` (`E7-001..003`) và `baseline_best` (`E7-004`): cả 4 đều dưới mốc `E1-014_s0` (`70.86%`) → chứng minh không thể cộng dồn mù siêu tham số giữa các kiến trúc; giữ `large_bn` 3 seed theo luật tiền đăng ký.
- Bước 2: Ensemble 3 seed tăng `70.60% → 72.36%` (NLL `1.2175 → 0.9780`), thêm **TTA lật ngang** lên **`72.50%`** (`2,602/3,589`, NLL **`0.9402`**). Ngược lại `ten_crop` giảm còn `69.91%` do cắt biên ảnh 48 px.

### Slide 12 — Kết quả PrivateTest duy nhất 1 lần *(TV3, ~60s)*
- Công bố kết quả khóa 1 lần `results/E7/TEST_DONE.json`: **`73.64%`** (`2,643/3,589`, Macro-F1 **`73.22%`**) — vượt mốc `69%` là `+4.64%`, vượt mốc `71%` là **`+2.64%`**, vượt báo cáo mẫu Keras (`64–67%`) tới `6.6–9.6%`.
- Nêu quy trình chứng kiến 3 bên (TV3 chạy 21.8s, TV2 chứng kiến, TV4 kiểm chạy lại `--eval-test` bị chặn ngay với `exit = 1`).
- Điểm bất ngờ: lớp hiếm nhất **`Disgust` đạt F1 `80.39%`** (đứng thứ 3/7 chỉ sau `Happy 90.78%` và `Surprise 84.12%`).

### Slide 13 — Phân tích lỗi trên PrivateTest *(TV3, ~55s)*
- Trong `946` lỗi (`26.36%`), cụm 4 lớp `{Angry, Fear, Sad, Neutral}` chiếm tới **`665/946` (`70.3%`)**.
- Chỉ ra 3 cặp nhầm đầu bảng: `Sad ↔ Neutral` (`173` mẫu), `Angry ↔ Sad` (`123` mẫu), `Fear → Sad` (`101` mẫu, `19.13%` tập Fear). Kết hợp lưới ảnh sai tự tin cao: phần lớn là mặt bị che, góc nghiêng hoặc biểu cảm mơ hồ khớp mức trần con người `65 ± 5%`.

### Slide 14 — Trực quan hóa Bộ lọc & Grad-CAM *(TV4, ~75s)*
- Thực hiện 4 kỹ thuật trên `E1-014_s0` (`large_bn`), hoàn toàn trên `PublicTest`:
  1. `conv1_1` (`64×1×3×3`): 64/64 bộ lọc có cả trọng số âm–dương (`|w| = 0.1337`), dò cạnh/tương phản rõ nét; trong khi `conv4_4` (`512×512×3×3`) lấy TB theo 512 kênh bị triệt tiêu `363.7×` → minh họa định lượng lý do cần dùng Feature map & Grad-CAM.
  2. Feature map `6×6` trên `#1623` (Happy): `16/16` kênh mạnh nhất có đỉnh trong vùng `4×4` giữa mặt (`13/16` vùng mũi–miệng).
  3. Activation Maximization (`100` bước): cho vân kết cấu cục bộ nhưng không ổn định giữa các lần chạy cuDNN/CPU → nhóm trung thực chỉ dùng minh họa định tính.
  4. **Grad-CAM 15 ca**: `6/7` ca đúng và `7/8` ca sai có đỉnh trong vùng mặt `y ∈ [16, 36]` — mô hình **không nhìn nền**; các ca sai tự tin `≥99.57%` như `#1774` (nhãn Neutral nhưng cười) hay `#441` (nhãn Sad nhưng nhíu mày) phản ánh chính nhiễu nhãn FER2013.

### Slide 15 — Kết luận, Hướng phát triển & Đóng góp nhóm *(TV4, ~45s)*
- Chốt 3 bài học cốt lõi (BatchNorm cho mạng sâu; không cộng dồn mù siêu tham số 1 seed; Ensemble + flip tăng độ chính xác chắc chắn nhất).
- Nêu 3 hướng phát triển (điều chuẩn `large_bn`, căn chỉnh mặt & xử lý nhiễu nhãn FER+, distillation sang `baseline_gap 1.28M`).
- Trình bày bảng đóng góp: 4 thành viên hoàn thành **100% nhiệm vụ R (`28/28`)**, mỗi người gánh **7 lượt kiểm thử độc lập `T`**, đóng góp đồng đều **25.0%**. Chuyển sang phần Demo trực tiếp & Q&A.

---

## 3. Bộ 12 Câu hỏi Q&A Dự kiến (Kèm Số liệu & Nguồn Truy vết)

1. **Hỏi: Vì sao nhóm tự viết CNN từ đầu mà vượt được cả ResNet18 và VGG16 pretrained ImageNet?**  
   - **Đáp**: Ảnh FER2013 là ảnh xám `48×48` cận mặt, khác xa ảnh màu `224×224` của ImageNet. Ở E6 (`report/tables/E6_summary.csv`), đóng băng backbone ResNet18 chỉ đạt `45.70%` chứng tỏ đặc trưng ImageNet bị lệch miền mạnh; khi fine-tune thì `resnet18_96_ft` chỉ đạt `68.74%` và `vgg16_96_ft` đạt `69.77%` (nặng `134.3M` tham số), trong khi `large_bn` (`20.41M`, thiết kế 4 tầng MaxPool xuống `3×3` vừa khít `48×48`) đạt `70.60% ± 0.23%` đơn lẻ và `72.50%` khi ensemble + flip.

2. **Hỏi: Tại sao `large` (16 Conv) không có BatchNorm lại bị sập đúng con số `24.94%`, còn `baseline_no_bn` (6 Conv) lại đạt `69.41%`?**  
   - **Đáp**: `24.94%` (`895/3,589`) đúng bằng tỷ lệ lớp đa số `Happy` trên `PublicTest` — mô hình ngừng học đặc trưng và dự đoán mọi ảnh là `Happy`. Nguyên nhân đã được tách chứng minh ở Chương 3 (`E1_summary.csv`): khởi tạo mặc định Kaiming uniform trong PyTorch có phương sai nhỏ hơn Glorot của Keras ~3×; qua 16 lớp Conv không BN, phương sai tín hiệu và gradient tắt dần trước khi về lớp đầu (khi đổi Glorot + Adam `1e-4`, `medium_01` hồi phục từ `24.94%` lên `67.21%`). Ở mạng nông 6 Conv (`baseline_no_bn`), đường truyền ngắn nên không tắt tín hiệu, ngược lại bỏ nhiễu thống kê batch của BN giúp nhỉnh lên `69.41%`.

3. **Hỏi: Nhóm chứng minh hiện tượng vanishing gradient ở E2 bằng cách nào, hay chỉ nói lý thuyết?**  
   - **Đáp**: Nhóm đo trực tiếp chuẩn L2 của gradient trên từng lớp trong suốt quá trình train (`src/visual/grad_norms.py`, lưu `figures/e2_grad_norms.csv` & `figures/e2_grad_norms.png`). Ở epoch 1 của `sigmoid_no_bn`, chuẩn gradient tại `features.conv1_1` là `3.21e-7` trong khi ở lớp ra `classifier.4` là `0.807`, tức suy giảm **~2.52 triệu lần**, khớp hoàn toàn kết quả `24.94%`.

4. **Hỏi: `bs_64`, `leaky_relu` và `WeightedRandomSampler` đều tốt hơn Baseline ở E2–E5, sao mô hình cuối E7 không dùng chúng?**  
   - **Đáp**: Nhóm đã kiểm nghiệm thực tế ở Bước 1 của E7 (`report/tables/E7_summary.csv`): khi ghép lần lượt vào `large_bn`, cả `E7-001` (`+bs64`: `69.71%`), `E7-002` (`+bs64+leaky`: `69.10%`) và `E7-003` (`+bs64+leaky+WRS`: `69.35%`) đều thấp hơn mốc `E1-014_s0` (`70.86%`). Hơn nữa các biến thể E2–E5 chỉ chạy 1 seed trên kiến trúc 6 Conv, không bảo đảm chuyển giao sang 16 Conv. Nhóm tuân thủ đúng luật tiền đăng ký trong `configs/E7_final.json`.

5. **Hỏi: Tại sao TTA `ten_crop` lại làm giảm accuracy từ `72.36%` xuống `69.91%`, trong khi `flip` tăng lên `72.50%`?**  
   - **Đáp**: Ảnh FER2013 chỉ `48×48` và khuôn mặt đã nằm sát khung hình. Cắt 4 góc ở `ten_crop` (ví dụ `42×42` phóng lại) dễ cắt mất phần lông mày, trán hoặc cằm — đúng các vùng quyết định mà Grad-CAM ở Chương 6 đã chỉ ra. Ngược lại, cảm xúc khuôn mặt đối xứng trái–phải nên lật ngang (`flip`) bảo toàn trọn vẹn thông tin, vừa tăng accuracy lên `72.50%` vừa giảm NLL từ `0.9780` xuống `0.9402`.

6. **Hỏi: Làm sao giảng viên tin được nhóm chỉ chạy `PrivateTest` đúng 1 lần chứ không chạy thử nhiều lần rồi chọn kết quả đẹp nhất?**  
   - **Đáp**: Có 4 tầng bằng chứng độc lập: (1) `make_loaders` chỉ trả `Training` và `PublicTest`, tách hẳn `make_test_loader` chỉ gọi qua `--eval-test`; (2) `eval_test` kiểm tra ngay `results/E7/TEST_DONE.json` — nếu tồn tại lập tức từ chối chạy (`exit = 1`); (3) lịch sử Git cho thấy `TEST_DONE.json` chỉ xuất hiện duy nhất ở commit `bb6d96a` (do TV3 chạy, TV2 chứng kiến ghi nhật ký, TV4 chạy thử lần 2 xác nhận bị chặn ở commit `57eacf2`); (4) TV1 kiểm thử độc lập `KT-024` trên bản clone sạch xác nhận `test_predictions.npz` khớp `73.64%`.

7. **Hỏi: Tại sao `PrivateTest` (`73.64%`) lại cao hơn `PublicTest` (`72.50%`) tới `1.14%`? Có rò rỉ dữ liệu không?**  
   - **Đáp**: Không rò rỉ — bộ dữ liệu kiểm tra `SHA-256` chuẩn Kaggle và `PrivateTest` tách biệt hoàn toàn. Thực tế ngay cả mốc cuộc thi gốc cũng có `PrivateTest` (`71%`) cao hơn `PublicTest` (`69%`) tới 2 điểm % do phân bố độ khó mẫu ngẫu nhiên giữa hai tập `3,589` ảnh. Chênh lệch `+1.14%` của nhóm còn xác nhận việc chọn mô hình trên `PublicTest` không gây overfit lạc quan vào tập validation.

8. **Hỏi: `Disgust` ít ảnh nhất (`1.52%`) nhưng trên `PrivateTest` lại đạt F1 `80.39%` (hạng 3/7), còn `Fear` nhiều ảnh gấp 10 lần lại chỉ có Recall `51.52%` — nhóm giải thích thế nào?**  
   - **Đáp**: Độ khó phân loại không chỉ phụ thuộc số lượng mẫu mà còn phụ thuộc độ phân tách đặc trưng thị giác. `Disgust` có tín hiệu cơ mặt rất đặc trưng (nhăn sống mũi, nhếch môi trên — thấy rõ ở Grad-CAM `#560`), giúp mô hình đạt Precision `87.23%` và Recall `74.55%`. Trái lại `Fear` giao thoa mạnh với 3 lớp khác (`101` ảnh nhầm sang `Sad`, `56` sang `Angry`, `42` sang `Surprise`), nên dù có `528` mẫu vẫn là lớp khó nhất.

9. **Hỏi: Vì sao trong Grad-CAM nhóm không vẽ trực tiếp trọng số `conv4_4` mà phải dùng tới 4 phương pháp, và vì sao Activation Maximization bị đánh dấu không tái lập?**  
   - **Đáp**: Lớp `conv4_4` có kích thước `(512, 512, 3, 3)` tức 4,608 trọng số/bộ lọc; khi lấy trung bình theo 512 kênh vào để vẽ lưới `3×3`, các giá trị âm–dương triệt tiêu nhau làm độ lớn trung bình giảm `363.7×` (`0.000368` so với `0.1337` ở `conv1_1`), gần như nhiễu. Do đó cần Feature map và Grad-CAM để giải thích lớp cuối. Với Activation Maximization, khi TV2 kiểm thử độc lập (`KT-022`), ảnh tối ưu thay đổi theo nhiễu khởi tạo và tính bất tất định cuDNN GPU nên nhóm ghi nhận trung thực giới hạn này thay vì suy diễn quá mức.

10. **Hỏi: Trong EDA có phát hiện 11 ảnh toàn đen ở Training và 1 ảnh ở PublicTest — vì sao nhóm không xóa đi?**  
    - **Đáp**: Tỷ lệ ảnh toàn đen rất nhỏ (`12/35,887 ≈ 0.03%`). Nhóm giữ nguyên tập dữ liệu gốc để con số `28,709 / 3,589 / 3,589` so sánh công bằng tuyệt đối với mốc cuộc thi Kaggle (`69% / 71%`) và các báo cáo chuẩn trên FER2013.

11. **Hỏi: Nhóm kiểm soát tái lập mã nguồn như thế nào khi 4 thành viên dùng 3 loại GPU khác nhau (`RTX 3050`, `3060`, `4060`)?**  
    - **Đáp**: Mọi lệnh đều cố định seed (`torch`, `numpy`, `random`), kiểm `SHA-256` dữ liệu, và mỗi nhiệm vụ có Tester (`T`) độc lập kiểm tra trên bản clone sạch (tổng 28 lượt kiểm thử chia đều 7 lượt/người). Ở nhiệm vụ QA `T27`, nhóm đã clone sạch repo, chạy đạt `128/128 pytest`, `10/10` lệnh trong `README.md`, và chạy lại từ đầu `notebooks/BTL02_reproducible.ipynb` (`5/5` cell `PASS`).

12. **Hỏi: Bảng 1.2 ghi đóng góp đều `25.0%` mỗi thành viên — cơ sở chia đều là gì?**  
    - **Đáp**: Theo ma trận phân công `tracking/tasks.csv`, cả 4 thành viên đều hoàn thành 100% nhiệm vụ phụ trách chính `R` được giao (`TV1 7/7`, `TV2 8/8`, `TV3 7/7`, `TV4 6/6`, tổng `28/28`), đồng thời mỗi người đều gánh đúng `7` lượt kiểm thử độc lập `T` và `13–15` lượt phối hợp `C` (các chỉ số thực làm về commit, run GPU, lượt kiểm thử đều được giữ nguyên trong Bảng 1.2 để minh bạch đối chiếu). Do khối lượng tổng thể cân bằng giữa code, huấn luyện GPU, báo cáo, QA và đóng gói, cả nhóm thống nhất tỷ lệ đóng góp `25.0%` cho mỗi thành viên.

---

## 4. Quy trình Demo Chạy Code Trực tiếp (3 – 4 phút, Bản Clone Sạch, KHÔNG Chạm `PrivateTest`)

> **Chuẩn bị trước giờ báo cáo (trên máy TV4 hoặc TV3)**:
> - Đảm bảo đã có `BTL02/data/fer2013.csv` và `BTL02/results/E1/E1-014_s0/best.pt` cục bộ (khớp SHA-256).
> - Mở sẵn terminal tại thư mục `BTL02/` và mở sẵn tab Google Sheets + GitHub repository trên trình duyệt.

### Bước 1 — Chứng minh khóa chống rò rỉ `PrivateTest` (30 giây)
Chạy lại lệnh `--eval-test` ngay trước lớp để chứng minh hệ thống **từ chối tuyệt đối lần chạy thứ hai**:
```powershell
python src/run_experiment.py --eval-test --member TV4 --config configs/E7_final.json --run-dirs results/E1/E1-014_s0 results/E1/E1-014_s1 results/E1/E1-014_s2 --tta flip
```
- **Kỳ vọng hiển thị**: Hệ thống in thông báo từ chối (`TEST_DONE.json` đã khóa `locked=true`, `test_acc=0.7364`, `exit code = 1`), không sửa 1 byte nào của 4 file bằng chứng.

### Bước 2 — Kiểm chứng chất lượng mã nguồn bằng `pytest` (30 giây)
Chạy nhanh nhóm test cốt lõi (kiểm tra khóa `PrivateTest`, metric đánh giá và trực quan hóa bộ lọc):
```powershell
python -m pytest -q tests/test_filters.py tests/test_evaluate.py tests/test_tv1_t20_hook.py
```
- **Kỳ vọng hiển thị**: Tất cả test đều `PASSED` (`0 failed`) chỉ trong vài giây. *(Ghi chú: toàn bộ 21 file test đạt `128/128 PASSED` trong ~2 phút như biên bản `report/QA_T27_report.md`.)*

### Bước 3 — Demo trực quan hóa bộ lọc & Grad-CAM trực tiếp trên `PublicTest` (60 giây)
Nạp `best.pt` thật của mô hình cuối (`E1-014_s0`, `large_bn`) và sinh lại Grad-CAM + trọng số lớp đầu ngay tại chỗ (tuyệt đối chỉ đọc `PublicTest`, không chạm `PrivateTest`):
```powershell
python src/visualize_filters.py --run-dir results/E1/E1-014_s0 --data-csv data/fer2013.csv --sample-idx 1623 --method gradcam --out-dir results/_demo_t26
```
- **Kỳ vọng hiển thị**: In tên lớp cuối `conv4_4` với shape trọng số `(512, 512, 3, 3)`, xuất `results/_demo_t26/filters_grad_cam.png` và `.csv`; mở nhanh ảnh để chỉ vùng kích hoạt tập trung vào miệng của ảnh Happy `#1623` (sau buổi demo xóa thư mục tạm `results/_demo_t26`).

### Bước 4 — Chứng minh truy vết tự động bảng kết quả & Notebook tái lập (45 giây)
```powershell
python src/t25_report_stats.py
python src/make_tables.py
```
- **Kỳ vọng hiển thị**: Tái sinh ngay `report/tables/T25_*.csv` và `report/tables/E1..E7_summary.csv` khớp 100% với số liệu trên slide và báo cáo Word; mở nhanh `notebooks/executed/` và tab Google Sheets (`Phan_cong`, `Thuc_nghiem`, `Kiem_thu`) để chứng minh toàn bộ lịch sử truy vết đồng bộ trực tiếp từ GitHub.
