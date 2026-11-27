# Chương 4. Phương pháp đánh giá, thực nghiệm tối ưu, mô hình cuối và phân tích lỗi

> Người viết: TV3 – Võ Hoàng Bảo (nhiệm vụ T23). Review chéo: TV1 (mục 4.1–4.2: phương pháp đánh giá & E3), TV4 (mục 4.3–4.5: E7, đánh giá cuối & phân tích lỗi). Kiểm thử: TV2.
>
> **Nguồn số liệu.** Mọi con số trong chương lấy trực tiếp từ file do code sinh ra, không gõ tay hay ước lượng:
> - bảng tổng hợp `report/tables/{baseline,E3,E7}_summary.csv` và `report/tables/E3_report.md`, sinh bằng `python src/make_tables.py` từ `tracking/experiments.csv`;
> - `figures/e3_comparison.csv` (dữ liệu của `figures/e3_comparison.png`), lệnh sinh hình ghi trong `tracking/figures_manifest.csv`;
> - `results/baseline/baseline_summary.json` (mốc đối chuẩn 3 seed và dung sai tái lập `T11`) và `tracking/test_log.csv` (`KT-003`, `KT-008`, `KT-016`);
> - các phần phụ thuộc vào G4 (`E7` – `T18`, `PrivateTest` – `T19`) hiện ghi rõ **`chưa thực hiện`** theo đúng nguyên tắc minh bạch số liệu.
>
> Các bảng trong chương giữ nguyên giá trị từ `make_tables.py` (mục `E3` tương ứng `Bảng 4.3` trong `report/tables/E3_report.md`). Mọi độ chính xác trong mục `4.2` và `4.3` đánh giá trên **PublicTest (validation)**; **PrivateTest** chỉ đánh giá một lần duy nhất ở mục `4.4` (`T19`).

---

## 4.1. Phương pháp đánh giá và các chỉ số

Để tránh sai lệch kết quả do dữ liệu mất cân bằng lớp mạnh và ngăn tuyệt đối rò rỉ dữ liệu kiểm thử, quy trình đánh giá được chuẩn hóa thống nhất trong `src/engine/evaluate.py` và `src/visual/plots.py` (`T08` – đã `Verified` qua `KT-003`):

1. **Nguyên tắc tách tập và thứ tự nhãn chuẩn Kaggle:**
   - Dữ liệu `fer2013.csv` (`35,887` ảnh xám `48×48`) được chia cố định theo cột `Usage`: `Training` (`28,709` ảnh) để huấn luyện, `PublicTest` (`3,589` ảnh) làm **tập validation** phục vụ early stopping, giảm learning rate và chọn cấu hình trong E1–E7, còn `PrivateTest` (`3,589` ảnh) bị khóa cho tới bước đánh giá cuối cùng `--eval-test` (`T19`).
   - Toàn bộ code dùng duy nhất hằng `EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]` (0..6 theo Kaggle), khắc phục lỗi đảo nhãn theo bảng chữ cái của `LabelEncoder` trong tài liệu mẫu.

2. **Bộ chỉ số định lượng (`evaluate.metrics`):**
   - **Accuracy tổng thể (Val / Test Accuracy):** Tỷ lệ dự đoán đúng trên toàn bộ 3,589 ảnh. Đây là tiêu chí chọn checkpoint `best.pt` và đối chiếu trực tiếp với hai mốc cuộc thi (**69% Public / 71% Private**).
   - **Macro-F1:** Trung bình cộng không trọng số của $F_1$ trên cả 7 lớp cảm xúc:
     $$\text{Macro-F1} = \frac{1}{7}\sum_{c=0}^{6} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
     Do lớp `Happy` (`7,215` ảnh train, `895` ảnh val) nhiều gấp **16.5×** lớp `Disgust` (`436` ảnh train, `56` ảnh val), Macro-F1 giúp phát hiện mô hình chỉ đoán tốt lớp đông mà bỏ rơi lớp thiểu số.
   - **Precision, Recall theo từng lớp và Ma trận nhầm lẫn chuẩn hóa theo hàng:** Đo trực tiếp độ nhạy trên các lớp khó (`Disgust`, `Fear`, `Sad`) và xác định các cặp cảm xúc dễ nhầm (`plot_confusion_matrix`, `plot_error_histogram`).
   - **Best validation loss & khoảng cách Train–Val:** `best_val_loss` phản ánh mức hiệu chỉnh xác suất (calibration), còn `Train − Val` (`train_acc_cuoi − val_acc`) chỉ dùng so sánh xu hướng quá khớp giữa các cấu hình (vì `train_acc_cuoi` đo khi bật augmentation và dropout).

3. **Ngưỡng phân biệt và dung sai tái lập (`T11`):**
   - Mốc đối chuẩn Baseline 3 seed (`baseline-002_s0/_s1/_s2`, `results/baseline/baseline_summary.json`): **`val_acc = 0.6815 ± 0.0075`**, **`val_macro_f1 = 0.6408 ± 0.0195`**, **`best_val_loss = 0.9109 ± 0.0094`**, `Train − Val = +0.0397`.
   - Với các cấu hình khảo sát 1 seed ở E1–E6, chênh lệch `|Δval_acc| < 0.0075` (`< 1 × std baseline`) được phân loại là **`< std baseline` (chưa đủ bằng chứng phân biệt rõ rệt)**; đồng thời `0.0075` cũng là dung sai chuẩn để Tester kiểm thử tái lập độc lập trên GPU khác.

4. **Test-Time Augmentation (TTA) và Ensemble cho mô hình cuối (`E7`):**
   - `evaluate.predict` hỗ trợ 3 chế độ suy luận trên cùng `model`: không TTA (`none`), TTA lật ngang (`flip`, trung bình xác suất softmax ảnh gốc và ảnh lật trái–phải), và TTA 10 vùng cắt (`ten_crop`, 4 góc + tâm kích thước `44×44` kèm lật ngang, nội suy bilinear về `48×48`, trung bình xác suất 10 lượt).
   - Ensemble đa seed (`src/eval_e7.py`) tính trung bình cộng phân phối xác suất softmax của các mô hình từ nhiều seed trước khi lấy `argmax`.

---

## 4.2. Thực nghiệm E3 – Thuật toán tối ưu, lịch learning rate và batch size

**Câu hỏi nghiên cứu.** Khi giữ nguyên kiến trúc Baseline (`5,867,847` tham số) và augmentation `flip_affine`, việc thay đổi thuật toán tối ưu × tốc độ học (learning rate), lịch điều chỉnh learning rate (scheduler) và kích thước lô (batch size) ảnh hưởng như thế nào tới khả năng hội tụ và độ chính xác trên PublicTest?

**Thiết kế thực nghiệm (`configs/E3_optim.json`, nhiệm vụ `T14` — đã `Verified` qua `KT-016`).** Khảo sát gồm **15 biến thể** (`seed = 0`, 100 epoch tối đa, early stopping patience = 15 theo `val_acc`), chia đều giữa TV1 (`sgd_*`, `adam_*`) và TV4 (`adamw_*`, `rmsprop_*`, `sched_*`, `bs_*`), do TV3 tổng hợp. Kết quả kiểm toán `KT-016` của TV2 xác nhận **16/16 run** (15 `Completed` + 1 run `E3-015_s0` bị ngắt phiên ở epoch 46 và tự động đánh `Aborted`, `DISCARD`, không đưa vào thống kê) đều **chỉ thay đổi đúng một yếu tố** so với `configs/baseline.json`.

**Bảng 4.1 (tương ứng Bảng 4.3 của `report/tables/E3_report.md`).** Kết quả toàn bộ 15 cấu hình E3 trên PublicTest (`report/tables/E3_summary.csv`, sắp theo thứ tự cấu hình `configs/E3_optim.json`; minh họa so sánh tại `figures/e3_comparison.png`).

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Train–Val* | GPU (s/ep) | So với baseline |
|---|---:|---:|---:|---:|---:|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | +0.0397 | — (nhiều GPU) | tham chiếu |
| `sgd_no_nesterov_1e-2` | 1 | 0.6843 | 0.6442 | 0.9134 | +0.0424 | RTX 3050 (37.4s) | +0.0028 (< std baseline) |
| `sgd_nesterov_3e-2` | 1 | 0.2494 | 0.0570 | 1.8059 | +0.0019 | RTX 3050 (23.0s) | -0.4322 (thấp hơn) |
| `sgd_nesterov_3e-3` | 1 | 0.6743 | 0.6376 | 0.9114 | +0.0186 | RTX 3050 (22.5s) | -0.0072 (< std baseline) |
| `adam_1e-3` | 1 | 0.6865 | 0.6592 | 0.8922 | +0.0472 | RTX 3050 (24.6s) | +0.0050 (< std baseline) |
| `adam_3e-4` | 1 | 0.6879 | 0.6587 | 0.9013 | +0.0607 | RTX 3050 (28.7s) | +0.0064 (< std baseline) |
| `adamw_1e-3` | 1 | 0.6723 | 0.5878 | 0.9626 | +0.0268 | RTX 3060 (35.9s) | -0.0092 (thấp hơn) |
| `adamw_3e-4` | 1 | 0.6704 | 0.5631 | 0.9176 | +0.0065 | RTX 3060 (49.7s) | -0.0111 (thấp hơn) |
| `rmsprop_1e-3` | 1 | 0.6857 | 0.6620 | **0.8809** | +0.0373 | RTX 3060 (25.1s) | +0.0042 (< std baseline) |
| `rmsprop_3e-4` | 1 | 0.6810 | 0.6332 | 0.9014 | +0.0292 | RTX 3060 (25.0s) | -0.0006 (< std baseline) |
| `sched_constant` | 1 | 0.6760 | 0.6376 | 0.9138 | +0.0334 | RTX 3060 (24.7s) | -0.0056 (< std baseline) |
| `sched_step` | 1 | 0.6648 | 0.5919 | 0.9208 | +0.0077 | RTX 3060 (25.4s) | -0.0167 (thấp hơn) |
| `sched_cosine` | 1 | 0.6771 | 0.6181 | 0.9126 | +0.0224 | RTX 3060 (26.7s) | -0.0045 (< std baseline) |
| `sched_onecycle` | 1 | 0.6863 | 0.6584 | 0.8997 | +0.0423 | RTX 3060 (25.1s) | +0.0047 (< std baseline) |
| **`bs_64`** | 1 | **0.6924** | **0.6642** | 0.9058 | +0.0774 | RTX 3060 (25.5s) | **+0.0109 (cao hơn)** |
| `bs_256` | 1 | 0.6726 | 0.6298 | 0.9169 | +0.0176 | RTX 3060 (24.6s) | -0.0089 (thấp hơn) |

*(Ghi chú: Giá trị gốc của `adamw_3e-4` `val_macro_f1` trong `E3_summary.csv` là `0.56305`, làm tròn 4 chữ số thành `0.5631`; bảng markdown gốc của `make_tables.py` hiển thị `0.5630` theo quy tắc làm tròn chẵn/gần nhất của Python.)*

**Hình 4.1.** So sánh Validation Accuracy của 15 cấu hình E3: `figures/e3_comparison.png` (dữ liệu `figures/e3_comparison.csv`).

### Phân tích theo 3 trục khảo sát

1. **Thuật toán tối ưu (Optimizer) × Learning rate:**
   - **Độ nhạy với learning rate của SGD Nesterov:** Tại `lr = 0.03` (`E3-012_s0`), bước cập nhật quá lớn ngay đầu huấn luyện khiến mạng mất ổn định và sụp hoàn toàn về dự đoán lớp đa số `Happy` (`val_acc = 0.2494`, macro-F1 `0.0570`, dừng sớm ở epoch 17). Giảm xuống `lr = 0.003` (`E3-013_s0`) giúp hội tụ an toàn nhưng bước học hơi nhỏ nên đạt `0.6743` (`-0.0072`). Điểm **`lr = 0.01`** của Baseline là tối ưu cho SGD trên mạng 6 Conv có BN (`0.6815 ± 0.0075`); bỏ Nesterov (`E3-011_s0`) đạt `0.6843` (`+0.0028 < std`, chưa phân biệt được ở 1 seed).
   - **Bộ tối ưu thích ứng (`Adam`, `RMSprop`) so với `AdamW`:** `adam_3e-4` (`0.6879`, F1 `0.6587`, `best_epoch = 100`), `adam_1e-3` (`0.6865`, F1 `0.6592`, loss `0.8922`) và `rmsprop_1e-3` (`0.6857`, F1 `0.6620`, **`best_val_loss = 0.8809` thấp nhất toàn E3**) đều đạt nhỉnh hơn trung bình Baseline `0.0042–0.0064`, nhưng vẫn nằm trong ngưỡng `< std baseline` (chỉ có giá trị định hướng ở 1 seed). Ngược lại, **`AdamW`** giảm rõ rệt cả accuracy lẫn macro-F1 (`0.6723` / F1 `0.5878` ở `1e-3`; `0.6704` / F1 `0.5631` ở `3e-4`, vượt quá `1 × std` theo chiều giảm). Nguyên nhân là trong AdamW, `weight_decay = 5e-4` bị tách khỏi bước thích ứng và nhân trực tiếp theo tỉ lệ cố định trên trọng số (`lr · wd`), làm co trọng số quá mạnh đối với các lớp đặc trưng ít kích hoạt hoặc lớp phân loại thiểu số, gây mất khả năng phân biệt lớp hiếm.

2. **Lịch điều chỉnh learning rate (Scheduler):**
   - Với `SGD lr=0.01`, **`OneCycleLR`** (`0.6863`, F1 `0.6584`, loss `0.8997`, `best_epoch = 100`) nhỉnh nhẹ hơn `ReduceLROnPlateau` của Baseline (`0.6815`, trong ngưỡng `< std`).
   - `CosineAnnealingLR` (`0.6771`) và `Constant` (`0.6760`) đều thấp nhẹ nhưng chưa vượt `std`, trong khi **`StepLR`** (`step_size=20, gamma=0.5`) tụt mạnh nhất nhóm xuống `0.6648` (`-0.0167`, macro-F1 `0.5919`, `Train−Val = +0.0077`): việc hạ bước học cứng nhắc xuống 1/2 sau mỗi 20 epoch (còn `0.000625` ở epoch 80) làm mô hình mất động lượng học sớm khi `val_acc` vẫn còn dư địa cải thiện.

3. **Kích thước batch (Batch size — nhân tố tạo khác biệt rõ rệt nhất trong E3):**
   - Kết quả cho thấy quy luật đơn điệu rõ ràng: **`bs_64` (`0.6924`, F1 `0.6642`) > `bs_128` (`0.6815`, F1 `0.6408`) > `bs_256` (`0.6726`, F1 `0.6298`)**.
   - **`bs_64` (`E3-009_s0`) là biến thể duy nhất trong toàn bộ 15 cấu hình E3 vượt ngưỡng `+1 × std baseline`** (`+0.0109 > 0.0075`), đồng thời tăng `macro-F1` thêm `+2.34` điểm %. Kết quả kiểm toán độc lập `KT-016` do TV2 chạy lại `bs_64` trên RTX 4060 đạt `0.6902` (độ lệch tái lập chỉ `|Δ| = 0.0022 ≤ 0.0075`, cao hơn mốc baseline `0.6815` là `+0.0086`).
   - Về cơ chế tối ưu hóa: batch size nhỏ hơn (`64` thay vì `128`) tạo số bước cập nhật mỗi epoch gấp đôi và tăng nhiễu ước lượng gradient mini-batch một cách có lợi, giúp SGD thoát các cực tiểu phẳng kém tổng quát hóa trên bộ dữ liệu FER-2013 có nhiễu nhãn cao; trái lại, batch lớn (`256`) làm gradient quá mượt và giảm khả năng tổng quát hóa (`-0.0089`).

**Quyết định chuyển sang E7 (`T18`).** Trong phạm vi khảo sát E3 trên kiến trúc Baseline, phương án mang lại cải thiện rõ rệt và đã tái lập tốt qua kiểm thử là giữ **`SGD Nesterov lr = 0.01`** kết hợp **`batch_size = 64`**. Tuy nhiên, như lưu ý ở `KT-016` và mục `3.9`, `bs_64` (`+0.0086` ở bản chạy lại) mới chạy 1 seed mỗi máy và các thí nghiệm E3 thực hiện trên mạng 6 Conv nên khi ghép vào `large_bn` ở E7 (`T18`) cần đánh giá đầy đủ 3 seed trước khi chốt cấu hình cuối.

---

## 4.3. Thực nghiệm E7 – Mô hình cuối, TTA và Ensemble

*(chưa thực hiện — chờ đóng cổng G3 và hoàn tất nhiệm vụ `T18`)*

- Bảng kết quả dự kiến: `report/tables/E7_report.md`, `report/tables/E7_tta_ensemble.md` (sinh từ `src/make_tables.py` và `src/eval_e7.py`).
- Biểu đồ dự kiến: `figures/e7_val_tta_ensemble.png` và các biểu đồ chuẩn trên PublicTest.

---

## 4.4. Đánh giá cuối cùng trên PrivateTest và so sánh mốc 69% / 71%

*(chưa thực hiện — chỉ đánh giá một lần duy nhất qua `python src/run_experiment.py --eval-test` và ghi khóa `results/E7/TEST_DONE.json` sau khi hoàn tất `T18`)*

---

## 4.5. Phân tích lỗi (ma trận nhầm lẫn, cặp lớp hay nhầm, ca sai điển hình)

*(chưa thực hiện — chờ dữ liệu dự đoán thật từ `T18` và `T19`; danh sách ca đúng/sai điển hình sẽ được xuất qua `results/E7/e7_gradcam_cases.csv` để liên kết với phân tích Grad-CAM ở `T20`)*
