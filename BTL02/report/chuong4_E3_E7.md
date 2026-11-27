# Chương 4. Phương pháp đánh giá, thực nghiệm tối ưu, mô hình cuối và phân tích lỗi

> Người viết: TV3 – Võ Hoàng Bảo (nhiệm vụ T23). Review chéo: TV1 (mục 4.1–4.2: phương pháp đánh giá & E3), TV4 (mục 4.3–4.5: E7, đánh giá cuối & phân tích lỗi). Kiểm thử: TV2.
>
> **Xác nhận review chéo mục 4.3–4.5 (TV4 – Huỳnh Viễn Thông, 2026-11-27):** Đã đối chiếu độc lập toàn bộ số liệu ở mục `4.3` (E7), `4.4` (PrivateTest) và `4.5` (phân tích lỗi) về đúng nguồn sinh từ code (`report/tables/E7_summary.csv`, `report/tables/E7_tta_ensemble.csv`, `results/E7/TEST_DONE.json`, `results/E7/test_per_class.csv`, `results/E7/test_predictions.npz`, `figures/e7_test_{confusion,error_hist,wrong_grid}.csv`): (1) **Bảng 4.2 & 4.3 (E7 PublicTest)**: 100% khớp (`E7-001..004`: `0.6971/0.6910/0.6935/0.7013`, `large_bn` 3 seed `0.7060 ± 0.0023`, Ensemble `flip` đạt đỉnh `0.7250`, F1 `0.7106`, NLL `0.9402`); (2) **Bảng 4.4 & 4.5 (PrivateTest `T19`)**: 100% khớp `TEST_DONE.json` & tái tính `test_predictions.npz` (`73.64%`, `2,643/3,589`, Macro-F1 `73.22%`, Macro-Prec `74.37%`, Macro-Rec `72.51%`, vượt mốc `71%` là `+2.64%`, `Disgust` F1 `80.39%`); (3) **Mục 4.5 (Phân tích lỗi)**: khớp 100% `946` lỗi, cụm `{Angry, Fear, Sad}` chiếm `640/946` (`67.7%`), cụm nội bộ 4 lớp `{Angry, Fear, Sad, Neutral}` chiếm `665/946` (`70.3%`), 9 cặp nhầm lẫn và toàn bộ 9 ca sai điển hình (`#6, #41, #12, #10, #44, #33, #8, #2, #14`). Trong quá trình rà soát, TV4 đã phát hiện và đính chính trực tiếp 4 chi tiết kỹ thuật nhỏ ở mục `4.3` cho khớp mã nguồn & config: (a) nguồn `WeightedRandomSampler` từ `E6` → `E5`; (b) bổ sung đủ siêu tham số `baseline_best` (`dropout_fc = 0.25`, `weight_decay = 0.001`, `label_smoothing = 0.1`) theo `configs/E7_final.json`; (c) số lớp Conv của `large_bn` là `16 Conv` (không phải `8 Conv`); (d) biên độ dịch chuyển của `flip_affine` là `±10%` theo `src/data/augment.py` (`translate=(0.1, 0.1)`). Sau đính chính, mục 4.3–4.5 khớp 100% nguồn, đạt yêu cầu kiểm thử `T23` và ghép báo cáo `T25`.
>
> **Nguồn số liệu.** Mọi con số trong chương lấy trực tiếp từ file do code sinh ra, không gõ tay hay ước lượng:
> - bảng tổng hợp `report/tables/{baseline,E3,E7}_summary.csv`, `report/tables/{E3,E7}_report.md` và `report/tables/E7_tta_ensemble.{csv,md}`, sinh bằng `python src/make_tables.py` và `python src/eval_e7.py`;
> - `figures/e3_comparison.csv`, `figures/e7_val_tta_ensemble.csv`, `figures/e7_test_{confusion,error_hist,correct_grid,wrong_grid}.csv`, lệnh sinh hình ghi đầy đủ trong `tracking/figures_manifest.csv`;
> - `results/baseline/baseline_summary.json` (`T11`), `results/E7/TEST_DONE.json`, `results/E7/test_per_class.csv`, `results/E7/e7_gradcam_cases.csv` và `tracking/test_log.csv` (`KT-003`, `KT-008`, `KT-016`, `KT-021`).
>
> Các bảng trong chương giữ nguyên giá trị từ `make_tables.py`, `eval_e7.py` và `run_experiment.py --eval-test`. Mọi độ chính xác trong mục `4.2` và `4.3` đánh giá trên **PublicTest (validation)**; **PrivateTest** chỉ đánh giá một lần duy nhất ở mục `4.4` (`T19`, có khóa cứng `results/E7/TEST_DONE.json`).

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

**Câu hỏi nghiên cứu.** (1) Việc ghép các yếu tố thắng từng bước từ E1–E6 (`large_bn` từ E1, `batch_size = 64` từ E3, `LeakyReLU` từ E2, `WeightedRandomSampler` từ E5) có tiếp tục cải thiện độ chính xác trên PublicTest so với `large_bn` gốc hay không? (2) Test-Time Augmentation (`flip`, `ten_crop`) và Ensemble 3 seed mang lại mức cải thiện cụ thể bao nhiêu trên PublicTest trước khi chốt mô hình đánh giá PrivateTest?

**Thiết kế thực nghiệm tiền đăng ký (`configs/E7_final.json`, nhiệm vụ `T18` — đã `Verified` qua `KT-021`).** Quy trình E7 chia làm 2 bước chặt chẽ nhằm tránh thử-sai trên PrivateTest:
- **Bước 1 (Sàng lọc tích lũy 1 seed trên cùng GPU RTX 4060):** Huấn luyện 4 cấu hình ghép trên `seed = 0` và đối chiếu trực tiếp với mốc cùng máy `E1-014_s0` (`large_bn`, `20,410,439` tham số: `val_acc = 0.7086`, `val_macro_f1 = 0.6953`, `best_val_loss = 0.9164`, `Train − Val = +0.1817`). Quy tắc chốt trước khi chạy: chỉ huấn luyện thêm `seed 1, 2` nếu biến thể ghép vượt `0.7086`; nếu không, chốt bộ **3 seed `large_bn` (`E1-014_s0/_s1/_s2`)** làm mô hình cuối.
- **Bước 2 (Đánh giá TTA & Ensemble trên 3 seed thắng qua `src/eval_e7.py`):** Đánh giá đồng bộ 6 tổ hợp `{none, flip, ten_crop} × {đơn lẻ 3 seed, ensemble 3 seed}` trên PublicTest (`3,589` ảnh), kèm kiểm tra chéo độc lập của TV1 (ensemble) và TV2 (TTA), và kiểm thử toàn diện `KT-021` của TV4.

### Kết quả Bước 1 – Sàng lọc cấu hình ghép trên PublicTest

**Bảng 4.2 (tương ứng Bảng 4.7 của `report/tables/E7_report.md`).** Kết quả 4 biến thể sàng lọc E7 trên PublicTest (`seed = 0`, RTX 4060) so với Baseline 3 seed và mốc cùng máy `large_bn` (`E1-014_s0`).

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| `large_bn` (`E1-014_s0`, mốc cùng GPU) | 1 | **0.7086** | **0.6953** | 0.9164 | 20,410,439 | +0.1817 | RTX 4060 (24.5s) | **+0.0270 (cao hơn)** |
| `large_bn_bs64` (`E7-001_s0`) | 1 | 0.6971 | 0.6778 | 0.9025 | 20,410,439 | +0.1868 | RTX 4060 (26.2s) | +0.0156 (cao hơn) |
| `large_bn_bs64_leaky` (`E7-002_s0`) | 1 | 0.6910 | 0.6725 | 0.9031 | 20,410,439 | +0.1589 | RTX 4060 (26.2s) | +0.0095 (cao hơn) |
| `large_bn_bs64_leaky_wrs` (`E7-003_s0`) | 1 | 0.6935 | 0.6874 | 0.9367 | 20,410,439 | +0.2053 | RTX 4060 (26.2s) | +0.0120 (cao hơn) |
| `baseline_best` (`E7-004_s0`) | 1 | 0.7013 | 0.6784 | **0.8693** | 5,867,847 | +0.0931 | RTX 4060 (14.9s) | +0.0198 (cao hơn) |

**Phát hiện quan trọng từ Bước 1 (tính phi tuyến khi chuyển siêu tham số giữa các kiến trúc):**
1. **Trên kiến trúc Baseline 6 Conv (`5.87M` tham số), tổ hợp `baseline_best`** (`LeakyReLU + dropout_fc = 0.25 + bs_64 + weight_decay = 0.001 + label_smoothing = 0.1 + flip_affine`, không WRS) đạt **`val_acc = 0.7013`** (`+1.98%` so với Baseline `0.6815`, vượt hơn `2.6 × std`), `val_macro_f1 = 0.6784` (`+3.76%`) và đặc biệt đạt **`best_val_loss = 0.8693` — mức loss thấp nhất toàn bộ dự án**, với tốc độ nhanh (`14.9 s/epoch`) và khoảng cách `Train − Val = +0.0931` kiểm soát tốt. Đây là bằng chứng rằng các yếu tố tìm ra trên Baseline cộng hưởng tốt trên chính kiến trúc Baseline.
2. **Ngược lại, trên mạng sâu `large_bn` (`20.41M` tham số), việc áp máy móc các siêu tham số thắng ở Baseline đều làm giảm độ chính xác so với `E1-014_s0` (`0.7086`):** giảm `batch_size` xuống `64` (`E7-001_s0`) làm tụt `-0.0114` xuống `0.6971` (vì trên mạng 16 Conv nhiều kênh kèm BatchNorm, batch nhỏ làm thống kê mean/variance của BN nhiễu hơn và tăng quá khớp `Train − Val = +0.1868`); đổi sang `LeakyReLU` (`E7-002_s0`) tụt tiếp xuống `0.6910` (`-0.0176`); thêm `WeightedRandomSampler` (`E7-003_s0`) giúp kéo `macro-F1` lên `0.6874` nhưng đẩy `val_loss` lên `0.9367` và `Train − Val = +0.2053` (`val_acc = 0.6935`).
3. **Chốt mô hình theo quy tắc tiền đăng ký:** Vì không có biến thể ghép nào vượt mốc `0.7086` của `E1-014_s0`, nhóm chốt bộ **3 seed `large_bn` (`E1-014_s0`, `E1-014_s1`, `E1-014_s2`)** bước sang Bước 2 đánh giá TTA và Ensemble.

### Kết quả Bước 2 – TTA và Ensemble 3 seed trên PublicTest

**Bảng 4.3 (tương ứng Bảng 4.7b của `report/tables/E7_tta_ensemble.md`).** So sánh 6 chế độ TTA × Ensemble của bộ 3 seed `large_bn` (`E1-014_s0/_s1/_s2`) trên PublicTest (`3,589` ảnh; minh họa tại `figures/e7_val_tta_ensemble.png`).

| Chế độ TTA | Loại đánh giá | Số seed | Val acc (mean ± std) | Val macro-F1 (mean ± std) | Val NLL |
|---|---|---:|---|---|---:|
| `none` | Đơn lẻ (trung bình) | 3 | 0.7060 ± 0.0023 | 0.6922 ± 0.0027 | 1.2175 |
| `none` | **Ensemble** | 3 | 0.7236 | **0.7134** | 0.9780 |
| `flip` | Đơn lẻ (trung bình) | 3 | 0.7150 ± 0.0024 | 0.7015 ± 0.0014 | 1.1157 |
| **`flip`** | **Ensemble (chọn cho T19)** | **3** | **0.7250** | **0.7106** | **0.9402** |
| `ten_crop` | Đơn lẻ (trung bình) | 3 | 0.6836 ± 0.0043 | 0.6686 ± 0.0061 | 1.1232 |
| `ten_crop` | **Ensemble** | 3 | 0.6991 | 0.6839 | 0.9716 |

*(Chi tiết từng seed trên PublicTest: ở chế độ `none`, `s0/s1/s2` lần lượt đạt `0.7086 / 0.7047 / 0.7047`; ở chế độ `flip`, `s0/s1/s2` lần lượt đạt `0.7166 / 0.7122 / 0.7161`, độ lệch chuẩn giữa các seed chỉ `0.0023–0.0024`, chứng tỏ độ ổn định rất cao.)*

**Hình 4.2.** So sánh Validation Accuracy giữa mô hình đơn lẻ (mean ± std) và Ensemble 3 seed theo 3 chế độ TTA trên PublicTest: `figures/e7_val_tta_ensemble.png` (dữ liệu `figures/e7_val_tta_ensemble.csv`). Các biểu đồ chẩn đoán trên PublicTest gồm `figures/e7_val_confusion.png`, `figures/e7_val_error_hist.png`, `figures/e7_val_correct_grid.png` và `figures/e7_val_wrong_grid.png`.

**Phân tích hiệu quả TTA và Ensemble:**
1. **Ensemble 3 seed tạo bước nhảy lớn nhất:** Riêng việc lấy trung bình xác suất softmax 3 seed (`none`) đã tăng `val_acc` thêm **`+1.76%`** (`0.7060 → 0.7236`), tăng `macro-F1` thêm **`+2.12%`** (`0.6922 → 0.7134`) và giảm mạnh độ quá tự tin trên các dự đoán sai (`Val NLL` giảm từ `1.2175 → 0.9780`).
2. **TTA `flip` cộng hưởng trực tiếp với Ensemble:** Trên mô hình đơn lẻ, lật ngang tăng `val_acc` thêm `+0.90%` (`0.7060 → 0.7150`) và giảm NLL từ `1.2175 → 1.1157`. Khi kết hợp **Ensemble 3 seed + TTA `flip`**, mô hình đạt đỉnh **`val_acc = 0.7250` (`72.50%`, vượt mốc 69% Public tới `+3.50%`)**, `val_macro_f1 = 0.7106` và **`Val NLL = 0.9402` thấp nhất toàn bảng** (chi phí suy luận chỉ tăng 2×).
3. **Vì sao TTA `ten_crop` làm giảm kết quả (`0.6836` đơn lẻ, `0.6991` ensemble):** Ảnh FER-2013 đã được cắt sát khuôn mặt ở độ phân giải rất nhỏ `48×48` và trong huấn luyện `flip_affine` chỉ dùng dịch chuyển nhẹ (`±10%`) + xoay `±10°` mà **không** dùng random crop phóng to. Khi `ten_crop` cắt 4 góc `44×44` rồi nội suy bilinear về `48×48`, các chi tiết biểu cảm then chốt ở rìa (đuôi mắt, lông mày, khóe miệng) bị mất lệch phân bố và làm mờ thêm, gây tụt `-2.24%` so với `none`.

**Quyết định chốt cho `T19`:** Chọn **Ensemble 3 seed `large_bn` (`E1-014_s0/_s1/_s2`) + TTA `flip`** làm cấu hình duy nhất đánh giá PrivateTest.

---

## 4.4. Đánh giá cuối cùng trên PrivateTest và so sánh mốc 69% / 71%

**Quy trình kiểm soát chống rò rỉ (`T19`, `results/E7/TEST_DONE.json`, `results/E7/test_run.log`).** Đánh giá trên `PrivateTest` (`3,589` ảnh) chỉ được thực thi **đúng 1 lần duy nhất** sau khi `T18` đạt `Verified` (`KT-021`), có thành viên TV2 chứng kiến trực tiếp lúc `2026-11-27 21:07:28–21:07:49 (+07:00)` (`21.8 s` trên máy `Hbao | RTX 3050 Laptop`, commit `08be17a0`), bằng đúng một lệnh:

```bash
python src/run_experiment.py --eval-test --member TV3 --config configs/E7_final.json --run-dirs results/E1/E1-014_s0 results/E1/E1-014_s1 results/E1/E1-014_s2 --tta flip
```

Ngay khi kết thúc, chương trình ghi khóa cứng `results/E7/TEST_DONE.json` (`"locked": true`) lưu dấu vân tay SHA-256 của bộ dữ liệu `fer2013.csv` và cả 3 file trọng số `best.pt`, đồng thời từ chối mọi lần chạy `--eval-test` tiếp theo.

### Kết quả chính thức trên PrivateTest và đối chiếu các mốc tham chiếu

**Bảng 4.4.** Tổng hợp kết quả đánh giá cuối cùng của mô hình E7 (Ensemble 3 seed `large_bn` + TTA `flip`) trên `PublicTest` và `PrivateTest` so với mốc Baseline, báo cáo mẫu và kỷ lục cuộc thi FER-2013 (nguồn: `results/E7/TEST_DONE.json`, `report/tables/E7_tta_ensemble.csv`).

| Hệ thống / Mốc đối chiếu | PublicTest Acc (%) | PrivateTest Acc (%) | PrivateTest Macro-F1 (%) | Chênh lệch so với mốc 71% Private |
|---|---:|---:|---:|---:|
| Báo cáo mẫu (`demo_report.pdf`) | 64.00 – 67.00 | — | — | — |
| Baseline nhóm (3 seed 6 Conv, `T11`) | 68.15 ± 0.75 | *(không đánh giá)* | *(không đánh giá)* | — |
| Mốc Kaggle FER-2013 (2013) | 69.00 | 71.00 | — | 0.00 |
| `E1-014_s0` đơn lẻ + TTA `flip` | 71.66 | 72.47 | 72.15 | +1.47 |
| `E1-014_s1` đơn lẻ + TTA `flip` | 71.22 | 71.97 | 71.74 | +0.97 |
| `E1-014_s2` đơn lẻ + TTA `flip` | 71.61 | 72.22 | 71.03 | +1.22 |
| **E7 Cuối (Ensemble 3 seed `large_bn` + TTA `flip`)** | **72.50** | **73.64** (`2,643 / 3,589`) | **73.22** | **+2.64** |

**Nhận xét đối chiếu:**
1. **Vượt chắc chắn cả hai mốc cuộc thi và báo cáo mẫu:** Mô hình cuối đạt **`72.50%`** trên `PublicTest` (**`+3.50%`** so với mốc `69%`) và **`73.64%`** (`2,643 / 3,589` ảnh đúng) trên `PrivateTest` (**`+2.64%`** so với mốc `71%`), đồng thời vượt khoảng **`+6.6%` đến `+9.6%`** so với báo cáo mẫu (`64%–67%`) mà vẫn hoàn toàn huấn luyện từ đầu (from scratch, không dùng pretrained weights hay dữ liệu ngoài).
2. **Tổng quát hóa hoàn toàn nhất quán — không có dấu hiệu overfitting lên PublicTest:** Kết quả PrivateTest (`73.64%`, Macro-F1 `73.22%`) cao hơn PublicTest (`72.50%`, Macro-F1 `71.06%`) là `+1.14%` Acc và `+2.16%` Macro-F1, bám sát chênh lệch tự nhiên của cuộc thi Kaggle (`69% → 71%`, tức `+2.0%`). Hơn nữa, ngay cả **từng mô hình đơn lẻ** (`71.97%–72.47%`, trung bình `72.22%`) đều đã vượt mốc `71%`, và Ensemble 3 seed tiếp tục cộng thêm **`+1.42%`** (`72.22% → 73.64%`).

### Hiệu năng chi tiết theo từng lớp cảm xúc trên PrivateTest

**Bảng 4.5.** Precision, Recall, F1-score và tỷ lệ sai theo từng lớp của mô hình cuối trên `PrivateTest` (`3,589` ảnh, nguồn: `results/E7/test_per_class.csv` và `figures/e7_test_error_hist.csv`).

| Lớp cảm xúc | Support | Số dự đoán đúng | Số dự đoán sai | Precision (%) | Recall (%) | Tỷ lệ sai (%) | F1-score (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Happy` | 879 | 802 | 77 | 90.32 | **91.24** | **8.76** | **90.78** |
| `Surprise` | 416 | 355 | 61 | 82.94 | 85.34 | 14.66 | 84.12 |
| `Disgust` | 55 | 41 | 14 | 87.23 | 74.55 | 25.45 | 80.39 |
| `Neutral` | 626 | 472 | 154 | 68.70 | 75.40 | 24.60 | 71.90 |
| `Angry` | 491 | 326 | 165 | 67.36 | 66.40 | 33.60 | 66.87 |
| `Sad` | 594 | 375 | 219 | 58.69 | 63.13 | 36.87 | 60.83 |
| `Fear` | 528 | 272 | 256 | 65.38 | **51.52** | **48.48** | **57.63** |
| **Tổng / Macro** | **3,589** | **2,643** | **946** | **74.37** | **72.51** | **26.36** | **73.22** |

Đáng chú ý, dù `Disgust` là lớp thiểu số nhất (chỉ `55` mẫu trên PrivateTest, chiếm `1.53%`), mô hình vẫn đạt **`Precision = 87.23%`, `Recall = 74.55%`, `F1 = 80.39%`** — cao thứ ba toàn bộ 7 lớp, giúp `Macro-F1` (`73.22%`) gần tiệm cận `Accuracy` tổng thể (`73.64%`).

---

## 4.5. Phân tích lỗi (ma trận nhầm lẫn, cặp lớp hay nhầm, ca sai điển hình)

Toàn bộ phân tích lỗi dưới đây dựa trực tiếp trên `946` mẫu dự đoán sai trên `PrivateTest` (nguồn: `figures/e7_test_confusion.csv`, `figures/e7_test_error_hist.csv`, `figures/e7_test_wrong_grid.csv`), đi kèm bộ 4 hình chuẩn:
- **Hình 4.3.** Ma trận nhầm lẫn chuẩn hóa theo hàng trên PrivateTest: `figures/e7_test_confusion.png` (`figures/e7_test_confusion.csv`).
- **Hình 4.4.** Tỷ lệ dự đoán sai theo từng lớp trên PrivateTest: `figures/e7_test_error_hist.png` (`figures/e7_test_error_hist.csv`).
- **Hình 4.5.** Lưới 16 mẫu dự đoán đúng (`figures/e7_test_correct_grid.png`) và 16 mẫu dự đoán sai (`figures/e7_test_wrong_grid.png`) trên PrivateTest.

### 1. Phân hóa rõ rệt giữa nhóm biểu cảm đặc trưng và cụm cảm xúc âm tính

Dựa trên Bảng 4.5 và Hình 4.4, 7 lớp cảm xúc tách thành 2 nhóm hiệu năng rất rõ:
- **Nhóm nhận diện tốt (tỷ lệ sai `< 26%`):** `Happy` (tỷ lệ sai `8.76%`, F1 `90.78%`), `Surprise` (`14.66%`, F1 `84.12%`), `Neutral` (`24.60%`, F1 `71.90%`) và `Disgust` (`25.45%`, F1 `80.39%`). Ba lớp `Happy`, `Surprise`, `Disgust` có đặc trưng hình thái khuôn mặt rất nổi bật ở ảnh `48×48` (khóe miệng nhếch rộng/lộ răng ở `Happy`, mắt mở tròn và miệng mở chữ O ở `Surprise`, mũi nhăn và môi trên kéo cao ở `Disgust`).
- **Nhóm khó nhất (tỷ lệ sai `33%–49%`):** `Fear` có tỷ lệ sai cao nhất (**`48.48%`**, `256/528` mẫu sai, Recall chỉ `51.52%`), kế tiếp là `Sad` (**`36.87%`**, `219/594` mẫu sai) và `Angry` (**`33.60%`**, `165/491` mẫu sai). Riêng 3 lớp này đã chiếm **`640 / 946` (`67.7%`)** tổng số lỗi toàn tập PrivateTest.

### 2. Các cặp lớp hay nhầm lẫn nhất (từ `figures/e7_test_confusion.csv`)

| Cặp nhầm lẫn (Nhãn thật → Dự đoán) | Số mẫu sai | Tỷ lệ theo hàng nhãn thật (%) | Cặp đối xứng 2 chiều (tổng mẫu sai) |
|---|---:|---:|---:|
| `Fear → Sad` | **101** | **19.13** | `Fear ↔ Sad`: `144` mẫu (`101 + 43`) |
| `Sad → Neutral` | **100** | **16.84** | **`Sad ↔ Neutral`: `173` mẫu (`100 + 73`)** |
| `Neutral → Sad` | 73 | 11.66 | *(nằm trong cặp `Sad ↔ Neutral`)* |
| `Angry → Sad` | 72 | 14.66 | `Angry ↔ Sad`: `123` mẫu (`72 + 51`) |
| `Fear → Angry` | 56 | 10.61 | `Fear ↔ Angry`: `94` mẫu (`56 + 38`) |
| `Sad → Angry` | 51 | 8.59 | *(nằm trong cặp `Angry ↔ Sad`)* |
| `Fear → Neutral` | 45 | 8.52 | `Fear ↔ Neutral`: `69` mẫu (`45 + 24`) |
| `Fear → Surprise` | 42 | 7.95 | `Fear ↔ Surprise`: `67` mẫu (`42 + 25`) |
| `Disgust → Angry` | 8 | 14.55 | `Disgust ↔ Angry`: `11` mẫu (`8 + 3`) |

**Nhận xét cấu trúc lỗi:**
- **Cụm 4 lớp `{Angry, Fear, Sad, Neutral}` chiếm tới `665 / 946` (`70.3%`) toàn bộ lỗi trên PrivateTest:**
  - **`Sad ↔ Neutral` (`173` mẫu sai, `18.3%` tổng lỗi):** Cặp nhầm đối xứng lớn nhất. Khi mắt khép nhẹ, miệng không cười rõ và ánh sáng ảnh `48×48` tương phản thấp, ranh giới giữa nét mặt buồn nhẹ và trung tính vốn mơ hồ ngay cả với người gắn nhãn (ước tính độ chính xác của con người trên FER-2013 chỉ vào khoảng `65 ± 5%`).
  - **`Fear` bị phân tán mạnh nhất sang 4 lớp lân cận:** `Fear → Sad` (`101`, `19.13%`), `Fear → Angry` (`56`, `10.61%`), `Fear → Neutral` (`45`, `8.52%`) và `Fear → Surprise` (`42`, `7.95%`). Về mặt giải phẫu biểu cảm (FACS), `Fear` chia sẻ chung mở rộng mắt/miệng với `Surprise`, chia sẻ nhíu mày căng thẳng với `Angry` và nét căng lo âu với `Sad`, nên khi mất chi tiết ở độ phân giải `48×48`, mô hình rất dễ kéo `Fear` sang lớp có prior hoặc tín hiệu hình thái rõ hơn.
  - **`Angry ↔ Sad` (`123` mẫu sai, `13.0%` tổng lỗi):** Đều có đặc trưng hạ lông mày và mím/trễ khóe miệng.
  - **`Disgust → Angry` (`8/14` lỗi của `Disgust`, chiếm `57.1%` lỗi lớp này):** Hai cảm xúc cùng có nếp nhăn vùng gian mày và căng cơ mũi–miệng, nhưng `Disgust` chỉ có `436` mẫu train so với `3,995` mẫu của `Angry` (ít hơn `9.2×`), nên các ca biên bị hút về `Angry`.

### 3. Ca sai điển hình có độ tự tin cao và liên kết với phân tích Grad-CAM (`T20`)

Trích xuất từ lưới sai trên PrivateTest (`figures/e7_test_wrong_grid.png` và `figures/e7_test_wrong_grid.csv`) cho thấy nhiều ca sai không phải do mô hình do dự (`confidence ≈ 0.40`), mà có độ tự tin rất cao (`0.70–0.97`):
- **Nhóm ranh giới nhãn mờ / nhiễu nhãn FER-2013:** Mẫu `#6` (`Sad → Neutral`, `conf = 97.18%`), mẫu `#41` (`Sad → Neutral`, `conf = 91.79%`) và mẫu `#12` (`Sad → Neutral`, `conf = 63.42%`) có tư thế gương mặt gần như trung tính không lộ nếp buồn rõ; mẫu `#10` (`Surprise → Happy`, `conf = 73.50%`) có miệng mở giống cười vui; mẫu `#44` (`Fear → Neutral`, `conf = 67.04%`).
- **Nhóm nhầm cấu trúc cơ mặt cục bộ:** Mẫu `#33` (`Angry → Fear`, `conf = 81.93%`), mẫu `#8` (`Angry → Fear`, `conf = 75.54%`), mẫu `#2` (`Neutral → Fear`, `conf = 70.48%`) và mẫu `#14` (`Angry → Sad`, `conf = 63.31%`) thường rơi vào ảnh chụp góc nghiêng, che khuất một phần khuôn mặt, độ tương phản kém hoặc vùng mắt/miệng bị mờ ở kích thước `48×48`.
- **Liên kết kiểm chứng định tính qua Grad-CAM (`T20`):** Để tránh sử dụng dữ liệu PrivateTest cho việc phân tích nội bộ lặp lại, nhiệm vụ `T18` đã xuất sẵn danh sách 30 ca điển hình trên `PublicTest` tại `results/E7/e7_gradcam_cases.csv` (gồm 14 ca dự đoán đúng trải đều cả 7 lớp và 16 ca dự đoán sai độ tự tin cao thuộc đúng các cặp lỗi trên, như `#2930` `Fear → Surprise`, `#441` `Sad → Angry`, `#1202` `Fear → Sad`, `#1892` `Disgust → Angry`) để Chương 5 (`T20` – Grad-CAM & trực quan hóa bản đồ đặc trưng của TV4) giải thích trực tiếp vùng khuôn mặt mà mô hình `large_bn` tập trung khi đoán đúng và khi nhầm lẫn.
