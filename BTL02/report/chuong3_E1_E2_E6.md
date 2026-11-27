# Chương 3. Mô hình CNN và quá trình huấn luyện

> Người viết: TV2 – Lê Phan Nguyên Hải (nhiệm vụ T22). Review chéo: TV3 (mục 3.1–3.5), TV4 (mục 3.6–3.8). Kiểm thử: TV1.
>
> **Xác nhận review chéo mục 3.1–3.5 (TV3 – Võ Hoàng Bảo, 2026-11-27):** Đã đối chiếu toàn bộ số liệu và mô tả mã nguồn trong mục 3.1–3.5 về đúng file sinh từ code — `figures/arch_cnn_baseline.csv` (64×24×24 / 128×12×12 / 256×6×6; `cls1_Linear` = `4,719,104` / `5,867,847` = `80.42%`), `figures/arch_cnn_comparison.csv` & `results/model_summary.csv` (`small` FC1 = `37,749,760` / `38,901,191` = `97.04%`), `report/tables/E6_summary.csv` (ResNet18 `11,180,103`, VGG16 `134,289,223`), `report/tables/baseline_summary.csv` & `results/baseline/baseline_summary.json` (`0.6815 ± 0.0075`, macro-F1 `0.6408 ± 0.0195`, best_val_loss `0.9109`, Train−Val `0.0397`), `src/models/cnn.py` (`bias=not bn`), `src/engine/trainer.py` (`ReduceLROnPlateau(mode="max", factor=0.5, patience=5)`, early stopping patience 15 theo `val_acc`), `tracking/experiments.csv` (`E5-003_s0` trùng sha256 `021c02e6…` với `baseline-002_s0`) và `tracking/test_log.csv` (`KT-016` lệch `0.0022`, `KT-019` lệch `0.0053`). Tất cả khớp 100% nguồn, văn phong rõ ràng, đạt yêu cầu chuẩn bị ghép báo cáo.
>
> **Nguồn số liệu.** Mọi con số trong chương lấy từ file do code sinh ra, không gõ tay hay ước lượng:
> - bảng tổng hợp `report/tables/{baseline,E1,E2,E6}_summary.csv`, sinh bằng `python src/make_tables.py` từ `tracking/experiments.csv`;
> - `report/tables/E6_analysis.md`, sinh bằng `python src/e6_analysis.py`;
> - `figures/*.csv` đi kèm từng hình, lệnh sinh hình ghi trong `tracking/figures_manifest.csv`;
> - `history.csv`, `grad_norms.csv`, `metrics.json` trong từng `results/<nhóm>/<exp_id>/`.
>
> Các bảng trong chương rút gọn từ các file trên, giữ nguyên giá trị. Nhận xét chi tiết từng run nằm ở `report/NHAT_KY_THUC_NGHIEM.md`. Khi ghép báo cáo, TV4 đánh lại số bảng/hình cho thống nhất (make_tables đặt tên "Bảng 4.x", chương này dùng "Bảng 3.x").
>
> Mọi độ chính xác là trên tập **PublicTest (validation)**. PrivateTest chỉ dùng một lần ở mô hình cuối (T19, Chương 4).

## 3.1. Các thành phần của CNN dùng trong bài

Mô hình nhận ảnh xám 1×48×48 và trả về 7 **logits** (chưa qua softmax). Các thành phần, theo thứ tự dữ liệu đi qua:

- **Tích chập (Conv2d 3×3, padding 1).** Mỗi bộ lọc trượt trên ảnh, tính tổng có trọng số của vùng 3×3 trên mọi kênh vào. Padding 1 giữ nguyên kích thước không gian, nên chỉ MaxPool làm ảnh nhỏ đi. Hai Conv 3×3 liên tiếp có vùng tiếp nhận 5×5 nhưng ít tham số hơn một Conv 5×5 và có thêm một lần phi tuyến (ý tưởng của VGG).
- **Batch Normalization (BatchNorm2d).** Chuẩn hóa mỗi kênh theo mean/variance của batch, rồi nhân và cộng hai tham số học được γ, β. Khi dùng BN, Conv bỏ bias (`bias=not bn` trong `cnn.py`) vì β đã làm việc đó. BN giữ phân phối đầu vào của mỗi lớp ổn định, nhờ đó mạng sâu học được với lr lớn và ít phụ thuộc cách khởi tạo (mục 3.6).
- **Hàm kích hoạt.** Mặc định là ReLU, f(x) = max(0, x): đạo hàm bằng 1 khi x > 0, nên gradient đi qua nhiều lớp không bị co lại. E2 so ReLU với LeakyReLU (hệ số 0.1), ELU, GELU, Tanh và Sigmoid. Đạo hàm Sigmoid lớn nhất chỉ 0.25 và đầu ra luôn dương; đạo hàm Tanh tại 0 bằng 1 và đầu ra đối xứng quanh 0. Khác biệt này giải thích kết quả E2 (mục 3.7).
- **Max Pooling 2×2, stride 2.** Lấy giá trị lớn nhất trong mỗi ô 2×2, chia đôi cạnh ảnh. Lớp này giảm tính toán và cho mạng chịu được dịch chuyển nhỏ. Với ảnh 48 px, số lần pool tối đa có giới hạn: VGG16 có 5 lần pool, đưa ảnh về 1×1 (mục 3.8).
- **Dropout.** Khi train, mỗi phần tử bị đặt về 0 với xác suất p và các phần tử còn lại được nhân 1/(1−p). Khi eval, dropout tắt. Baseline đặt Dropout 0.25 sau mỗi khối Conv và 0.5 trước lớp FC cuối.
- **Phần phân loại.** Mặc định là Flatten, rồi FC 512, ReLU, Dropout 0.5, rồi FC 7. Phương án thay thế là **Global Average Pooling (GAP)**: lấy trung bình mỗi kênh, bỏ thông tin vị trí, nhưng giảm mạnh số tham số (mục 3.6).
- **Hàm mất mát.** Dùng Cross-Entropy, L = −log softmax(z)_y, trên logits; `nn.CrossEntropyLoss` đã gồm softmax nên mô hình không có lớp softmax. Loss train có thể thêm label smoothing hoặc trọng số lớp (E4, E5). Loss val luôn là CE thường để các run so được với nhau.
- **Tối ưu.** Mặc định là SGD với momentum Nesterov 0.9, lr 0.01, weight decay 5×10⁻⁴ (phạt L2 cộng vào gradient). Ở E1, Adam lr 1×10⁻⁴ (cấu hình của notebook mẫu) được dùng để tách nguyên nhân mạng sâu không học được. So sánh optimizer có hệ thống nằm ở E3 (Chương 4).

## 3.2. Kiến trúc baseline của nhóm

Baseline là một CNN kiểu VGG gồm 3 khối, mỗi khối có 2 Conv (cấu hình `"baseline"` trong `src/models/cnn.py`). Kích thước bản đồ đặc trưng và số tham số được đo bằng forward thật (`describe()`), không tính tay.

**Hình 3.1.** Kiến trúc baseline: `figures/arch_cnn_baseline.png` (dữ liệu `figures/arch_cnn_baseline.csv`, sinh bằng `python -m src.visual.architecture_diagram`).

**Bảng 3.1.** Kích thước đầu ra và số tham số theo khối (rút từ `figures/arch_cnn_baseline.csv`).

| Khối | Các lớp | Đầu ra | Tham số tích lũy |
|---|---|---|---|
| 1 | 2 × [Conv3×3 64, BN, ReLU], MaxPool, Dropout 0.25 | 64×24×24 | 37,696 |
| 2 | 2 × [Conv3×3 128, BN, ReLU], MaxPool, Dropout 0.25 | 128×12×12 | 259,392 |
| 3 | 2 × [Conv3×3 256, BN, ReLU], MaxPool, Dropout 0.25 | 256×6×6 | 1,145,152 |
| Phân loại | Flatten (9216), Linear 512, ReLU, Dropout 0.5, Linear 7 | 7 | 5,867,847 |

Nhận xét: lớp `Linear(9216, 512)` có 4,719,104 tham số, chiếm **80.4%** của toàn mô hình. Sáu lớp Conv chỉ chiếm khoảng 1.1 M tham số. Vì vậy thay Flatten bằng GAP giảm mô hình còn 1,280,327 tham số (mục 3.6).

## 3.3. Các mô hình đối chiếu

**Bốn CNN của notebook mẫu** (`demo_code.html`, Keras) được tái hiện trong `cnn.py` với tên `small`, `medium`, `medium_01`, `large`. Các mô hình này không có BN, dùng Dropout 0.2 sau mỗi khối và 0.5 trước lớp cuối.

**Bảng 3.2.** So sánh kiến trúc (`figures/arch_cnn_comparison.csv`, Hình 3.2 `figures/arch_cnn_comparison.png`).

| Mô hình | Số lớp Conv | BN | FC | Tham số |
|---|---|---|---|---|
| `baseline` | 6 | có | 512 | 5,867,847 |
| `small` | 6 | không | 1024 | 38,901,191 |
| `medium` | 8 | không | 1024 | 11,327,303 |
| `medium_01` | 11 | không | 1024 | 12,101,895 |
| `large` | 16 | không | 2048 | 20,406,599 |

`small` nặng nhất dù chỉ có 6 lớp Conv: nó chỉ có 2 lần MaxPool nên bản đồ cuối là 256×12×12, và lớp FC đầu một mình có 37,749,760 tham số (97% mô hình). Đây là lý do số tham số tăng theo độ phân giải của bản đồ cuối chứ không theo độ sâu.

**Mô hình transfer learning** (`src/models/transfer.py`, dùng ở E6): ResNet18 và VGG16 của torchvision, thay lớp cuối bằng `Linear(…, 7)`. Có tùy chọn nạp trọng số ImageNet (`pretrained`) và đóng băng backbone (`freeze_backbone`; khi đóng băng, BN của backbone cũng giữ thống kê cũ). Ảnh được phóng lên 48 hoặc 96 px và nhân thành 3 kênh. Số tham số: ResNet18 11,180,103, VGG16 134,289,223 (`report/tables/E6_summary.csv`).

## 3.4. Quy trình huấn luyện

**Hình 3.3.** Pipeline dữ liệu và huấn luyện: `figures/pipeline_overview.png`.

- **Dữ liệu.** Chia theo cột `Usage` của `fer2013.csv`: Training 28,709 ảnh để train, PublicTest 3,589 ảnh làm validation, PrivateTest 3,589 ảnh bị khóa tới T19. Ảnh được chuẩn hóa bằng mean/std tính **chỉ trên Training** (`src/data/loader.py`) để không rò rỉ thông tin từ val/test.
- **Augmentation mặc định** (`flip_affine`, `src/data/augment.py`): lật ngang p = 0.5, RandomAffine xoay ±10°, dịch ±10%, tỉ lệ 0.9–1.1. Chỉ áp cho Training.
- **Vòng lặp** (`src/engine/trainer.py`, hàm `fit`): batch 128, tối đa 100 epoch, AMP (FP16 trên GPU). Sau mỗi epoch, mạng được đánh giá trên val (`model.eval()`, `torch.no_grad()`).
- **Chọn mô hình theo val_acc.** `best.pt` được lưu khi val_acc tăng. `ReduceLROnPlateau` theo dõi **val_acc**: không tăng 5 epoch thì lr nhân 0.5. **Early stopping** dừng khi val_acc không tăng 15 epoch. Do đó `best_epoch` là epoch có val_acc cao nhất, và run có thể train thêm khi val_loss đã bắt đầu tăng (thấy rõ ở `large_bn`, mục 3.6).
- **Ghi vết.** Mỗi run sinh `run.log`, `history.csv` (loss/acc/F1/lr/thời gian từng epoch), `metrics.json`, `config.json`, `curves.png`. Toàn bộ siêu tham số, git commit, máy, GPU và sha256 dữ liệu được ghi một dòng vào `tracking/experiments.csv`. Lệnh chạy: `python src/run_experiment.py --config configs/<nhóm>.json --member TVx [--only <biến thể>] [--seeds 0 1 2]`.
- **Ghi chuẩn gradient** (chỉ E2): `log_grad_norms = true` ghi `grad_norms.csv`, chứa chuẩn L2 trung bình của gradient từng lớp có trọng số ở mỗi epoch. Giá trị lấy sau khi bỏ hệ số scale của AMP, nên là gradient thật.
- **Tái lập.** Cùng máy, cùng seed thì run trùng tới từng chữ số: `E5-003_s0` trùng `baseline-002_s0` tới sha256 của `best.pt`. Khác GPU thì số học dấu phẩy động lệch từ epoch đầu. Với cấu hình ổn định, độ lệch nằm trong std của baseline: ví dụ `bs_64` lệch 0.0022 (KT-016), `aug_none` lệch 0.0053 (KT-019). Với cấu hình bất ổn như Sigmoid có BN, độ lệch lớn hơn nhiều (mục 3.7).

## 3.5. Mốc so sánh: baseline 3 seed

**Bảng 3.3.** Baseline (`report/tables/baseline_summary.csv`, cấu hình đóng băng `configs/baseline.json`).

| Mô hình | Số run | Tham số | Val acc | Val macro-F1 | Best val loss | Train − Val |
|---|---|---|---|---|---|---|
| `baseline` | 3/3 | 5,867,847 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 0.0397 |

Quy ước cho E1–E6: mỗi thí nghiệm đổi **một yếu tố** so với baseline. Chênh lệch val_acc nhỏ hơn std baseline (0.0075) coi là **chưa phân biệt được**. Biến thể có 1 seed chỉ dùng để định hướng; muốn kết luận cần 3 seed. Cột "Train − Val" là `train_acc_cuoi − val_acc`. `train_acc_cuoi` đo trong chế độ train (có augmentation và dropout), nên cột này chỉ dùng để so xu hướng quá khớp giữa các cấu hình.

## 3.6. Thực nghiệm E1 – Kiến trúc

**Câu hỏi.** Độ sâu, độ rộng, BatchNorm và kiểu phần phân loại ảnh hưởng thế nào? Bốn CNN của notebook mẫu có tái hiện được không? Cấu hình: `configs/E1_arch.json` (T12).

**Bảng 3.4.** Kết quả E1 (rút từ `report/tables/E1_summary.csv`; cột "Số run" là số run Completed trên tổng số run).

| Biến thể | Số run | Tham số | Val acc | Val macro-F1 | Best val loss | Train − Val | So với baseline |
|---|---|---|---|---|---|---|---|
| `small` | 3/4 | 38,901,191 | 0.6934 ± 0.0027 | 0.6773 ± 0.0045 | 0.9143 | 0.1227 | +0.0119 (cao hơn) |
| `medium` | 3/3 | 11,327,303 | 0.6919 ± 0.0031 | 0.6730 ± 0.0059 | 0.9119 | 0.101 | +0.0104 (cao hơn) |
| `medium_01` | 1/1 | 12,101,895 | 0.2494 | 0.0570 | 1.8107 | 0.0019 | -0.4322 (thấp hơn) |
| `large` | 1/1 | 20,406,599 | 0.2494 | 0.0570 | 1.8107 | 0.0019 | -0.4322 (thấp hơn) |
| `baseline_w0.5` | 3/3 | 2,650,279 | 0.6747 ± 0.0049 | 0.6347 ± 0.0103 | 0.9042 | 0.0162 | -0.0068 (< std baseline) |
| `baseline_w2` | 3/3 | 14,017,159 | 0.6654 ± 0.0070 | 0.5717 ± 0.0196 | 0.9663 | 0.0234 | -0.0162 (thấp hơn) |
| `baseline_no_bn` | 3/3 | 5,866,951 | 0.6941 ± 0.0033 | 0.6682 ± 0.0044 | 0.9059 | 0.0588 | +0.0125 (cao hơn) |
| `baseline_gap` | 3/3 | 1,280,327 | 0.6839 ± 0.0013 | 0.6478 ± 0.0040 | 0.9077 | 0.0463 | +0.0024 (< std baseline) |
| `medium_01_adam1e-4` | 1/1 | 12,101,895 | 0.6392 | 0.5993 | 0.9729 | 0.0145 | -0.0424 (thấp hơn) |
| `large_adam1e-4` | 0/1 | — | — | — | — | — | — |
| `medium_01_bn` | 1/2 | 12,103,623 | 0.6913 | 0.6686 | 0.9204 | 0.1669 | +0.0098 (cao hơn) |
| `large_bn` | 3/3 | 20,410,439 | **0.7060 ± 0.0023** | **0.6922 ± 0.0027** | 0.9084 | 0.1952 | +0.0244 (cao hơn) |
| `medium_01_glorot_adam1e-4` | 1/1 | 12,101,895 | 0.6721 | 0.6383 | 0.9064 | 0.0545 | -0.0095 (thấp hơn) |
| `large_glorot_adam1e-4` | 1/1 | 20,406,599 | 0.6236 | 0.5545 | 1.0056 | 0.0313 | -0.0580 (thấp hơn) |

`large_adam1e-4` (`E1-011_s0`) bị dừng tay ở epoch 4 vì đứng yên ở 0.2494 (Aborted), nên không có số.

**3.6.1. Mạng sâu không BN sập vì khởi tạo, không phải vì optimizer.** Với SGD lr 0.01 và khởi tạo mặc định của PyTorch, `medium_01` (11 Conv) và `large` (16 Conv) không học được: loss đứng ở 1.81 từ epoch 1 và mạng đoán toàn lớp Happy, ra đúng tỉ lệ lớp Happy trong val là 0.2494. Bảng 3.5 tách nguyên nhân bằng cách mỗi dòng đổi một yếu tố:

**Bảng 3.5.** Tách nguyên nhân (1 seed mỗi dòng, trừ `large_bn`).

| Mô hình | Init | Optimizer | BN | Val acc | Run |
|---|---|---|---|---|---|
| MEDIUM1 | mặc định | SGD 0.01 | không | 0.2494 | `E1-003_s0` |
| MEDIUM1 | mặc định | Adam 1e-4 | không | 0.6392 | `E1-010_s0` |
| MEDIUM1 | Glorot | Adam 1e-4 | không | 0.6721 | `E1-015_s0` |
| MEDIUM1 | mặc định | SGD 0.01 | có | 0.6913 | `E1-013_s0` |
| LARGE | mặc định | SGD 0.01 | không | 0.2494 | `E1-004_s0` |
| LARGE | mặc định | Adam 1e-4 | không | kẹt 0.2494 (dừng tay) | `E1-011_s0` |
| LARGE | Glorot | Adam 1e-4 | không | 0.6236 | `E1-016_s0` |
| LARGE | mặc định | SGD 0.01 | có | 0.7060 ± 0.0023 (3 seed) | `E1-014_s0`–`_s2` |

- Keras mặc định khởi tạo Glorot, còn PyTorch dùng kaiming_uniform với a = √5, phương sai nhỏ hơn khoảng 3 lần. Ở LARGE không BN, gradient tới `conv1_1` lúc khởi tạo chỉ 7.6×10⁻⁸ (Glorot: 1.0×10⁻³; số đo ghi trong `configs/E1_arch.json`). Con số này ngang `eps = 1e-8` của Adam, nên Adam chỉ cứu được MEDIUM1 chứ không cứu được LARGE.
- Đổi sang Glorot cùng Adam 1e-4 tái hiện được thiết lập của notebook mẫu: MEDIUM1 đạt 0.6721, mẫu báo cáo 0.65. LARGE đạt 0.6236 (mẫu 0.67) và vẫn thiếu khớp sau 100 epoch.
- **BatchNorm bỏ hẳn sự phụ thuộc vào khởi tạo:** giữ nguyên SGD lr 0.01 và init mặc định, chỉ thêm BN là cả hai độ sâu đều cho kết quả tốt nhất.

**3.6.2. Độ sâu khi có BN.** `large_bn` đạt **0.7060 ± 0.0023**, cao nhất E1 và hơn baseline +0.0244 (hơn 3 lần std baseline). Ba seed là 0.7086, 0.7047 và 0.7047; seed thấp nhất vẫn cao hơn seed tốt nhất của baseline (0.6890). Macro-F1 0.6922 ± 0.0027 hơn baseline khoảng 0.05, và dao động giữa các seed nhỏ hơn baseline. Cái giá là quá khớp mạnh nhất E1: Train − Val 0.1952 (baseline 0.0397). val_loss thấp nhất ở khoảng epoch 40 rồi tăng dần trong khi val_acc vẫn nhích lên, vì checkpoint và early stopping dựa trên val_acc. Mô hình có 20.4 M tham số, gấp 3.5 lần baseline.

**3.6.3. Độ rộng.** Giữ 6 Conv của baseline:
- `baseline_w0.5` (một nửa số kênh, 2.65 M tham số) đạt 0.6747 ± 0.0049, chênh −0.0068, dưới std baseline. Thu hẹp một nửa gần như không mất gì, nghĩa là baseline còn dư sức chứa.
- `baseline_w2` (gấp đôi kênh, 14.0 M tham số) **thấp hơn** baseline: 0.6654 ± 0.0070, và macro-F1 chỉ 0.5717 ± 0.0196. Nguyên nhân không phải quá khớp (Train − Val 0.0234, nhỏ hơn baseline) mà là học chậm. Cả 3 seed đạt đỉnh ở epoch 99–100, tức chưa hội tụ sau 100 epoch với lr 0.01.
- Kết luận: với cùng cấu hình huấn luyện, **tăng sâu (có BN) có lợi hơn tăng rộng**.

**3.6.4. BatchNorm và phần phân loại ở độ sâu baseline.**
- `baseline_no_bn` đạt 0.6941 ± 0.0033 (+0.0125): với 6 Conv và ReLU, bỏ BN không làm mạng sập mà còn cao hơn. BN chỉ cần thiết khi mạng sâu (mục 3.6.1).
- `baseline_gap` đạt 0.6839 ± 0.0013 (+0.0024, dưới std) chỉ với 1,280,327 tham số, bằng 22% baseline. Lớp FC sau Flatten chiếm 80% tham số (mục 3.2) nhưng không làm tăng độ chính xác.

**Kết luận E1.** Ứng viên kiến trúc cho E7 là `large_bn`, đã đủ 3 seed. Nên kết hợp nó với điều chuẩn của E4 vì khoảng Train − Val lớn. Dự phòng: `baseline_no_bn` và `small`, cũng đủ 3 seed. Nếu cần mô hình nhỏ thì dùng head GAP. Baseline vẫn là mốc so sánh của E2–E6.

## 3.7. Thực nghiệm E2 – Hàm kích hoạt và vanishing gradient

**Câu hỏi.** Hàm kích hoạt ảnh hưởng thế nào tới độ chính xác? Sigmoid có gây **vanishing gradient** trên mạng 6 Conv không, và BN có cứu được không? Cấu hình: `configs/E2_activation.json` (T13). Tất cả các run bật `log_grad_norms`.

**Bảng 3.6.** Kết quả E2 (rút từ `report/tables/E2_summary.csv`). `relu` trùng cấu hình baseline, chỉ thêm việc ghi gradient.

| Biến thể | Số run | Tham số | Val acc | Val macro-F1 | Best val loss | Train − Val | So với baseline |
|---|---|---|---|---|---|---|---|
| `relu` | 1/1 | 5,867,847 | 0.6863 | 0.6542 | 0.9033 | 0.05 | +0.0047 (< std baseline) |
| `leaky_relu` | 1/1 | 5,867,847 | 0.6935 | 0.6664 | 0.9016 | 0.0661 | +0.0120 (cao hơn) |
| `elu` | 1/1 | 5,867,847 | 0.6553 | 0.6169 | 0.9436 | -0.0113 | -0.0262 (thấp hơn) |
| `gelu` | 1/1 | 5,867,847 | 0.6918 | 0.6614 | 0.8807 | 0.0632 | +0.0103 (cao hơn) |
| `tanh` | 1/1 | 5,867,847 | 0.6414 | 0.5952 | 0.9688 | -0.0025 | -0.0401 (thấp hơn) |
| `sigmoid` | 4/4 | 5,867,847 | 0.4765 ± 0.0220 | 0.3638 ± 0.0291 | 1.3658 | -0.051 | -0.2050 (thấp hơn) |
| `relu_no_bn` | 1/1 | 5,866,951 | 0.6913 | 0.6675 | 0.8969 | 0.0602 | +0.0098 (cao hơn) |
| `tanh_no_bn` | 1/1 | 5,866,951 | 0.6523 | 0.6201 | 0.9667 | -0.0161 | -0.0293 (thấp hơn) |
| `sigmoid_no_bn` | 1/1 | 5,866,951 | 0.2494 | 0.0570 | 1.8108 | 0.0019 | -0.4322 (thấp hơn) |

**Hình 3.4.** Chuẩn gradient: `figures/e2_grad_norms.png` (dữ liệu `figures/e2_grad_norms.csv`, lệnh `python src/visual/grad_norms.py --group E2 --epoch 1`). Hình (a) vẽ chuẩn L2 gradient từng lớp ở epoch 1, từ lớp vào (`conv1_1`) tới lớp ra (`fc2`). Hình (b) vẽ gradient của `conv1_1` theo epoch.

**3.7.1. Vanishing gradient của Sigmoid không BN.**

**Bảng 3.7.** Chuẩn gradient lớp Conv đầu ở epoch 1 (`figures/e2_grad_norms.csv` và `results/E2/*/grad_norms.csv`).

| Biến thể | `conv1_1` | `conv1_1` / `fc2` | Diễn biến về sau (Hình 3.4b) |
|---|---|---|---|
| `relu` | 5.9e-2 | 0.056 | tăng lên 0.72 ở epoch 100 |
| `relu_no_bn` | 1.3e-2 | 0.042 | tăng lên 0.42 |
| `tanh_no_bn` | 1.5e-2 | 0.035 | tăng lên 0.34 |
| `sigmoid` (có BN) | 1.8e-3 | 2.5e-3 | hồi phục, 0.28 ở epoch 10 |
| `sigmoid_no_bn` | **3.2e-7** | **4.0e-7** | giảm còn 1.0e-7 ở epoch 16, không hồi phục |

- `sigmoid_no_bn` **không học được**. Gradient giảm đều từ lớp ra về lớp vào: `fc2` 8.1e-1, `fc1` 6.5e-1, `conv3_2` 4.6e-2, `conv3_1` 4.9e-3, `conv2_2` 6.9e-4, `conv2_1` 7.9e-5, `conv1_2` 1.1e-5, `conv1_1` 3.2e-7. Mỗi lớp Conv nhân gradient thêm khoảng 0.1 lần, khớp với việc đạo hàm sigmoid không vượt 0.25. Lớp đầu nhận gradient nhỏ hơn lớp ra khoảng 2.5 triệu lần, nên các lớp Conv gần như đứng yên.
- Hệ quả đo được: train_loss đứng ở 1.81 suốt 16 epoch, bằng entropy của phân bố lớp trong Training (1.8096). Tức mạng chỉ học được tần suất các lớp qua bias của lớp cuối, không học gì từ ảnh. Mạng đoán toàn lớp Happy, val_acc 0.2494 (`E2-009_s0`).
- Đây là **đặc tính của Sigmoid**, không phải do bỏ BN hay do hàm bão hòa nói chung. `tanh_no_bn`, cũng bão hòa, có tỉ lệ lớp đầu/lớp ra 0.035, ngang `relu_no_bn` (0.042) và lớn hơn `sigmoid_no_bn` khoảng 100,000 lần. Lý do: đạo hàm tanh tại 0 bằng 1 và đầu ra đối xứng quanh 0, còn sigmoid chỉ 0.25 và đầu ra luôn dương.

**3.7.2. BN sửa được gradient nhưng không sửa được độ chính xác của Sigmoid.** Có BN, gradient lớp đầu của sigmoid ở epoch 1 vẫn nhỏ hơn ReLU khoảng 33 lần (1.8e-3 so với 5.9e-2). Nhưng tới epoch 10 nó hồi phục ngang ReLU, vì BN giữ đầu vào mỗi lớp quanh 0, vùng đạo hàm sigmoid lớn nhất. Dù vậy val_acc chỉ đạt 0.4765 ± 0.0220 (4 run), `train_acc_cuoi` 0.41–0.44, tức thiếu khớp nặng. Ở bản có BN, nút thắt không còn là vanishing gradient mà là sigmoid học rất chậm với SGD lr 0.01, khiến plateau cắt lr 3–8 lần và early stopping dừng ở epoch 44–84.

**3.7.3. Độ ổn định của Sigmoid có BN và tái lập.** Sigmoid có 4 run: seed 0, 1, 2, và một lần chạy lại seed 0 trên GPU khác (`E2-010_s0`) cho kiểm thử KT-017. Đường val_acc dao động trung bình 0.02–0.03 mỗi epoch, có lần nhảy 0.10–0.18. Cùng seed 0, run gốc trên RTX 3060 đạt 0.4511, còn trên RTX 4060 đạt 0.4890. Ba seed trên cùng một máy chênh nhau 0.034. Vì vậy dung sai 0.0075 (std của baseline, cấu hình ổn định) không áp dụng được cho biến thể này. Phần định tính thì tái lập ở mọi run: gradient `conv1_1` epoch 1 trong khoảng 1.8–2.1×10⁻³, hồi phục khoảng 0.23–0.31 ở epoch 10. Chi tiết ở nhật ký, Kết luận E2 mục 6.

**3.7.4. Các hàm còn lại.**
- `leaky_relu` (0.6935) và `gelu` (0.6918, best_val_loss thấp nhất E2 là 0.8807) vượt std baseline, nhưng mới 1 seed và chỉ nhỉnh hơn ngưỡng.
- `elu` (0.6553) thiếu khớp (`train_acc_cuoi` thấp hơn val), dù gradient không có vấn đề.
- `tanh` (0.6414) và `tanh_no_bn` (0.6523) kém ReLU 3–4 điểm.
- `relu_no_bn` (0.6913) khớp kết quả `baseline_no_bn` của E1 (0.6941 ± 0.0033) đo trên máy khác, xác nhận chéo rằng ở 6 Conv với ReLU, bỏ BN không gây hại.

**Kết luận E2.** Giữ họ ReLU cho E7. `leaky_relu` và `gelu` là ứng viên thay ReLU, nhưng cần thêm seed 1, 2 trước khi chọn. Loại `sigmoid`, `tanh`, `elu`. `E2-009_s0` cùng Hình 3.4 là bằng chứng vanishing gradient của Sigmoid so với ReLU mà đề bài yêu cầu.

## 3.8. Thực nghiệm E6 – Transfer learning

**Câu hỏi.** Trọng số ImageNet có giúp ích với ảnh khuôn mặt xám 48 px không? Vì sao VGG16 ở 48 px bị coi là "fail case" trong báo cáo mẫu? Cấu hình: `configs/E6_transfer.json` (T17). Base giữ nguyên tối ưu và augmentation như baseline; ảnh phóng lên 96 px (hoặc 48 px) và nhân 3 kênh.

**Bảng 3.8.** Kết quả E6 (rút từ `report/tables/E6_summary.csv`; chi tiết ở `report/tables/E6_analysis.md`, Bảng E6.1).

| Biến thể | Số run | Tham số | Val acc | Val macro-F1 | Best val loss | Train − Val | So với baseline |
|---|---|---|---|---|---|---|---|
| `vgg16_48_ft` | 1/3 | 134,289,223 | 0.6938 | 0.6812 | 0.9373 | 0.2571 | +0.0123 (cao hơn) |
| `vgg16_96_ft` (30 epoch) | 1/1 | 134,289,223 | 0.6977 | 0.6818 | 0.922 | 0.1876 | +0.0162 (cao hơn) |
| `resnet18_96_scratch` | 1/1 | 11,180,103 | 0.6771 | 0.6672 | 1.0006 | 0.2437 | -0.0045 (< std baseline) |
| `resnet18_96_ft` | 1/1 | 11,180,103 | 0.6874 | 0.6698 | 0.9718 | 0.2983 | +0.0059 (< std baseline) |
| `resnet18_96_frozen` | 1/1 | 11,180,103 | 0.4570 | 0.3683 | 1.4507 | -0.0133 | -0.2246 (thấp hơn) |

Hậu tố `_ft`: pretrained ImageNet, fine-tune toàn bộ. `_scratch`: khởi tạo ngẫu nhiên. `_frozen`: pretrained, chỉ train lớp cuối.

**3.8.1. Vì sao VGG16 ở 48 px là "fail case".** VGG16 có 5 lần MaxPool, nên ở 48 px cạnh ảnh đi 48 → 24 → 12 → 6 → 3 → 1. Các số dưới đây đo bằng forward thật, trong Bảng E6.3 của `E6_analysis.md`:
- Khối conv5 chỉ chạy trên bản đồ 3×3. pool5 chỉ dùng cửa sổ 2×2 góc trên trái, tức 4/9 ô, và bỏ hàng, cột thứ ba. Mỗi kênh còn lại đúng một giá trị, nên mạng **mất hết thông tin vị trí** (mắt, miệng nằm đâu).
- torchvision không báo lỗi vì `AdaptiveAvgPool2d(7, 7)` lặp bản đồ 1×1 thành 7×7. Do đó FC1 nhận 25,088 đầu vào nhưng chỉ có 512 số thật. 102,764,544 tham số của FC1 tương đương một lớp `Linear(512, 4096)` với 2,101,248 tham số hiệu dụng; khoảng 100.7 M tham số là dư thừa nhưng vẫn phải lưu và cập nhật. Ở 96 px, pool5 còn 3×3, dùng đủ 36/36 ô conv5.

**3.8.2. "Fail" về hiệu quả, không phải về độ chính xác.** Fine-tune toàn bộ từ ImageNet, VGG16 48 px vẫn đạt 0.6938, chỉ kém bản 96 px 0.0039 (dưới std). Vector đặc trưng toàn cục của ImageNet vẫn đủ phân biệt cảm xúc ở mức này. Điểm "fail" thật nằm ở chi phí: 134.3 M tham số (22.9 lần baseline), `best.pt` 512.3 MB, quá khớp mạnh (Train − Val 0.2571 so với 0.0397), và khoảng 100.7 M tham số FC1 không mang thông tin. Cách sửa nếu vẫn dùng VGG16 ở 48 px: bỏ pool5, hoặc thay classifier bằng GAP và `Linear(512, 7)`. Cách khác là phóng ảnh lên ≥ 96 px như `vgg16_96_ft`.

**3.8.3. ResNet18: từ đầu, fine-tune và đóng băng** (Bảng E6.2 của `E6_analysis.md`).
- Fine-tune từ ImageNet (0.6874) hơn train từ đầu (0.6771) +0.0103, nhưng mới 1 seed.
- Lợi ích rõ nhất là **hội tụ nhanh**. Sau epoch 1, bản fine-tune đạt 0.5612, bản từ đầu 0.2862. val ≥ 0.65 ở epoch 14 so với 45; best epoch 42 so với 60.
- Đóng băng backbone chỉ đạt 0.4570, train acc cuối 0.4436 còn thấp hơn val: **thiếu khớp**. Một lớp `Linear(512, 7)` trên đặc trưng ImageNet cố định không tách được biểu cảm khuôn mặt xám, nên phải fine-tune cả backbone.

**Kết luận E6.** Không biến thể transfer nào vừa hơn baseline rõ ràng vừa rẻ. VGG16 hơn 0.012–0.016 nhưng nặng gấp 22.9 lần; ResNet18 fine-tune chỉ hơn 0.0059, dưới std. E7 giữ họ CNN của nhóm làm mô hình chính. Nếu cần đa dạng cho ensemble, `resnet18_96_ft` là ứng viên rẻ nhất, nhưng phải chạy đủ 3 seed trước khi chọn.

## 3.9. Tóm tắt các quyết định chuyển sang E7

| Thí nghiệm | Kết luận chính | Đưa vào E7 |
|---|---|---|
| E1 | BN là điều kiện để mạng sâu học được; `large_bn` 0.7060 ± 0.0023 cao nhất; tăng sâu tốt hơn tăng rộng; GAP giảm 78% tham số mà không mất độ chính xác | `large_bn` làm kiến trúc chính, kèm điều chuẩn E4; `baseline_no_bn`, `small` dự phòng |
| E2 | Sigmoid không BN gây vanishing gradient (lớp đầu nhỏ hơn lớp ra khoảng 2.5 triệu lần); BN cứu gradient nhưng sigmoid vẫn thiếu khớp; họ ReLU tốt nhất | Giữ ReLU; thử `leaky_relu`/`gelu` nếu đủ 3 seed |
| E6 | Transfer không hơn rõ ràng so với CNN nhỏ; VGG16 48 px tốn 100.7 M tham số vô ích; đóng băng backbone thiếu khớp | Không dùng làm mô hình chính; `resnet18_96_ft` chỉ là ứng viên ensemble |

Lưu ý: kết quả E2–E6 đo trên baseline nên chưa chắc còn đúng khi đổi sang `large_bn`. E7 nên so cả "baseline + lựa chọn tốt nhất" lẫn "large_bn + lựa chọn tốt nhất", mỗi bên 3 seed, thay vì cộng dồn các mức cải thiện.

## Phụ lục 3.A. Lệnh tái lập

```bash
# Bảng tổng hợp (Bảng 3.3, 3.4, 3.6, 3.8) và tự điền dòng Kết quả trong nhật ký
python src/make_tables.py --update-log report/NHAT_KY_THUC_NGHIEM.md
# Bảng phân tích E6 (E6.1–E6.3)
python src/e6_analysis.py
# Hình kiến trúc và pipeline (Hình 3.1–3.3)
python -m src.visual.architecture_diagram
# Hình chuẩn gradient (Hình 3.4)
python src/visual/grad_norms.py --group E2 --epoch 1
# Chạy lại một biến thể, ví dụ large_bn 3 seed
python src/run_experiment.py --config configs/E1_arch.json --member TVx --only large_bn --seeds 0 1 2
```
