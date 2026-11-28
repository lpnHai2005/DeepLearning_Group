# Chương 5. Tóm tắt mã nguồn

> **Xác nhận review chéo Chương 5–8 (`C2` — TV3 – Võ Hoàng Bảo, 2026-11-28, commit `95c3dce`):** Đã đối chiếu độc lập toàn bộ số liệu, bảng và hình trong Chương 5–8 (và rà soát Chương 1 Bảng 1.2, phần chỉnh giọng văn Chương 4 ở `8006efe`) với các nguồn sinh từ code (`report/tables/T25_filters_stats.csv`, `report/tables/T25_runs_stats.csv`, `report/tables/T25_contribution.csv`, `report/tables/E7_tta_ensemble.csv`, `figures/filters_grad_cam_{correct,wrong}.csv`, `results/E7/TEST_DONE.json`, `tracking/experiments.csv`): (1) **Chương 5**: `21/21` file `tests/test_*.py` khớp thực tế, Bảng 5.1 khớp kiến trúc mã nguồn trong `src/` và `tracking/`; (2) **Chương 6**: `15/15` dòng Bảng 6.1 khớp 100% (`0` sai lệch về nhãn, độ tự tin 6 chữ số thập phân và tọa độ đỉnh `(y, x)`), toàn bộ thống kê trọng số/feature map/Grad-CAM (`|w| = 0.1337`, `0.000368`, tỷ lệ `363.7×`, `16/16`, `13/16`, độ phủ `0.2555` vs `0.2355`, `6/7` & `7/8`) khớp 100% `T25_filters_stats.csv`, `7/7` ảnh tồn tại; (3) **Chương 7**: quy mô `92` dòng (`86` chính thức + `1` PrivateTest + `5` hạ tầng; `80` Completed / `4` Aborted / `2` Failed), `51.8` giờ GPU (`20.2 + 16.5 + 15.1`), số run `4/30/12/16/10/3/7/4`, E7 (`0.7060 → 0.7236 → 0.7250`, NLL `1.2175 → 0.9780 → 0.9402`, `ten_crop 0.6991`), tham chiếu `Bảng 3.10` và `Hình 7.1` đều khớp 100% nguồn; (4) **Chương 8 & Chương 1**: khớp `72.50%` Public / `73.64%` Private (`Macro-F1 73.22%`) và Bảng 1.2 (`28/28` nhiệm vụ R, `25.0%`/thành viên). **2 góp ý biên tập nhỏ gửi `R` (TV4) trước khi `T` (TV1) kiểm thử `T25`**: (a) `chuong8_ket_luan.md` dòng 5 đang ghi `"127 test tự động"` — đề xuất cập nhật thành **`128 test`** cho đồng nhất với `QA_T27_report.md` (`a0cf536`) và `Slide 1/15` (`b9da663`), kèm ghi chú rõ trong `chuong5_ma_nguon.md` dòng 124; (b) `chuong5_ma_nguon.md` dòng 3 khẳng định mọi file `.py` có dòng `Phụ trách: TVx`, hiện còn 2 file test (`tests/test_baseline_summary.py`, `tests/test_e6_analysis.py`) chưa có dòng này (đề xuất bổ sung trước khi đóng gói `T28`). Đạt yêu cầu review chéo `C2` của `T25`.

Mỗi file `.py` có dòng "Phụ trách: TVx" ở đầu file; cột người phụ trách trong Bảng 5.1 lấy theo dòng đó.

## 5.1. Tổ chức kho mã và luồng chạy

Toàn bộ code nằm trong thư mục `BTL02/` và chỉ dùng Python, PyTorch, torchvision cùng các thư viện số học (numpy, pandas, scikit-learn, matplotlib). Không có dòng nào dùng Keras hay TensorFlow (kiểm tra tĩnh ở QA T27, `report/QA_T27_report.md`). Luồng chạy một thực nghiệm như sau:

1. File cấu hình JSON trong `configs/` khai báo mô hình, siêu tham số và danh sách biến thể (ví dụ `configs/E1_arch.json`). Mỗi biến thể chỉ đổi một yếu tố so với `configs/baseline.json`.
2. `src/run_experiment.py` đọc cấu hình, tạo mô hình bằng `build_model` hoặc `build_transfer`, tạo loader bằng `make_loaders`, gọi `fit` cho từng seed, rồi ghi một dòng vào `tracking/experiments.csv` và vẽ `curves.png`.
3. `src/make_tables.py` gom `tracking/experiments.csv` thành các bảng `report/tables/*_summary.csv` mà Chương 2 đến 4 trích dẫn.
4. Mô hình cuối được đánh giá trên PublicTest bằng `src/eval_e7.py` (TTA, ensemble) và trên PrivateTest đúng một lần bằng `python src/run_experiment.py --eval-test`.
5. Hình bộ lọc và Grad-CAM sinh bằng `src/visualize_filters.py` từ `best.pt` của mô hình cuối.

**Bảng 5.1.** Các module chính, người phụ trách và nhiệm vụ.

| Module | Chức năng | Phụ trách | Nhiệm vụ |
|---|---|---|---|
| `src/data/fer2013.py`, `check_fer2013.py` | Đọc `fer2013.csv` thành mảng `uint8 (N, 48, 48)`, kiểm tra schema, số split và SHA-256 | TV1 | T03, T04 |
| `src/data/loader.py`, `augment.py` | Chuẩn hóa theo thống kê Training, DataLoader, augmentation, WeightedRandomSampler | TV1 | T04, T16 |
| `src/data/eda.py` | Hình và CSV phân tích dữ liệu (phân bố lớp, ảnh mẫu, ảnh trung bình) | TV1 | T05 |
| `src/models/cnn.py`, `transfer.py` | CNN kiểu VGG có cấu hình, ResNet18 và VGG16 của torchvision | TV2 | T06, T17 |
| `src/engine/trainer.py` | Vòng huấn luyện `fit`, ghi log, early stopping, luật giữ/xóa weights | TV2 | T07 |
| `src/run_experiment.py` | Chạy nhóm thực nghiệm từ config, ghi `experiments.csv`, đánh giá PrivateTest một lần | TV2 (TV3 viết phần `--eval-test`) | T07, T19 |
| `src/visual/grad_norms.py`, `src/e6_analysis.py` | Chuẩn gradient từng lớp (E2), phân tích VGG16 ở 48 px (E6) | TV2 | T13, T17 |
| `src/engine/evaluate.py`, `src/visual/plots.py` | Dự đoán có TTA, chỉ số, ma trận nhầm lẫn, lưới ca đúng/sai, đăng ký hình | TV3 | T08 |
| `src/make_tables.py`, `src/eval_e7.py` | Bảng tổng hợp mean ± std, TTA và ensemble E7 | TV3 | T09, T18 |
| `notebooks/BTL02_reproducible.ipynb` | Notebook gọi hàm trong `src/`, chạy lại từ đầu đến cuối có lưu output | TV3 | T10 |
| `src/visual/filters.py`, `src/visualize_filters.py` | Trọng số bộ lọc, feature map, activation maximization, Grad-CAM | TV4 | T20 |
| `src/visual/architecture_diagram.py`, `e4_overfit.py`, `e5_recall.py` | Sơ đồ kiến trúc và pipeline, hình E4, E5 | TV4 | T24, T15, T16 |
| `src/t25_report_stats.py` | Số liệu đóng góp và thống kê hình bộ lọc cho báo cáo | TV4 | T25 |
| `tracking/*.py` | Dựng bảng phân công, sinh nhật ký git, tự suy trạng thái nhiệm vụ | TV2 | T01 |

## 5.2. Dữ liệu (TV1)

`load_fer2013(csv_path)` đọc cột `pixels` thành ảnh 48 × 48 và trả về ba split `train`, `val`, `test` theo cột `Usage`, giữ nguyên thứ tự nhãn của Kaggle qua một hằng `EMOTIONS` duy nhất. `make_loaders(config)` chỉ trả loader Training và PublicTest; mean và độ lệch chuẩn tính trên Training rồi áp cho mọi split. PrivateTest chỉ đọc được qua `make_test_loader`, hàm này chỉ được gọi trong lệnh `--eval-test`.

Augmentation mặc định `flip_affine` dựng bằng `torchvision.transforms.v2`, chỉ gắn vào loader Training (trích rút gọn từ `build_augmentation`):

```python
operations.append(transforms.RandomHorizontalFlip(p=0.5))
operations.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1),
                                          scale=(0.9, 1.1), fill=0))
if mode == "random_erasing":
    operations.append(transforms.RandomErasing(p=erase_probability, value=0))
```

Khi cấu hình đặt `"sampler": "weighted_random"` (E5), `_training_sampler` tạo `WeightedRandomSampler` với trọng số mỗi mẫu tỷ lệ nghịch với tần suất lớp.

## 5.3. Mô hình (TV2)

Lớp `CNN` trong `src/models/cnn.py` dựng mạng từ danh sách khối, mỗi khối là danh sách số kênh của các lớp Conv 3 × 3, sau đó MaxPool 2 × 2 và Dropout. Đoạn cốt lõi (trích nguyên văn):

```python
for b, widths in enumerate(blocks, start=1):
    for i, c_out in enumerate(widths, start=1):
        feats.append((f"conv{b}_{i}", nn.Conv2d(c_in, c_out, 3, padding=1, bias=not bn)))
        if bn:
            feats.append((f"bn{b}_{i}", nn.BatchNorm2d(c_out)))
        feats.append((f"act{b}_{i}", _act(activation)))
        c_in = c_out
    feats.append((f"pool{b}", nn.MaxPool2d(2)))
    feats.append((f"drop{b}", nn.Dropout(dropout_conv)))
```

Kích thước đầu vào của phần phân loại được đo bằng một lần chạy thử thay vì tính tay, nên đổi số khối hay kích thước ảnh không phải sửa code. Mô hình trả logits, không có softmax, vì `nn.CrossEntropyLoss` đã gồm softmax. Năm cấu hình có sẵn: `small`, `medium`, `medium_01`, `large` tái hiện bốn CNN của notebook mẫu, `baseline` là cấu hình của nhóm. Mọi thí nghiệm E1, E2 chỉ truyền tham số ghi đè, ví dụ `build_model("large", bn=True)` là `large_bn`, `build_model("baseline", activation="sigmoid", bn=False)` là `sigmoid_no_bn`. Tên lớp được đặt rõ (`conv4_4`, `bn4_4`) để hàm trực quan hóa tìm đúng lớp conv cuối qua `model.last_conv()`.

`build_transfer(name, pretrained, freeze_backbone)` trong `transfer.py` lấy ResNet18 hoặc VGG16 của torchvision, thay lớp cuối bằng `Linear(..., 7)`; khi đóng băng backbone, BatchNorm của backbone cũng giữ thống kê ImageNet.

## 5.4. Huấn luyện và ghi vết (TV2)

`fit(model, train_loader, val_loader, cfg, run_dir)` trong `src/engine/trainer.py` là vòng lặp duy nhất của dự án, dùng cho mọi thực nghiệm và cả notebook. Mỗi epoch, hàm huấn luyện một lượt, đánh giá trên PublicTest, ghi ngay một dòng `history.csv` và một dòng `run.log`, lưu `best.pt` khi val_acc tăng (trích rút gọn, bỏ phần ghi log và chuẩn gradient):

```python
for epoch in range(1, epochs + 1):
    tl, ta, _ = _run_epoch(model, train_loader, device, criterion, amp, optimizer, scaler, ...)
    vl, va, conf = _run_epoch(model, val_loader, device, val_criterion, amp)
    if not (math.isfinite(tl) and math.isfinite(vl)):
        raise _NanLoss(epoch)
    writer.writerow([epoch, tl, vl, ta, va, macro_f1(conf), lr, round(dt, 3)]); hf.flush()
    if va > best_acc:
        best_acc, stale = va, 0
        atomic_torch_save(model.state_dict(), best_path)
    else:
        stale += 1
    scheduler.step(va)                      # ReduceLROnPlateau theo val_acc
    if patience and stale >= patience:      # early stopping
        break
```

Huấn luyện dùng AMP (FP16) trên GPU; loss huấn luyện có thể thêm trọng số lớp và label smoothing, còn loss validation luôn là cross-entropy thường để các run so được với nhau. Khối `finally` gọi `finalize_weights`, luật giữ hay xóa `best.pt` kiểm theo thứ tự và không xét metric: có exception, loss NaN, bị ngắt, không có checkpoint, hoặc nạp lại `best.pt` mà val_acc tính lại không khớp `history.csv` thì xóa weights (DISCARD) nhưng giữ log; còn lại thì giữ (KEEP). Mã lý do được ghi vào `experiments.csv`, nhờ đó run lỗi vẫn để lại dấu vết để phân tích.

`src/run_experiment.py` ghép các phần trên: đặt seed, ghi `config.json` gồm siêu tham số, mã thành viên (`--member`), hostname, GPU, commit git, SHA-256 dữ liệu, rồi gọi `fit` và thêm một dòng vào `tracking/experiments.csv`. Lệnh mẫu: `python src/run_experiment.py --config configs/E1_arch.json --member TV2 --only large_bn --seeds 0 1 2`.

## 5.5. Đánh giá và tổng hợp kết quả (TV3)

`predict(model, loader, tta)` trong `src/engine/evaluate.py` trả nhãn thật và xác suất softmax, hỗ trợ ba chế độ suy luận. Chế độ `flip` lấy trung bình xác suất của ảnh gốc và ảnh lật ngang:

```python
if tta_mode == "flip":
    p1 = torch.softmax(model(x), dim=1)
    p2 = torch.softmax(model(torch.flip(x, dims=[-1])), dim=1)
    return 0.5 * (p1 + p2)
```

`metrics` tính accuracy, macro-F1, precision và recall từng lớp và ma trận nhầm lẫn (thô và chuẩn hóa theo hàng) bằng numpy. `src/eval_e7.py` chạy ba chế độ TTA trên từng seed và trên ensemble (trung bình xác suất của các seed rồi lấy argmax), ghi bảng `report/tables/E7_tta_ensemble.csv` và danh sách ca đúng/sai `results/E7/e7_gradcam_cases.csv` cho Chương 6. Hàm `eval_test` trong `run_experiment.py` kiểm tra file khóa `results/E7/TEST_DONE.json` ngay đầu hàm; nếu đã có thì từ chối chạy, nếu chưa có thì đánh giá PrivateTest, ghi dự đoán, chỉ số từng lớp và tạo file khóa kèm SHA-256 của dữ liệu và của ba file `best.pt`.

`src/make_tables.py` đọc `tracking/experiments.csv`, nhóm các run theo biến thể, tính mean ± std (độ lệch chuẩn mẫu) và so với baseline: chênh lệch nhỏ hơn độ lệch chuẩn của baseline (0.0075) được ghi là "< std baseline", tức chưa phân biệt được. `src/visual/plots.py` vẽ đường cong, ma trận nhầm lẫn chuẩn hóa theo hàng, histogram tỷ lệ sai, lưới ca đúng/sai; mỗi hình kèm CSV dữ liệu và được `register_figure` ghi vào `tracking/figures_manifest.csv` với lệnh sinh, run liên quan và SHA-256.

## 5.6. Trực quan hóa (TV4)

`src/visual/filters.py` cài đặt bốn cách trực quan hóa ở Chương 6: `plot_weights` (trọng số lớp conv đầu và conv cuối), `feature_maps` (16 kênh kích hoạt mạnh nhất của lớp conv cuối), `activation_max` (gradient ascent trên ảnh đầu vào) và `compute_grad_cam_map` (Grad-CAM). Lõi của Grad-CAM lấy kích hoạt và gradient tại lớp conv cuối bằng hook, lấy trung bình gradient theo không gian làm trọng số của từng kênh:

```python
weights = grad.mean(dim=(1, 2), keepdim=True)                 # alpha_k
cam = F.relu((weights * act).sum(dim=0, keepdim=True)).unsqueeze(0)
cam = F.interpolate(cam, size=(48, 48), mode="bilinear", align_corners=False)
```

`src/visualize_filters.py` là lệnh dòng lệnh nạp `best.pt`, đọc mẫu PublicTest và gọi các hàm trên theo `--method`. `src/visual/architecture_diagram.py` vẽ sơ đồ kiến trúc từ mô hình thật (Hình 3.1 đến 3.3).

## 5.7. Kiểm thử tự động và tái lập

Thư mục `tests/` có 21 file `test_*.py` chạy bằng `pytest`, kiểm tra shape và số tham số mô hình, thứ tự nhãn, việc PrivateTest không lọt vào loader huấn luyện, luật giữ/xóa weights với run lỗi, run bị ngắt và run bình thường, công thức mean ± std, khóa `TEST_DONE.json` và các hàm trực quan hóa. Ở lần QA tái lập toàn repo trên bản clone sạch (T27, commit `208e0e3`), toàn bộ 127 test đạt, 10/10 lệnh trong README chạy thành công và notebook chạy lại từ đầu không lỗi (`report/QA_T27_report.md`). Notebook `notebooks/BTL02_reproducible.ipynb` gọi đúng các hàm trong `src/`, không chép lại logic, và ô cuối dùng `assert` để kiểm số dòng epoch trong `run.log` bằng số dòng `history.csv`.

Các script trong `tracking/` không tham gia huấn luyện nhưng giữ cho phân công và tiến độ tự cập nhật: `git_activity.py` sinh `activity_log.csv` từ lịch sử git, `task_status.py` suy trạng thái nhiệm vụ từ commit và nhật ký kiểm thử. Workflow GitHub Actions chạy hai script này sau mỗi lần push.
