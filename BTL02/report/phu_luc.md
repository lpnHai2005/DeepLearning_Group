# Phụ lục

## Phụ lục A. Lệnh tái lập

Mọi lệnh chạy từ thư mục `BTL02/` sau khi cài môi trường theo `README.md` và đặt `data/fer2013.csv` (SHA-256 `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`). Tham số `--member` nhận mã thành viên trong `tracking/members.csv`.

### A.1. Dữ liệu (Chương 2)

```bash
# Kiểm tra schema, số split và SHA-256
python -X utf8 -m src.data.check_fer2013 --csv data/fer2013.csv --output results/data_check.json --download-date 2026-11-24
# Năm hình EDA và CSV dữ liệu
python -X utf8 -m src.data.eda --csv data/fer2013.csv --output-dir figures --seed 0
```

### A.2. Huấn luyện các nhóm thực nghiệm

```bash
# Baseline 3 seed
python src/run_experiment.py --config configs/baseline.json --member TVx --seeds 0 1 2
# Một biến thể của một nhóm, ví dụ large_bn 3 seed (E1)
python src/run_experiment.py --config configs/E1_arch.json --member TVx --only large_bn --seeds 0 1 2
# Các nhóm còn lại: configs/E2_activation.json, E3_optim.json, E4_aug_reg.json, E5_imbalance.json, E6_transfer.json
python src/run_experiment.py --config configs/E2_activation.json --member TVx
# Bước 1 của E7 (sàng lọc cấu hình ghép, seed 0)
python src/run_experiment.py --config configs/E7_final.json --member TVx --only large_bn_bs64 large_bn_bs64_leaky large_bn_bs64_leaky_wrs baseline_best --seeds 0
```

### A.3. Bảng, hình và phân tích (Chương 2 đến 4)

```bash
# Bảng tổng hợp mọi nhóm và tự điền dòng Kết quả trong nhật ký thực nghiệm
python src/make_tables.py --update-log report/NHAT_KY_THUC_NGHIEM.md
# Mốc baseline 3 seed
python src/baseline_summary.py
# Phân tích VGG16 ở 48 px (E6)
python src/e6_analysis.py
# Sơ đồ kiến trúc và pipeline
python -m src.visual.architecture_diagram
# Chuẩn gradient từng lớp (E2)
python src/visual/grad_norms.py --group E2 --epoch 1
# Khoảng cách train/val (E4) và recall theo lớp (E5)
python src/visual/e4_overfit.py
python src/visual/e5_recall.py --run-dir results/E5/E5-001_s0
python src/visual/e5_recall.py --run-dir results/E5/E5-003_s0 --no-figure
python src/visual/e5_recall.py --run-dir results/E5/E5-002_s0 --no-figure
# TTA và ensemble 3 seed trên PublicTest (E7)
python src/eval_e7.py --run-dirs results/E1/E1-014_s0 results/E1/E1-014_s1 results/E1/E1-014_s2 --tta none flip ten_crop
```

### A.4. Đánh giá PrivateTest (chỉ chạy một lần)

```bash
python src/run_experiment.py --eval-test --member TV3 --config configs/E7_final.json --run-dirs results/E1/E1-014_s0 results/E1/E1-014_s1 results/E1/E1-014_s2 --tta flip
```

Lệnh này đã chạy ngày 2026-11-27 và tạo `results/E7/TEST_DONE.json`; mọi lần chạy lại đều bị từ chối.

### A.5. Trực quan hóa bộ lọc (Chương 6)

```bash
python src/visualize_filters.py --run-dir results/E1/E1-014_s0 --data-csv data/fer2013.csv --sample-idx 1623 --cases-csv results/E7/e7_gradcam_cases.csv --method weights --layer first
python src/visualize_filters.py --run-dir results/E1/E1-014_s0 --data-csv data/fer2013.csv --sample-idx 1623 --cases-csv results/E7/e7_gradcam_cases.csv --method weights --layer last
python src/visualize_filters.py --run-dir results/E1/E1-014_s0 --data-csv data/fer2013.csv --sample-idx 1623 --cases-csv results/E7/e7_gradcam_cases.csv --method fmaps
python src/visualize_filters.py --run-dir results/E1/E1-014_s0 --data-csv data/fer2013.csv --sample-idx 1623 --cases-csv results/E7/e7_gradcam_cases.csv --method actmax --steps 100
python src/visualize_filters.py --run-dir results/E1/E1-014_s0 --data-csv data/fer2013.csv --sample-idx 1623 --cases-csv results/E7/e7_gradcam_cases.csv --method gradcam
```

### A.6. Số liệu Chương 1, 6, 7 và kiểm thử tự động

```bash
# Bảng đóng góp, thống kê bộ lọc, thống kê run; tự điền Bảng 1.2
python src/t25_report_stats.py
# Kiểm thử tự động toàn kho mã nguồn
python -m pytest -q
```

## Phụ lục B. Các file và links google sheets nộp kèm và nơi truy vết

**Bảng B.1.** Các file truy vết đi kèm báo cáo.

| File | Nội dung |
|---|---|
| `report/NHAT_KY_THUC_NGHIEM.md` | Nhật ký thực nghiệm: giả thuyết, kết quả, nhận xét và quyết định cho từng run |
| `tracking/experiments.csv` | Mỗi run một dòng: người chạy, máy, GPU, siêu tham số, kết quả, quyết định giữ/xóa weights |
| `tracking/tasks.csv` | 28 nhiệm vụ với vai R, C, T và trạng thái tự suy |
| `tracking/test_log.csv` | Nhật ký kiểm thử độc lập, kết quả Pass hoặc Fail |
| `tracking/activity_log.csv` | Nhật ký hoạt động sinh từ lịch sử git |
| `tracking/figures_manifest.csv` | Mỗi hình: script, lệnh, run liên quan, SHA-256 |
| `report/tables/*.csv` | Bảng tổng hợp sinh từ `experiments.csv` |
| `results/<nhóm>/<exp_id>/` | `config.json`, `run.log`, `history.csv`, `metrics.json` của từng run |
| `results/E7/TEST_DONE.json` | Khóa và kết quả đánh giá PrivateTest duy nhất |
| `report/QA_T27_report.md` | Biên bản QA tái lập toàn repo trên bản clone sạch |

Link sheets: https://docs.google.com/spreadsheets/d/1mgwx1TGTL0fQewJ3Qto-K_HcZEUbgN4_P7Cy2-Z-aFk/edit?usp=sharing
