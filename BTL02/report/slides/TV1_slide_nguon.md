# Dò nguồn số liệu: slide phần TV1 (T26)

Sinh tự động bởi `report/slides/build_slides_tv1.py`; mỗi dòng là một số hiển thị trên
`report/slides/Slide_BTL02_TV1.pptx` (cả tiêu đề, bảng, biểu đồ) và vị trí của nó trong file nguồn.
Đường dẫn tính từ thư mục `BTL02/`. Phép tính (÷, −, × 100) ghi rõ trong cột vị trí.

| Slide | Số liệu | Hiển thị | File nguồn | Dòng/cột |
|---|---|---:|---|---|
| 1 | Tỷ lệ Happy/Disgust trên Training | 16,55 | `figures/eda_class_distribution.csv` | count Training/Happy ÷ count Training/Disgust |
| 1 | Tổng số ảnh | 35.887 | `results/data_check.json` | total_rows |
| 1 | Số ảnh Training | 28.709 | `results/data_check.json` | observed_split_counts.Training |
| 1 | Số ảnh PublicTest | 3.589 | `results/data_check.json` | observed_split_counts.PublicTest |
| 1 | Số ảnh PrivateTest | 3.589 | `results/data_check.json` | observed_split_counts.PrivateTest |
| 1 | Trạng thái kiểm tra dữ liệu | PASS | `results/data_check.json` | status |
| 1 | SHA-256 (rút gọn) | 3b8d9617…f871b | `results/data_check.json` | sha256 |
| 1 | Training Angry (%, số ảnh) | 13,92% / 3.995 | `figures/eda_class_distribution.csv` | split=Training, emotion=Angry: percent, count |
| 1 | Training Disgust (%, số ảnh) | 1,52% / 436 | `figures/eda_class_distribution.csv` | split=Training, emotion=Disgust: percent, count |
| 1 | Training Fear (%, số ảnh) | 14,27% / 4.097 | `figures/eda_class_distribution.csv` | split=Training, emotion=Fear: percent, count |
| 1 | Training Happy (%, số ảnh) | 25,13% / 7.215 | `figures/eda_class_distribution.csv` | split=Training, emotion=Happy: percent, count |
| 1 | Training Sad (%, số ảnh) | 16,82% / 4.830 | `figures/eda_class_distribution.csv` | split=Training, emotion=Sad: percent, count |
| 1 | Training Surprise (%, số ảnh) | 11,05% / 3.171 | `figures/eda_class_distribution.csv` | split=Training, emotion=Surprise: percent, count |
| 1 | Training Neutral (%, số ảnh) | 17,29% / 4.965 | `figures/eda_class_distribution.csv` | split=Training, emotion=Neutral: percent, count |
| 1 | PublicTest Disgust (số ảnh) | 56 | `figures/eda_class_distribution.csv` | split=PublicTest, emotion=Disgust: count |
| 1 | Tổng PublicTest | 3.589 | `results/data_check.json` | observed_split_counts.PublicTest |
| 1 | PublicTest Disgust (%) | 1,56% | `figures/eda_class_distribution.csv` | split=PublicTest, emotion=Disgust: percent |
| 2 | Gap tại best epoch aug_flip_affine (điểm %) | 4,72 | `figures/e4_overfit_summary.csv` | variant=aug_flip_affine: acc_gap_at_best × 100 |
| 2 | Gap tại best epoch aug_none (điểm %) | 28,09 | `figures/e4_overfit_summary.csv` | variant=aug_none: acc_gap_at_best × 100 |
| 2 | Gap baseline-002_s0 (điểm %) | 4,72 | `figures/e4_overfit_summary.csv` | exp_id=baseline-002_s0: acc_gap_at_best × 100 |
| 2 | Val acc baseline-002_s0 | 0,6815 | `figures/e4_overfit_summary.csv` | exp_id=baseline-002_s0: best_val_acc |
| 2 | Gap aug_none (điểm %) | 28,09 | `figures/e4_overfit_summary.csv` | variant=aug_none: acc_gap_at_best × 100 |
| 2 | Val acc aug_none | 0,6656 | `report/tables/E4_summary.csv` | variant=aug_none: val_acc_mean |
| 2 | Gap aug_flip (điểm %) | 22,04 | `figures/e4_overfit_summary.csv` | variant=aug_flip: acc_gap_at_best × 100 |
| 2 | Val acc aug_flip | 0,6693 | `report/tables/E4_summary.csv` | variant=aug_flip: val_acc_mean |
| 2 | Gap aug_flip_affine (điểm %) | 4,72 | `figures/e4_overfit_summary.csv` | variant=aug_flip_affine: acc_gap_at_best × 100 |
| 2 | Val acc aug_flip_affine | 0,6863 | `report/tables/E4_summary.csv` | variant=aug_flip_affine: val_acc_mean |
| 2 | Gap aug_random_erasing (điểm %) | 0,49 | `figures/e4_overfit_summary.csv` | variant=aug_random_erasing: acc_gap_at_best × 100 |
| 2 | Val acc aug_random_erasing | 0,6799 | `report/tables/E4_summary.csv` | variant=aug_random_erasing: val_acc_mean |
| 2 | Gap dropout_0 (điểm %) | 10,97 | `figures/e4_overfit_summary.csv` | variant=dropout_0: acc_gap_at_best × 100 |
| 2 | Val acc dropout_0 | 0,6896 | `report/tables/E4_summary.csv` | variant=dropout_0: val_acc_mean |
| 2 | Gap dropout_0_25 (điểm %) | 5,55 | `figures/e4_overfit_summary.csv` | variant=dropout_0_25: acc_gap_at_best × 100 |
| 2 | Val acc dropout_0_25 | 0,6888 | `report/tables/E4_summary.csv` | variant=dropout_0_25: val_acc_mean |
| 2 | Gap dropout_0_5 (điểm %) | −1,42 | `figures/e4_overfit_summary.csv` | variant=dropout_0_5: acc_gap_at_best × 100 |
| 2 | Val acc dropout_0_5 | 0,6662 | `report/tables/E4_summary.csv` | variant=dropout_0_5: val_acc_mean |
| 2 | Gap weight_decay_0 (điểm %) | 2,97 | `figures/e4_overfit_summary.csv` | variant=weight_decay_0: acc_gap_at_best × 100 |
| 2 | Val acc weight_decay_0 | 0,6818 | `report/tables/E4_summary.csv` | variant=weight_decay_0: val_acc_mean |
| 2 | Gap weight_decay_1e-3 (điểm %) | 5,24 | `figures/e4_overfit_summary.csv` | variant=weight_decay_1e-3: acc_gap_at_best × 100 |
| 2 | Val acc weight_decay_1e-3 | 0,6840 | `report/tables/E4_summary.csv` | variant=weight_decay_1e-3: val_acc_mean |
| 2 | Gap label_smoothing_0_1 (điểm %) | 3,32 | `figures/e4_overfit_summary.csv` | variant=label_smoothing_0_1: acc_gap_at_best × 100 |
| 2 | Val acc label_smoothing_0_1 | 0,6829 | `report/tables/E4_summary.csv` | variant=label_smoothing_0_1: val_acc_mean |
| 2 | Macro-F1 aug_random_erasing | 0,6406 | `report/tables/E4_summary.csv` | variant=aug_random_erasing: val_macro_f1_mean |
| 2 | Macro-F1 dropout_0 (cao nhất E4) | 0,6746 | `report/tables/E4_summary.csv` | max val_macro_f1_mean → variant=dropout_0 |
| 3 | Δ macro-F1 WRS so với CE (điểm %) | +3,01 | `report/tables/E5_summary.csv` | (weighted_random_sampler − ce): val_macro_f1_mean × 100 |
| 3 | Val acc ce | 0,6815 | `report/tables/E5_summary.csv` | variant=ce: val_acc_mean |
| 3 | Macro-F1 ce | 0,6416 | `report/tables/E5_summary.csv` | variant=ce: val_macro_f1_mean |
| 3 | Val acc class_weight | 0,6612 | `report/tables/E5_summary.csv` | variant=class_weight: val_acc_mean |
| 3 | Δ acc class_weight vs CE (điểm %) | −2,03 | `report/tables/E5_summary.csv` | (class_weight − ce): val_acc_mean × 100 |
| 3 | Macro-F1 class_weight | 0,6397 | `report/tables/E5_summary.csv` | variant=class_weight: val_macro_f1_mean |
| 3 | Val acc weighted_random_sampler | 0,6824 | `report/tables/E5_summary.csv` | variant=weighted_random_sampler: val_acc_mean |
| 3 | Δ acc weighted_random_sampler vs CE (điểm %) | +0,08 | `report/tables/E5_summary.csv` | (weighted_random_sampler − ce): val_acc_mean × 100 |
| 3 | Macro-F1 weighted_random_sampler | 0,6717 | `report/tables/E5_summary.csv` | variant=weighted_random_sampler: val_macro_f1_mean |
| 3 | Recall Disgust (WRS) | 67,86% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Disgust: recall × 100 |
| 3 | Đúng/support Disgust | 38/56 | `results/E5/E5-001_s0/per_class_recall.csv` | label=Disgust: tp/support |
| 3 | Recall Fear (WRS) | 44,96% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Fear: recall × 100 |
| 3 | Đúng/support Fear | 223/496 | `results/E5/E5-001_s0/per_class_recall.csv` | label=Fear: tp/support |
| 3 | Recall Angry (WRS, biểu đồ) | 65,3% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Angry: recall × 100 |
| 3 | Recall Disgust (WRS, biểu đồ) | 67,9% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Disgust: recall × 100 |
| 3 | Recall Fear (WRS, biểu đồ) | 45,0% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Fear: recall × 100 |
| 3 | Recall Happy (WRS, biểu đồ) | 85,1% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Happy: recall × 100 |
| 3 | Recall Sad (WRS, biểu đồ) | 55,9% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Sad: recall × 100 |
| 3 | Recall Surprise (WRS, biểu đồ) | 85,5% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Surprise: recall × 100 |
| 3 | Recall Neutral (WRS, biểu đồ) | 66,1% | `results/E5/E5-001_s0/per_class_recall.csv` | label=Neutral: recall × 100 |

Tổng: 63 số liệu, 100% có nguồn trong repo.
