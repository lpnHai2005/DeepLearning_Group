**Bảng 4.8: Tổng hợp toàn bộ kết quả thực nghiệm (Baseline và E1–E7)**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| E1 – small | 1 | 0.6916 | 0.6731 | 0.9094 | 38,901,191 | +0.1252 | RTX 4060 (21.1s) | +0.0100 (cao hơn) |
| E1 – medium | 1 | 0.6952 | 0.6789 | 0.9162 | 11,327,303 | +0.1104 | RTX 4060 (16.8s) | +0.0137 (cao hơn) |
| E1 – medium_01 | 1 | 0.2494 | 0.0570 | 1.8107 | 12,101,895 | +0.0019 | RTX 4060 (19.6s) | -0.4322 (thấp hơn) |
| E1 – large | 1 | 0.2494 | 0.0570 | 1.8107 | 20,406,599 | +0.0019 | RTX 4060 (24.0s) | -0.4322 (thấp hơn) |
| E1 – baseline_w0.5 | 1 | 0.6701 | 0.6329 | 0.9147 | 2,650,279 | +0.0166 | RTX 4060 (13.9s) | -0.0114 (thấp hơn) |
| E1 – baseline_w2 | 1 | 0.6701 | 0.5824 | 0.9473 | 14,017,159 | +0.0347 | RTX 4060 (25.9s) | -0.0114 (thấp hơn) |
| E1 – baseline_no_bn | 3 | 0.6941 ± 0.0033 | 0.6682 ± 0.0044 | 0.9059 | 5,866,951 | +0.0588 | — (nhiều GPU) | +0.0125 (cao hơn) |
| E1 – baseline_gap | 3 | 0.6839 ± 0.0013 | 0.6478 ± 0.0040 | 0.9077 | 1,280,327 | +0.0463 | — (nhiều GPU) | +0.0024 (< std baseline) |
| E3 – adamw_1e-3 | 1 | 0.6723 | 0.5878 | 0.9626 | 5,867,847 | +0.0268 | RTX 3060 (35.9s) | -0.0092 (thấp hơn) |
| E3 – adamw_3e-4 | 1 | 0.6704 | 0.5630 | 0.9176 | 5,867,847 | +0.0065 | RTX 3060 (49.7s) | -0.0111 (thấp hơn) |
| E3 – rmsprop_1e-3 | 1 | 0.6857 | 0.6620 | 0.8809 | 5,867,847 | +0.0373 | RTX 3060 (25.1s) | +0.0042 (< std baseline) |
| E3 – rmsprop_3e-4 | 1 | 0.6810 | 0.6332 | 0.9014 | 5,867,847 | +0.0292 | RTX 3060 (25.0s) | -0.0006 (< std baseline) |
| E3 – sched_constant | 1 | 0.6760 | 0.6376 | 0.9138 | 5,867,847 | +0.0334 | RTX 3060 (24.7s) | -0.0056 (< std baseline) |
| E3 – sched_step | 1 | 0.6648 | 0.5919 | 0.9208 | 5,867,847 | +0.0077 | RTX 3060 (25.4s) | -0.0167 (thấp hơn) |
| E3 – sched_cosine | 1 | 0.6771 | 0.6181 | 0.9126 | 5,867,847 | +0.0224 | RTX 3060 (26.7s) | -0.0045 (< std baseline) |
| E3 – sched_onecycle | 1 | 0.6863 | 0.6584 | 0.8997 | 5,867,847 | +0.0423 | RTX 3060 (25.1s) | +0.0047 (< std baseline) |
| E3 – bs_64 | 1 | 0.6924 | 0.6642 | 0.9058 | 5,867,847 | +0.0774 | RTX 3060 (25.5s) | +0.0109 (cao hơn) |
| E3 – bs_256 | 1 | 0.6726 | 0.6298 | 0.9169 | 5,867,847 | +0.0176 | RTX 3060 (24.6s) | -0.0089 (thấp hơn) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
