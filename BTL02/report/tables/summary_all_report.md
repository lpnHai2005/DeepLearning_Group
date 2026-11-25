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
| E1 – baseline_no_bn | 1 | 0.6907 | 0.6640 | 0.9086 | 5,866,951 | +0.0610 | RTX 4060 (14.0s) | +0.0092 (cao hơn) |
| E1 – baseline_gap | 1 | 0.6829 | 0.6449 | 0.9089 | 1,280,327 | +0.0443 | RTX 4060 (14.2s) | +0.0014 (< std baseline) |
| E3 – adamw_1e-3 | 1 | 0.6723 | 0.5878 | 0.9626 | 5,867,847 | +0.0268 | RTX 3060 (35.9s) | -0.0092 (thấp hơn) |
| E3 – adamw_3e-4 | 1 | 0.6704 | 0.5630 | 0.9176 | 5,867,847 | +0.0065 | RTX 3060 (49.7s) | -0.0111 (thấp hơn) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
