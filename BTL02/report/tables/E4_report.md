**Bảng 4.4: Kết quả thực nghiệm E4 – Khảo sát data augmentation và điều chuẩn**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| aug_none | 1 | 0.6656 | 0.6495 | 1.0512 | 5,867,847 | +0.2962 | RTX 3050 (13.5s) | -0.0159 (thấp hơn) |
| aug_flip | 1 | 0.6693 | 0.6470 | 1.0016 | 5,867,847 | +0.2341 | RTX 3050 (13.2s) | -0.0123 (thấp hơn) |
| aug_flip_affine | 1 | 0.6863 | 0.6542 | 0.9033 | 5,867,847 | +0.0500 | RTX 3050 (20.5s) | +0.0047 (< std baseline) |
| aug_random_erasing | 1 | 0.6799 | 0.6406 | 0.9010 | 5,867,847 | +0.0069 | RTX 3050 (22.2s) | -0.0017 (< std baseline) |
| weight_decay_0 | 1 | 0.6818 | 0.6381 | 0.9103 | 5,867,847 | +0.0297 | RTX 3060 (24.8s) | +0.0003 (< std baseline) |
| weight_decay_1e-3 | 1 | 0.6840 | 0.6510 | 0.9000 | 5,867,847 | +0.0597 | RTX 3060 (24.5s) | +0.0025 (< std baseline) |
| label_smoothing_0_1 | 1 | 0.6829 | 0.6280 | 0.8897 | 5,867,847 | +0.0375 | RTX 3060 (24.4s) | +0.0014 (< std baseline) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
