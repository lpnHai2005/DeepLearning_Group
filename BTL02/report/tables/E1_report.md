**Bảng 4.1: Kết quả thực nghiệm E1 – Khảo sát kiến trúc CNN**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| small | 1 | 0.6916 | 0.6731 | 0.9094 | 38,901,191 | +0.1252 | RTX 4060 (21.1s) | +0.0100 (cao hơn) |
| medium | 1 | 0.6952 | 0.6789 | 0.9162 | 11,327,303 | +0.1104 | RTX 4060 (16.8s) | +0.0137 (cao hơn) |
| medium_01 | 1 | 0.2494 | 0.0570 | 1.8107 | 12,101,895 | +0.0019 | RTX 4060 (19.6s) | -0.4322 (thấp hơn) |
| large | 1 | 0.2494 | 0.0570 | 1.8107 | 20,406,599 | +0.0019 | RTX 4060 (24.0s) | -0.4322 (thấp hơn) |
| baseline_w0.5 | 1 | 0.6701 | 0.6329 | 0.9147 | 2,650,279 | +0.0166 | RTX 4060 (13.9s) | -0.0114 (thấp hơn) |
| baseline_w2 | 1 | 0.6701 | 0.5824 | 0.9473 | 14,017,159 | +0.0347 | RTX 4060 (25.9s) | -0.0114 (thấp hơn) |
| baseline_no_bn | 3 | 0.6941 ± 0.0033 | 0.6682 ± 0.0044 | 0.9059 | 5,866,951 | +0.0588 | — (nhiều GPU) | +0.0125 (cao hơn) |
| baseline_gap | 3 | 0.6839 ± 0.0013 | 0.6478 ± 0.0040 | 0.9077 | 1,280,327 | +0.0463 | — (nhiều GPU) | +0.0024 (< std baseline) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
