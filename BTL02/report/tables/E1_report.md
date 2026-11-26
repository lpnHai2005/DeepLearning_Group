**Bảng 4.1: Kết quả thực nghiệm E1 – Khảo sát kiến trúc CNN**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| small | 3 | 0.6934 ± 0.0027 | 0.6773 ± 0.0045 | 0.9143 | 38,901,191 | +0.1227 | — (nhiều GPU) | +0.0119 (cao hơn) |
| medium | 3 | 0.6919 ± 0.0031 | 0.6730 ± 0.0059 | 0.9119 | 11,327,303 | +0.1010 | — (nhiều GPU) | +0.0104 (cao hơn) |
| medium_01 | 1 | 0.2494 | 0.0570 | 1.8107 | 12,101,895 | +0.0019 | RTX 4060 (19.6s) | -0.4322 (thấp hơn) |
| large | 1 | 0.2494 | 0.0570 | 1.8107 | 20,406,599 | +0.0019 | RTX 4060 (24.0s) | -0.4322 (thấp hơn) |
| baseline_w0.5 | 3 | 0.6747 ± 0.0049 | 0.6347 ± 0.0103 | 0.9042 | 2,650,279 | +0.0162 | RTX 4060 (13.7s) | -0.0068 (< std baseline) |
| baseline_w2 | 3 | 0.6654 ± 0.0070 | 0.5717 ± 0.0196 | 0.9663 | 14,017,159 | +0.0234 | RTX 4060 (26.4s) | -0.0162 (thấp hơn) |
| baseline_no_bn | 3 | 0.6941 ± 0.0033 | 0.6682 ± 0.0044 | 0.9059 | 5,866,951 | +0.0588 | — (nhiều GPU) | +0.0125 (cao hơn) |
| baseline_gap | 3 | 0.6839 ± 0.0013 | 0.6478 ± 0.0040 | 0.9077 | 1,280,327 | +0.0463 | — (nhiều GPU) | +0.0024 (< std baseline) |
| medium_01_adam1e-4 | 1 | 0.6392 | 0.5993 | 0.9729 | 12,101,895 | +0.0145 | RTX 4060 (20.5s) | -0.0424 (thấp hơn) |
| medium_01_bn | 1 | 0.6913 | 0.6686 | 0.9204 | 12,103,623 | +0.1669 | RTX 4060 (21.6s) | +0.0098 (cao hơn) |
| large_bn | 3 | 0.7060 ± 0.0023 | 0.6922 ± 0.0027 | 0.9084 | 20,410,439 | +0.1952 | RTX 4060 (25.6s) | +0.0244 (cao hơn) |
| medium_01_glorot_adam1e-4 | 1 | 0.6721 | 0.6383 | 0.9064 | 12,101,895 | +0.0545 | RTX 4060 (20.7s) | -0.0095 (thấp hơn) |
| large_glorot_adam1e-4 | 1 | 0.6236 | 0.5545 | 1.0056 | 20,406,599 | +0.0313 | RTX 4060 (24.5s) | -0.0580 (thấp hơn) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
