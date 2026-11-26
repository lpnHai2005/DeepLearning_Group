**Bảng 4.5: Kết quả thực nghiệm E5 – Xử lý mất cân bằng lớp**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| ce | 1 | 0.6815 | 0.6416 | 0.9150 | 5,867,847 | +0.0431 | RTX 4060 (15.5s) | +0.0000 (< std baseline) |
| class_weight | 1 | 0.6612 | 0.6397 | 0.9414 | 5,867,847 | +0.0041 | RTX 4060 (15.0s) | -0.0203 (thấp hơn) |
| weighted_random_sampler | 1 | 0.6824 | 0.6717 | 0.9361 | 5,867,847 | +0.0821 | RTX 3060 (24.5s) | +0.0008 (< std baseline) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
