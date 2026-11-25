**Bảng 4.3: Kết quả thực nghiệm E3 – Khảo sát thuật toán tối ưu, lịch learning rate và batch size**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| adamw_1e-3 | 1 | 0.6723 | 0.5878 | 0.9626 | 5,867,847 | +0.0268 | RTX 3060 (35.9s) | -0.0092 (thấp hơn) |
| adamw_3e-4 | 1 | 0.6704 | 0.5630 | 0.9176 | 5,867,847 | +0.0065 | RTX 3060 (49.7s) | -0.0111 (thấp hơn) |
| rmsprop_1e-3 | 1 | 0.6857 | 0.6620 | 0.8809 | 5,867,847 | +0.0373 | RTX 3060 (25.1s) | +0.0042 (< std baseline) |
| rmsprop_3e-4 | 1 | 0.6810 | 0.6332 | 0.9014 | 5,867,847 | +0.0292 | RTX 3060 (25.0s) | -0.0006 (< std baseline) |
| sched_constant | 1 | 0.6760 | 0.6376 | 0.9138 | 5,867,847 | +0.0334 | RTX 3060 (24.7s) | -0.0056 (< std baseline) |
| sched_step | 1 | 0.6648 | 0.5919 | 0.9208 | 5,867,847 | +0.0077 | RTX 3060 (25.4s) | -0.0167 (thấp hơn) |
| sched_cosine | 1 | 0.6771 | 0.6181 | 0.9126 | 5,867,847 | +0.0224 | RTX 3060 (26.7s) | -0.0045 (< std baseline) |
| sched_onecycle | 1 | 0.6863 | 0.6584 | 0.8997 | 5,867,847 | +0.0423 | RTX 3060 (25.1s) | +0.0047 (< std baseline) |
| bs_64 | 1 | 0.6924 | 0.6642 | 0.9058 | 5,867,847 | +0.0774 | RTX 3060 (25.5s) | +0.0109 (cao hơn) |
| bs_256 | 1 | 0.6726 | 0.6298 | 0.9169 | 5,867,847 | +0.0176 | RTX 3060 (24.6s) | -0.0089 (thấp hơn) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
