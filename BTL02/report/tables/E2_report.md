**Bảng 4.2: Kết quả thực nghiệm E2 – Khảo sát hàm kích hoạt**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| relu | 1 | 0.6863 | 0.6542 | 0.9033 | 5,867,847 | +0.0500 | RTX 3050 (34.0s) | +0.0047 (< std baseline) |
| leaky_relu | 1 | 0.6935 | 0.6664 | 0.9016 | 5,867,847 | +0.0661 | RTX 3050 (36.8s) | +0.0120 (cao hơn) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
