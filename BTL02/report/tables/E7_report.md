**Bảng 4.7: Kết quả thực nghiệm E7 – Mô hình cuối, TTA và ensemble**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| large_bn_bs64 | 1 | 0.6971 | 0.6778 | 0.9025 | 20,410,439 | +0.1868 | RTX 4060 (26.2s) | +0.0156 (cao hơn) |
| large_bn_bs64_leaky | 1 | 0.6910 | 0.6725 | 0.9031 | 20,410,439 | +0.1589 | RTX 4060 (26.2s) | +0.0095 (cao hơn) |
| large_bn_bs64_leaky_wrs | 1 | 0.6935 | 0.6874 | 0.9367 | 20,410,439 | +0.2053 | RTX 4060 (26.2s) | +0.0120 (cao hơn) |
| baseline_best | 1 | 0.7013 | 0.6784 | 0.8693 | 5,867,847 | +0.0931 | RTX 4060 (14.9s) | +0.0198 (cao hơn) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
