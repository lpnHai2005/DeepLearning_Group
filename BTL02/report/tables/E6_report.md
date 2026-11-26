**Bảng 4.6: Kết quả thực nghiệm E6 – Transfer learning (VGG16 và ResNet18)**

| Cấu hình | Seed | Val acc (mean ± std) | Val macro-F1 | Best val loss | Số tham số | Train–Val* | GPU (s/ep) | So với baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline (tham chiếu) | 3 | 0.6815 ± 0.0075 | 0.6408 ± 0.0195 | 0.9109 | 5,867,847 | +0.0397 | — (nhiều GPU) | tham chiếu |
| vgg16_48_ft [E6-003] | 0 | — | — | — | — | — | — | — |
| vgg16_48_ft [E6-004] | 1 | 0.6938 | 0.6812 | 0.9373 | 134,289,223 | +0.2571 | RTX 3050 (60.5s) | +0.0123 (cao hơn) |
| vgg16_48_ft [E6-005] | 0 | — | — | — | — | — | — | — |
| resnet18_96_scratch | 1 | 0.6771 | 0.6672 | 1.0006 | 11,180,103 | +0.2437 | RTX 3050 (55.5s) | -0.0045 (< std baseline) |
| resnet18_96_ft | 1 | 0.6874 | 0.6698 | 0.9718 | 11,180,103 | +0.2983 | RTX 3060 (79.4s) | +0.0059 (< std baseline) |
| vgg16_96_ft | 1 | 0.6977 | 0.6818 | 0.9220 | 134,289,223 | +0.1876 | RTX 4060 (71.3s) | +0.0162 (cao hơn) |

_* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau)._
