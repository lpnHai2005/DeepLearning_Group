# Bảng 4.1: Kết quả thực nghiệm E1 – Khảo sát kiến trúc CNN

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | T12 | small | 1 | 1 | 38901191 | 0.691558 |  | 0.6916 | 0.67314 |  | 0.6731 | 0.9094 | 0.1252 | RTX 4060 | 21.1 | +0.0100 (cao hơn) |
| E1 | T12 | medium | 1 | 1 | 11327303 | 0.69518 |  | 0.6952 | 0.67889 |  | 0.6789 | 0.9162 | 0.1104 | RTX 4060 | 16.8 | +0.0137 (cao hơn) |
| E1 | T12 | medium_01 | 1 | 1 | 12101895 | 0.249373 |  | 0.2494 | 0.057028 |  | 0.0570 | 1.8107 | 0.0019 | RTX 4060 | 19.6 | -0.4322 (thấp hơn) |
| E1 | T12 | large | 1 | 1 | 20406599 | 0.249373 |  | 0.2494 | 0.057028 |  | 0.0570 | 1.8107 | 0.0019 | RTX 4060 | 24.0 | -0.4322 (thấp hơn) |
| E1 | T12 | baseline_w0.5 | 1 | 1 | 2650279 | 0.670103 |  | 0.6701 | 0.63292 |  | 0.6329 | 0.9147 | 0.0166 | RTX 4060 | 13.9 | -0.0114 (thấp hơn) |
| E1 | T12 | baseline_w2 | 1 | 1 | 14017159 | 0.670103 |  | 0.6701 | 0.582386 |  | 0.5824 | 0.9473 | 0.0347 | RTX 4060 | 25.9 | -0.0114 (thấp hơn) |
| E1 | T12 | baseline_no_bn | 1 | 1 | 5866951 | 0.690722 |  | 0.6907 | 0.663963 |  | 0.6640 | 0.9086 | 0.061 | RTX 4060 | 14.0 | +0.0092 (cao hơn) |
| E1 | T12 | baseline_gap | 1 | 1 | 1280327 | 0.68292 |  | 0.6829 | 0.644917 |  | 0.6449 | 0.9089 | 0.0443 | RTX 4060 | 14.2 | +0.0014 (< std baseline) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).
