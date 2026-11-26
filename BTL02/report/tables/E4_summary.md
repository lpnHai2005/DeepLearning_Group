# Bảng 4.4: Kết quả thực nghiệm E4 – Khảo sát data augmentation và điều chuẩn

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E4 | T15 | weight_decay_0 | 1 | 1 | 5867847 | 0.681806 |  | 0.6818 | 0.638066 |  | 0.6381 | 0.9103 | 0.0297 | RTX 3060 | 24.8 | +0.0003 (< std baseline) |
| E4 | T15 | weight_decay_1e-3 | 1 | 1 | 5867847 | 0.684035 |  | 0.6840 | 0.651002 |  | 0.6510 | 0.9 | 0.0597 | RTX 3060 | 24.5 | +0.0025 (< std baseline) |
| E4 | T15 | label_smoothing_0_1 | 1 | 1 | 5867847 | 0.68292 |  | 0.6829 | 0.627969 |  | 0.6280 | 0.8897 | 0.0375 | RTX 3060 | 24.4 | +0.0014 (< std baseline) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).
