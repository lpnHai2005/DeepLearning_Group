# Bảng 4.5: Kết quả thực nghiệm E5 – Xử lý mất cân bằng lớp

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E5 | T16 | weighted_random_sampler | 1 | 1 | 5867847 | 0.682363 |  | 0.6824 | 0.671689 |  | 0.6717 | 0.9361 | 0.0821 | RTX 3060 | 24.5 | +0.0008 (< std baseline) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).
