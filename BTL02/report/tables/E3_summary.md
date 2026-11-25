# Bảng 4.3: Kết quả thực nghiệm E3 – Khảo sát thuật toán tối ưu, lịch learning rate và batch size

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3 | T14 | adamw_1e-3 | 1 | 1 | 5867847 | 0.672332 |  | 0.6723 | 0.587766 |  | 0.5878 | 0.9626 | 0.0268 | RTX 3060 | 35.9 | -0.0092 (thấp hơn) |
| E3 | T14 | adamw_3e-4 | 1 | 1 | 5867847 | 0.670382 |  | 0.6704 | 0.56305 |  | 0.5630 | 0.9176 | 0.0065 | RTX 3060 | 49.7 | -0.0111 (thấp hơn) |
| E3 | T14 | sched_step | 1 | 1 | 5867847 | 0.664809 |  | 0.6648 | 0.591861 |  | 0.5919 | 0.9208 | 0.0077 | RTX 3060 | 25.4 | -0.0167 (thấp hơn) |
| E3 | T14 | sched_cosine | 1 | 1 | 5867847 | 0.677069 |  | 0.6771 | 0.61807 |  | 0.6181 | 0.9126 | 0.0224 | RTX 3060 | 26.7 | -0.0045 (< std baseline) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).
