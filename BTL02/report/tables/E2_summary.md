# Bảng 4.2: Kết quả thực nghiệm E2 – Khảo sát hàm kích hoạt

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2 | T13 | relu | 1 | 1 | 5867847 | 0.686264 |  | 0.6863 | 0.654228 |  | 0.6542 | 0.9033 | 0.05 | RTX 3050 | 34.0 | +0.0047 (< std baseline) |
| E2 | T13 | leaky_relu | 1 | 1 | 5867847 | 0.693508 |  | 0.6935 | 0.666391 |  | 0.6664 | 0.9016 | 0.0661 | RTX 3050 | 36.8 | +0.0120 (cao hơn) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).
