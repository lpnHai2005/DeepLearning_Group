# Bảng 4.7: Kết quả thực nghiệm E7 – Mô hình cuối, TTA và ensemble

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E7 | T18 | large_bn_bs64 | 1 | 1 | 20410439 | 0.69713 |  | 0.6971 | 0.677833 |  | 0.6778 | 0.9025 | 0.1868 | RTX 4060 | 26.2 | +0.0156 (cao hơn) |
| E7 | T18 | large_bn_bs64_leaky | 1 | 1 | 20410439 | 0.691 |  | 0.6910 | 0.672465 |  | 0.6725 | 0.9031 | 0.1589 | RTX 4060 | 26.2 | +0.0095 (cao hơn) |
| E7 | T18 | large_bn_bs64_leaky_wrs | 1 | 1 | 20410439 | 0.693508 |  | 0.6935 | 0.687361 |  | 0.6874 | 0.9367 | 0.2053 | RTX 4060 | 26.2 | +0.0120 (cao hơn) |
| E7 | T18 | baseline_best | 1 | 1 | 5867847 | 0.70131 |  | 0.7013 | 0.678364 |  | 0.6784 | 0.8693 | 0.0931 | RTX 4060 | 14.9 | +0.0198 (cao hơn) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).
