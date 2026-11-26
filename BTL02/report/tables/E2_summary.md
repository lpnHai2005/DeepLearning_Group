# Bảng 4.2: Kết quả thực nghiệm E2 – Khảo sát hàm kích hoạt

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2 | T13 | relu | 1 | 1 | 5867847 | 0.686264 |  | 0.6863 | 0.654228 |  | 0.6542 | 0.9033 | 0.05 | RTX 3050 | 34.0 | +0.0047 (< std baseline) |
| E2 | T13 | leaky_relu | 1 | 1 | 5867847 | 0.693508 |  | 0.6935 | 0.666391 |  | 0.6664 | 0.9016 | 0.0661 | RTX 3050 | 36.8 | +0.0120 (cao hơn) |
| E2 | T13 | elu | 1 | 1 | 5867847 | 0.655336 |  | 0.6553 | 0.61688 |  | 0.6169 | 0.9436 | -0.0113 | RTX 3050 | 18.9 | -0.0262 (thấp hơn) |
| E2 | T13 | gelu | 1 | 1 | 5867847 | 0.691836 |  | 0.6918 | 0.661445 |  | 0.6614 | 0.8807 | 0.0632 | RTX 3050 | 18.9 | +0.0103 (cao hơn) |
| E2 | T13 | tanh | 1 | 1 | 5867847 | 0.641404 |  | 0.6414 | 0.595193 |  | 0.5952 | 0.9688 | -0.0025 | RTX 3060 | 35.7 | -0.0401 (thấp hơn) |
| E2 | T13 | sigmoid | 1 | 1 | 5867847 | 0.451101 |  | 0.4511 | 0.330282 |  | 0.3303 | 1.4288 | -0.0359 | RTX 3060 | 29.3 | -0.2304 (thấp hơn) |
| E2 | T13 | relu_no_bn | 1 | 1 | 5866951 | 0.691279 |  | 0.6913 | 0.667516 |  | 0.6675 | 0.8969 | 0.0602 | RTX 3050 | 18.1 | +0.0098 (cao hơn) |
| E2 | T13 | tanh_no_bn | 1 | 1 | 5866951 | 0.652271 |  | 0.6523 | 0.620114 |  | 0.6201 | 0.9667 | -0.0161 | RTX 3060 | 24.8 | -0.0293 (thấp hơn) |
| E2 | T13 | sigmoid_no_bn | 1 | 1 | 5866951 | 0.249373 |  | 0.2494 | 0.057028 |  | 0.0570 | 1.8108 | 0.0019 | RTX 3060 | 28.4 | -0.4322 (thấp hơn) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).
