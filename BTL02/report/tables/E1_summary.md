# Bảng 4.1: Kết quả thực nghiệm E1 – Khảo sát kiến trúc CNN

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | T12 | small | 4 | 3 | 38901191 | 0.693415 | 0.002749 | 0.6934 ± 0.0027 | 0.677265 | 0.004478 | 0.6773 ± 0.0045 | 0.9143 | 0.1227 | nhiều GPU (RTX 3050/RTX 4060) |  | +0.0119 (cao hơn) |
| E1 | T12 | medium | 3 | 3 | 11327303 | 0.691929 | 0.003082 | 0.6919 ± 0.0031 | 0.672983 | 0.005923 | 0.6730 ± 0.0059 | 0.9119 | 0.101 | nhiều GPU (RTX 3050/RTX 4060) |  | +0.0104 (cao hơn) |
| E1 | T12 | medium_01 | 1 | 1 | 12101895 | 0.249373 |  | 0.2494 | 0.057028 |  | 0.0570 | 1.8107 | 0.0019 | RTX 4060 | 19.6 | -0.4322 (thấp hơn) |
| E1 | T12 | large | 1 | 1 | 20406599 | 0.249373 |  | 0.2494 | 0.057028 |  | 0.0570 | 1.8107 | 0.0019 | RTX 4060 | 24.0 | -0.4322 (thấp hơn) |
| E1 | T12 | baseline_w0.5 | 3 | 3 | 2650279 | 0.674747 | 0.004893 | 0.6747 ± 0.0049 | 0.634663 | 0.010277 | 0.6347 ± 0.0103 | 0.9042 | 0.0162 | RTX 4060 | 13.7 | -0.0068 (< std baseline) |
| E1 | T12 | baseline_w2 | 3 | 3 | 14017159 | 0.665366 | 0.007032 | 0.6654 ± 0.0070 | 0.571711 | 0.01957 | 0.5717 ± 0.0196 | 0.9663 | 0.0234 | RTX 4060 | 26.4 | -0.0162 (thấp hơn) |
| E1 | T12 | baseline_no_bn | 3 | 3 | 5866951 | 0.694065 | 0.003344 | 0.6941 ± 0.0033 | 0.668242 | 0.004356 | 0.6682 ± 0.0044 | 0.9059 | 0.0588 | nhiều GPU (RTX 3050/RTX 4060) |  | +0.0125 (cao hơn) |
| E1 | T12 | baseline_gap | 3 | 3 | 1280327 | 0.683942 | 0.001317 | 0.6839 ± 0.0013 | 0.647752 | 0.003999 | 0.6478 ± 0.0040 | 0.9077 | 0.0463 | nhiều GPU (RTX 3050/RTX 4060) |  | +0.0024 (< std baseline) |
| E1 | T12 | medium_01_adam1e-4 | 1 | 1 | 12101895 | 0.639175 |  | 0.6392 | 0.599278 |  | 0.5993 | 0.9729 | 0.0145 | RTX 4060 | 20.5 | -0.0424 (thấp hơn) |
| E1 | T12 | large_adam1e-4 | 1 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| E1 | T12 | medium_01_bn | 2 | 1 | 12103623 | 0.691279 |  | 0.6913 | 0.668633 |  | 0.6686 | 0.9204 | 0.1669 | RTX 4060 | 21.6 | +0.0098 (cao hơn) |
| E1 | T12 | large_bn | 1 | 1 | 20410439 | 0.708554 |  | 0.7086 | 0.695289 |  | 0.6953 | 0.9164 | 0.1817 | RTX 4060 | 25.9 | +0.0270 (cao hơn) |
| E1 | T12 | medium_01_glorot_adam1e-4 | 1 | 1 | 12101895 | 0.672053 |  | 0.6721 | 0.638302 |  | 0.6383 | 0.9064 | 0.0545 | RTX 4060 | 20.7 | -0.0095 (thấp hơn) |
| E1 | T12 | large_glorot_adam1e-4 | 1 | 1 | 20406599 | 0.623572 |  | 0.6236 | 0.554473 |  | 0.5545 | 1.0056 | 0.0313 | RTX 4060 | 24.5 | -0.0580 (thấp hơn) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).
