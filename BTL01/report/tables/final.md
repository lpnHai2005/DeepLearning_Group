<!-- sinh tự động bởi src/make_tables.py từ results/final/summary.csv; không sửa tay -->
| value | params | seeds | train_mse | val_mse | test_mse | val_mae | test_mae | test_rmse | test_r2 | best_epoch | epochs_run | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 41 | 3 | 2.21e-05 ± 6.1e-06 | 1.89e-05 ± 5.9e-06 | 2.28e-05 ± 1.4e-05 | 3.35e-03 ± 5.6e-04 | 3.53e-03 ± 1.1e-03 | 4.63e-03 ± 1.4e-03 | 0.99030 ± 0.00609 | 1653 ± 411 | 1792 ± 346 | 36.3 ± 7.2 |
| 93 | 373 | 3 | 1.11e-06 ± 1.6e-07 | 1.46e-06 ± 4.3e-07 | 1.41e-06 ± 2.0e-07 | 9.06e-04 ± 9.8e-05 | 8.83e-04 ± 2.0e-05 | 1.18e-03 ± 8.7e-05 | 0.99940 ± 0.00009 | 1266 ± 445 | 1466 ± 445 | 29.6 ± 9.1 |
| 10-10-10 | 261 | 3 | 6.18e-08 ± 1.2e-08 | 6.02e-08 ± 2.1e-08 | 7.13e-08 ± 2.4e-08 | 1.80e-04 ± 2.6e-05 | 1.94e-04 ± 3.4e-05 | 2.64e-04 ± 4.7e-05 | 0.99997 ± 0.00001 | 1667 ± 208 | 1835 ± 156 | 47.2 ± 4.2 |

Nguồn: `results/final/summary.csv` (mỗi run 1 dòng; log: `results/final/run.log`). Giá trị: trung bình ± độ lệch chuẩn (ddof=1) qua các seed; MSE/MAE trên thang y gốc.
