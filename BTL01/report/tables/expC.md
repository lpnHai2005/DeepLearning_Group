<!-- sinh tự động bởi src/make_tables.py từ results/expC/summary.csv; không sửa tay -->
| value | params | seeds | train_mse | val_mse | test_mse | val_mae | test_mae | test_rmse | test_r2 | best_epoch | epochs_run | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 41 | 3 | 2.21e-05 ± 6.1e-06 | 1.89e-05 ± 5.9e-06 | 2.28e-05 ± 1.4e-05 | 3.35e-03 ± 5.6e-04 | 3.53e-03 ± 1.1e-03 | 4.63e-03 ± 1.4e-03 | 0.99030 ± 0.00609 | 1653 ± 411 | 1792 ± 346 | 36.3 ± 7.2 |
| 10-10 | 151 | 3 | 1.17e-07 ± 3.4e-08 | 1.05e-07 ± 2.7e-08 | 1.07e-07 ± 3.9e-08 | 2.35e-04 ± 3.0e-05 | 2.27e-04 ± 2.1e-05 | 3.24e-04 ± 5.8e-05 | 0.99995 ± 0.00002 | 1653 ± 315 | 1806 ± 255 | 41.8 ± 4.7 |
| 10-10-10 | 261 | 3 | 6.18e-08 ± 1.2e-08 | 6.02e-08 ± 2.1e-08 | 7.13e-08 ± 2.4e-08 | 1.80e-04 ± 2.6e-05 | 1.94e-04 ± 3.4e-05 | 2.64e-04 ± 4.7e-05 | 0.99997 ± 0.00001 | 1667 ± 208 | 1835 ± 156 | 47.2 ± 4.2 |
| 10-10-10-10 | 371 | 3 | 9.09e-08 ± 7.5e-08 | 9.21e-08 ± 8.2e-08 | 7.30e-08 ± 5.9e-08 | 2.25e-04 ± 1.1e-04 | 1.92e-04 ± 9.0e-05 | 2.56e-04 ± 1.1e-04 | 0.99997 ± 0.00002 | 1298 ± 442 | 1498 ± 442 | 45.5 ± 12.9 |

Nguồn: `results/expC/summary.csv` (mỗi run 1 dòng; log: `results/expC/run.log`, `results/expC/posthoc_test.log`). Giá trị: trung bình ± độ lệch chuẩn (ddof=1) qua các seed; MSE/MAE trên thang y gốc.
