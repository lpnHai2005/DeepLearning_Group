<!-- sinh tự động bởi src/make_tables.py từ results/grid_batch/summary.csv; không sửa tay -->
| value | params | seeds | train_mse | val_mse | test_mse | val_mae | test_mae | test_rmse | test_r2 | best_epoch | epochs_run | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 41 | 3 | 1.91e-05 ± 4.5e-06 | 2.00e-05 ± 4.0e-06 | 2.02e-05 ± 7.1e-06 | 3.48e-03 ± 4.5e-04 | 3.48e-03 ± 7.6e-04 | 4.45e-03 ± 7.8e-04 | 0.99142 ± 0.00304 | 1003 ± 177 | 1203 ± 177 | 47.8 ± 11.0 |
| 32 | 41 | 3 | 2.21e-05 ± 6.1e-06 | 1.89e-05 ± 5.9e-06 | 2.28e-05 ± 1.4e-05 | 3.35e-03 ± 5.6e-04 | 3.53e-03 ± 1.1e-03 | 4.63e-03 ± 1.4e-03 | 0.99030 ± 0.00609 | 1653 ± 411 | 1792 ± 346 | 41.6 ± 4.3 |
| 64 | 41 | 3 | 2.42e-05 ± 8.9e-06 | 2.38e-05 ± 8.3e-06 | 2.36e-05 ± 8.2e-06 | 3.78e-03 ± 7.2e-04 | 3.68e-03 ± 6.2e-04 | 4.80e-03 ± 9.0e-04 | 0.98997 ± 0.00349 | 1653 ± 537 | 1745 ± 442 | 23.2 ± 6.1 |
| 128 | 41 | 3 | 2.04e-05 ± 9.8e-06 | 2.27e-05 ± 9.0e-06 | 2.32e-05 ± 1.2e-05 | 3.70e-03 ± 7.4e-04 | 3.56e-03 ± 8.9e-04 | 4.72e-03 ± 1.1e-03 | 0.99014 ± 0.00491 | 1965 ± 2 | 2000 ± 0 | 20.3 ± 2.6 |

Nguồn: `results/grid_batch/summary.csv` (mỗi run 1 dòng; log: `results/grid_batch/run.log`, `results/grid_batch/posthoc_test.log`). Giá trị: trung bình ± độ lệch chuẩn (ddof=1) qua các seed; MSE/MAE trên thang y gốc.
