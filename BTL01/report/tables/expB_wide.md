<!-- sinh tự động bởi src/make_tables.py từ results/expB_wide/summary.csv; không sửa tay -->
| value | params | seeds | train_mse | val_mse | test_mse | val_mae | test_mae | test_rmse | test_r2 | best_epoch | epochs_run | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 65 | 3 | 5.58e-06 ± 2.7e-06 | 5.63e-06 ± 3.2e-06 | 7.19e-06 ± 3.5e-06 | 1.74e-03 ± 4.0e-04 | 1.95e-03 ± 4.7e-04 | 2.63e-03 ± 6.2e-04 | 0.99694 ± 0.00148 | 1921 ± 55 | 2000 ± 0 | 45.1 ± 4.6 |
| 32 | 129 | 3 | 2.82e-06 ± 1.3e-06 | 2.72e-06 ± 1.2e-06 | 2.76e-06 ± 7.6e-07 | 1.32e-03 ± 2.7e-04 | 1.25e-03 ± 1.9e-04 | 1.65e-03 ± 2.3e-04 | 0.99883 ± 0.00032 | 1277 ± 531 | 1477 ± 531 | 36.6 ± 11.8 |
| 64 | 257 | 3 | 2.28e-06 ± 7.4e-07 | 2.09e-06 ± 7.6e-07 | 2.32e-06 ± 1.3e-06 | 1.10e-03 ± 2.0e-04 | 1.14e-03 ± 3.1e-04 | 1.49e-03 ± 4.1e-04 | 0.99901 ± 0.00056 | 1355 ± 133 | 1555 ± 133 | 32.9 ± 2.7 |

Nguồn: `results/expB_wide/summary.csv` (mỗi run 1 dòng; log: `results/expB_wide/run.log`, `results/expB_wide/posthoc_test.log`). Giá trị: trung bình ± độ lệch chuẩn (ddof=1) qua các seed; MSE/MAE trên thang y gốc.
