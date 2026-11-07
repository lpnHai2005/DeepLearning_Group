<!-- sinh tự động bởi src/make_tables.py từ results/expC_ctrl/summary.csv; không sửa tay -->
| value | params | seeds | train_mse | val_mse | test_mse | val_mae | test_mae | test_rmse | test_r2 | best_epoch | epochs_run | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 38 | 153 | 3 | 2.38e-06 ± 1.1e-06 | 2.42e-06 ± 1.6e-06 | 2.96e-06 ± 1.8e-06 | 1.18e-03 ± 3.8e-04 | 1.30e-03 ± 3.7e-04 | 1.67e-03 ± 5.1e-04 | 0.99874 ± 0.00078 | 1378 ± 582 | 1526 ± 506 | 32.3 ± 13.2 |
| 65 | 261 | 3 | 1.87e-06 ± 9.5e-07 | 1.68e-06 ± 8.7e-07 | 1.92e-06 ± 1.2e-06 | 9.95e-04 ± 2.6e-04 | 1.06e-03 ± 3.6e-04 | 1.34e-03 ± 4.6e-04 | 0.99918 ± 0.00051 | 1084 ± 221 | 1284 ± 221 | 26.0 ± 5.2 |
| 93 | 373 | 3 | 1.11e-06 ± 1.6e-07 | 1.46e-06 ± 4.3e-07 | 1.41e-06 ± 2.0e-07 | 9.06e-04 ± 9.8e-05 | 8.83e-04 ± 2.0e-05 | 1.18e-03 ± 8.7e-05 | 0.99940 ± 0.00009 | 1266 ± 445 | 1466 ± 445 | 29.6 ± 9.1 |

Nguồn: `results/expC_ctrl/summary.csv` (mỗi run 1 dòng; log: `results/expC_ctrl/run.log`, `results/expC_ctrl/posthoc_test.log`). Giá trị: trung bình ± độ lệch chuẩn (ddof=1) qua các seed; MSE/MAE trên thang y gốc.
