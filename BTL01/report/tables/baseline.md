<!-- sinh tự động bởi src/make_tables.py từ results/baseline/summary.csv; không sửa tay -->
| value | params | seeds | train_mse | val_mse | val_mae | best_epoch | epochs_run | train_time_s |
|---|---|---|---|---|---|---|---|---|
| base | 41 | 3 | 2.21e-05 ± 6.1e-06 | 1.89e-05 ± 5.9e-06 | 3.35e-03 ± 5.6e-04 | 1653 ± 411 | 1792 ± 346 | 34.5 ± 5.7 |

Nguồn: `results/baseline/summary.csv` (mỗi run 1 dòng; log: `results/baseline/run.log`). Giá trị: trung bình ± độ lệch chuẩn (ddof=1) qua các seed; MSE/MAE trên thang y gốc.
