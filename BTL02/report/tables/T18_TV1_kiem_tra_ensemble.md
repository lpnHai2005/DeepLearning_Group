# T18 – Kiểm tra độc lập ensemble của TV1

**Người thực hiện:** TV1 – Đoàn Trung Kiên (C1)
**Ngày:** 2026-11-27
**Phạm vi:** dự đoán ensemble trên PublicTest; không huấn luyện lại và không truy cập PrivateTest.

## Kết quả kiểm tra

Theo nhánh fallback đã đăng ký trước trong `configs/E7_final.json`, bộ ba run `E1-014_s0`, `E1-014_s1`, `E1-014_s2` được dùng để tổng hợp TTA và ensemble; vì vậy TV1 không cần chạy lại seed 1.

TV1 đọc trực tiếp `results/E7/val_predictions_ens_{none,flip,ten_crop}.npz`, tự giải mã các mảng `y_true`, `y_prob`, `y_pred` và tính lại accuracy, macro-F1, NLL bằng phép tính độc lập. Kết quả:

| TTA | Số mẫu | Accuracy tính lại | Macro-F1 tính lại | NLL tính lại | Kết quả đối chiếu |
|---|---:|---:|---:|---:|---|
| `none` | 3.589 | 0.723600 | 0.713421 | 0.978011 | Khớp `e7_val_summary.json` trong sai số < 1e-6 |
| `flip` | 3.589 | 0.724993 | 0.710649 | 0.940189 | Khớp `e7_val_summary.json` trong sai số < 1e-6 |
| `ten_crop` | 3.589 | 0.699081 | 0.683870 | 0.971639 | Khớp `e7_val_summary.json` trong sai số < 1e-6 |

Các kiểm tra cấu trúc cũng đạt:

- `y_true` có cùng thứ tự ở cả ba file; mỗi file có 3.589 nhãn và 3.589 vector xác suất gồm 7 lớp.
- `y_pred` bằng `argmax(y_prob)` ở cả ba file.
- F1 tính lại cho từng lớp trong dự đoán tốt nhất khớp cả 7/7 giá trị trong JSON.
- `e7_val_best_predictions.npz` trùng chính xác với `val_predictions_ens_flip.npz`; `flip` có accuracy ensemble cao nhất trong ba chế độ.
- Metadata báo cáo ghi đúng ba run `E1-014_s0/s1/s2` và ba SHA-256 checkpoint riêng biệt.

## Bằng chứng đầu vào

| Tệp | SHA-256 |
|---|---|
| `results/E7/val_predictions_ens_none.npz` | `5baef458e3739bb78189efc5b309e548842f4b06ef2edc4b2e52d18b58c3a1ae` |
| `results/E7/val_predictions_ens_flip.npz` | `90cb1caa119a11cf8715ae35f8cca691c1d18e89156c1a724a60b36286e528d` |
| `results/E7/val_predictions_ens_ten_crop.npz` | `34cd737207cea9222234ea7540d4536685680c5f529f16251bee797c04763359` |

Ba checkpoint trong bộ `E1-014_s0/s1/s2` không có trong checkout này, và các xác suất riêng từng seed không được xuất thành tệp riêng. Vì vậy phép kiểm tra này tái tính độc lập các chỉ số từ đầu ra ensemble đã lưu, nhưng không phát lại forward pass hoặc tính lại phép trung bình xác suất từ ba seed. Đây là giới hạn của bằng chứng NPZ hiện có.

**Kết luận phần TV1:** kiểm tra độc lập các đầu ra ensemble/TTA đã lưu đạt; báo cáo khớp. Không cập nhật `tracking/test_log.csv` hoặc `tracking/activity_log.csv`, và không đặt trạng thái Verified thay tester.
