# Mục 4.2 – Thực nghiệm A: ảnh hưởng của số mẫu huấn luyện

> Kết quả tổng hợp từ 30 run (10 giá trị A × 3 seed). Mọi metric dưới đây ở thang y gốc; std là độ lệch chuẩn mẫu giữa ba seed.

## Thiết kế

Thay đổi số mẫu huấn luyện \(A\in\{200,400,\ldots,2000\}\). Mỗi mức A lấy A mẫu đầu của cùng train pool 2.000 mẫu. Các tập con vì vậy lồng nhau; validation cố định 200 mẫu. Thực hiện ba seed huấn luyện (0, 1, 2), giữ nguyên kiến trúc và các siêu tham số còn lại.

Baseline được chọn bằng lưới 3 learning rate × 3 activation × 3 seed, chỉ trên validation. Mean validation MSE thấp nhất là sigmoid, learning rate 0.01; cấu hình được khóa trong `configs/baseline.json` và dùng cố định trong `configs/expA.json`. Bảng tổng hợp đủ 27 run nằm ở `results/grid_baseline/summary.csv`.

| Activation | Learning rate | Validation MSE (mean ± std, 3 seed) |
|---|---:|---:|
| Sigmoid | 0.01 | 1.8908e-5 ± 5.8935e-6 |
| Tanh | 0.003 | 2.9563e-5 ± 1.1088e-5 |
| Sigmoid | 0.003 | 3.7491e-5 ± 1.8026e-5 |
| Tanh | 0.01 | 3.8619e-5 ± 2.5905e-5 |
| Tanh | 0.001 | 4.1553e-5 ± 1.3443e-5 |
| Sigmoid | 0.001 | 1.1747e-4 ± 2.7726e-5 |
| ReLU | 0.003 | 3.3040e-4 ± 3.3453e-4 |
| ReLU | 0.001 | 3.3498e-4 ± 3.3700e-4 |
| ReLU | 0.01 | 3.4692e-4 ± 2.8886e-4 |

Thực nghiệm A chạy với Adam, batch size 32, tối đa 2.000 epoch và early stopping theo validation loss với patience 200. Mỗi run lưu cấu hình, history, metric và best checkpoint.

Báo cáo train/validation MSE, MAE, RMSE trên thang y gốc dưới dạng mean ± độ lệch chuẩn qua seed. Đường cong A hiển thị error bar và so sánh với hằng số dự đoán bằng trung bình nhãn của từng tập train con. Test không tham gia chọn baseline, checkpoint hay kết luận về A.

## Kết quả

| A | Train MSE | Validation MSE | Train MAE | Validation MAE | Train RMSE | Validation RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 5.43e-5 ± 1.37e-5 | 7.06e-5 ± 1.80e-5 | 5.64e-3 ± 9.26e-4 | 6.26e-3 ± 8.86e-4 | 7.33e-3 ± 9.41e-4 | 8.35e-3 ± 1.10e-3 |
| 400 | 3.45e-5 ± 1.35e-5 | 3.93e-5 ± 1.30e-5 | 4.51e-3 ± 7.99e-4 | 4.73e-3 ± 8.25e-4 | 5.81e-3 ± 1.10e-3 | 6.21e-3 ± 1.07e-3 |
| 600 | 1.95e-5 ± 8.27e-6 | 2.16e-5 ± 7.77e-6 | 3.39e-3 ± 6.25e-4 | 3.56e-3 ± 6.17e-4 | 4.35e-3 ± 8.98e-4 | 4.60e-3 ± 8.10e-4 |
| 800 | 2.74e-5 ± 1.75e-5 | 2.73e-5 ± 1.67e-5 | 3.99e-3 ± 1.32e-3 | 4.05e-3 ± 1.35e-3 | 5.06e-3 ± 1.64e-3 | 5.07e-3 ± 1.55e-3 |
| 1000 | 2.21e-5 ± 6.15e-6 | 1.89e-5 ± 5.89e-6 | 3.63e-3 ± 5.97e-4 | 3.35e-3 ± 5.64e-4 | 4.67e-3 ± 6.77e-4 | 4.31e-3 ± 6.83e-4 |
| 1200 | 2.19e-5 ± 9.43e-6 | 2.23e-5 ± 9.32e-6 | 3.58e-3 ± 6.29e-4 | 3.56e-3 ± 6.52e-4 | 4.61e-3 ± 1.00e-3 | 4.66e-3 ± 9.49e-4 |
| 1400 | 1.80e-5 ± 7.14e-6 | 1.62e-5 ± 4.69e-6 | 3.27e-3 ± 6.11e-4 | 3.13e-3 ± 3.34e-4 | 4.19e-3 ± 8.12e-4 | 4.00e-3 ± 5.66e-4 |
| 1600 | 1.72e-5 ± 4.48e-6 | 1.52e-5 ± 1.42e-6 | 3.28e-3 ± 2.69e-4 | 3.06e-3 ± 9.08e-5 | 4.13e-3 ± 5.24e-4 | 3.89e-3 ± 1.81e-4 |
| 1800 | 2.01e-5 ± 2.64e-6 | 1.75e-5 ± 3.47e-6 | 3.38e-3 ± 2.43e-4 | 3.20e-3 ± 3.33e-4 | 4.47e-3 ± 3.00e-4 | 4.16e-3 ± 4.28e-4 |
| 2000 | 3.24e-5 ± 1.22e-5 | 2.57e-5 ± 8.43e-6 | 4.26e-3 ± 7.38e-4 | 3.91e-3 ± 5.95e-4 | 5.63e-3 ± 1.04e-3 | 5.03e-3 ± 8.16e-4 |

## Hình và nhận xét

![MSE theo kích thước tập train](../figures/expA_validation_mse.png)

![Khoảng cách validation–train](../figures/expA_train_val_gap.png)

- Validation MSE giảm từ 7.06e-5 tại A=200 xuống 2.16e-5 tại A=600, rồi dao động quanh khoảng 1.5e-5–2.7e-5. Mức thấp nhất trong lưới là A=1600 (1.52e-5 ± 1.42e-6); lỗi tăng lại tại A=2000 (2.57e-5 ± 8.43e-6), nên quan hệ không giảm đơn điệu.
- Train và validation MSE gần nhau ở phần lớn kích thước; khoảng dao động giữa seed chồng lấp ở nhiều mức. Chưa có bằng chứng rõ về quá khớp tăng theo A từ đường cong này. Mức A=2000 có lỗi train và validation cùng cao hơn các điểm lân cận, nên không nên diễn giải như quá khớp riêng.
- Baseline dự đoán hằng số bằng trung bình nhãn train có validation MSE xấp xỉ 2.49e-3 ở mọi A; mọi mô hình A đều thấp hơn rõ rệt.
- Không đánh giá test trong thí nghiệm A; test vẫn được giữ cho lần đánh giá cuối mô hình do nhóm chọn bằng validation. Dữ liệu không thêm nhiễu và mô hình giữ dung lượng cố định, nên hiện tượng quá khớp có thể không rõ như dữ liệu nhiễu.
- Muốn khẳng định mô hình cuối tổng quát hóa ra sao, nhóm cần chọn cấu hình bằng validation rồi mới đánh giá test một lần.
