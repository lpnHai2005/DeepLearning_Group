# Nhận xét TV1 — T12 / E1

## Phạm vi và nguồn số liệu

TV1 chạy `small` và `medium` với seed 1, 2 theo `configs/E1_arch.json`. Các nhận xét dưới đây dùng `metrics.json` của những run có `status=Completed`; seed 0 do TV2 chạy được dùng để đối chiếu đủ ba seed. Run `E1-001_s1` dừng ở epoch 77, được ghi `Aborted` và không đưa vào thống kê. Run hoàn tất cho `small`, seed 1 là `E1-009_s1`.

| Mô hình | Tham số | Run hoàn tất (seed 0, 1, 2) | Best validation accuracy, mean ± SD | Best validation macro-F1, mean ± SD |
|---|---:|---|---:|---:|
| `small` | 38,901,191 | `E1-001_s0`, `E1-009_s1`, `E1-001_s2` | 0.6934 ± 0.0027 | 0.6773 ± 0.0045 |
| `medium` | 11,327,303 | `E1-002_s0`, `E1-002_s1`, `E1-002_s2` | 0.6919 ± 0.0031 | 0.6730 ± 0.0059 |

Mean và độ lệch chuẩn mẫu được tính trực tiếp từ `best_val_acc` và `best_val_macro_f1` trong các `metrics.json` Completed. Không dùng PrivateTest.

## Nhận xét

Trên ba seed hiện có, `small` cao hơn `medium` khoảng 0.0015 về best validation accuracy và 0.0043 về best validation macro-F1. `small` có khoảng 3.44 lần số tham số của `medium`, nên mức tăng validation quan sát được khá nhỏ so với chi phí mô hình. Độ lệch chuẩn của accuracy ở hai mô hình thấp (khoảng 0.003), nhưng đây chỉ là kết quả validation trên ba seed; TV4 vẫn cần kiểm thử độc lập theo tiêu chí T12 trước khi task được Verified.

## Run TV1

| Run | Mô hình | Seed | Trạng thái | Epoch tốt nhất | Best validation accuracy | Best validation macro-F1 |
|---|---|---:|---|---:|---:|---:|
| `E1-009_s1` | `small` | 1 | Completed | 98 | 0.6921 | 0.6766 |
| `E1-001_s2` | `small` | 2 | Completed | 83 | 0.6966 | 0.6820 |
| `E1-002_s1` | `medium` | 1 | Completed | 100 | 0.6890 | 0.6730 |
| `E1-002_s2` | `medium` | 2 | Completed | 98 | 0.6916 | 0.6670 |
