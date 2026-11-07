# Nhật ký quyết định (mục 4.1 và Phụ lục)

> Mỗi quyết định gồm: nội dung, lý do, file minh chứng, và **thứ tự**:
> - **Khảo sát → chọn**: chạy lưới trước, chọn theo val MSE thấp nhất.
> - **Chọn trước → kiểm chứng sau**: chọn theo kinh nghiệm hoặc mặc định phổ biến, sau đó mới chạy thí nghiệm kiểm tra lại. Nhóm ghi rõ để không trình bày như là đã khảo sát từ đầu.
> - **Phát hiện từ log**: không có trong kế hoạch ban đầu, phát sinh khi đọc kết quả.
>
> Mọi số liệu lấy từ `report/tables/*.md` (sinh tự động bởi `src/make_tables.py`) hoặc CSV trong `results/`. Val MSE ghi dạng trung bình ± độ lệch chuẩn qua 3 seed, thang y gốc. Tập test **không** được dùng cho bất kỳ quyết định nào dưới đây.

## 1. Bảng tóm tắt

| # | Quyết định | Thứ tự | Minh chứng chính |
|---|---|---|---|
| D1 | Chia train_pool / val / test = 2000 / 200 / 100, seed 2026 / 2027 / 2028 | Chọn trước | `data/meta.json`, `results/eda/data_summary.csv` |
| D2 | Lấy x đều trên miền [−2, 2] × [−4, 4] | Chọn trước | `src/data/gaussian.py::sample_uniform`, `figures/eda_*.png` |
| D3 | Tập train của A là tập con lồng nhau (n mẫu đầu của train_pool) | Chọn trước | `src/data/dataset.py::subset` |
| D4 | Chuẩn hóa X_SCALE = (2, 4), Y_SCALE = y_max ≈ 0.2690 | Chọn trước | `src/data/gaussian.py` |
| D5 | Hàm kích hoạt sigmoid, lr 0.01 | Khảo sát → chọn | `results/grid_baseline/` (27 run), `figures/grid_baseline_val_mse.png`, `report/tables/grid_baseline.md` |
| D6 | Optimizer Adam | **Chọn trước → kiểm chứng sau** | `results/grid_optimizer/` (126 run), `figures/optimizer_loss.png`, `report/tables/grid_optimizer.md` |
| D7 | Batch size 32 | Chọn trước → kiểm chứng sau | `results/grid_batch/` (12 run), `report/tables/grid_batch.md` |
| D8 | Patience 200 | Chọn trước → kiểm chứng sau | `results/stopping/patience_sim.csv`, `figures/stopping_patience.png` |
| D9 | max_epochs 2000 | Chọn trước; **là hạn chế** | `results/stopping/epoch_cap.csv` |
| D10 | 3 seed (0, 1, 2) cho mỗi cấu hình | Chọn trước; **là hạn chế** | mọi `configs/*.json` |
| D11 | Đối chứng A cùng ngân sách bước cập nhật | Phát hiện từ log | `configs/expA_steps.json`, `results/expA_steps/`, `figures/expA_steps.png` |
| D12 | Mở rộng B lên 16/32/64 và chạy B ở A = 200 | Phát hiện từ log | `results/expB_wide/`, `results/expB_A200/`, `figures/expB_A200.png` |
| D13 | Đối chứng C: mạng 1 tầng cùng số tham số (B = 38/65/93) | Chọn trước | `configs/expC_ctrl.json`, `results/expC_ctrl/` |
| D14 | Đo chuẩn gradient từng tầng để xét vanishing gradient | Chọn trước | `results/expC/grad_norms.csv`, `figures/expC_grad_norms.png` |
| D15 | Mô hình cuối = argmin val MSE trên 53 cấu hình | Khảo sát → chọn | `results/summary_all.csv`, `configs/final.json`, `tests/test_config_evidence.py` |
| D16 | Test chỉ dùng để hậu kiểm và đánh giá cuối | Quy tắc | `results/*/posthoc_test.log`, `results/final/run.log`, cờ `--eval-test` trong `src/run_experiment.py` |
| D17 | Tìm siêu tham số theo từng tọa độ, không thử mọi tổ hợp | Chọn trước; **là hạn chế** | — |

## 2. Chi tiết từng quyết định

### D1. Chia dữ liệu 2000 / 200 / 100, seed riêng cho từng tập
- **Lý do**: ba tập sinh độc lập bằng ba seed khác nhau nên không trùng điểm và không rò rỉ. train_pool 2000 đủ cho Thực nghiệm A (A tối đa 2000). Val 200 đủ ổn định để so sánh cấu hình; test 100 theo đề bài.
- **Minh chứng**: `data/meta.json` ghi seed và kích thước. Phân phối y gần giống nhau giữa ba tập: tỉ lệ y < 0.01 là 62.6% (train_pool), 66% (val), 61% (test).

### D2. Lấy x đều trên miền, không lấy theo phân phối Gaussian
- **Lý do**: bài toán là xấp xỉ hàm y = f(x) trên cả miền, nên cần phủ đều miền, kể cả vùng đuôi y ≈ 0. Nếu lấy x theo Gaussian thì mẫu dồn vào quanh đỉnh và vùng biên gần như không có dữ liệu.
- **Trung thực**: lựa chọn này **khác** hình scatter (h) trong slide của thầy (điểm dồn quanh tâm). Hệ quả là 62.6% mẫu train có y < 0.01, nên MSE trung bình bị kéo xuống bởi vùng phẳng. Vì vậy nhóm báo cáo thêm R² và Grid MSE.

### D3. Tập con lồng nhau cho Thực nghiệm A
- **Lý do**: A = 200 ⊂ A = 400 ⊂ … ⊂ A = 2000. Khi tăng A chỉ có thêm mẫu, không đổi mẫu cũ, nên chênh lệch kết quả là do số mẫu chứ không do lấy mẫu khác nhau.

### D4. X_SCALE = (2, 4), Y_SCALE = 0.2690
- **Lý do**: đưa x về [−1, 1] và y về [0, 1] để sigmoid không bão hòa và gradient có cùng cỡ ở hai chiều.
- **Không rò rỉ**: các hằng số lấy từ **đề bài** (biên miền và y_max = 1/(2π√det C)), không ước lượng từ dữ liệu, nên không dùng thông tin của val/test.
- **Trung thực**: đây là lựa chọn của nhóm, đề bài không yêu cầu.

### D5. sigmoid, lr 0.01 — khảo sát rồi chọn
- **Lưới**: activation ∈ {tanh, sigmoid, relu} × lr ∈ {0.001, 0.003, 0.01}, mạng `[10]`, Adam, batch 32, A = 1000, 3 seed → 27 run. Chỉ xét val (không bật `--eval-test`).
- **Kết quả**: sigmoid + lr 0.01 có val MSE thấp nhất: 1.89e-5 ± 5.9e-6. Tốt thứ hai là tanh + lr 0.003: 2.96e-5 ± 1.1e-5. relu kém nhất ở mọi lr (khoảng 3.3e-4), vì mạng chỉ có 10 nơ-ron nên xấp xỉ tuyến tính từng khúc rất thô.
- Cấu hình được đóng băng vào `configs/baseline.json`. `tests/test_config_evidence.py` kiểm tra config khớp với argmin trong log.

### D6. Adam — chọn trước, kiểm chứng sau
- **Thứ tự thực tế**: lưới D5 đã dùng Adam (commit `a5e0f01`, 02/11). So sánh optimizer chỉ chạy sau đó (commit `3565482`, 05/11). Vì vậy Adam là lựa chọn mặc định, **không** phải kết quả khảo sát ban đầu.
- **Kiểm chứng**: 6 optimizer (SGD, momentum, Nesterov, RMSprop, Adam, Nadam) × 7 lr (0.001 → 1.0) × 3 seed = 126 run, chạy đủ 2000 epoch. Adam + lr 0.01 vẫn tốt nhất: 1.88e-5 ± 5.9e-6. Tốt thứ hai là Nadam + lr 0.01: 2.51e-5 ± 1.2e-5. SGD tốt nhất (lr 0.3) chỉ đạt 3.57e-4.
- **Lưu ý**: `results/grid_optimizer/` chỉ giữ thư mục run của lr tốt nhất mỗi optimizer (để tiết kiệm dung lượng). `summary.csv` vẫn đủ 126 dòng.
- **Hạn chế**: lr tốt nhất của Adam được tìm khi đã cố định sigmoid; chưa thử lại activation với optimizer khác.

### D7. Batch 32 — chọn trước, kiểm chứng sau
- **Kiểm chứng** (`grid_batch`, mạng `[10]`, A = 1000): val MSE của batch 16 / 32 / 64 / 128 lần lượt 2.00e-5 / **1.89e-5** / 2.38e-5 / 2.27e-5.
- **Trung thực**: batch 32 thấp nhất nhưng chênh lệch **nhỏ hơn độ lệch chuẩn** (khoảng 4–9e-6), nên không khẳng định được batch 32 tốt hơn có ý nghĩa. Giữ batch 32 vì không có lý do để đổi.
- Batch 128 chạm trần 2000 epoch ở cả 3 seed (best_epoch 1965 ± 2): batch lớn ít bước cập nhật hơn nên cần nhiều epoch hơn.

### D8. Patience 200 — chọn trước, kiểm chứng sau
- **Cách kiểm chứng**: `src/analyze_stopping.py` mô phỏng các mức patience trên đường val của `results/grid_optimizer/adam_0.01` (chạy đủ 2000 epoch, không dừng sớm), không cần train lại.
- **Kết quả** (tỉ số val MSE so với chạy đủ 2000 epoch, trung bình 3 seed): patience 25 → 2.82, 50 → 2.66, 100 → 1.31, **200 → 1.007**, 400 → 1.0005, 800 → 1.00.
- Như vậy patience 200 chỉ kém 0.7% so với không dừng sớm, trong khi patience 100 kém 31%. Patience 200 là mức nhỏ nhất gần như không mất độ chính xác.

### D9. max_epochs 2000 — là hạn chế
- Số run có best_epoch nằm trong 200 epoch cuối (tức là có thể vẫn đang cải thiện khi bị cắt), theo `results/stopping/epoch_cap.csv`:

  | Thí nghiệm | Run chạm trần / tổng | % |
  |---|---|---|
  | expA | 11 / 30 | 36.7 |
  | expA_steps | 7 / 30 | 23.3 |
  | expB | 2 / 30 | 6.7 |
  | expB_wide | 3 / 9 | 33.3 |
  | expB_A200 | 17 / 39 | 43.6 |
  | expC | 3 / 12 | 25.0 |
  | expC_ctrl | 1 / 9 | 11.1 |
  | grid_batch | 6 / 12 | 50.0 |

- **Hệ quả**: lỗi của một số cấu hình (nhất là A nhỏ và mạng rộng) có thể bị **đánh giá cao hơn thực tế**. D11 xử lý một phần vấn đề này cho Thực nghiệm A.

### D10. 3 seed — là hạn chế
- Cân đối giữa thời gian chạy trên CPU và độ tin cậy. Với 3 seed, nhiều chênh lệch (D7, A ≥ 600, A = 2000 tăng nhẹ) nhỏ hơn std nên không kết luận được.

### D11. Đối chứng A cùng ngân sách bước cập nhật — phát hiện từ log
- **Phát hiện**: batch cố định 32 nên A = 200 chỉ có 7 bước/epoch, còn A = 1000 có 32 bước/epoch. Trong expA, các run A = 200…600 chạm trần 2000 epoch với best_epoch nằm trong 200 epoch cuối. Như vậy A nhỏ vừa ít mẫu vừa được train ít bước hơn (hai yếu tố lẫn vào nhau).
- **Cách tách**: giữ cùng ngân sách khoảng 64 000 bước, nhân max_epochs và patience với 32/⌈A/32⌉ (ví dụ A = 200: 9143 epoch, patience 914). A = 1000 giữ nguyên để kiểm tra tái lập.
- **Kết quả**: A = 200 giảm từ 7.06e-5 xuống 4.04e-5 (giảm 43%). Như vậy một phần lỗi ở A nhỏ là do số bước, phần còn lại là do số mẫu. Từ A ≥ 600, giới hạn chính là dung lượng của mạng `[10]`.

### D12. Mở rộng B và chạy B ở A = 200 — phát hiện từ log
- **Phát hiện**: ở A = 1000, val giảm đều theo B = 1…10, chưa thấy đường chữ U, nên cần xét B lớn hơn và dữ liệu ít hơn để tìm overfit.
- **Kết quả**: ở A = 1000, B = 64 đạt 2.09e-6 và B = 93 (từ D13) đạt 1.46e-6, không có overfit. Ở A = 200, tỉ số Val/Train tăng từ 1.30 (B = 10) lên 1.74 (B = 64): có khoảng cách tổng quát hóa, nhưng val vẫn giảm theo B nên vẫn chưa có chữ U. 17/39 run chạm trần epoch (D9).

### D13. Đối chứng C cùng số tham số
- **Lý do**: mạng sâu hơn cũng có nhiều tham số hơn. Muốn biết "sâu" có lợi hơn "rộng" hay không thì phải so ở cùng số tham số.
- **Cách chọn B**: mạng `[2, B, 1]` có 4B + 1 tham số. Chọn B = 38 / 65 / 93 → 153 / 261 / 373 tham số, gần với C = 2 / 3 / 4 (151 / 261 / 371).
- **Kết quả** (`report/tables/expC.md`, `expC_ctrl.md`): mạng sâu tốt hơn mạng 1 tầng cùng số tham số khoảng 16–28 lần.

  | Số tham số | Mạng sâu | Val MSE | Mạng 1 tầng | Val MSE | Tỉ số |
  |---|---|---|---|---|---|
  | ~152 | `[10,10]` | 1.05e-7 | `[38]` | 2.42e-6 | 23 |
  | 261 | `[10,10,10]` | 6.02e-8 | `[65]` | 1.68e-6 | 28 |
  | ~372 | `[10,10,10,10]` | 9.21e-8 | `[93]` | 1.46e-6 | 16 |

### D14. Đo gradient từng tầng
- **Lý do**: sigmoid có đạo hàm tối đa 0.25 nên mạng sâu có nguy cơ vanishing gradient.
- **Kết quả** (`results/expC/grad_norms.csv`, cột `grad_epoch1`): ở epoch 1, chuẩn gradient tầng đầu nhỏ hơn tầng cuối rất nhiều. Với C = 3, tầng 1 là 1.5e-4, còn tầng ra là 0.205 (khoảng 1300 lần). Với C = 4, tầng 1 chỉ còn 1.7e-5. Như vậy có vanishing gradient lúc khởi tạo, nhưng không cản trở việc huấn luyện (C = 3, 4 vẫn tốt nhất), vì Adam chuẩn hóa bước cập nhật theo từng tham số.

### D15. Chọn mô hình cuối theo argmin val trên 53 cấu hình
- `results/summary_all.csv` gộp 53 cấu hình từ mọi thí nghiệm. Ba cấu hình tốt nhất: C = 3 `[10,10,10]` 6.02e-8, C = 4 9.21e-8, C = 2 1.05e-7.
- Chọn C = 3 (261 tham số). Đóng băng vào `configs/final.json`, chỉ sau đó mới chạy `--eval-test`. `tests/test_config_evidence.py` kiểm tra `final.json` khớp argmin.
- **Kết quả test** (chỉ để báo cáo): MSE 7.13e-8 ± 2.36e-8, R² = 0.99997, Grid MSE 7.34e-8. Checkpoint `results/final/best.pt` lấy từ seed 2.

### D16. Test chỉ dùng để hậu kiểm
- Trong quá trình chọn cấu hình, không lệnh nào bật `--eval-test`. Sau khi chọn xong, `src/posthoc_test.py` nạp các `best.pt` đã đóng băng, đo test và ghi vào `posthoc_test.log`. Kết quả này chỉ để trình bày, **không** dùng để quay lại đổi cấu hình.

### D17. Tìm siêu tham số theo từng tọa độ — là hạn chế
- Nhóm cố định từng siêu tham số một (activation + lr → optimizer → batch → patience), không thử mọi tổ hợp, do giới hạn thời gian chạy trên CPU.
- **Hệ quả**: có thể bỏ sót tổ hợp tốt hơn, ví dụ relu với batch khác hoặc tanh với optimizer khác. Kết luận "sigmoid tốt nhất" chỉ đúng trong vùng đã khảo sát.

## 3. Tóm tắt hạn chế (đưa vào Ch.5)

1. Trần 2000 epoch: 6.7–50% run chạm trần tùy thí nghiệm (D9).
2. Chỉ 3 seed: nhiều chênh lệch nhỏ hơn std (D7, D10).
3. Tìm siêu tham số theo từng tọa độ (D17); Adam, batch 32, patience 200 là chọn trước rồi mới kiểm chứng (D6–D8).
4. Lấy x đều trên miền khác với hình minh họa của đề (D2).
5. Chỉ chạy CPU. Thời gian của 90 run bổ sung đo khi 4 tiến trình chạy song song nên không so sánh được với các run khác.
