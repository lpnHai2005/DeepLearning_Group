# Ghi chú slide TV3 – Thực nghiệm B, Đánh giá mô hình cuối & Review chéo

## Slide 9 – Kết quả Thực nghiệm B (Độ rộng tầng ẩn $B = 1 \dots 10$)

- **Thiết lập:** Mạng 1 tầng ẩn ($C = 1$, kiến trúc $2 \rightarrow [B] \rightarrow 1$), $B = 1 \dots 10$ (tham số $4B + 1$ từ $5 \rightarrow 41$), cố định $A = 1000$, sigmoid, Adam `lr = 0.01`, 3 seed.
- **Hình chính trên slide:** Chèn `figures/expB_error.png` và `figures/expB_surfaces.png`.
- **Ý chính cần nói:**
  1. **Giảm lỗi đơn điệu hơn 120 lần:** Validation MSE giảm liên tục từ $2.29 \times 10^{-3} \pm 1.51 \times 10^{-4}$ ($B = 1$) xuống $1.89 \times 10^{-5} \pm 5.89 \times 10^{-6}$ ($B = 10$).
  2. **Giải thích hình học vùng Underfitting ($B$ nhỏ):**
     - $B = 1$ chỉ là một hàm bước đơn điệu theo một hướng duy nhất (*ridge function*), không thể tạo cực đại cục bộ $\rightarrow$ bề mặt phẳng/bậc thang ($\text{Grid } R^2 = 0.103$).
     - $B = 2$ ghép hai sườn dốc ngược chiều tạo thành "dải sống núi" hở hai đầu ($\text{Grid } R^2 = 0.463$).
     - Từ $B = 3$ trở lên các mặt phẳng cắt nhau mới khép kín được đỉnh chuông; đến $B = 10$ đạt $\text{Grid } R^2 = 0.994$.
  3. **Vì sao không thấy Overfitting chữ U?** Dữ liệu sinh từ hàm giải tích không nhiễu ($\epsilon = 0$), số mẫu $A = 1000 \gg 41$ tham số, kết hợp early stopping theo validation. Ngay cả khi mở rộng 1 tầng lên $B = 93$ ($373$ tham số), lỗi vẫn giảm về $1.46 \times 10^{-6}$ nhưng vẫn kém xa mạng sâu 3 tầng ($6.02 \times 10^{-8}$).

---

## Slide 11 – Lựa chọn mô hình cuối và Đánh giá trên tập Test

- **Quy trình chọn công bằng (chống rò rỉ):** So sánh mọi ứng viên từ A, B, C hoàn toàn bằng **Validation MSE** (không chạm `test.csv`). Cấu hình tốt nhất là **$C = 3$ (`[10, 10, 10]`, $261$ tham số)** với $\text{Val MSE} = 6.02 \times 10^{-8} \pm 2.07 \times 10^{-8}$ (checkpoint tốt nhất: `seed = 2`, lưu tại `results/final/best.pt`).
- **Hình chính trên slide:** Chèn `figures/final_parity_test.png`, `figures/final_surface_comparison.png` và `figures/final_error_map.png`.
- **Kết quả định lượng trên tập Test độc lập ($N = 100$) và lưới dày ($100 \times 100$):**
  - **Trung bình 3 seed (`[10, 10, 10]`):** $\text{Test MSE} = 7.13 \times 10^{-8} \pm 2.36 \times 10^{-8}$, $\text{Test RMSE} = 2.64 \times 10^{-4}$, $\text{Test MAE} = 1.94 \times 10^{-4}$, $R^2 = 0.999970$.
  - **Checkpoint `best.pt` (`seed = 2`):** $\text{Test MSE} = 4.42 \times 10^{-8}$, $\text{Test RMSE} = 2.10 \times 10^{-4}$, $R^2 = 0.999981$; trên lưới $10.000$ điểm đạt $\text{Grid MSE} = 5.73 \times 10^{-8}$, sai số tuyệt đối cực đại toàn miền $\max |y - \hat{y}| = 1.07 \times 10^{-3}$ ($< 0.4\%$ đỉnh hàm $0.2690$).
- **Nhận xét giới hạn:** Tập test $100$ mẫu chỉ có $11$ điểm có $y > 0.1$, nên việc đánh giá bổ sung trên lưới $100 \times 100$ và bản đồ nhiệt $|y - \hat{y}|$ là cần thiết để xác nhận mô hình khớp chính xác cả ở đỉnh lẫn biên.

---

## Biên bản Review chéo của TV3 (T3.7 & T3.8)

1. **Review `src/engine/trainer.py` và `src/run_experiment.py` (T3.7):**
   - Hàm `fit(model, train_loader, val_loader, cfg, run_dir)` hoàn toàn không nhận `test_loader` hay dữ liệu test.
   - Trọng số `best_state` được sao chép bằng `copy.deepcopy(model.state_dict())` mỗi khi `val_loss` giảm và nạp lại trước khi trả về.
   - Tập `test.csv` chỉ được đọc khi có cờ `--eval-test` hoặc trong bước tổng hợp cuối `src/make_report_assets.py`. Không có rò rỉ dữ liệu test.
2. **Review so sánh Optimizer T2.8 của TV2 (`src/summarize_optimizer.py` – T3.8):**
   - Đã kiểm tra `configs/grid_optimizer.json` và `results/grid_optimizer/best_per_optimizer.csv`: cả 6 bộ tối ưu (Adam, NAdam, RMSProp, Momentum, Nesterov, SGD) đều được dò công bằng trên cùng lưới 7 mức learning rate $\{10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, 10^{-2}, 3\times 10^{-2}, 10^{-1}\} \times 3\text{ seed}$ chỉ dựa trên Validation MSE.
   - Adam (`lr = 0.01`) đạt mean Validation MSE thấp nhất ($1.88 \times 10^{-5} \pm 6.08 \times 10^{-6}$), nhất quán với lựa chọn baseline của nhóm.

---

## Câu hỏi vấn đáp dự kiến cho TV3 (Chuẩn bị bảo vệ)

1. **Vì sao $B = 1$ không thể xấp xỉ được đỉnh hàm Gaussian 2D?**
   $\rightarrow$ Vì $\hat{y} = w_1 \sigma(u_1 x_1 + u_2 x_2 + b_1) + c$ là hàm đơn điệu theo tổ hợp tuyến tính $z = u_1 x_1 + u_2 x_2 + b_1$. Mọi đường thẳng $u_1 x_1 + u_2 x_2 = \text{const}$ đều có cùng giá trị đầu ra, nên đồ thị chỉ là một mặt bậc thang kéo dài vô hạn (ridge function), không thể bao kín một đỉnh cục bộ 2 chiều. Cần tối thiểu $B \ge 3$ mặt cắt giao nhau để bao quanh một vùng lồi trên mặt phẳng $\mathbb{R}^2$.
2. **Vì sao MSE nhỏ (ví dụ $2.3 \times 10^{-3}$ ở $B = 1$) vẫn là mô hình rất tệ?**
   $\rightarrow$ Vì giá trị cực đại của hàm chỉ là $y_{\max} \approx 0.2690$ và hơn $62\%$ mẫu trên miền chữ nhật có $y < 0.01$ ($\text{std}(y) \approx 0.050$). Ngay cả mô hình tầm thường luôn dự đoán hằng số $\hat{y} = \bar{y}_{\text{train}} \approx 0.0315$ cũng đã có Validation MSE $= 2.49 \times 10^{-3}$ ($R^2 = 0$). Do đó phải đối chiếu với baseline dự đoán trung bình, xem chỉ số $R^2$ và vẽ bề mặt 3D.
3. **Underfitting và Overfitting được nhận biết như thế nào trong bảng số liệu của nhóm?**
   $\rightarrow$ Underfitting thể hiện rõ ở $B = 1 \dots 4$ khi cả Train MSE và Validation MSE đều rất cao ($10^{-3} \rightarrow 4\times 10^{-4}$) và bề mặt 3D bị bẹp đỉnh. Overfitting (nếu có) sẽ làm Train MSE giảm sâu nhưng Validation MSE tăng lên (khoảng cách $\text{Val} - \text{Train}$ dương lớn). Trong bài toán này, do dữ liệu sạch không nhiễu và có early stopping, khoảng cách $\text{Val} - \text{Train}$ xấp xỉ $0$, không xảy ra overfitting.
4. **Vì sao phải đổi lại thang $y$ gốc trước khi tính MSE, MAE, RMSE?**
   $\rightarrow$ Trong lúc train, nhãn được chia cho $Y_{\text{SCALE}} = y_{\max} \approx 0.2690$ để đưa về $[0, 1]$ giúp ổn định gradient. Nếu tính MSE trên thang đã scale thì MSE sẽ bị phóng đại khoảng $1 / 0.2690^2 \approx 13.82$ lần và không phản ánh đúng đơn vị mật độ xác suất ban đầu của bài toán.
5. **Tập Test 100 mẫu có đủ tin cậy không? Nhóm đã khắc phục như thế nào?**
   $\rightarrow$ $100$ mẫu lấy ngẫu nhiên đều trên diện tích $32$ chỉ có $11$ điểm rơi vào vùng đỉnh $y > 0.1$, nên có phương sai lấy mẫu nhất định. Nhóm đã bổ sung đánh giá trên lưới đều $100 \times 100 = 10.000$ điểm (`evaluate_grid`) và vẽ bản đồ nhiệt $|y - \hat{y}|$ để kiểm chứng sai số trên toàn bộ mặt phẳng.

