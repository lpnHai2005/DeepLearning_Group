# Ghi chú slide TV1 – Dữ liệu

## Slide 1 – Bài toán và miền dữ liệu

- Xấp xỉ mật độ Gaussian 2D bằng MLP PyTorch.
- Nêu μ = [0, 0], ma trận C và miền x1 ∈ [-2,2], x2 ∈ [-4,4].
- Nhấn mạnh y_max ≈ 0.2690, không phải 1.
- Hình gợi ý: BTL01/figures/eda_surface_contour.png.
- Lời trình bày: các mẫu được lấy đều trên miền chữ nhật; nhãn được tính trực tiếp từ hàm nên không có nhiễu.

## Slide 2 – Các tập dữ liệu và phân bố

- Train pool: 2.000 mẫu, seed 2026; validation: 200, seed 2027; test: 100, seed 2028.
- Các mức A lấy A điểm đầu của cùng train pool; validation dùng chọn mô hình, test để đánh giá cuối.
- Khoảng 62.6% nhãn train pool nhỏ hơn 0.01; do đó MSE nhỏ cần đọc cùng metric khác và chất lượng dự đoán quanh vùng đỉnh.
- Hình gợi ý: BTL01/figures/eda_scatter_3d.png và eda_histograms.png.

## Slide 3 – Kết quả Thực nghiệm A

- Baseline MLP được chọn bằng validation: sigmoid, learning rate 0.01; ba seed có validation MSE 1.891e-5 ± 5.893e-6 tại A=1000.
- Chạy đủ A = 200…2000 với 3 seed. Validation MSE thấp nhất tại A=1600: 1.517e-5 ± 1.423e-6; A=200 đạt 7.057e-5 ± 1.803e-5.
- Chèn `figures/expA_validation_mse.png`; nói rõ sai số giảm mạnh đến khoảng A=600 rồi dao động, không giảm đều; tại A=2000 lỗi tăng so với A=1600.
- Baseline hằng số theo mean train có validation MSE xấp xỉ 2.49e-3; train/validation MSE gần nhau ở nhiều mức A.
- Chỉ kết luận theo validation; test không được dùng để chọn A và chưa được đánh giá trong lượt này.
- Bảng đủ MSE/MAE/RMSE cùng mean ± std nằm trong `report/THUC_NGHIEM_A.md` và `results/expA/aggregate.csv`.
