# Ghi chú slide TV2 – Mô hình, huấn luyện, Thực nghiệm C

## Slide 5 – Sơ đồ MLP và công thức

- Hình: **figures/mlp_1hidden_B10.png** (baseline [2, 10, 1], sigmoid, 41 tham số).
- Forward: z = W a + b, a = σ(z); đầu ra **tuyến tính** vì là hồi quy.
- Loss MSE; backprop: δ_out = (2/N)(ŷ − y), δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ σ′(z⁽ˡ⁾), ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ a⁽ˡ⁻¹⁾ᵀ.
- Nói thêm: σ′ ≤ 0.25 nên mạng sigmoid sâu có nguy cơ vanishing gradient → Thực nghiệm C đo trực tiếp.
- Kiểm chứng: **tests/test_backprop.py** so backprop tính tay và đạo hàm số với autograd.

## Slide 6 – Pipeline PyTorch

- Dữ liệu: scale bằng hằng số bài toán → TensorDataset → DataLoader (batch 32, xáo có seed).
- Vòng lặp: zero_grad → forward → loss → backward → step; train()/eval(); validate dưới no_grad.
- Hai bẫy đã chặn: shape (N,1) với (N,) trong MSELoss (assert shape); state_dict là tham chiếu (deepcopy khi lưu best).
- Early stopping theo val (patience 200), lưu best.pt; fit() không nhận test.
- Tái lập: chạy lại trên cùng máy cho cùng số; khác máy thì val MSE lệch tương đối dưới 0.01%.

## Slide 7 – Baseline (phần TV2 gửi TV4)

- Hình: **figures/grid_baseline_val_mse.png** (chọn lr và activation) và **figures/optimizer_loss.png** (so sánh optimizer).
- Chọn baseline chỉ theo val: sigmoid, lr 0.01, val MSE 1.89e-5 ± 5.89e-6. Tanh lr 0.003 (2.96e-5 ± 1.11e-5) chồng khoảng ± std, nên đây là chọn theo mean.
- So 6 optimizer, mỗi cái dò 7 lr, 126 run: Adam 1.88e-5, Nadam 2.51e-5, RMSProp 3.38e-5, Momentum 8.35e-5, Nesterov 1.01e-4, SGD 3.57e-4.
- Câu chốt: Adam, Nadam, RMSProp chưa tách được với 3 seed; không có optimizer nào tốt hơn Adam nên giữ baseline, không chạy lại A, B, C.
- Nói rõ hạn chế: sigmoid được chọn dưới Adam, và phép so chỉ trên mạng một tầng ẩn.

## Slide 10 – Kết quả C (kèm thời gian)

- Chèn **figures/expC_error.png** (trái: lỗi theo C; phải: sâu so với rộng cùng số tham số).
- C=1 → C=2: val MSE giảm ~180 lần (1.89e-5 → 1.05e-7); C=2–4 cùng bậc 1e-7, error bar C=3/C=4 chồng nhau.
- Cùng số tham số, mạng sâu tốt hơn mạng rộng 16–28 lần → lợi ích do độ sâu, không chỉ do nhiều tham số.
- Vanishing gradient **có** lúc khởi tạo (C=4: lớp đầu nhỏ hơn lớp ra ~3.600 lần, **figures/expC_grad_norms.png**) nhưng **không** cản huấn luyện với Adam.
- Thời gian/epoch: 20.2 → 30.5 ms (C=1 → 4); mạng rộng 373 tham số vẫn 20.2 ms → chi phí theo số tầng.
- Câu chốt: ngược với Sample_Report; nêu hạn chế val loss dao động mạnh, checkpoint theo val nên số hơi lạc quan.

## Demo (Slide 15)

```
python src/run_experiment.py --config configs/demo.json
```

Chạy khoảng 10 giây (1 seed, 300 epoch, baseline). In ra val MSE, epoch tốt nhất, thời gian; kết quả ở **results/demo/base/seed_0/** (history.csv, best.pt).

## Câu hỏi vấn đáp cần chuẩn bị

1. Vì sao đầu ra không dùng sigmoid? → Hồi quy; sigmoid giới hạn (0,1) và làm gradient nhỏ ở vùng bão hòa.
2. train() và eval() khác gì, no_grad để làm gì?
3. Viết ∂L/∂W cho tầng cuối và tầng đầu (dùng công thức slide 5).
4. Checkpoint chọn theo tập nào, vì sao không theo test?
5. Vì sao thời gian tăng theo C mà không tăng theo B?
6. Có vanishing gradient không? Bằng chứng nào? Vì sao vẫn học được?
7. Nếu loss giảm mà dự đoán gần hằng số thì nghi lỗi gì? → shape (N,1) với (N,).
8. Vì sao chọn Adam? → Mean val thấp nhất khi mỗi optimizer dò lr riêng; Nadam và RMSProp chưa tách được với 3 seed, nên giữ Adam chứ không khẳng định Adam tốt hơn hẳn.
