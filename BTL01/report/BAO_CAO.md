<!--
Nguồn nội dung của báo cáo Word. 
Cú pháp riêng:
  [@fig:key] / [@tab:key]          -> "Hình 2.1" / "Bảng 4.1" (đánh số tự động theo chương)
  ![chú thích](figures/x.png){#fig:key width=16}
  Table: chú thích {#tab:key}      (dòng ngay trước một bảng markdown)
  @table <csv> {#tab:key} | col=Tiêu đề; ... | chú thích     (bảng sinh từ report/tables/<csv>.csv, không gõ tay)
       tùy chọn sau chú thích: | rows=v1,v2   (lọc và giữ thứ tự dòng)
       col dạng a/b: tỉ số mean_a/mean_b;  csv có thể ghép bằng +, hoặc là đường dẫn results/...csv
  $$ công thức $$                  -> dòng công thức căn giữa
Số liệu trong văn bản phải có trong report/tables/, results/ hoặc report/NHAT_KY_QUYET_DINH.md.
-->

# Chương 1. Giới thiệu

## 1.1. Mục tiêu

Bài tập lớn yêu cầu xây dựng và huấn luyện mạng perceptron đa tầng (MLP) bằng PyTorch để xấp xỉ một hàm số hai biến, rồi làm thực nghiệm có kiểm soát để trả lời ba câu hỏi: số mẫu huấn luyện (A), số nơ-ron của tầng ẩn (B) và số tầng ẩn (C) ảnh hưởng thế nào đến sai số. Ngoài việc chạy được mô hình, nhóm đặt mục tiêu mọi con số trong báo cáo phải truy vết được về file log, CSV hoặc hình trong repo, và tập test chỉ được dùng ở bước đánh giá cuối.

## 1.2. Phát biểu bài toán

Cho hàm mật độ Gaussian hai chiều với μ = (0, 0) và ma trận hiệp phương sai C = [[0.3, −0.5], [−0.5, 2]]. Lấy ngẫu nhiên x = (x₁, x₂) trong miền −2 ≤ x₁ ≤ 2, −4 ≤ x₂ ≤ 4 và tính nhãn y = f(x₁, x₂). Cần xây dựng MLP hai đầu vào, một đầu ra ŷ sao cho ŷ ≈ y; có tập validation riêng để chọn mô hình và một tập test độc lập 100 mẫu. Đây là bài toán hồi quy, đánh giá bằng MSE cùng MAE, RMSE và R².

## 1.3. Kết quả chính

- Mạng sâu tốt hơn rõ rệt so với mạng nông: mô hình cuối `[10,10,10]` (261 tham số) đạt Test MSE 7.13e-8 ± 2.36e-8 và R² = 0.99997, thấp hơn khoảng 320 lần so với baseline một tầng `[10]`.
- Ở cùng số tham số, mạng nhiều tầng tốt hơn mạng một tầng rộng 16–28 lần, nên lợi ích đến từ độ sâu chứ không chỉ từ số tham số.
- Với dữ liệu không nhiễu, nhóm không quan sát được đường chữ U của quá khớp, kể cả khi giảm còn A = 200 mẫu; chỉ thấy khoảng cách train–validation tăng dần. Chương 5 phân tích nguyên nhân.
- Một phần lỗi cao ở A nhỏ là do được huấn luyện ít bước cập nhật hơn, không chỉ do ít dữ liệu; nhóm tách hai yếu tố này bằng một thí nghiệm đối chứng.

## 1.4. Thành viên và đóng góp

Table: Thành viên, nhiệm vụ và tỉ lệ đóng góp. {#tab:members}

| TV | Họ tên | MSSV | Nhiệm vụ chính | Đóng góp |
|---|---|---|---|---:|
| TV1 | [Họ tên – Trung Kiên] | [MSSV] | Sinh dữ liệu, EDA, khung repo, README; Thực nghiệm A và đối chứng cùng ngân sách bước; Ch.2, mục 4.2 | 25% |
| TV2 | [Họ tên – Nguyên Hải] | [MSSV] | Lớp MLP, vòng huấn luyện, early stopping, checkpoint; chọn baseline, so sánh optimizer; Thực nghiệm C và đối chứng cùng số tham số; Ch.3, mục 4.4 | 25% |
| TV3 | [Họ tên – Viễn Thông] | [MSSV] | Hàm đánh giá, thư viện vẽ hình; Thực nghiệm B (A = 1000 và A = 200); chọn mô hình cuối và đánh giá test; mục 4.3, 4.5 | 25% |
| TV4 | [Họ tên – Hoàng Bảo] | [MSSV] | Sơ đồ, bảng tự động, kiểm tra tái lập; mục 4.1, Ch.1, Ch.5, Ch.6; ghép báo cáo và slide | 25% |

# Chương 2. Dữ liệu

## 2.1. Hàm mục tiêu

$$ f(x) = 1 / (2π·√det C) · exp(−½ · (x − μ)ᵀ C⁻¹ (x − μ)),   μ = (0, 0),   C = [[0.3, −0.5], [−0.5, 2]] $$

Ta có det C = 0.3·2 − 0.5² = 0.35, nên giá trị lớn nhất tại tâm là y_max = 1/(2π√0.35) ≈ 0.2690. Nhãn vì vậy nằm trong (0; 0.269], không phải [0; 1]. Ma trận nghịch đảo là C⁻¹ = (1/0.35)·[[2, 0.5], [0.5, 0.3]] ≈ [[5.714, 1.429], [1.429, 0.857]]. Khai triển dạng toàn phương trong hàm mũ:

$$ Q(x) = (x − μ)ᵀ C⁻¹ (x − μ) = 5.714·x₁² + 2.857·x₁x₂ + 0.857·x₂² $$

Hệ số của x₁² lớn hơn x₂² khoảng 6.7 lần, nên hàm giảm nhanh theo x₁ và chậm theo x₂: "đồi" Gaussian hẹp theo x₁, dài theo x₂. Số hạng chéo 2.857·x₁x₂ dương (do hiệp phương sai −0.5 âm) làm Q nhỏ hơn khi x₁ và x₂ trái dấu, tức đồi nghiêng về phía góc phần tư II và IV. Hệ số tương quan ρ = −0.5/√(0.3·2) ≈ −0.645. Trị riêng của C là 2.136 và 0.164; vectơ riêng ứng với trị riêng lớn là (−0.263, 0.965), lệch khoảng 15° so với trục x₂. [@tab:handcalc] tính tay vài điểm để kiểm tra hàm cài đặt trong `src/data/gaussian.py`.

Table: Giá trị f tại một số điểm (tính bằng `gaussian_pdf`, khớp công thức trên). {#tab:handcalc}

| (x₁, x₂) | Q(x) | f(x) | Nhận xét |
|---|---:|---:|---|
| (0, 0) | 0 | 0.2690 | Đỉnh, bằng y_max |
| (1, 0) | 5.714 | 0.0155 | Ra xa 1 đơn vị theo x₁: còn 5.7% đỉnh |
| (0, 2) | 3.429 | 0.0484 | Ra xa 2 đơn vị theo x₂ vẫn còn 18% đỉnh |
| (−0.5, 1) | 0.857 | 0.1753 | Trái dấu: nằm dọc trục chính của đồi |
| (0.5, 1) | 3.714 | 0.0420 | Cùng dấu: cùng khoảng cách nhưng thấp hơn 4 lần |

Miền lấy mẫu [−2, 2] × [−4, 4] chứa 99.52% thể tích của mật độ (tích phân số trên lưới), nên gần như toàn bộ phần có giá trị của hàm nằm trong miền.

## 2.2. Lấy mẫu và chia dữ liệu

Mỗi điểm x được lấy **đều** và độc lập trên hình chữ nhật, nhãn tính trực tiếp từ f, không thêm nhiễu. Nhóm chọn lấy mẫu đều (không lấy theo phân phối Gaussian) vì mục tiêu là xấp xỉ hàm trên toàn miền, kể cả vùng đuôi. Ba tập được sinh độc lập bằng `numpy.random.default_rng` với seed riêng ([@tab:split]); script `src/generate_data.py` ghi ba file CSV và `data/meta.json`. Các mức A của Thực nghiệm A lấy A mẫu **đầu tiên** của train pool, nên các tập train lồng nhau: tăng A chỉ thêm mẫu mới, không thay mẫu cũ.

Table: Chia dữ liệu. {#tab:split}

| Tập | Số mẫu | Seed | Vai trò |
|---|---:|---:|---|
| Train pool | 2000 | 2026 | Lấy tập con A = 200, 400, …, 2000 |
| Validation | 200 | 2027 | Early stopping, chọn checkpoint và chọn cấu hình |
| Test | 100 | 2028 | Chỉ đánh giá mô hình cuối, sau khi đã chọn xong bằng validation |

**Tiền xử lý.** Đầu vào chia cho biên miền (x₁/2, x₂/4) để về [−1, 1]; nhãn chia cho y_max ≈ 0.2690 để về [0, 1]. Đây là lựa chọn của nhóm, không phải yêu cầu của đề. Các hằng số này lấy từ đề bài, không ước lượng từ dữ liệu, nên không dùng thông tin của validation hay test. Mọi chỉ số trong báo cáo được tính trên thang y gốc.

## 2.3. Thống kê và trực quan hóa

Table: Thống kê mô tả của nhãn y (nguồn `data/meta.json`, `results/eda/data_summary.csv`). {#tab:ystats}

| Tập | Min | Max | Mean | Std | Tỉ lệ y < 0.01 |
|---|---:|---:|---:|---:|---:|
| Train pool | 8.9e-14 | 0.2687 | 0.0314 | 0.0581 | 62.6% |
| Validation | 1.1e-11 | 0.2526 | 0.0264 | 0.0497 | 66.0% |
| Test | 4.5e-13 | 0.2269 | 0.0268 | 0.0485 | 61.0% |

x₁ và x₂ phủ gần hết miền ở cả ba tập (min/max sát biên, độ lệch chuẩn gần 4/√12 ≈ 1.15 và 8/√12 ≈ 2.31 của phân phối đều). Nhãn mất cân bằng mạnh: 62.6% mẫu train có y < 0.01, đỉnh chỉ xuất hiện ở một vùng hẹp quanh tâm. Hệ quả là MSE tuyệt đối rất nhỏ ngay cả với mô hình kém; mô hình luôn đoán bằng trung bình đã có validation MSE 2.49e-3. Vì vậy nhóm luôn đặt mô hình hằng số vào bảng làm mốc và báo cáo thêm R², sai số trên lưới đều và bản đồ sai số.

![Bề mặt Gaussian thật (trái) và đường đồng mức (phải) trên miền lấy mẫu.](figures/eda_surface_contour.png){#fig:surface width=16}

![Histogram của x₁, x₂ và y; trục đếm của y theo thang log để thấy các bin thưa ở vùng đỉnh.](figures/eda_histograms.png){#fig:hist width=16}

![Scatter 3D của train pool, validation và test.](figures/eda_scatter_3d.png){#fig:scatter width=8.5}

# Chương 3. Mô hình và cài đặt

## 3.1. Kiến trúc MLP và lan truyền thuận

Lớp `MLP` (`src/model/mlp.py`) nhận danh sách số nơ-ron từng tầng ẩn `hidden_sizes` và tên hàm kích hoạt, nên một lớp dùng chung cho cả ba thực nghiệm: A giữ `[10]`, B đổi `[B]`, C đổi `[10]×C`. Tầng ra tuyến tính vì đây là hồi quy.

$$ z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾,   a⁽ˡ⁾ = σ(z⁽ˡ⁾),   a⁽⁰⁾ = (x₁/2, x₂/4),   ŷ = W⁽ᴸ⁾ a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾ $$

với σ(z) = 1/(1 + e⁻ᶻ). Mạng một tầng [2, B, 1] có 4B + 1 tham số; mạng C tầng, mỗi tầng 10 nơ-ron, có 41, 151, 261, 371 tham số với C = 1…4. Hàm `n_params` đếm trực tiếp từ model và được kiểm thử khớp công thức.

![Kiến trúc baseline [2, 10, 1], sigmoid, 41 tham số (vẽ từ model thật bằng `src/visual/draw_mlp.py`).](figures/mlp_1hidden_B10.png){#fig:mlp1 width=9}

![Mạng ba tầng ẩn [10, 10, 10] (261 tham số), là mô hình được chọn cuối cùng.](figures/mlp_C3_10.png){#fig:mlp3 width=14}

```python
ACTS = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}

class MLP(nn.Module):
    def __init__(self, hidden_sizes=(10,), activation="tanh"):
        super().__init__()
        layers, d = [], 2
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), ACTS[activation]()]
            d = h
        layers.append(nn.Linear(d, 1))              # đầu ra tuyến tính (hồi quy)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)              # (N,) cùng shape với y
```

## 3.2. Hàm mất mát, lan truyền ngược và Adam

Hàm mất mát là MSE trên một batch N mẫu, L = (1/N)·Σ(ŷₙ − yₙ)². Gradient lan ngược từ tầng ra:

$$ δ⁽ᴸ⁾ = (2/N)(ŷ − y),   δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ σ′(z⁽ˡ⁾),   ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ a⁽ˡ⁻¹⁾ᵀ,   ∂L/∂b⁽ˡ⁾ = Σₙ δₙ⁽ˡ⁾ $$

với σ′(z) = σ(z)(1 − σ(z)) ≤ 0.25. Mỗi tầng sigmoid nhân gradient với một hệ số không quá 0.25, nên mạng sâu có nguy cơ vanishing gradient; mục 4.4 đo chuẩn gradient từng tầng để kiểm tra bằng số liệu. PyTorch tính gradient bằng autograd; `tests/test_backprop.py` so autograd với lời giải tay bằng NumPy cho mạng 2→2→1 (sai khác < 10⁻¹²) và với đạo hàm số qua `torch.autograd.gradcheck`. Adam cập nhật tham số theo

$$ mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,   vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²,   θₜ = θₜ₋₁ − η·m̂ₜ / (√v̂ₜ + ε) $$

với m̂, v̂ là các ước lượng đã hiệu chỉnh độ lệch, β = (0.9, 0.999), ε = 10⁻⁸ (mặc định PyTorch) và η = 0.01.

## 3.3. Cài đặt PyTorch và quy trình huấn luyện

[@fig:pipeline] tóm tắt quy trình. `make_loader` (`src/data/dataset.py`) scale dữ liệu, gói vào `TensorDataset` và `DataLoader` batch 32; loader train xáo trộn bằng `torch.Generator` gắn seed nên thứ tự batch tái lập được. Vòng lặp một epoch theo đúng thứ tự đề yêu cầu:

```python
def train_one_epoch(model, loader, opt, crit):
    model.train()
    tot = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        assert pred.shape == yb.shape      # (N,1) với (N,) sẽ broadcast thành (N,N)
        loss = crit(pred, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0)    # batch cuối có thể nhỏ hơn 32
    return tot / len(loader.dataset)
```

- `validate` chạy trong `model.eval()` và `@torch.no_grad()`, không xây đồ thị tính toán.
- `forward` trả shape (N,) và có `assert` shape: nếu dự đoán (N,1) còn nhãn (N,), `MSELoss` broadcast thành ma trận N×N, loss vẫn giảm nhưng mô hình học ra hằng số.
- `fit` sau mỗi epoch tính val loss; khi val tốt hơn mức tốt nhất thì lưu `copy.deepcopy(model.state_dict())` (state_dict là tham chiếu, không deepcopy thì bản "tốt nhất" bị ghi đè). Nếu val không cải thiện sau patience = 200 epoch thì dừng, nạp lại trọng số tốt nhất và ghi `best.pt`. `fit` không nhận dữ liệu test.
- Mỗi run lưu `config.json`, `history.csv` (train loss, val loss, chuẩn gradient từng lớp theo epoch), `metrics.json`, `best.pt` trong `results/<thí nghiệm>/<giá trị>/seed_<k>/`; `run_experiment.py` gom lại thành `summary.csv`. Test chỉ được tính khi chạy với cờ `--eval-test`.

![Quy trình train/validation/test. Test chỉ đi vào bước đánh giá cuối (vẽ bằng `src/visual/draw_pipeline.py`).](figures/pipeline.png){#fig:pipeline width=16}

Table: Tổ chức mã nguồn. {#tab:code}

| File | Chức năng |
|---|---|
| `src/data/gaussian.py`, `src/generate_data.py` | Hàm Gaussian, hằng số bài toán, sinh và lưu ba tập dữ liệu |
| `src/data/dataset.py`, `src/data/eda.py` | Đọc CSV, scale, DataLoader, tập con lồng nhau; thống kê và hình EDA |
| `src/model/mlp.py` | Lớp `MLP`, đếm tham số |
| `src/engine/trainer.py`, `src/engine/evaluate.py` | Vòng train/validate, early stopping, checkpoint; MSE, MAE, RMSE, R², dự đoán trên lưới |
| `src/run_experiment.py`, `configs/*.json` | Chạy một thí nghiệm theo config, lặp qua giá trị × seed, ghi `summary.csv` |
| `src/posthoc_test.py`, `src/make_report_assets.py` | Hậu kiểm test trên `best.pt` đã đóng băng; chọn mô hình cuối, đánh giá test và lưới |
| `src/summarize_*.py`, `src/analyze_stopping.py`, `src/make_tables.py` | Tổng hợp, vẽ hình, phân tích dừng sớm, sinh bảng `report/tables/` |
| `tests/` | Kiểm thử hàm Gaussian, dữ liệu, backprop, trainer, và config khớp log |

## 3.4. Môi trường và tái lập

Mã chạy trên CPU: Windows 11, Python 3.12.10, PyTorch 2.6.0, NumPy 2.5.2, Pandas 3.0.3, Matplotlib 3.10.1 (`results/env.txt`). Mạng chỉ có 5–373 tham số nên CPU là đủ. `set_seed` cố định seed của random, NumPy và PyTorch; `torch.set_num_threads(1)` cho mọi run để thời gian so sánh được. Trên cùng máy, chạy lại cùng cấu hình cho ra cùng số đến từng chữ số (run A = 1000 của Thực nghiệm A trùng run sigmoid/lr 0.01 của lưới baseline); giữa hai máy, validation MSE lệch tương đối dưới 0.01%. Lệnh tái lập ở Phụ lục A.

# Chương 4. Thực nghiệm

## 4.1. Thiết lập và quy trình đánh giá công bằng

**Nguyên tắc.** (1) Mỗi nhóm thực nghiệm chỉ thay đổi một yếu tố so với baseline ([@tab:hparams]). (2) Mỗi cấu hình chạy 3 seed (0, 1, 2), khác nhau ở khởi tạo trọng số và thứ tự batch; dữ liệu giữ nguyên; báo cáo trung bình ± độ lệch chuẩn mẫu. (3) Mọi lựa chọn (siêu tham số, checkpoint, cấu hình cuối) chỉ dựa trên validation. Cột test của A, B, C là **hậu kiểm**: `src/posthoc_test.py` nạp `best.pt` đã đóng băng và đo trên test sau khi mọi lựa chọn đã xong, chỉ để trình bày đủ ba đường train/validation/test. (4) `tests/test_config_evidence.py` kiểm tra `configs/baseline.json` và `configs/final.json` khớp với cấu hình có val MSE thấp nhất trong log.

Table: Siêu tham số baseline và cách chọn. "Chọn trước" nghĩa là chọn theo thông lệ, sau đó mới chạy thí nghiệm kiểm chứng. {#tab:hparams}

| Siêu tham số | Giá trị | Cách chọn | Minh chứng |
|---|---|---|---|
| Kiến trúc | [2, 10, 1], đầu ra tuyến tính | Đề bài | — |
| Hàm kích hoạt, learning rate | sigmoid, 0.01 | Khảo sát 3 × 3 rồi chọn | [@tab:gridbase] |
| Optimizer | Adam, β = (0.9, 0.999) | Chọn trước, kiểm chứng sau | [@tab:opt] |
| Batch size | 32 | Chọn trước, kiểm chứng sau | [@tab:batch] |
| Patience early stopping | 200 epoch theo val loss | Chọn trước, kiểm chứng sau | [@fig:patience] |
| Số epoch tối đa | 2000 | Chọn trước; là hạn chế (mục 5.4) | `results/stopping/epoch_cap.csv` |
| Số mẫu train | A = 1000 (trừ Thực nghiệm A) | Giữa lưới A | — |
| Seed | huấn luyện 0, 1, 2; dữ liệu 2026/2027/2028 | Chọn trước | `data/meta.json` |
| Khởi tạo trọng số | mặc định `nn.Linear` | Mặc định PyTorch | — |

**Chọn hàm kích hoạt và learning rate.** Lưới {tanh, sigmoid, relu} × {0.001, 0.003, 0.01} × 3 seed (27 run), mạng `[10]`, Adam, A = 1000. Sigmoid với lr 0.01 có val MSE trung bình thấp nhất ([@tab:gridbase]); tanh với lr 0.003 đứng thứ hai nhưng khoảng ± std của hai cấu hình chồng nhau, nên đây là lựa chọn theo trung bình, chưa chứng minh sigmoid tốt hơn tanh. ReLU kém nhất ở mọi lr: mạng 10 nơ-ron ReLU chỉ tạo được xấp xỉ tuyến tính từng khúc rất thô cho một bề mặt cong.

@table grid_baseline {#tab:gridbase} | value=lr_activation; val_mse=Val MSE; val_mae=Val MAE; best_epoch=Epoch tốt nhất | Lưới chọn hàm kích hoạt và learning rate (27 run, chỉ xét validation). Nguồn `results/grid_baseline/summary.csv`. | sort=val_mse

**Kiểm chứng optimizer.** Adam được chọn trước (lưới trên đã dùng Adam), sau đó nhóm so sánh 6 optimizer, mỗi optimizer dò riêng 7 lr từ 0.001 đến 1.0 (126 run, tắt early stopping để chạy đủ 2000 epoch). Adam ở lr 0.01 vẫn tốt nhất ([@tab:opt], [@fig:opt]), nên baseline giữ nguyên. Khoảng ± std của Adam, Nadam và RMSProp chồng nhau; Momentum và Nesterov kém Adam 4–5 lần, SGD kém khoảng 19 lần.

@table grid_optimizer {#tab:opt} | value=Optimizer_lr; val_mse=Val MSE; best_epoch=Epoch tốt nhất | Kết quả ở lr tốt nhất của từng optimizer (126 run). Nguồn `results/grid_optimizer/summary.csv`. | rows=adam_0.01,nadam_0.01,rmsprop_0.003,momentum_0.3,nesterov_0.1,sgd_0.3

![Train loss và val loss theo epoch ở lr tốt nhất của mỗi optimizer (trung bình 3 seed, trung vị trượt 25 epoch).](figures/optimizer_loss.png){#fig:opt width=16}

**Kiểm chứng batch size và patience.** Batch 32 có val MSE thấp nhất trong {16, 32, 64, 128} ([@tab:batch]), nhưng chênh lệch nhỏ hơn độ lệch chuẩn nên chỉ có thể nói batch 32 không kém các lựa chọn khác. Để kiểm chứng patience, `src/analyze_stopping.py` mô phỏng các mức patience trên đường val của run Adam lr 0.01 chạy đủ 2000 epoch ([@fig:patience]): so với không dừng sớm, patience 200 chỉ kém 0.7% val MSE, còn patience 100 kém 31%.

@table grid_batch {#tab:batch} | value=Batch; val_mse=Val MSE; best_epoch=Epoch tốt nhất; epochs_run=Số epoch chạy | Kiểm chứng batch size trên baseline (12 run). Nguồn `results/grid_batch/summary.csv`.

![Mô phỏng patience trên đường val chạy đủ 2000 epoch: val MSE tương đối so với không dừng sớm.](figures/stopping_patience.png){#fig:patience width=15}

## 4.2. Thực nghiệm A: số mẫu huấn luyện

**Thiết kế.** A ∈ {200, 400, …, 2000}, mỗi mức lấy A mẫu đầu của train pool, mạng `[10]`, các siêu tham số khác như baseline, 3 seed (30 run).

@table expA {#tab:expA} | value=A; train_mse=Train MSE; val_mse=Val MSE; test_mse=Test MSE*; val_mae=Val MAE; test_mae=Test MAE*; test_rmse=Test RMSE* | Kết quả Thực nghiệm A (* hậu kiểm). Nguồn `results/expA/summary.csv`, `posthoc_test.log`.

![Train, validation và test (hậu kiểm) MSE theo A, error bar ± std qua 3 seed; đường ngang là mô hình hằng số.](figures/expA_validation_mse.png){#fig:expA width=13}

**Nhận xét.** Val MSE giảm từ 7.06e-5 (A = 200) xuống 2.16e-5 (A = 600) rồi dao động trong khoảng 1.5e-5–2.7e-5; thấp nhất ở A = 1600 (1.52e-5), tăng lại ở A = 2000 (2.57e-5). Train và validation gần nhau ở mọi mức, không thấy quá khớp tăng theo A. Test (hậu kiểm) đi cùng xu hướng với validation và nằm trong khoảng ± std của nó, cho thấy validation 200 mẫu đại diện tốt cho dữ liệu chưa thấy.

**Đối chứng cùng ngân sách bước cập nhật.** Log cho thấy batch cố định 32 làm A nhỏ có ít bước cập nhật hơn: A = 200 chỉ 7 bước/epoch, tức khoảng 14 000 bước trong 2000 epoch, so với 64 000 bước của A = 1000; 8/9 run ở A = 200–600 chạm trần epoch khi vẫn đang cải thiện. Nhóm chạy lại A với max_epochs và patience nhân 32/⌈A/32⌉ để mọi mức A có cùng khoảng 64 000 bước (`configs/expA_steps.json`). Kết quả ([@tab:expAsteps], [@fig:expAsteps]): A = 200 giảm từ 7.06e-5 xuống 4.04e-5 (−43%), các mức A ≤ 800 giảm 7–43%. Như vậy một phần đáng kể của "lỗi cao khi ít dữ liệu" là do huấn luyện chưa đủ. Sau khi cân bằng số bước, A = 200 vẫn kém A = 1000 khoảng 2 lần, nên số mẫu vẫn có ảnh hưởng thật. Từ A ≥ 600, val MSE nằm trong dải 1.6e-5–2.6e-5 với ± std chồng nhau: giới hạn chính là dung lượng của mạng `[10]` (41 tham số), không phải dữ liệu.

@table expA_steps {#tab:expAsteps} | value=A; epochs_run=Số epoch chạy; val_mse=Val MSE; test_mse=Test MSE* | Đối chứng A cùng ngân sách khoảng 64 000 bước cập nhật (* hậu kiểm). Nguồn `results/expA_steps/summary.csv`.

![Val và test MSE theo A: liền nét là Thực nghiệm A (2000 epoch), đứt nét là đối chứng cùng ngân sách bước.](figures/expA_steps.png){#fig:expAsteps width=13}

## 4.3. Thực nghiệm B: số nơ-ron của tầng ẩn

**Thiết kế.** Mạng một tầng ẩn [2, B, 1], B = 1…10 ở A = 1000 (30 run). Để xem xu hướng khi rộng hơn, nhóm chạy thêm B = 16, 32, 64 (`expB_wide`) và dùng B = 38, 65, 93 từ đối chứng của Thực nghiệm C. Để tìm quá khớp, toàn bộ dải B được lặp lại ở A = 200 (`expB_A200`, 39 run).

@table expB+expB_wide+expC_ctrl {#tab:expB} | value=B; n_params=Tham số; train_mse=Train MSE; val_mse=Val MSE; test_mse=Test MSE*; val_mae=Val MAE; best_epoch=Epoch tốt nhất | Thực nghiệm B tại A = 1000 (B = 16–64 mở rộng; B = 38, 65, 93 lấy từ đối chứng C; * hậu kiểm).

![Trái: MSE theo B tại A = 1000 (trục B theo log₂). Phải: khoảng cách val − train MSE.](figures/expB_error.png){#fig:expB width=16}

![Bề mặt thật và bề mặt xấp xỉ của mạng một tầng với B = 1, 2, 3, 5, 10 (seed 0).](figures/expB_surfaces.png){#fig:expBsurf width=15}

**Thiếu khớp khi B nhỏ.** Val MSE giảm đều hơn 120 lần từ B = 1 (2.29e-3, gần mức hằng số 2.49e-3) đến B = 10 (1.89e-5). [@fig:expBsurf] giải thích bằng hình học: với B = 1, đầu ra w·σ(u₁x₁ + u₂x₂ + b) + c là một "sườn dốc" chỉ thay đổi theo một hướng, không thể tạo đỉnh; B = 2 tạo được một dải chạy dọc trục chính của elip nhưng chưa khép kín hai đầu; từ B ≈ 5 trở lên, nhiều sườn dốc xoay theo các hướng khác nhau mới ghép được thành đỉnh elip nghiêng.

**Không có quá khớp ở A = 1000.** Val MSE tiếp tục giảm khi mở rộng: 5.63e-6 (B = 16), 2.09e-6 (B = 64), 1.46e-6 (B = 93); khoảng cách val − train tiến về 0. Ở B nhỏ, val thấp hơn train một chút vì tập validation có tỉ lệ điểm y ≈ 0 cao hơn (66% so với 62.6%), nên mô hình thiếu khớp bị phạt ít hơn trên validation.

@table expB_A200 {#tab:expBA200} | value=B; n_params=Tham số; train_mse=Train MSE; val_mse=Val MSE; test_mse=Test MSE*; val_mse/train_mse=Val/Train | Thực nghiệm B lặp lại tại A = 200 (* hậu kiểm). Nguồn `results/expB_A200/summary.csv`.

![Thực nghiệm B tại A = 1000 (liền nét) và A = 200 (đứt nét). Phải: khoảng cách val − train (thang symlog).](figures/expB_A200.png){#fig:expBA200 width=16}

**Ít dữ liệu: có khoảng cách tổng quát hóa nhưng chưa có chữ U.** Ở A = 200, từ B ≥ 8 val MSE luôn lớn hơn train và tỉ số Val/Train tăng theo B, từ 1.30 (B = 10) lên 1.74 (B = 64), trong khi ở A = 1000 tỉ số này xấp xỉ 1 ([@tab:expBA200], [@fig:expBA200]). Đây là dấu hiệu sớm của quá khớp. Tuy vậy val và test vẫn giảm theo B (val từ 7.06e-5 xuống 9.21e-6), tức chưa có đường chữ U. 17/39 run chạm trần 2000 epoch, nên khoảng cách có thể rõ hơn nếu huấn luyện lâu hơn.

## 4.4. Thực nghiệm C: số tầng ẩn

**Thiết kế.** C ∈ {1, 2, 3, 4} tầng ẩn, mỗi tầng 10 nơ-ron, các yếu tố khác như baseline (12 run). Tăng C cũng tăng số tham số (41 → 371), nên nhóm thêm **đối chứng cùng số tham số**: mạng một tầng B = 38, 65, 93 (153, 261, 373 tham số, theo công thức 4B + 1). Thời gian đo bằng `time.perf_counter()` quanh `fit()`, cùng máy, 1 luồng, các run chạy tuần tự. Mỗi epoch ghi chuẩn gradient từng lớp.

@table expC+expC_ctrl {#tab:expC} | value=Mạng; n_params=Tham số; train_mse=Train MSE; val_mse=Val MSE; test_mse=Test MSE*; val_mae=Val MAE; best_epoch=Epoch tốt nhất; train_time_s=Thời gian (s) | Thực nghiệm C và đối chứng mạng một tầng cùng số tham số (* hậu kiểm). Nguồn `results/expC/`, `results/expC_ctrl/`. | arch

![Trái: MSE theo C. Phải: val MSE của mạng sâu và mạng một tầng rộng theo số tham số.](figures/expC_error.png){#fig:expC width=16}

![Train loss và val loss theo epoch với C = 1…4 (trung bình 3 seed, trung vị trượt 25 epoch).](figures/expC_val_curves.png){#fig:expCcurve width=16}

**Thêm tầng làm lỗi giảm mạnh, nhưng chỉ đến C = 2–3.** Val MSE giảm khoảng 180 lần từ C = 1 (1.89e-5) sang C = 2 (1.05e-7). Từ C = 2 đến C = 4 lỗi cùng bậc 1e-7, khoảng ± std của C = 3 và C = 4 chồng nhau. Với C ≥ 2, loss xuống dưới 1e-5 trong khoảng 200 epoch đầu, còn C = 1 giảm chậm về quanh 3e-5 ([@fig:expCcurve]). Train, validation và test gần bằng nhau ở mọi C, không có quá khớp.

**Lợi ích đến từ độ sâu, không chỉ từ số tham số.** Ở cùng số tham số, mạng sâu có val MSE thấp hơn mạng một tầng 16–28 lần (151 so với 153 tham số: 1.05e-7 so với 2.42e-6; 261: 6.02e-8 so với 1.68e-6; 371/373: 9.21e-8 so với 1.46e-6). Mạng rộng gần như chững ở khoảng 1.5e-6 khi B tăng từ 38 lên 93.

![Chuẩn gradient từng lớp ở epoch đầu và trung vị qua các epoch (trung bình 3 seed).](figures/expC_grad_norms.png){#fig:grad width=12}

**Vanishing gradient có xuất hiện lúc khởi tạo nhưng không cản trở huấn luyện.** Ở epoch 1, chuẩn gradient giảm dần khi lan ngược qua từng tầng, mỗi tầng khoảng 5–20 lần ([@fig:grad]): với C = 4, lớp đầu chỉ 1.7e-5 so với 6.0e-2 ở lớp ra, chênh khoảng 3600 lần; với C = 1 chênh khoảng 25 lần. Trong quá trình huấn luyện, trung vị chuẩn gradient của các tầng ẩn chỉ còn chênh dưới 3.5 lần, và mạng sâu vẫn đạt lỗi thấp nhất.

![Tổng thời gian huấn luyện (trái) và thời gian mỗi epoch (phải) theo C.](figures/expC_time.png){#fig:time width=15}

**Thời gian.** Thời gian mỗi epoch tăng từ 20.2 ms (C = 1) lên 30.5 ms (C = 4), khoảng +3.4 ms mỗi tầng, trong khi mạng một tầng 93 nơ-ron (373 tham số) vẫn chỉ 20.2 ms ([@fig:time]). Với mạng nhỏ, chi phí bị chi phối bởi số phép tính tuần tự theo tầng, không bởi kích thước ma trận. Tổng thời gian (khoảng 36–47 s) dao động nhiều hơn vì phụ thuộc số epoch đến khi dừng sớm.

## 4.5. Mô hình cuối và đánh giá trên tập test

**Chọn mô hình.** `results/summary_all.csv` gộp 53 cấu hình của mọi thực nghiệm. Cấu hình có val MSE trung bình thấp nhất là C = 3 `[10,10,10]` (6.02e-8), tiếp theo C = 4 (9.21e-8) và C = 2 (1.05e-7). Nhóm chọn C = 3 (261 tham số, sigmoid, Adam lr 0.01, A = 1000), đóng băng vào `configs/final.json`, rồi mới chạy `--eval-test` một lần. Checkpoint `results/final/best.pt` lấy từ seed có val thấp nhất (seed 2, epoch 1490). Vì tập test chỉ có 100 điểm và 61 điểm có y < 0.01, nhóm đánh giá thêm trên lưới đều 100 × 100 phủ toàn miền.

@table results/final/aggregate.csv {#tab:final} | candidate=Mô hình; n_params=Tham số; val_mse=Val MSE; test_mse=Test MSE; test_mae=Test MAE; test_rmse=Test RMSE; test_r2=Test R²; grid_mse=Grid MSE | Đánh giá cuối trên test (N = 100) và lưới 100 × 100 (mean ± std qua 3 seed). Nguồn `results/final/aggregate.csv`. | labels=Hằng số (trung bình train),Baseline [10],Một tầng [93],Mô hình chọn [10,10,10]

Checkpoint seed 2 (`best.pt`) đạt Test MSE 4.42e-8, Test MAE 1.54e-4, R² = 0.99998, Grid MSE 5.73e-8 và sai số tuyệt đối lớn nhất trên lưới 1.07e-3, dưới 0.4% chiều cao đỉnh (`results/final/metrics.json`).

![Trái: y thật và ŷ dự đoán trên 100 mẫu test cùng đường ŷ = y. Phải: phần dư theo y.](figures/final_parity_test.png){#fig:parity width=15}

![Bề mặt Gaussian thật (trái) và bề mặt xấp xỉ của MLP [10, 10, 10] (phải), cùng thang z.](figures/final_surface_comparison.png){#fig:finalsurf width=15}

![Trái: đường đồng mức của hàm thật (liền) và hàm xấp xỉ (đứt). Phải: bản đồ |y − ŷ| trên lưới kèm vị trí 100 mẫu test.](figures/final_error_map.png){#fig:errmap width=15}

**Nhận xét.** Test MSE (7.13e-8) và Grid MSE (7.34e-8) rất sát Val MSE (6.02e-8), nên việc chọn theo validation không làm mô hình khớp riêng tập validation. Các điểm test nằm sát đường ŷ = y trên cả dải 0–0.269 ([@fig:parity]); đường đồng mức của mô hình chồng khít hàm thật ([@fig:errmap]). Sai số lớn nhất tập trung ở sườn dốc quanh đỉnh, nơi hàm thay đổi nhanh nhất.

# Chương 5. Thảo luận

## 5.1. Vì sao không thấy đường chữ U của quá khớp

Hình minh họa trong đề và nhiều giáo trình cho thấy lỗi validation giảm rồi tăng lại khi mô hình quá lớn. Nhóm không quan sát được điều này, kể cả khi giảm còn A = 200 và tăng B tới 64. Có ba lý do:

1. **Nhãn không có nhiễu.** Quá khớp nặng xảy ra khi mô hình dùng dung lượng thừa để ghi nhớ nhiễu ở các điểm train. Ở đây y = f(x) chính xác, nên khớp đúng điểm train cũng là khớp đúng hàm.
2. **Hàm mục tiêu trơn và dữ liệu phủ đủ dày.** Một hàm Gaussian 2D trơn, không có chi tiết nhỏ; 200–2000 điểm đều trên miền 2D đủ để mạng nội suy tốt giữa các điểm.
3. **Early stopping theo validation.** Checkpoint luôn lấy ở epoch có val loss thấp nhất, nên mô hình không kịp đi vào vùng quá khớp.

Điều nhóm thấy được là dấu hiệu sớm: ở A = 200, tỉ số Val/Train tăng từ 1.30 lên 1.74 khi B tăng. Muốn thấy rõ chữ U, cần thêm nhiễu vào nhãn hoặc giảm A mạnh hơn nữa (mục 5.5).

## 5.2. Vì sao mạng sâu tốt hơn mạng rộng

Mỗi nơ-ron sigmoid ở tầng đầu chỉ tạo một "sườn dốc" theo một hướng trong mặt phẳng (x₁, x₂). Mạng một tầng phải cộng tuyến tính rất nhiều sườn dốc mới tạo được một đỉnh khép kín hình elip, nên lỗi giảm chậm và chững lại quanh 1.5e-6 dù tăng tới 93 nơ-ron. Tầng thứ hai lấy các sườn dốc làm đầu vào và kết hợp phi tuyến chúng thành vùng lồi bao quanh đỉnh, nên biểu diễn hình dạng này hiệu quả hơn nhiều với cùng số tham số. Định lý xấp xỉ phổ quát (Cybenko, 1989; Hornik, 1989) chỉ bảo đảm mạng một tầng *có thể* xấp xỉ, không nói nó cần bao nhiêu nơ-ron; số liệu cho thấy với hàm này, cái giá đó cao hơn hẳn so với thêm tầng.

## 5.3. Những kết quả không như kỳ vọng

- **Ngược với Sample_Report.** Báo cáo mẫu kết luận tăng tầng làm lỗi tăng do vanishing gradient. Nhóm đo được vanishing gradient lúc khởi tạo (lớp đầu nhỏ hơn lớp ra khoảng 3600 lần với C = 4), nhưng nó không làm hỏng huấn luyện. Giả thuyết hợp lý là Adam chia bước cập nhật cho √v̂ₜ nên tham số có gradient nhỏ vẫn được cập nhật với bước cỡ lr; nhóm chưa chạy SGD thuần trên mạng sâu nên chưa kiểm chứng được giả thuyết này.
- **Yếu tố lẫn giữa số mẫu và số bước ở A.** Kết quả gốc của Thực nghiệm A phóng đại ảnh hưởng của số mẫu, vì A nhỏ được huấn luyện ít bước hơn. Thí nghiệm đối chứng cho thấy khoảng 43% lỗi thừa ở A = 200 biến mất khi cấp đủ bước.
- **A = 2000 có lỗi tăng nhẹ.** Val MSE ở A = 2000 (2.57e-5) cao hơn A = 1600 (1.52e-5). Đây không phải do ngân sách bước: ở đối chứng, A = 2000 cho kết quả trùng khớp vì cả 3 seed đều dừng sớm trước giới hạn mới. Chênh lệch này nằm trong biến động giữa seed (std 8.43e-6), nên nhóm không coi là xu hướng.
- **Validation thấp hơn train khi thiếu khớp.** Ở B nhỏ, val MSE thấp hơn train MSE; nguyên nhân là validation có nhiều điểm y ≈ 0 hơn, không phải mô hình tổng quát hóa tốt hơn.

## 5.4. Hạn chế

- **Trần 2000 epoch.** Tùy thí nghiệm, 6.7–50% run có epoch tốt nhất nằm trong 200 epoch cuối, tức có thể vẫn đang cải thiện khi bị cắt (11/30 run A, 17/39 run B ở A = 200). Lỗi của một số cấu hình có thể bị đánh giá cao hơn thực tế.
- **Chỉ 3 seed.** Nhiều chênh lệch (batch size, A ≥ 600, C = 3 so với C = 4, Adam so với Nadam, RMSProp) nhỏ hơn độ lệch chuẩn nên không kết luận được.
- **Tìm siêu tham số theo từng tọa độ.** Nhóm cố định lần lượt activation + lr, rồi optimizer, batch, patience thay vì thử mọi tổ hợp; Adam, batch 32 và patience 200 là chọn trước rồi mới kiểm chứng. Có thể bỏ sót tổ hợp tốt hơn (ví dụ tanh với optimizer khác), và kết luận về optimizer chỉ làm trên mạng một tầng.
- **Phạm vi C.** Chỉ xét tối đa 4 tầng, sigmoid và Adam; với mạng sâu hơn hoặc SGD, vanishing gradient có thể ảnh hưởng rõ hơn.
- **Đánh giá thời gian.** Chỉ chạy CPU một luồng. Thời gian của 90 run bổ sung (`expA_steps`, `expB_wide`, `expB_A200`, `grid_batch`) đo khi 4 tiến trình chạy song song, nên không so sánh được với các run khác; báo cáo chỉ dùng thời gian của Thực nghiệm C.
- **Lấy mẫu đều trên miền** khác hình scatter minh họa trong đề (điểm dồn quanh tâm), làm 62.6% nhãn gần 0; nhóm bù lại bằng R², đánh giá trên lưới và bản đồ sai số.

## 5.5. Đề xuất cải tiến

- Thêm nhiễu Gaussian vào nhãn hoặc giảm A xuống 50–100 để kiểm tra có xuất hiện đường chữ U hay không.
- Tăng trần epoch, hoặc cố định tổng số bước cập nhật cho mọi thí nghiệm, để loại hẳn ảnh hưởng của việc huấn luyện chưa đủ.
- Chạy 5–10 seed cho các cấu hình gần nhau để tách được C = 3 với C = 4 và các optimizer.
- So sánh Adam với SGD trên mạng sâu C = 4–8 để kiểm chứng giả thuyết Adam bù vanishing gradient.
- Giảm learning rate theo lịch (learning-rate schedule) để làm mượt đường val loss của mạng sâu, vốn dao động mạnh ở lr 0.01 khi lỗi đã rất nhỏ.

# Chương 6. Kết luận

Nhóm đã cài đặt MLP tùy biến bằng PyTorch với vòng lặp train/validation tự viết, early stopping và checkpoint theo validation, và chạy hơn 320 run có ghi log để khảo sát ba yếu tố A, B, C. Kết luận chính:

1. **Số mẫu (A):** lỗi giảm mạnh đến A ≈ 600 rồi chững; một phần lỗi ở A nhỏ là do ít bước cập nhật. Với mạng `[10]`, giới hạn chính là dung lượng mô hình chứ không phải dữ liệu.
2. **Độ rộng (B):** B nhỏ thiếu khớp rõ, lỗi giảm hơn 120 lần từ B = 1 đến B = 10 và tiếp tục giảm tới B = 93. Không có quá khớp ở A = 1000; ở A = 200 xuất hiện khoảng cách train–validation nhưng chưa có đường chữ U, do nhãn không nhiễu và có early stopping.
3. **Độ sâu (C):** thêm tầng giảm lỗi khoảng 180 lần từ C = 1 sang C = 2, và mạng sâu tốt hơn mạng rộng cùng số tham số 16–28 lần. Vanishing gradient có xuất hiện lúc khởi tạo nhưng không cản trở huấn luyện với Adam.
4. **Mô hình cuối** `[10,10,10]`, chọn hoàn toàn bằng validation, đạt Test MSE 7.13e-8 ± 2.36e-8, R² = 0.99997 và Grid MSE 7.34e-8.

Mọi lựa chọn được ghi trong nhật ký quyết định (Phụ lục B) cùng file minh chứng; mọi bảng số liệu sinh tự động từ log bằng `src/make_tables.py`.

# Tài liệu tham khảo

1. PyTorch Documentation, *torch.nn*, *torch.optim*, *torch.utils.data*. https://pytorch.org/docs/stable/
2. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
3. D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization," *ICLR*, 2015.
4. X. Glorot, Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," *AISTATS*, 2010.
5. G. Cybenko, "Approximation by superpositions of a sigmoidal function," *Mathematics of Control, Signals and Systems*, 2(4), 1989.
6. K. Hornik, M. Stinchcombe, H. White, "Multilayer feedforward networks are universal approximators," *Neural Networks*, 2(5), 1989.
7. Đỗ Như Tài, *Bài tập xấp xỉ hàm số dùng mạng MLP* và *Sample_Report*, tài liệu môn Học sâu, Trường Đại học Sài Gòn.

# Phụ lục A. Lệnh tái lập

Cài đặt: `python -m venv .venv`, kích hoạt môi trường, `pip install -r requirements.txt`, rồi `pytest -q`. Trên Windows đặt `PYTHONUTF8=1`. Thứ tự chạy (chi tiết trong `README.md`):

```
python src/generate_data.py                                   # sinh data/*.csv
python src/data/eda.py                                        # thống kê, hình EDA
python src/run_experiment.py --config configs/grid_baseline.json
python src/run_experiment.py --config configs/grid_optimizer.json
python src/run_experiment.py --config configs/grid_batch.json
python src/run_experiment.py --config configs/expA.json       # tương tự: expA_steps, expB,
                                                              # expB_wide, expB_A200, expC, expC_ctrl
python src/posthoc_test.py                                    # hậu kiểm test trên best.pt
python src/analyze_stopping.py
python src/data/summarize_expA.py
python src/summarize_expB.py
python src/summarize_expC.py
python src/make_report_assets.py                              # chọn mô hình cuối, đánh giá test
python src/make_tables.py                                     # bảng report/tables/
python src/run_experiment.py --config configs/demo.json       # demo khoảng 10 giây
```

# Phụ lục B. Nhật ký quyết định

Bảng dưới tóm tắt `report/NHAT_KY_QUYET_DINH.md` (lý do và số liệu chi tiết của từng quyết định nằm trong file đó). Cột "Thứ tự" ghi rõ quyết định được khảo sát rồi mới chọn, hay chọn trước rồi kiểm chứng sau.

Table: Nhật ký quyết định. {#tab:decisions}

| # | Quyết định | Thứ tự | Minh chứng |
|---|---|---|---|
| D1 | Chia 2000/200/100, seed 2026/2027/2028 | Chọn trước | `data/meta.json` |
| D2 | Lấy x đều trên miền | Chọn trước | `src/data/gaussian.py` |
| D3 | Tập train của A lồng nhau | Chọn trước | `src/data/dataset.py::subset` |
| D4 | Scale x theo biên miền, y theo y_max | Chọn trước | `src/data/gaussian.py` |
| D5 | sigmoid, lr 0.01 | Khảo sát → chọn | `results/grid_baseline/` |
| D6 | Adam | Chọn trước → kiểm chứng | `results/grid_optimizer/` |
| D7 | Batch 32 | Chọn trước → kiểm chứng | `results/grid_batch/` |
| D8 | Patience 200 | Chọn trước → kiểm chứng | `results/stopping/patience_sim.csv` |
| D9 | max_epochs 2000 | Chọn trước; hạn chế | `results/stopping/epoch_cap.csv` |
| D10 | 3 seed | Chọn trước; hạn chế | `configs/*.json` |
| D11 | Đối chứng A cùng ngân sách bước | Phát hiện từ log | `results/expA_steps/` |
| D12 | B mở rộng 16/32/64 và B ở A = 200 | Phát hiện từ log | `results/expB_wide/`, `results/expB_A200/` |
| D13 | Đối chứng C cùng số tham số | Chọn trước | `results/expC_ctrl/` |
| D14 | Đo chuẩn gradient từng lớp | Chọn trước | `results/expC/grad_norms.csv` |
| D15 | Mô hình cuối = argmin val trên 53 cấu hình | Khảo sát → chọn | `results/summary_all.csv`, `configs/final.json` |
| D16 | Test chỉ dùng hậu kiểm và đánh giá cuối | Quy tắc | `results/*/posthoc_test.log` |
| D17 | Tìm siêu tham số theo từng tọa độ | Chọn trước; hạn chế | — |
