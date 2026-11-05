# BTL01 – Nhóm 10: Chi tiết nhiệm vụ, kiến thức cần có và tài liệu cho từng thành viên

Tài liệu này mở rộng `PHAN_CONG_NHIEM_VU.md`. Quy ước: `SGU_DL\` là thư mục `D:\SGU_HK1_2026-2027\SGU_DL\` trên máy (chỉ dùng để tìm tài liệu, **không** đưa đường dẫn này vào code). Số trang là số trang của file PDF khi mở bằng trình đọc.

## 0. Gợi ý ghép người (theo vai trò Lab03)

Trong Lab03 nhóm đã chia: Trung Kiên (EDA/tiền xử lý), Viễn Thông (nhánh sklearn + đánh giá), Nguyên Hải (PyTorch MLP), Hoàng Bảo (run_all, sơ đồ, báo cáo). Nếu nhóm giữ nguyên 4 người, cách ghép tự nhiên là:

| Vai | Gợi ý | Lý do |
|---|---|---|
| TV1 Dữ liệu + Thực nghiệm A | Trung Kiên | Đã làm EDA/tiền xử lý |
| TV2 Mô hình + Huấn luyện + Thực nghiệm C | Nguyên Hải | Đã viết `model.py`, `train_mlp.py` |
| TV3 Đánh giá + Vẽ hình + Thực nghiệm B | Viễn Thông | Đã làm đánh giá/CV, biểu đồ so sánh |
| TV4 Báo cáo + Slide + QA | Hoàng Bảo | Đã làm `run_all.py`, `diagrams.py`, báo cáo tổng hợp |

Đây chỉ là gợi ý, nhóm tự quyết.

---

## 1. Đọc trước: kết quả chạy thử sơ bộ (ảnh hưởng đến cách thiết kế)

Mình đã chạy thử một script tạm (không nằm trong sản phẩm) để biết trước các "bẫy". Điều kiện: 1 seed, Adam lr = 1e-3, batch 32, tối đa 300 epoch, patience 50, tanh (trừ khi ghi khác), chỉ dùng tập validation 200 mẫu, **không đụng test**. MSE tính trên thang y gốc. **Đây không phải kết quả chính thức của nhóm**, chỉ để đặt kỳ vọng.

| Nhóm | Cấu hình | Val MSE | Ghi chú |
|---|---|---|---|
| Baseline tầm thường | luôn dự đoán trung bình | ≈ 3.2e-3 | = phương sai của y (ước lượng trên 200 000 mẫu) |
| B (A=1000) | B=1 / 2 / 3 / 5 / 10 / 32 | 2.5e-3 / 1.3e-3 / 7.1e-4 / 4.0e-4 / 6.2e-4 / 1.2e-4 | B=1 dừng sớm ở epoch 22; B=10 tệ hơn B=5 (nhiễu seed/chưa hội tụ) |
| A (B=10) | A=200 / 1000 / 2000 | 1.5e-3 / 6.2e-4 / 6.0e-4 | 1000→2000 gần như không đổi |
| C (A=1000, rộng 10) | C=1 / 2 / 3 / 4 | 6.2e-4 / 9.7e-6 / 1.7e-6 / 1.9e-6 | **Tăng tầng làm giảm lỗi rất mạnh** |
| Activation (A=1000, B=10) | tanh / sigmoid / relu | 6.2e-4 / 1.3e-3 / 1.1e-4 | |
| Thời gian | A=1000, B=10, 300 epoch | ≈ 5 s | ≈ 0.5 ms/bước tối ưu (CPU, 1 luồng) |

Những điều cần rút ra:

1. **300 epoch với lr = 1e-3 chưa hội tụ** (epoch tốt nhất ≈ 290–299). Baseline phải được chỉnh bằng validation (lr lớn hơn và/hoặc nhiều epoch hơn, patience ≥ 100–200) trước khi chạy A/B/C.
2. **MSE rất nhỏ về số tuyệt đối** vì y ≤ 0.269 (không phải 1 như slide đề ghi). B=1 chỉ đạt 2.5e-3, gần mức tầm thường 3.2e-3 (R² ≈ 0.2). Luôn đưa baseline "dự đoán trung bình" vào bảng để người đọc có thang so sánh.
3. **Kết quả nhóm C ngược với Sample_Report** (mẫu nói tăng tầng làm lỗi tăng, do vanishing gradient). Ở đây tăng tầng làm lỗi giảm. Nhưng mô hình sâu hơn có nhiều tham số hơn (41 → 151 → 261 → 371) và mới chỉ 1 seed/300 epoch. Phải kiểm chứng và so sánh công bằng theo số tham số (xem TV2). **Không được chép nhận xét "vanishing gradient" của mẫu nếu số liệu của nhóm không chứng minh.**
4. **Đường cong theo A có thể gần phẳng** khi mô hình nhỏ (B=10): lỗi bị giới hạn bởi dung lượng/độ hội tụ chứ không phải thiếu dữ liệu. Đây là giả thuyết cần kiểm tra ở nhóm A bằng cách chạy thêm một mô hình có dung lượng lớn hơn.
5. **Bẫy shape trong MSELoss**: nếu `pred` có shape `(N,1)` còn `y` có shape `(N,)`, PyTorch chỉ in `UserWarning` rồi broadcast thành `(N,N)`. Mình đã thử: loss vẫn giảm nhưng mô hình học dự đoán hằng số (std của dự đoán 0.0034 so với 0.24 khi đúng; mean(pred) = mean(y)). Cách phòng: luôn `assert pred.shape == target.shape`.
6. **Windows**: `print` tiếng Việt báo lỗi `UnicodeEncodeError` (cp1252) nếu không đặt `PYTHONUTF8=1` hoặc `sys.stdout.reconfigure(encoding="utf-8")`. Ngoài ra `fig.gca(projection="3d")` (dùng trong Sample_Report) **không còn chạy** ở Matplotlib hiện tại (3.10.1 trên máy này báo `TypeError`); dùng `fig.add_subplot(projection="3d")`.
7. **Ngân sách tính toán**: ≈ 0.5 ms/bước. Với 2000 epoch, A=1000 mất ≈ 30 s/lần chạy; toàn bộ A (10 giá trị) + B (khoảng 15 giá trị × 2 mức A) + C (khoảng 7 giá trị), mỗi cấu hình ≥ 3 seed, ước tính **khoảng 1–1.5 giờ CPU**, có thể chạy song song theo nhóm thí nghiệm.

Ghi chú hạ tầng: máy này đã có `torch 2.6.0+cu124`, `matplotlib 3.10.1`, `numpy`, `pandas`, `scipy`, `tqdm`, `python-docx`. **Chưa có** `torchinfo` và `python-pptx` (cần `pip install` nếu dùng).

---

## 2. Hợp đồng giao diện chung (để 4 người làm song song)

Chốt ngay ở giai đoạn G1. Ai muốn đổi phải báo cả nhóm.

```
src/data/gaussian.py      gaussian_pdf(x:(N,2)) -> (N,)         # hằng MU, C, X_SCALE, Y_SCALE
src/data/dataset.py       load_split(name) -> (x, y)            # numpy, thang GỐC
                          make_loader(x, y, batch_size, shuffle, seed) -> DataLoader   # đã scale x, y
src/model/mlp.py          class MLP(nn.Module): __init__(hidden_sizes, activation="tanh")
src/engine/trainer.py     fit(model, train_loader, val_loader, cfg, run_dir) -> dict(best_epoch, epochs_run, train_time_s)
src/engine/evaluate.py    predict(model, x) -> y_hat (thang gốc)
                          metrics(y, y_hat) -> dict(mse, mae, rmse, r2)
src/run_experiment.py     --config configs/expX.json [--eval-test]
```

Quy tắc bắt buộc:

- `fit()` **không nhận** dữ liệu test. Test chỉ được tính trong bước `--eval-test` sau cùng (chống rò rỉ, tránh chọn cấu hình theo test).
- x scale theo biên miền cố định (x1/2, x2/4), y scale theo hằng y_max lý thuyết = 1/(2π√det C) ≈ 0.269. Đây là hằng số của bài toán, không fit từ dữ liệu nên không có rò rỉ. Mọi chỉ số báo cáo trên **thang y gốc**.
- Mỗi lần chạy lưu: `results/<exp>/<giá trị>/seed_<k>/{config.json, history.csv, metrics.json, best.pt}`; tổng hợp vào `results/<exp>/summary.csv` với các cột: `exp, value, seed, n_params, train_mse, val_mse, test_mse, test_mae, test_rmse, test_r2, best_epoch, epochs_run, train_time_s, time_per_epoch_ms`.
- Đường dẫn tính từ vị trí file (`Path(__file__).resolve()`), không viết đường dẫn tuyệt đối; không import `tensorflow`/`keras`.
- Git như Lab03: nhánh theo người (`feat/data`, `feat/train`, `feat/eval`, `docs/report`), commit kiểu `feat:`/`fix:`, review chéo trước khi gộp.

Ví dụ file cấu hình (giá trị baseline là **placeholder**, chốt ở G2):

```json
{"exp": "A",
 "vary": {"n_train": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
 "base": {"hidden_sizes": [10], "activation": "tanh", "optimizer": "adam",
          "lr": 0.003, "batch_size": 32, "max_epochs": 2000, "patience": 200},
 "seeds": [0, 1, 2], "out_dir": "results/expA"}
```

---

## 3. Bảng tra cứu tài liệu dùng chung

| Tài liệu (trong `SGU_DL\`) | Trang / mục | Nội dung | Ai cần |
|---|---|---|---|
| `DL08a - Deep Neural Network (MLP).pdf` | **p7** | **Ví dụ xấp xỉ Gaussian 2D bằng MLP sigmoid (160 mẫu, 8 nơ-ron ẩn), có in trọng số** | Cả nhóm |
| | p9–15 | Trọng số d×p×m; đạo hàm ∂E/∂v_jk, ∂E/∂u_ij cho MLP 2 tầng | TV2, TV4 |
| | p17–21, p23–34 | Deep MLP: forward, lỗi, lan truyền ngược; minh họa trực quan | TV2 |
| | p35–36 | Khó khăn khi huấn luyện MLP sâu (Hinton), bài báo Nature 2015 | TV2, TV4 |
| `DL07b - Deep_networks_1.pdf` | p4–18 | Backprop với scalar/tensor, minibatch | TV2 |
| | p36–42 | Kiến trúc MLP, thuật toán huấn luyện SGD | TV2, TV4 |
| | p46–54 | Biểu diễn hàm, **Universal Approximation Theorem (p48)**, space folding (p52), **ảnh hưởng độ sâu (p54)** | TV2, TV3, TV4 |
| `DL08b - Deep_networks_2.pdf` | p15, p21 | MSE xuất phát từ Gaussian; **đầu ra tuyến tính cho hồi quy** | TV2 |
| | p42–56 | Hàm kích hoạt: sigmoid, tanh, ReLU, Leaky ReLU | TV2, TV3 |
| | p57–62 | Đạo hàm số (dùng để **gradient check**) | TV2 |
| | p63–68 | Tiền xử lý dữ liệu | TV1 |
| | p69–80 | Khởi tạo trọng số (Xavier cho tanh, He cho ReLU) | TV2 |
| `DL04a - Model Selection.pdf` | p6–11 | Bias/variance, under/overfitting, sai số huấn luyện vs tổng quát | TV1, TV3 |
| | **p28** | **Learning curve theo kích thước tập huấn luyện (chính là nhóm A)** | TV1 |
| | p29–32 | Dùng test đúng cách, k-fold, siêu tham số, quy trình train/val/test | Cả nhóm |
| `DL02b - Modeling.pdf` | p20, p22 | Đánh giá mô hình cuối, so với baseline | TV3 |
| `DL06a - Computation graphs.pdf` | p24–56, p60–64 | Đồ thị tính toán, backprop, framework giáo dục NumPy | TV2 |
| `DLCoBan - Chapter03 - MLP.pdf` (19 trang, tiếng Việt) | 3.4 | Biểu diễn mạng bằng vector/ma trận, backprop dạng ma trận | TV2, TV4 |
| `Sridhar - Chapter10 - ANN.pdf` | p280– | MLP, backprop | TV2 |
| `deep_learning_with_pytorch.pdf` (343 trang, số trang PDF) | Ch.4 p47 | Autograd | TV2 |
| | Ch.6–8, p61–91 | Khối xây dựng MLP, mạng đầu tiên, **vòng lặp huấn luyện** | TV2 |
| | Ch.9, p92–103 | Đánh giá mô hình, tách train/val, k-fold | TV2, TV3 |
| | Ch.12, p131 | Bài toán hồi quy bằng PyTorch | TV2 |
| | Ch.13, p143 | Lưu/nạp mô hình, `state_dict` | TV2 |
| | Ch.14–15, p152, p168 | Hàm kích hoạt, hàm loss (MSELoss) | TV2 |
| | Ch.17, p188 | Lịch học learning rate (tùy chọn) | TV2 |
| | Ch.18, p198 | `Dataset`, `DataLoader` | TV1, TV2 |
| | Ch.20.5–20.10, p224–235 | Chọn optimizer, lr, activation, số nơ-ron | TV2, TV3 |
| | Ch.21, p240 | **Checkpoint + Early Stopping** | TV2 |
| | Ch.22, p250 | Vẽ đồ thị mô hình bằng Netron | TV3, TV4 |
| | Ch.23, p255 | Thu thập và vẽ metric huấn luyện | TV3 |
| `DLPytorch - Chapter 13-18 - MLP Param Fintune.pdf` | 64 trang | Bản trích riêng Ch.13–18 của sách trên | TV2 |
| `Lab02\pytorch_p1_background.pdf` | p28 | Listing vẽ bề mặt bằng matplotlib (mẫu cho hình 3D) | TV1, TV3 |
| `DL03a - Data Exploration.pdf` | p31–36, p70–94 | Lấy mẫu & kích thước mẫu; thống kê mô tả, Pearson (p76), scatter (p85), boxplot (p88) | TV1 |
| `[Tutorial] EDA-Python.ipynb` | toàn bộ | Mẫu EDA bằng pandas/matplotlib/seaborn (histogram, scatter, correlogram, heatmap) | TV1 |
| `Chong_Data_Leakage_Overfitting.docx` | toàn bộ | Quy tắc chống rò rỉ (scaler chỉ fit trên train), early stopping, weight decay | Cả nhóm |
| `paper\!! NatureDeepReview.pdf` | toàn bộ | LeCun–Bengio–Hinton 2015 (tài liệu tham khảo) | TV4 |
| `Lab08\BT07_MLP_...docx`, `Lab08\verify_math.py` | Bài 1 | Mẫu tính tay forward/backward và script đối chiếu số tay với code | TV2 |
| `Lab08\.agents\skills\sgu-deeplearning-assignment\SKILL.md`, `Lab08\build_doc.py` | toàn bộ | Quy chuẩn Word (Times New Roman, căn trái, bảng chống vỡ dòng) và mẫu code `python-docx` | TV4 |
| `lab03_house_price_...\ml_hybrid_project\` | `prj\pytorch_mlp\*.py`, `README.md`, `NGHIEMTHU.md`, `prj\model\diagrams.py` | Cấu trúc repo, `EarlyStopping`, `set_seed`, `train_one_epoch`, `validate`, README/nghiệm thu, vẽ sơ đồ quy trình | TV2, TV4 |
| `BTL01_Nhom10\Sample_Report.pdf`, `Sample_Code.html` | | Bố cục báo cáo; code **Keras** (cấm dùng, chỉ tham khảo cách tổ chức) | TV4 |

Lưu ý khi tái sử dụng code Lab03: `MLPRegressor` có BatchNorm, Dropout, ReLU, `squeeze(-1)`. **Không dùng nguyên** cho BTL01 vì BatchNorm/Dropout làm nhiễu việc so sánh B/C; chỉ mượn khung `EarlyStopping`, `set_seed`, vòng lặp train/validate, lưu history.

---

## 4. TV1 – Dữ liệu, repo, Thực nghiệm A (Rubric 1, 3 và một phần 2: 10 + 10 điểm)

### 4.1 Kiến thức cần có

| Kiến thức | Mức | Học ở đâu |
|---|---|---|
| Gaussian đa biến: det C, C⁻¹, dạng toàn phương (x−μ)ᵀC⁻¹(x−μ), hệ số tương quan ρ = −0.5/√(0.3·2) ≈ −0.645, trục chính của elip | Hiểu rõ, tự tính được | `DL08a` p7; công thức trong đề |
| Lấy mẫu đều bằng `np.random.default_rng(seed)`, tách luồng seed cho từng tập | Làm được | numpy docs |
| Vector hóa bằng `np.einsum("ni,ij,nj->n", d, C_inv, d)` | Làm được | Sample_Report trang 4 (hàm `multivariate_gaussian`) |
| Tách train/val/test, rò rỉ dữ liệu, dùng test đúng cách | Hiểu rõ | `DL04a` p29–32; `Chong_Data_Leakage_Overfitting.docx` |
| Learning curve theo kích thước mẫu | Hiểu rõ | `DL04a` p28 |
| EDA, vẽ 3D scatter/surface/histogram/contour | Làm được | `DL03a` p70–94; `[Tutorial] EDA-Python.ipynb`; `pytorch_p1_background` p28 |
| `Dataset`/`TensorDataset`/`DataLoader` | Làm được | Sách Ch.18 p198 |

Số liệu để nắm sẵn (đã tính): det C = 0.35; C⁻¹ = [[5.714, 1.429], [1.429, 0.857]]; **y_max = 0.2690** tại gốc; tích phân của f trên miền chữ nhật ≈ 0.995; khi lấy mẫu đều thì ≈ 40% mẫu có y < 0.001, ≈ 62% có y < 0.01, mean(y) ≈ 0.031, std ≈ 0.057.

### 4.2 Công việc chi tiết

| # | Việc | Đầu ra | Xong khi |
|---|---|---|---|
| T1.1 | Dựng khung repo theo `PHAN_CONG_NHIEM_VU.md` mục 4, `requirements.txt`, README bản đầu | Repo chạy `pip install -r requirements.txt` | TV2–TV4 clone về chạy được |
| T1.2 | `gaussian.py` + `tests/test_gaussian.py` | Hàm `gaussian_pdf` | Khớp `scipy.stats.multivariate_normal(MU, C).pdf` (`np.allclose`), đỉnh tại gốc ≈ 0.2690 |
| T1.3 | `generate_data.py`: pool train 2000 (seed 2026), val 200 (seed 2027), test 100 (seed 2028) | `data/train_pool.csv`, `val.csv`, `test.csv`, `data/meta.json` (seed, μ, C, miền, thống kê) | Chạy lại ra file giống hệt; ba tập không trùng điểm |
| T1.4 | `dataset.py`: đọc CSV, scale (x/2, x/4, y/y_max), `subset(A)` lấy **A mẫu đầu** của pool, `make_loader` có `generator` seed | API trong mục 2 | TV2 dùng được ngay |
| T1.5 | EDA | Hình và bảng ở mục 4.4 | Số liệu trong báo cáo truy được về CSV |
| T1.6 | **Thực nghiệm A** | `configs/expA.json`, `results/expA/*`, hình, bảng, nhận xét | Xem 4.3 |
| T1.7 | README đầy đủ (mục đích, cấu trúc thư mục, lệnh chạy từng thí nghiệm, môi trường) | `README.md` | TV4 chạy lại từ máy sạch thành công |

Gợi ý code lõi:

```python
MU = np.zeros(2); C = np.array([[0.3, -0.5], [-0.5, 2.0]])
C_INV = np.linalg.inv(C); NORM = 1 / (2 * np.pi * np.sqrt(np.linalg.det(C)))   # NORM = y_max
def gaussian_pdf(x):                      # x: (N,2) -> (N,)
    d = x - MU
    return NORM * np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, C_INV, d))
def sample_uniform(n, rng):
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-4, 4, n)])
```

### 4.3 Thiết kế Thực nghiệm A (so sánh công bằng)

- Biến độc lập: A = 200, 400, …, 2000. Các tập con **lồng nhau** (A mẫu đầu của cùng một pool) để khác biệt chỉ do số mẫu, không do may rủi chọn mẫu. (Code mẫu bốc ngẫu nhiên có lặp mỗi lần nên đồ thị rất nhiễu.)
- Giữ nguyên: kiến trúc baseline, optimizer, lr, batch size, tập val (200) và test (100).
- Vì A nhỏ thì số bước cập nhật/epoch ít hơn (A=200 chỉ 7 bước/epoch, A=2000 có 63), dùng **early stopping theo val với max_epochs đủ lớn** và ghi cả số bước đã chạy. Nêu rõ lựa chọn này trong báo cáo.
- Chạy ≥ 3 seed (khác nhau ở khởi tạo trọng số và thứ tự batch). Nên chạy **thêm một mô hình dung lượng lớn hơn** (ví dụ B=32 hoặc [10,10]) để kiểm tra giả thuyết ở mục 1 điểm 4.
- Kết quả bắt buộc: bảng train/val/test MSE, MAE hoặc RMSE (mean ± std) theo A; đồ thị lỗi theo A (trục y log, có error bar, kèm đường baseline "dự đoán trung bình"); nhận xét xu hướng.
- Phân tích phải trả lời: lỗi giảm tới mức nào thì chững lại? Khoảng cách train–val có đổi không? Error bar có chồng nhau không (nếu có, đừng kết luận quá mức)? Vì sao đường test có thể **không** tăng trở lại như hình minh họa của thầy (dữ liệu không nhiễu, dung lượng cố định)?
- (Tùy chọn, ghi rõ là ngoài đề) thêm nhiễu Gaussian vào nhãn train để minh họa quá khớp.

### 4.4 Phần báo cáo và slide của TV1

- Báo cáo Ch.2 (khoảng 2–3 trang): công thức Gaussian, μ, C, miền; cách lấy mẫu, bảng kích thước và seed từng tập; **Bảng 2.1** thống kê x₁, x₂, y (min, max, mean, std, tỷ lệ y < 0.01); **Hình 2.1** bề mặt thật + contour; **Hình 2.2** scatter 3D train/val/test (3 màu cố định); **Hình 2.3** histogram x₁, x₂, y (y dùng thang log); nhận xét mất cân bằng (nhiều mẫu ≈ 0) và hệ quả với MSE.
- Báo cáo mục 4.2: kết quả A.
- Slide: bài toán + dữ liệu (1–2 slide), kết quả A (1 slide).

### 4.5 Lỗi hay gặp

- Đề/slide ghi 0 ≤ y ≤ 1 nhưng thực tế y ≤ 0.269; ghi đúng trong báo cáo.
- Dùng chung một seed cho cả ba tập (sinh ra tập giống nhau/tương quan).
- Fit scaler trên cả val/test (nếu nhóm dùng scaler thống kê thay vì hằng số).
- Mở CSV bằng `open(...)` với đường dẫn cứng.

### 4.6 Câu hỏi vấn đáp dự kiến

1. Vì sao det C = 0.35 và đỉnh hàm bằng bao nhiêu? 2. Vì sao chia ba tập và test dùng khi nào? 3. Vì sao tập con A lồng nhau? 4. Vì sao lỗi theo A không tăng trở lại? 5. Scale x, y có gây rò rỉ không, vì sao?

---

## 5. TV2 – Mô hình, vòng huấn luyện, Thực nghiệm C (Rubric 2, 5: 25 + 10 điểm)

### 5.1 Kiến thức cần có

| Kiến thức | Mức | Học ở đâu |
|---|---|---|
| Forward: z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾, a⁽ˡ⁾ = σ(z⁽ˡ⁾); đầu ra hồi quy **tuyến tính** | Viết được công thức | `DL08a` p4, p17–19; `DL08b` p21 |
| Hàm mất mát MSE và gradient; backprop: δ_out = (2/N)(ŷ−y), δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ σ′(z⁽ˡ⁾), ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾a⁽ˡ⁻¹⁾ᵀ | Tự suy ra được, giải thích khi vấn đáp | `DL08a` p10–15, p20–21; `DL07b` p4–18; `BT07` Bài 1 |
| Đồ thị tính toán và autograd | Hiểu | `DL06a` p24–56; sách Ch.4 p47 |
| `nn.Module`, `nn.Linear`, `nn.Sequential`, `forward` | Làm được | Sách Ch.6–7 p61–73 |
| Vòng lặp: `zero_grad → forward → loss → backward → step`; `train()/eval()`; `torch.no_grad()` | Làm được, giải thích | Sách Ch.8 p80; yêu cầu đề 3.2 |
| Adam (m, v, hiệu chỉnh bias) và SGD | Hiểu công thức | Bài báo Adam; sách Ch.20.5 p224 |
| Hàm kích hoạt, ảnh hưởng đến gradient | Hiểu | `DL08b` p42–56; sách Ch.14 p152 |
| Khởi tạo Xavier/He | Biết khi nào dùng | `DL08b` p69–80 |
| Checkpoint theo val loss, early stopping, `state_dict` | Làm được | Sách Ch.13 p143, Ch.21 p240; `train_mlp.py` Lab03 (`EarlyStopping`) |
| Vanishing/exploding gradient và cách **đo** | Hiểu | `DL08a` p35–36; `DL08b` p45 |
| Số tham số: mạng [2,B,1] có 4B+1; C tầng rộng 10 có 30 + 110(C−1) + 11 | Tính được | Tự tính (41/151/261/371 với C=1..4) |

### 5.2 Công việc chi tiết

| # | Việc | Đầu ra | Xong khi |
|---|---|---|---|
| T2.1 | `mlp.py`: `MLP(hidden_sizes, activation)` | `nn.Module` tùy biến số tầng/nơ-ron | `print(model)` và đếm tham số đúng công thức |
| T2.2 | `trainer.py`: `train_one_epoch`, `validate`, `fit`; log CSV theo epoch; early stopping; lưu `best.pt` và nạp lại | Hàm `fit` theo mục 2 | Nạp `best.pt` cho lại đúng val loss đã log |
| T2.3 | `run_experiment.py` đọc JSON, lặp giá trị × seed, ghi `summary.csv`, tùy chọn `--eval-test` | CLI | Chạy lại ra cùng số (sai khác < 1e-6) |
| T2.4 | **Gradient check**: mạng 2→2→1 tính tay (hoặc NumPy) so với autograd; thêm kiểm tra đạo hàm số (`DL08b` p57–62) | `tests/test_backprop.py` | `torch.allclose` đạt; dùng để giải thích khi vấn đáp |
| T2.5 | **Chốt baseline** (cùng nhóm, G2): lưới nhỏ lr ∈ {1e-3, 3e-3, 1e-2} × activation ∈ {tanh, sigmoid, relu}, chỉ xét **val** | `configs/baseline.json` đóng băng | Có bảng lưới được lưu lại làm minh chứng |
| T2.6 | **Thực nghiệm C** | `results/expC/*`, hình, bảng | Xem 5.3 |
| T2.7 | Chuẩn bị demo chạy code trực tiếp | Lệnh chạy 1 cấu hình < 1 phút | Giải thích được từng dòng vòng lặp |
| T2.8 | **So sánh optimizer** (làm trước khi TV3 chạy B): giữ kiến trúc baseline, sigmoid, A = 1000, 3 seed; so sánh SGD, Momentum, Nesterov, RMSProp, Adam, Nadam; **mỗi optimizer dò lr riêng**; chỉ xét **val** | `configs/grid_optimizer.json`, `results/grid_optimizer/*`, hình train/val loss theo epoch, thêm vào mục 3.5 và 4.2 | TV3 review xong; nhóm chốt giữ Adam hay đổi baseline |

Khung code (giữ đúng thứ tự các bước mà đề yêu cầu):

```python
ACTS = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}
class MLP(nn.Module):
    def __init__(self, hidden_sizes=(10,), activation="tanh"):
        super().__init__()
        layers, d = [], 2
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), ACTS[activation]()]
            d = h
        layers.append(nn.Linear(d, 1))                 # đầu ra tuyến tính
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)                 # (N,) cùng shape với y

def train_one_epoch(model, loader, opt, crit):
    model.train(); tot = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb); assert pred.shape == yb.shape
        loss = crit(pred, yb); loss.backward(); opt.step()
        tot += loss.item() * xb.size(0)                # trọng số theo kích thước batch
    return tot / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, crit):
    model.eval(); tot = 0.0
    for xb, yb in loader:
        tot += crit(model(xb), yb).item() * xb.size(0)
    return tot / len(loader.dataset)
```

### 5.3 Thiết kế Thực nghiệm C

- Biến độc lập: C = 1, 2, 3, 4 (mở rộng 5, 6, 8), **mỗi tầng 10 nơ-ron**, A cố định (ví dụ 1000 hoặc 2000), baseline giữ nguyên.
- Vì C tăng làm số tham số tăng (41 → 151 → 261 → 371), chạy thêm **nhóm đối chứng cùng số tham số**: C=1 với B ≈ 38, 65, 93 (4B+1 = 153, 261, 373; file `configs/expC_ctrl.json`). Nhờ đó tách được "sâu hơn" khỏi "nhiều tham số hơn" (`DL07b` p54: tăng độ sâu hiệu quả hơn tăng tham số).
- Đo cho từng lần chạy: test MSE/MAE/RMSE, số tham số, **thời gian huấn luyện** (`time.perf_counter()` chỉ quanh `fit`, cùng CPU, cùng số luồng), thời gian/epoch (ms), số epoch tới điểm tốt nhất, đường val loss theo epoch (trục y log).
- Để nói về vanishing gradient **phải có bằng chứng**: ghi chuẩn gradient từng lớp (`[p.grad.norm() for p in model.parameters()]`) ở epoch đầu và cuối. Nếu số liệu không cho thấy gradient lớp đầu nhỏ hơn rõ rệt thì không được kết luận vanishing gradient.
- Có thể so thêm activation (tanh/sigmoid/relu) ở C=4: sigmoid thường khó huấn luyện hơn khi sâu (`DL08b` p45).
- Khi giải thích kết quả "sâu hơn tốt hơn" cần nêu cơ chế: nhiều lớp biểu diễn đỉnh cục bộ bằng ít nơ-ron hơn (`DL07b` p52, p54) và hạn chế: kết quả có thể thay đổi khi lr/số epoch khác.

### 5.4 Phần báo cáo và slide của TV2

- Báo cáo Ch.3 (khoảng 3–4 trang): **Hình 3.1** sơ đồ kiến trúc (dùng `src/visual/draw_mlp.py` đã có trong repo); công thức forward, MSE, backprop, Adam; bảng siêu tham số baseline (seed, batch size, epochs, optimizer, lr, activation, số tầng/nơ-ron, điều kiện dừng); môi trường (phiên bản Python/PyTorch, CPU/GPU); đoạn code quan trọng kèm giải thích (đề yêu cầu "giải thích các đoạn mã quan trọng").
- Báo cáo mục 4.4: kết quả C.
- Slide: sơ đồ MLP + code PyTorch chính (2 slide), kết quả C (1 slide). Chuẩn bị demo.

### 5.5 Lỗi hay gặp

- Shape `(N,1)` so với `(N,)` trong `MSELoss` (mục 1 điểm 5).
- Quên `model.eval()`/`torch.no_grad()` khi validate; quên `zero_grad()`.
- Tính loss trung bình epoch không theo trọng số batch (batch cuối nhỏ hơn).
- Lưu `best.pt` bằng tham chiếu thay vì `copy.deepcopy(state_dict)`.
- Dừng sớm quá vội: lần chạy thử B=1 dừng ở epoch 22 vì plateau; đặt `patience` lớn và/hoặc `min_epochs`.
- So sánh thời gian khi máy đang chạy việc khác; chạy lại ≥ 3 seed và báo cáo mean ± std.

### 5.6 Câu hỏi vấn đáp dự kiến

1. Vì sao đầu ra không dùng sigmoid? 2. `train()` và `eval()` khác gì, `no_grad` để làm gì (mạng này không có Dropout nhưng vẫn phải đúng ngữ cảnh)? 3. Viết ∂L/∂W cho tầng cuối và tầng đầu. 4. Checkpoint tốt nhất chọn theo tập nào, vì sao không theo test? 5. Vì sao thời gian huấn luyện tăng theo C? 6. Nếu thấy loss giảm mà dự đoán gần hằng số thì nghi ngờ lỗi gì?

---

## 6. TV3 – Đánh giá, trực quan hóa, Thực nghiệm B, chọn mô hình cuối (Rubric 4, 6: 10 + 10 điểm)

### 6.1 Kiến thức cần có

| Kiến thức | Mức | Học ở đâu |
|---|---|---|
| MSE, MAE, RMSE = √MSE, R² = 1 − SS_res/SS_tot; ý nghĩa khi y rất nhỏ | Tính được, diễn giải được | `DL02b` p20–22; `DL04a` |
| Underfitting, overfitting, bias–variance, khoảng cách train–val | Hiểu rõ | `DL04a` p6–11; `DL07b` p46–54 |
| Universal Approximation: một tầng ẩn đủ nơ-ron xấp xỉ hàm liên tục | Hiểu, nêu được giới hạn (cần đủ B và tối ưu được) | `DL07b` p48 |
| mean ± std qua nhiều seed, error bar, ý nghĩa thống kê thô | Làm được | `DL04a` p28 |
| Matplotlib: 3D scatter, `plot_surface`, `contourf`, `errorbar`, trục log | Làm được | `pytorch_p1_background` p28; sách Ch.23 p255 |
| Dự đoán trên lưới và vẽ bề mặt xấp xỉ | Làm được | Tự làm (np.meshgrid → model → reshape) |

### 6.2 Công việc chi tiết

| # | Việc | Đầu ra | Xong khi |
|---|---|---|---|
| T3.1 | `evaluate.py`: `predict`, `metrics` (thang gốc), `predict_grid`, đánh giá thêm trên **lưới dày** (ví dụ 100×100) | API mục 2 | Kiểm thử với mô hình giả (ŷ = y cho ra MSE = 0, R² = 1) |
| T3.2 | `plots.py`: loss curve, y–ŷ (đường y = x), bề mặt thật so với xấp xỉ (cùng giới hạn trục z), bản đồ sai số \|y−ŷ\| (`contourf`), lỗi theo A/B/C có error bar, cột thời gian theo C | Hàm vẽ dùng chung, một style duy nhất (màu train/val/test cố định, nhãn trục, đơn vị, 300 dpi) | TV1, TV2 gọi được |
| T3.3 | **Thực nghiệm B** | `results/expB/*`, hình, bảng | Xem 6.3 |
| T3.4 | Hình "bề mặt xấp xỉ theo B" (B = 1, 2, 3, 5, 10) cùng seed, cùng trục | **Hình 4.x** | Thấy trực quan B nhỏ không tạo được đỉnh |
| T3.5 | `aggregate.py`: gom mọi `summary.csv` thành bảng tổng hợp (mean ± std) | `results/summary_all.csv` | TV4 lấy số cho báo cáo |
| T3.6 | **Chọn mô hình cuối theo val** (không theo test), chạy `--eval-test` **một lần**, lưu `best.pt` mô hình chọn | `results/final/`, hình y–ŷ và bề mặt trên test | Báo cáo nêu rõ quy trình chọn |
| T3.7 | **Review chéo** code `fit()` của TV2 để chắc không đụng test | Biên bản review | Không có tham chiếu test trong lúc train/chọn |
| T3.8 | **Review T2.8** của TV2: lr được dò công bằng cho từng optimizer, không đụng test | Biên bản review | Xong trước khi chạy Thực nghiệm B |

### 6.3 Thiết kế Thực nghiệm B

- B = 1, 2, …, 10 (mở rộng: 12, 16, 24, 32, 64), **một tầng ẩn**, baseline giữ nguyên, A = 1000; chạy thêm A = 200 để có cơ hội thấy quá khớp.
- ≥ 3 seed, nên 5 vì B nhỏ dễ kẹt ở cực trị xấu (B=1 trong lần chạy thử). Với B = 1 nhận xét cơ chế: đầu ra là một hàm tăng/giảm đơn điệu theo một hướng duy nhất (ridge function) nên không thể tạo đỉnh cục bộ; B nhỏ chỉ ghép được vài "sườn dốc" nên underfit. Xác nhận bằng **Hình bề mặt xấp xỉ** (T3.4), đừng chỉ khẳng định bằng lời.
- Bảng: train/val/test MSE, MAE hoặc RMSE (mean ± std) theo B; đồ thị lỗi theo B (có error bar, trục log); thêm đồ thị khoảng cách val − train.
- Nhận xét underfitting (lỗi train và val cùng cao khi B nhỏ) và quá khớp (khoảng cách tăng khi B lớn, A nhỏ). **Nếu không thấy quá khớp, nói thẳng là không thấy và giải thích** (dữ liệu không nhiễu, early stopping theo val), thay vì ép kết luận theo hình minh họa của thầy.
- Lưu ý số lần chạy cho B lớn: kiểm tra B=32/64 có hội tụ trong cùng ngân sách epoch không.

### 6.4 Quy trình chọn mô hình cuối (đề xuất)

1. Ứng viên: cấu hình tốt nhất theo **val** (mean qua seed) trong từng nhóm A, B, C; so sánh ở cùng A (ví dụ A = 2000) để công bằng.
2. Chọn một mô hình theo val, huấn luyện lại hoặc lấy checkpoint tốt nhất theo val.
3. Chạy `--eval-test` cho mô hình này và báo cáo MSE, MAE/RMSE, R²; bổ sung kết quả trên lưới dày (ghi rõ là bổ sung, không dùng để chọn).
4. Báo cáo bảng các cấu hình đã xét, tiêu chí chọn, giới hạn (test chỉ 100 mẫu nên các chỉ số nhiễu; chỉ khoảng 11% mẫu có y > 0.1 nên vùng đỉnh được đánh giá yếu; vì vậy cần lưới dày làm bổ sung).

### 6.5 Phần báo cáo và slide của TV3

- Báo cáo mục 4.3 (kết quả B) và 4.5 (mô hình chọn, đánh giá test, y–ŷ, bề mặt xấp xỉ, bản đồ sai số) và đóng góp vào Ch.5 (diễn giải B, phân tích trường hợp không như kỳ vọng).
- Slide: kết quả B (1 slide), minh họa mô hình xấp xỉ và dữ liệu thật (1 slide).

### 6.6 Lỗi hay gặp

- Dùng `fig.gca(projection="3d")` (lỗi ở Matplotlib hiện tại).
- Hai bề mặt (thật/xấp xỉ) khác trục z nên nhìn không so sánh được.
- Vẽ lỗi thang tuyến tính khiến mọi điểm dồn về sát 0; dùng trục log.
- Chọn mô hình dựa trên test, hoặc chạy `--eval-test` nhiều lần rồi chọn.
- Quên đổi lại thang y về gốc trước khi tính metric.
- Dùng font không có dấu tiếng Việt (DejaVu Sans mặc định hỗ trợ).

### 6.7 Câu hỏi vấn đáp dự kiến

1. Vì sao B=1 không xấp xỉ được đỉnh? 2. Vì sao RMSE nhỏ vẫn chưa chắc tốt (so với baseline trung bình, R²)? 3. Underfitting/overfitting nhìn thấy ở đâu trong bảng? 4. Vì sao không chọn mô hình theo test? 5. Test 100 mẫu có đủ tin cậy không, làm gì để bổ sung?

---

## 7. TV4 – Báo cáo, slide, QA tái lập, đóng gói (Rubric 7, 8: 15 + 10 điểm)

### 7.1 Kiến thức cần có

| Kiến thức | Mức | Học ở đâu |
|---|---|---|
| Bố cục báo cáo khoa học và 6 chương theo đề | Làm được | `Sample_Report.pdf`; đề mục 6 |
| Word: mục lục tự động, caption đánh số hình/bảng, tham chiếu chéo, header/footer | Làm được | Word / quy chuẩn `SKILL.md` |
| Tự động sinh Word bằng `python-docx` từ CSV để số liệu truy vết được | Làm được | `Lab08\build_doc.py` |
| Trích dẫn tài liệu (kiểu IEEE) | Làm được | Mục 9 |
| Hiểu toàn bộ pipeline đủ để viết Ch.1, 5, 6 | Hiểu | Đọc code TV1–TV3 |
| Tạo slide (PowerPoint hoặc `python-pptx`), xuất PDF | Làm được | Bản thân |
| Tái lập: tạo venv sạch, chạy theo README | Làm được | `Lab03\README.md` (mẫu) |
| Sơ đồ quy trình | Làm được | `prj\model\diagrams.py` Lab03 (`draw_card`, `draw_arrow_clean`, `create_diagram_3`) hoặc draw.io |

Quy chuẩn Word có sẵn từ các bài trước (`SKILL.md`): Times New Roman, thân 13 pt, căn trái, giãn dòng 1.25, bảng viền mảnh không tô nền, gắn `cantSplit` và `tblHeader`, công thức không đóng khung nền xám. Nhóm có thể dùng lại nếu thấy phù hợp; điều chỉnh trang bìa thành bìa nhóm (tên thành viên, MSSV, lớp, giảng viên).

### 7.2 Công việc chi tiết

| # | Việc | Đầu ra | Xong khi |
|---|---|---|---|
| T4.1 | Template Word (bìa, mục lục tự động, danh mục hình/bảng, style caption) | `report/BaoCao_Nhom10.docx` (khung) | Gửi cả nhóm ở G1 |
| T4.2 | Sơ đồ quy trình (sinh dữ liệu → chia 3 tập → huấn luyện/val/checkpoint → đánh giá test) | Hình dùng cho báo cáo, slide, README | Khớp với code thật |
| T4.3 | `make_report_assets.py`: gom CSV → bảng Word/Markdown, bảng siêu tham số | Bảng sinh tự động | Mọi số trong báo cáo truy được về file log |
| T4.4 | Viết Ch.1, mục 4.1, Ch.5, Ch.6, tài liệu tham khảo, phụ lục lệnh tái lập | Bản nháp, rồi bản cuối | Theo mục 7.3 |
| T4.5 | Ghép phần của TV1–TV3, thống nhất giọng văn, đánh số hình/bảng | Báo cáo 10–20 trang | Xuất `.docx` và `.pdf` |
| T4.6 | Slide 10–15 trang theo mục 7.4, bản PDF | `.pptx` + `.pdf` | Tập dượt ≤ 15 phút |
| T4.7 | **QA tái lập** trên thư mục/venv mới theo mục 7.5 | Biên bản QA | Pass hết checklist |
| T4.8 | Đóng gói zip, bảng thành viên – nhiệm vụ – % đóng góp (tổng 100%) | `BTL01_Nhom_HoTen_MSSV.zip` | Đúng danh mục nộp |
| T4.9 | Đưa kết quả T2.8 vào Ch.5 và phụ lục lệnh tái lập | Đoạn thảo luận chọn optimizer | Số liệu truy được về `results/grid_optimizer/` |

### 7.3 Nội dung các chương do TV4 viết

| Chương | Nội dung bắt buộc | Dung lượng gợi ý |
|---|---|---|
| Ch.1 Giới thiệu | Mục tiêu; phát biểu bài toán (công thức, μ, C, miền); **đóng góp của nhóm**; bảng phân công và % đóng góp | 1–1.5 trang |
| 4.1 Thiết lập và quy trình công bằng | Bảng siêu tham số baseline; quy tắc "chỉ đổi một yếu tố"; seed; early stopping; test chỉ dùng cuối; ≥ 3 seed | 1 trang |
| Ch.5 Thảo luận | Xu hướng A/B/C; **trường hợp không như kỳ vọng** (ví dụ lỗi nhóm C giảm khi tăng tầng, không có chữ U ở A và B); hạn chế (test 100 mẫu, chỉ một hàm Gaussian, dữ liệu không nhiễu, một họ kiến trúc); đề xuất cải tiến | 2 trang |
| Ch.6 Kết luận | Tóm tắt kết quả định lượng + mô hình chọn | 0.5 trang |
| Tài liệu tham khảo, phụ lục | Mục 9; lệnh tái lập | 1 trang |

### 7.4 Kế hoạch slide (10–15 slide, 10–15 phút)

| # | Slide | Phụ trách nội dung |
|---|---|---|
| 1 | Tiêu đề, thành viên | TV4 |
| 2 | Mục tiêu và bài toán | TV4 |
| 3 | Gaussian 2D: công thức, μ, C, bề mặt | TV1 |
| 4 | Dữ liệu: scatter train/val/test, cách chia và seed | TV1 |
| 5 | Sơ đồ MLP và công thức | TV2 |
| 6 | Pipeline PyTorch (Dataset/DataLoader, vòng lặp, checkpoint) | TV2 |
| 7 | Baseline và quy trình đánh giá công bằng | TV4 |
| 8 | Kết quả A | TV1 |
| 9 | Kết quả B | TV3 |
| 10 | Kết quả C (kèm thời gian) | TV2 |
| 11 | Mô hình chọn: y–ŷ, bề mặt xấp xỉ so với thật | TV3 |
| 12 | Thảo luận, trường hợp bất ngờ | TV4 |
| 13 | Kết luận, hạn chế, hướng cải tiến | TV4 |
| 14 | Phân công và đóng góp | TV4 |
| 15 | Q&A / demo | Cả nhóm |

Ưu tiên đồ thị rõ nhãn, trục, đơn vị; mỗi slide một ý chính.

### 7.5 Checklist QA tái lập

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py
python src/run_experiment.py --config configs/baseline.json
python src/run_experiment.py --config configs/expA.json
python src/run_experiment.py --config configs/expB.json
python src/run_experiment.py --config configs/expC.json
python src/make_report_assets.py
```

- [ ] Chạy từ thư mục sạch, không lỗi; không còn đường dẫn tuyệt đối (`grep` các chuỗi `C:\`, `D:\`, `/Users/`)
- [ ] Không có `import tensorflow`/`keras`/AutoML
- [ ] Chạy hai lần cho cùng kết quả (sai khác < 1e-6)
- [ ] `fit()` không tham chiếu test; test chỉ xuất hiện trong bước `--eval-test`
- [ ] Mọi bảng/hình trong báo cáo có file nguồn trong `results/` hoặc `figures/`
- [ ] README có: mô tả thư mục, lệnh từng thí nghiệm, môi trường
- [ ] Có `.pt` của mô hình chọn, CSV log từng lần chạy
- [ ] Báo cáo `.docx` + `.pdf` 10–20 trang, hình/bảng đánh số và có chú thích
- [ ] Slide `.pptx` + `.pdf` 10–15 slide

### 7.6 Câu hỏi vấn đáp dự kiến

1. Nhóm đóng góp gì ngoài việc chạy code? 2. Quy trình công bằng gồm những gì? 3. Vì sao kết quả khác với hình minh họa của thầy/Sample_Report? 4. Hạn chế lớn nhất của thí nghiệm? 5. Làm sao chứng minh số trong báo cáo không chép từ mẫu?

---

## 8. Mốc kiểm tra chéo

| Giai đoạn | Điều kiện qua mốc | Người kiểm |
|---|---|---|
| G1 Khởi động | Khung repo chạy; hợp đồng giao diện được chốt; template Word đã gửi | TV4 kiểm TV1 |
| G2 Baseline | `baseline.json` đóng băng có bảng lưới minh chứng; đạo hàm autograd khớp tay; có bảng so sánh optimizer (T2.8) | TV3 kiểm TV2 |
| G3 Thực nghiệm | `summary.csv` đủ A/B/C ≥ 3 seed; hình đủ nhãn | TV1 kiểm TV3 (và ngược lại) |
| G4 Viết | Bản nháp báo cáo + slide; mọi số truy vết được | TV2 kiểm TV4 |
| G5 Hoàn thiện | QA tái lập pass; tập dượt trả lời vấn đáp | Cả nhóm |

Quy tắc: đổi baseline sau G2 thì phải chạy lại cả A, B, C.

---

## 9. Tài liệu tham khảo gợi ý cho báo cáo (TV4 kiểm tra lại chi tiết trước khi nộp)

1. G. Cybenko, "Approximation by superpositions of a sigmoidal function," *Math. Control Signals Systems*, vol. 2, no. 4, pp. 303–314, 1989.
2. K. Hornik, M. Stinchcombe, H. White, "Multilayer feedforward networks are universal approximators," *Neural Networks*, vol. 2, no. 5, pp. 359–366, 1989.
3. D. E. Rumelhart, G. E. Hinton, R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, pp. 533–536, 1986.
4. Y. LeCun, Y. Bengio, G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436–444, 2015. (`paper\!! NatureDeepReview.pdf`)
5. D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization," arXiv:1412.6980 (ICLR 2015).
6. X. Glorot, Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," *AISTATS*, 2010.
7. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," *NeurIPS*, 2019.
8. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.
9. J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.
10. P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020.
11. Tài liệu môn học (TS. Đỗ Như Tài): `DL07b - Deep_networks_1`, `DL08a - Deep Neural Network (MLP)`, `DL08b - Deep_networks_2`, `DL04a - Model Selection`; sách `deep_learning_with_pytorch.pdf`; đề bài `MLP_Exercise.pdf`.
