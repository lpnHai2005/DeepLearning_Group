# BTL01 – Xấp xỉ hàm Gaussian 2D bằng MLP (PyTorch) — Phân tích & Phân công

## 1. Tóm tắt yêu cầu (từ `MLP_Exercise - Yeu Cau.pdf`, `MLP_Exercise.pdf`)

| Hạng mục | Nội dung bắt buộc |
|---|---|
| Hàm mục tiêu | f(x) = 1/(2π√det C) · exp(−½ (x−μ)ᵀ C⁻¹ (x−μ)), μ = [0,0], C = [[0.3,−0.5],[−0.5,2]] |
| Miền dữ liệu | x₁ ∈ [−2,2], x₂ ∈ [−4,4] (lấy mẫu đều); y = f(x₁,x₂) |
| Chia tập | train (A mẫu) / validation riêng / **test độc lập 100 mẫu**; ghi seed; **test chỉ dùng đánh giá cuối** |
| Mô hình | `nn.Module` + `nn.Linear`, tùy biến số tầng ẩn và số nơ-ron/tầng; 2 input → 1 output |
| Huấn luyện | Tensor, `TensorDataset/DataLoader`, `MSELoss`, Adam/SGD; vòng lặp tự viết `forward → loss → zero_grad → backward → step`; `model.train()/eval()`, `torch.no_grad()` |
| Theo dõi | log loss theo epoch (CSV), checkpoint best theo val loss (.pt), chạy được trên CPU |
| Đánh giá | MSE + MAE/RMSE trên test; biểu đồ y–ŷ; bề mặt xấp xỉ |
| Thực nghiệm A | A = 200, 400, …, 2000 → bảng + đồ thị sai số theo A |
| Thực nghiệm B | 1 tầng ẩn, B = 1..10 (có thể mở rộng) → bảng/đồ thị, nhận xét under/overfitting |
| Thực nghiệm C | C = 1..4 tầng ẩn (ít nhất) → test error, đường hội tụ loss, **thời gian huấn luyện** |
| Khuyến khích | ≥ 3 seeds, báo cáo mean ± std |
| Mã nguồn | tổ chức theo chức năng, `requirements.txt`, `README.md`, lệnh chạy từng thí nghiệm, không đường dẫn cá nhân, **không nộp chỉ file HTML** |
| Báo cáo | .docx + .pdf, 10–20 trang, 6 chương (Giới thiệu, Dữ liệu, Mô hình, Thực nghiệm, Thảo luận, Kết luận + TLTK + phụ lục) |
| Slide | .pptx (+pdf), 10–15 slide, 10–15 phút, sẵn sàng demo code & hỏi đáp |
| Nộp | `BTL01_Nhom_HoTen_MSSV.zip`, ghi thành viên – nhiệm vụ – % đóng góp (tổng 100%) |
| **Cấm** | TensorFlow/Keras, AutoML |

### Rubric (100 đ) → ai chịu trách nhiệm chính

| Tiêu chí | Điểm | Phụ trách chính |
|---|---|---|
| 1. Dữ liệu & bài toán | 10 | TV1 |
| 2. Triển khai PyTorch | 25 | TV2 (TV1 phần README/chạy lại) |
| 3. Thực nghiệm A | 10 | TV1 |
| 4. Thực nghiệm B | 10 | TV3 |
| 5. Thực nghiệm C | 10 | TV2 |
| 6. Đánh giá & phân tích chung | 10 | TV3 |
| 7. Báo cáo viết | 15 | TV4 (mọi người viết phần mình) |
| 8. Slide & thuyết trình | 10 | TV4 (mọi người trình bày phần mình) |

## 2. Phân tích kỹ thuật – những điểm cần lưu ý

**Đặc điểm hàm mục tiêu** (đã tính kiểm tra):
- det C = 0.35 → giá trị cực đại y_max = 1/(2π√0.35) ≈ **0.2690** (tại gốc). Vậy y ∈ (0, 0.269], không phải [0,1].
- Hệ số tương quan ρ = −0.5/√(0.3·2) ≈ −0.645 → "sống núi" nghiêng, kéo dài theo trục riêng (−0.26, 0.96) (phương sai 2.14) và hẹp theo trục (0.96, 0.26) (phương sai 0.16).
- Lấy mẫu đều trên hình chữ nhật → **dữ liệu mất cân bằng**: ~40% mẫu có y < 0.001, ~62% có y < 0.01; mean(y) ≈ 0.031, std ≈ 0.057. Hệ quả: MSE rất nhỏ về số tuyệt đối, mô hình dễ "lười" dự đoán ≈ 0 mà MSE vẫn thấp → cần thêm MAE/RMSE, R², và xem lỗi tại vùng đỉnh.
- Dữ liệu **không nhiễu** → overfitting khó xuất hiện rõ như hình minh họa của giảng viên; nhóm phải **phân tích trung thực** hiện tượng thực tế (rubric có mục "trường hợp không như kỳ vọng").

**Thiết kế để so sánh công bằng & tránh rò rỉ** (các lỗi có trong báo cáo/code mẫu cần tránh):
- Code mẫu dùng Keras (cấm) và dùng **test làm validation** (rò rỉ) → phải có tập val riêng.
- Thực nghiệm A trong mẫu bốc ngẫu nhiên có lặp (`randint`) mỗi lần → đường cong rất nhiễu. Đề xuất: sinh **1 pool train 2000 mẫu cố định**, tập A = **A mẫu đầu tiên** của pool (tập lồng nhau), val/test cố định.
- Với số epoch cố định, A nhỏ ⇒ ít bước cập nhật hơn ⇒ nhiễu so sánh. Đề xuất dùng max_epochs lớn + early stopping theo val (hoặc cố định tổng số iteration) và ghi rõ trong báo cáo.
- Chuẩn hóa: x có thể scale theo biên miền đã biết về [−1,1] (không rò rỉ); y có thể chia cho y_max hoặc thống kê từ train. **Báo cáo chỉ số trên thang y gốc.**
- Chạy ≥ 3 seed (khác nhau ở khởi tạo trọng số và thứ tự batch, dữ liệu giữ nguyên) → mean ± std.
- Thực nghiệm C: tăng tầng làm tăng số tham số → nêu rõ hạn chế/so sánh thêm (tùy chọn) phiên bản cân bằng số tham số; với sigmoid/tanh sâu cần bàn về vanishing gradient.

**Baseline đề xuất (chốt sau khi chạy thử, tune chỉ dựa trên val):**

| Tham số | Giá trị đề xuất |
|---|---|
| Seed dữ liệu | train_pool=2026, val=2027, test=2028 (ghi trong config) |
| Kích thước | train_pool 2000, val 200, test 100 |
| Kiến trúc | 2 → [10] → 1, output tuyến tính; activation chốt bằng val trong {tanh, sigmoid, relu} (chạy thử sơ bộ: relu < tanh < sigmoid về MSE, xem `CHI_TIET_NHIEM_VU.md` mục 1) |
| Optimizer | Adam, lr chốt bằng val trong {1e-3, 3e-3, 1e-2} (lr = 1e-3 với 300 epoch chưa hội tụ) |
| Batch size | 32 |
| Epochs / dừng | max ≥ 2000, early stopping patience ≥ 100–200 theo val MSE, lưu best checkpoint |
| Seeds huấn luyện | 0, 1, 2 (khuyến khích thêm 3, 4) |
| Cấu hình cố định khi khảo sát | A: B=10, C=1 · B: A=1000 (thêm A=200 để thấy overfit), C=1 · C: A=1000, 10 nơ-ron/tầng |
| Mở rộng | B ∈ {16, 32, 64}; C ∈ {5, 6, 8} |

## 3. Tham khảo repo `sowwnn/3DDualDomainAttention`

Repo là bài toán phân đoạn u não 3D (ICIP 2023, nhóm tác giả của TS. Đỗ Như Tài – cùng tác giả `Sample_Report.pdf`), **khác hẳn bài toán**, chỉ nên học **cách tổ chức & trình bày**:

| Học từ repo | Áp dụng cho BTL |
|---|---|
| Cấu trúc `src/libs/data`, `src/model`, `src/stat`, script train riêng | Tách `data/`, `model/`, `engine/`, `visual/`, `experiments/` |
| `seg_train.py`: `train_step()`, `valid_step()`, `run()`, `main()` + `argparse --input exp.json` | Trainer cùng khuôn: config JSON cho baseline/A/B/C, chạy bằng 1 lệnh |
| Lưu `output.json` (lịch sử loss/metric) + `config.json` mỗi lần chạy | Mỗi run lưu `config.json`, `history.csv`, `metrics.json` → truy vết được số liệu |
| Checkpoint dạng dict `{'epoch','model','optimizer'}` → `best_metric_model.pth`, `last_model.pth` | Lưu `best.pt` theo val MSE (+ `last.pt`) |
| `get_model(name, ...)` factory | `build_mlp(hidden_sizes, activation)` |
| Thư mục `static/` với hình overview + detail kiến trúc nhúng trong README | Vẽ hình kiến trúc MLP tổng quát + chi tiết 1 neuron (draw.io/TikZ) cho báo cáo, slide, README |
| README có badge Python/PyTorch, tác giả, phương pháp, hình, cách chạy | README của nhóm theo cùng bố cục |
| Notebook EDA trong `src/stat` | `notebooks/01_EDA.ipynb` cho thống kê dữ liệu |

Không dùng: MONAI, wandb (tùy chọn), AMP, sliding window.

## 4. Cấu trúc thư mục đề xuất

```
BTL01_Nhom10/
├── README.md, requirements.txt
├── configs/            baseline.json, expA.json, expB.json, expC.json
├── src/
│   ├── data/           gaussian.py (f, generate, save/load CSV), dataset.py (scale, DataLoader)
│   ├── model/          mlp.py (class MLP(nn.Module))
│   ├── engine/         trainer.py (train_step, valid_step, fit, early stop, checkpoint), evaluate.py (MSE/MAE/RMSE/R², predict)
│   ├── visual/         plots.py (scatter 3D, surface, loss curve, y–ŷ, error map, error-vs-A/B/C)
│   ├── utils/          seed.py, io.py, timer.py
│   ├── generate_data.py
│   ├── run_experiment.py   (--config configs/expA.json)
│   └── make_report_assets.py (gom summary.csv → bảng + hình)
├── notebooks/          01_EDA.ipynb, 02_demo.ipynb
├── data/               train_pool.csv, val.csv, test.csv
├── results/            expA|expB|expC/<value>/seed_<k>/{config.json, history.csv, metrics.json, best.pt}, summary.csv
├── figures/            hình dùng cho báo cáo/slide (đánh số)
└── report/             BaoCao_Nhom10.docx/.pdf, Slide_Nhom10.pptx/.pdf
```

## 5. Phân công 4 thành viên

Nguyên tắc: mỗi người **vừa code, vừa chạy thí nghiệm, vừa viết phần báo cáo + slide của mình**, và phải giải thích được code của mình khi demo. Tỷ lệ đóng góp dự kiến 25% mỗi người (điều chỉnh theo thực tế).

### TV1 – Dữ liệu, hạ tầng repo & Thực nghiệm A (Rubric 1, 3, một phần 2)
- `src/data/gaussian.py`: hàm Gaussian đúng công thức (đối chiếu `scipy.stats.multivariate_normal.pdf` bằng unit test nhỏ), sinh train_pool/val/test với seed riêng, lưu CSV.
- `src/data/dataset.py`: scale x/y, `TensorDataset` + `DataLoader`, hàm lấy A mẫu đầu pool.
- EDA: scatter 3D train/val/test, bề mặt hàm thật + contour, histogram x₁, x₂, y, bảng thống kê (min/max/mean/std, tỷ lệ y≈0).
- Khung repo, `requirements.txt`, `README.md` (cấu trúc, lệnh chạy từng thí nghiệm), `utils/seed.py`; kiểm tra chạy lại từ đầu trên máy sạch.
- **Thực nghiệm A**: A = 200..2000 × ≥3 seeds → bảng MSE/MAE/RMSE (train/val/test, mean±std), đồ thị sai số theo A, nhận xét xu hướng.
- Viết báo cáo: **Ch.2 Dữ liệu**, **mục 4.2 Thực nghiệm A**; slide: bài toán + dữ liệu + kết quả A.

### TV2 – Mô hình, vòng huấn luyện & Thực nghiệm C (Rubric 2, 5)
- `src/model/mlp.py`: `MLP(nn.Module)` nhận `hidden_sizes: list[int]`, `activation`, khởi tạo trọng số; in số tham số.
- `src/engine/trainer.py`: `train_step`/`valid_step`/`fit` đúng chuẩn (`train()/eval()`, `no_grad`, zero_grad→backward→step), log loss theo epoch ra CSV, early stopping, checkpoint best/last, đo thời gian, chọn CPU/GPU.
- `src/run_experiment.py` đọc config JSON, lặp qua giá trị khảo sát × seeds, ghi `results/.../summary.csv`. **Chốt interface sớm (cuối giai đoạn 1)** để TV1, TV3 dùng.
- Chạy baseline và chốt baseline cùng nhóm (chỉ dựa trên val).
- **Thực nghiệm C**: C = 1..4 (+5,6,8) → bảng test error, số tham số, thời gian train/epoch & tổng, đường hội tụ loss từng C; phân tích độ sâu (chỉ kết luận vanishing gradient nếu có số liệu chuẩn gradient từng lớp; chạy thử sơ bộ cho thấy tăng tầng làm lỗi giảm, ngược với Sample_Report).
- Viết báo cáo: **Ch.3 Mô hình & cài đặt** (công thức forward, MSE, backprop, Adam, môi trường), **mục 4.4 Thực nghiệm C**; slide: kiến trúc + code PyTorch chính + kết quả C. Chuẩn bị **demo chạy code**.

### TV3 – Thực nghiệm B, đánh giá & trực quan hóa chung (Rubric 4, 6)
- `src/engine/evaluate.py`: MSE, MAE, RMSE, R² (thang y gốc), dự đoán trên lưới.
- `src/visual/plots.py`: loss curve train/val, scatter y–ŷ (đường y=x), bề mặt xấp xỉ vs bề mặt thật, bản đồ sai số |y−ŷ| trên lưới, đồ thị error vs biến khảo sát có error bar. Dùng chung cho A/B/C (thống nhất style, nhãn trục, đơn vị).
- **Thực nghiệm B**: B = 1..10 (+16, 32, 64) × ≥3 seeds, ở A=1000 và A=200 → bảng + đồ thị train/val/test, chỉ ra vùng underfitting (B nhỏ) và overfitting/khoảng cách train–val.
- **Tổng hợp chọn mô hình cuối** (theo val), đánh giá test 1 lần, xuất `best.pt` + hình y–ŷ / bề mặt của mô hình được chọn.
- Viết báo cáo: **mục 4.3 Thực nghiệm B**, **mục 4.5 Mô hình được chọn & đánh giá test**; slide: kết quả B + minh họa xấp xỉ.

### TV4 – Báo cáo, slide, thảo luận & kiểm định tái lập (Rubric 7, 8)
- Lập template Word (trang bìa, mục lục tự động, đánh số hình/bảng, caption, header/footer) theo bố cục `Sample_Report.pdf` + 6 chương của đề; gửi cho nhóm ngay giai đoạn 1.
- Vẽ hình kiến trúc MLP tổng quát (1 tầng – B nơ-ron; C tầng) và sơ đồ pipeline train/val/test (draw.io), dùng cho báo cáo/slide/README (tương tự `static/` của repo tham khảo).
- `src/make_report_assets.py`: gom `summary.csv` của A/B/C → bảng LaTeX/Word & bảng siêu tham số; kiểm tra mọi số liệu trong báo cáo **truy vết được** về file log.
- **QA tái lập**: clone repo về thư mục khác, tạo venv mới, chạy theo README từ đầu; báo lỗi cho người phụ trách.
- Viết báo cáo: **Ch.1 Giới thiệu** (mục tiêu, bài toán, bảng phân công/% đóng góp), **4.1 Thiết lập & quy trình đánh giá công bằng**, **Ch.5 Thảo luận** (tổng hợp nhận xét từ TV1–3, trường hợp không như kỳ vọng, hạn chế, cải tiến), **Ch.6 Kết luận**, TLTK, phụ lục lệnh tái lập; xuất .docx/.pdf.
- Dựng slide .pptx 10–15 slide từ nội dung các thành viên, kịch bản thuyết trình, phân vai, tổ chức tập dượt + bộ câu hỏi Q&A dự kiến. Đóng gói file zip nộp.

## 6. Tiến độ & phụ thuộc

| Giai đoạn | Việc chính | Đầu ra / mốc |
|---|---|---|
| G1 – Khởi động | Đọc đề, chốt cấu trúc repo, interface hàm (`get_dataloaders`, `MLP`, `fit`), template báo cáo | TV1: data CSV + dataset.py; TV2: mlp.py + trainer.py chạy được baseline; TV4: template |
| G2 – Baseline | Chạy thử, chốt baseline (dựa trên val), hoàn thiện evaluate/plots | `configs/baseline.json` được **đóng băng** |
| G3 – Thực nghiệm | A (TV1), B (TV3), C (TV2) song song, ≥3 seeds | `results/exp*/summary.csv` + hình |
| G4 – Viết | Mỗi người viết phần mình + gửi 2–3 slide; TV4 ghép, viết Thảo luận/Kết luận | Bản nháp báo cáo + slide |
| G5 – Hoàn thiện | Review chéo (TV1↔TV3, TV2↔TV4), QA tái lập, tập thuyết trình, đóng gói zip | Nộp bài |

Phụ thuộc chính: TV3, TV1 (phần A) cần trainer của TV2 → TV2 ưu tiên xong `trainer.py` sớm nhất; mọi thay đổi baseline sau G2 phải chạy lại cả A/B/C.

## 7. Checklist nộp bài
- [ ] Code chạy lại từ đầu bằng lệnh trong README, không đường dẫn tuyệt đối, không TF/Keras
- [ ] `data/*.csv` hoặc script sinh dữ liệu với seed cố định
- [ ] Log CSV từng run, `summary.csv`, hình, `best.pt` của mô hình được chọn
- [ ] Test chỉ dùng đánh giá cuối; val dùng cho early stopping/chọn cấu hình
- [ ] Bảng + đồ thị A/B/C có mean ± std, nhãn trục, đơn vị, đánh số & caption
- [ ] Báo cáo .docx + .pdf (10–20 trang), slide .pptx (+pdf) 10–15 slide
- [ ] Bảng thành viên – nhiệm vụ – % đóng góp (tổng 100%)
- [ ] Zip `BTL01_Nhom_HoTen_MSSV.zip`
