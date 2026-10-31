# BTL01 – Nhóm 10: Xấp xỉ hàm Gaussian 2D bằng MLP (PyTorch)

Môn Deep Learning, SGU, giảng viên TS. Đỗ Như Tài. Dùng MLP (PyTorch) để xấp xỉ mật độ Gaussian 2D
với μ = [0, 0], C = [[0.3, −0.5], [−0.5, 2]] trên miền x₁ ∈ [−2, 2], x₂ ∈ [−4, 4].
**Không dùng TensorFlow, Keras, AutoML.**

Ba nhóm thí nghiệm: **A** thay đổi số mẫu huấn luyện (200…2000), **B** thay đổi số nơ-ron của một tầng ẩn (1…10),
**C** thay đổi số tầng ẩn (1…4). Chạy ≥ 3 seed, báo cáo mean ± std.

## Trạng thái

Khung repo đã dựng sẵn. Các hàm trong `src/` hiện là **stub** (`raise NotImplementedError("TVx - Tx.x")`),
mỗi thành viên điền phần của mình theo `CHI_TIET_NHIEM_VU.md`.

| Phần | Người | File |
|---|---|---|
| Dữ liệu, repo, Thực nghiệm A | TV1 | `src/data/*`, `src/generate_data.py`, `configs/expA.json` |
| Mô hình, huấn luyện, Thực nghiệm C | TV2 | `src/model/mlp.py`, `src/engine/trainer.py`, `src/run_experiment.py`, `src/utils/*`, `configs/expC.json` |
| Đánh giá, vẽ hình, Thực nghiệm B | TV3 | `src/engine/evaluate.py`, `src/visual/plots.py`, `configs/expB.json` |
| Báo cáo, slide, QA, đóng gói | TV4 | `report/`, `src/make_report_assets.py` |

## Cài đặt

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # kiểm tra khung repo
```

Trên Windows, đặt `PYTHONUTF8=1` nếu `print` tiếng Việt báo lỗi mã hóa.

## Chạy (khi các stub đã được điền)

```
python src/generate_data.py                                  # sinh data/*.csv
python src/run_experiment.py --config configs/grid_baseline.json      # T2.5: lưới chốt baseline (chỉ xem val)
python src/run_experiment.py --config configs/baseline.json
python src/run_experiment.py --config configs/expA.json
python src/run_experiment.py --config configs/expB.json
python src/run_experiment.py --config configs/expC.json
python src/run_experiment.py --config configs/expC_ctrl.json         # đối chứng C: 1 tầng, cùng số tham số
python src/make_report_assets.py                             # gom bảng/hình cho báo cáo
```

Sơ đồ mạng vẽ từ model thật: `python src/visual/draw_mlp.py` (lưu vào `figures/`).

## Cấu trúc thư mục

```
configs/      baseline.json, expA.json, expB.json, expC.json (giá trị baseline là PLACEHOLDER, chốt ở G2)
src/data/     gaussian.py, dataset.py
src/model/    mlp.py
src/engine/   trainer.py, evaluate.py
src/visual/   plots.py, draw_mlp.py
src/utils/    seed.py, io.py, timer.py
src/          generate_data.py, run_experiment.py, make_report_assets.py
tests/        test_skeleton.py (+ test_gaussian.py, test_backprop.py khi làm xong)
notebooks/    EDA, demo
data/         train_pool.csv, val.csv, test.csv, meta.json
results/      <exp>/<giá trị>/seed_<k>/{config.json, history.csv, metrics.json, best.pt}, summary.csv
figures/      hình dùng cho báo cáo/slide
report/       báo cáo (.docx/.pdf), slide (.pptx/.pdf)
```

## Quy tắc chung

- `fit()` không nhận dữ liệu test. Test chỉ được đánh giá ở bước `--eval-test` sau khi đã chọn mô hình theo val.
- x scale theo biên miền (x₁/2, x₂/4), y scale theo hằng y_max = 1/(2π√det C) ≈ 0.269 (hằng số của bài toán, không rò rỉ).
  Mọi chỉ số báo cáo trên thang y gốc.
- Mỗi lần chạy lưu config, history, metrics và `best.pt`; tổng hợp vào `results/<exp>/summary.csv`.
- Đường dẫn tính từ vị trí file, không viết đường dẫn tuyệt đối.
- Khóa bắt đầu bằng `_` trong file cấu hình (ví dụ `_note`) được bỏ qua.
- Git: nhánh theo người (`feat/data`, `feat/train`, `feat/eval`, `docs/report`), commit kiểu `feat:` / `fix:`, review chéo trước khi gộp.

Chi tiết nhiệm vụ: `PHAN_CONG_NHIEM_VU.md`, `CHI_TIET_NHIEM_VU.md`.
