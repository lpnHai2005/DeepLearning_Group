# BTL01 – Nhóm 10: Xấp xỉ hàm Gaussian 2D bằng MLP (PyTorch)

Môn Deep Learning, SGU, giảng viên TS. Đỗ Như Tài. Dùng MLP (PyTorch) để xấp xỉ mật độ Gaussian 2D
với μ = [0, 0], C = [[0.3, −0.5], [−0.5, 2]] trên miền x₁ ∈ [−2, 2], x₂ ∈ [−4, 4].
**Không dùng TensorFlow, Keras, AutoML.**

Ba nhóm thí nghiệm: **A** thay đổi số mẫu huấn luyện (200…2000), **B** thay đổi số nơ-ron của một tầng ẩn (1…10),
**C** thay đổi số tầng ẩn (1…4). Chạy ≥ 3 seed, báo cáo mean ± std.

## Trạng thái

Khung repo đã dựng sẵn; các thành viên hoàn thiện phần được giao theo `CHI_TIET_NHIEM_VU.md`.

**TV1:** Đã sinh dữ liệu, hoàn thiện loader/EDA, chọn ứng viên baseline bằng validation và chạy đủ 30 run A. Ứng viên hiện tại là sigmoid/lr 0.01; chờ TV2/TV3 review và TV4 QA/bàn giao.

**TV2:** Đã chạy đủ Thực nghiệm C (12 run) và đối chứng cùng số tham số (9 run) với baseline đã chốt; có bản nháp Ch.3, mục 4.4 và ghi chú slide; demo `configs/demo.json` chạy khoảng 10 giây.

**TV3:** Đã hoàn thiện `src/engine/evaluate.py`, `src/visual/plots.py`, tổng hợp/vẽ biểu đồ Thực nghiệm B (`src/summarize_expB.py`), chọn mô hình cuối theo validation và đánh giá tập test + lưới 100×100 (`src/make_report_assets.py`, `results/final/`), cùng bản thảo mục 4.3, mục 4.5 và ghi chú slide TV3.

| Phần | Người | File |
|---|---|---|
| Dữ liệu, repo, Thực nghiệm A | TV1 | `src/data/*`, `src/generate_data.py`, `src/data/eda.py`, `src/data/summarize_expA.py`, `configs/expA.json` |
| Mô hình, huấn luyện, Thực nghiệm C | TV2 | `src/model/mlp.py`, `src/engine/trainer.py`, `src/run_experiment.py`, `src/utils/*`, `configs/expC.json` |
| Đánh giá, vẽ hình, Thực nghiệm B | TV3 | `src/engine/evaluate.py`, `src/visual/plots.py`, `src/summarize_expB.py`, `src/make_report_assets.py`, `configs/expB.json` |
| Báo cáo, slide, QA, đóng gói | TV4 | `report/`, `src/make_report_assets.py` |

## Cài đặt

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # kiểm tra khung repo
```

Trên Windows, đặt `PYTHONUTF8=1` nếu `print` tiếng Việt báo lỗi mã hóa.

## Chạy

```
python src/env_info.py                                       # ghi môi trường chạy -> results/env.txt
python src/generate_data.py                                  # sinh data/*.csv
python src/data/eda.py                                       # TV1: thống kê và hình dữ liệu
# --- chọn baseline (chỉ xem val) ---
python src/run_experiment.py --config configs/grid_baseline.json      # T2.5: lưới lr × activation
python src/summarize_grid_baseline.py                        # T2.5: hình bằng chứng chọn lr, activation
python src/run_experiment.py --config configs/grid_optimizer.json     # T2.8: 6 optimizer × 7 lr × 3 seed (~2 giờ tuần tự)
python src/summarize_optimizer.py                            # TV2: chọn lr theo val, vẽ figures/optimizer_loss.png
python src/run_experiment.py --config configs/grid_batch.json         # minh chứng batch size 16/32/64/128
python src/run_experiment.py --config configs/baseline.json
# --- ba nhóm thực nghiệm + đối chứng ---
python src/run_experiment.py --config configs/expA.json
python src/run_experiment.py --config configs/expA_steps.json         # đối chứng A: cùng ngân sách ~64k bước cập nhật
python src/run_experiment.py --config configs/expB.json
python src/run_experiment.py --config configs/expB_wide.json          # B = 16/32/64 ở A=1000
python src/run_experiment.py --config configs/expB_A200.json          # B ở A=200 (kiểm tra quá khớp)
python src/run_experiment.py --config configs/expC.json
python src/run_experiment.py --config configs/expC_ctrl.json         # đối chứng C: 1 tầng, cùng số tham số
# --- hậu kiểm test (sau khi đã chạy xong mọi run; chạy lại run_experiment thì chạy lại bước này) ---
python src/posthoc_test.py                                   # điền test_* từng seed vào summary.csv/metrics.json + posthoc_test.log
# --- tổng hợp, hình, bảng ---
python src/analyze_stopping.py                               # minh chứng patience/max_epochs -> results/stopping/
python src/data/summarize_expA.py                            # TV1: tổng hợp/đồ thị A (+ expA_steps)
python src/summarize_expB.py                                 # TV3: tổng hợp/đồ thị B + bề mặt theo B (+ A=200, B rộng)
python src/summarize_expC.py                                 # TV2: tổng hợp/đồ thị C + chuẩn gradient
python src/make_report_assets.py                             # chọn mô hình cuối theo val, test 1 lần, hình final_*
python src/make_tables.py                                    # bảng báo cáo tự động -> report/tables/*.md
python src/run_experiment.py --config configs/demo.json      # demo trước lớp (~10 s)
```

Sơ đồ mạng vẽ từ model thật: `python src/visual/draw_mlp.py` (lưu vào `figures/`).

## Cấu trúc thư mục

```
configs/      baseline.json, expA.json, expB.json, expC.json (ứng viên sigmoid/lr 0.01 chọn bằng validation; chờ nhóm duyệt)
src/data/     gaussian.py, dataset.py
src/model/    mlp.py
src/engine/   trainer.py, evaluate.py
src/visual/   plots.py, draw_mlp.py
src/utils/    seed.py, io.py, timer.py
src/          generate_data.py, run_experiment.py, posthoc_test.py, analyze_stopping.py,
              summarize_exp*.py, make_report_assets.py, make_tables.py, env_info.py
tests/        test_skeleton.py, test_gaussian.py, test_dataset.py, test_evaluate.py, test_trainer.py, test_backprop.py
notebooks/    EDA, demo
data/         train_pool.csv, val.csv, test.csv, meta.json
results/      <exp>/<giá trị>/seed_<k>/{config.json, history.csv, metrics.json, best.pt}, summary.csv,
              run.log (log train), posthoc_test.log (log test hậu kiểm), aggregate.csv; stopping/; env.txt
figures/      hình dùng cho báo cáo/slide
report/       báo cáo (.docx/.pdf), slide (.pptx/.pdf); tables/ (bảng sinh tự động)
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
