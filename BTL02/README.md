# BTL02 – Nhận dạng cảm xúc khuôn mặt FER2013 bằng CNN (PyTorch)

Kế hoạch, phân công và quy chuẩn: [`PHAN_CONG_NHIEM_VU.md`](PHAN_CONG_NHIEM_VU.md). Bảng theo dõi: `tracking/` và Google Sheets (link ở mục 10 của file kế hoạch).

**Hiện trạng (2026-11-23):** repo có loader/augmentation và kiểm tra dữ liệu (`src/data/`), model (`src/models/`), trainer/evaluate/plots (`src/engine/`, `src/visual/`), bảng (`src/make_tables.py`), trực quan hóa bộ lọc (`src/visual/filters.py`) và pytest. Theo `tracking/tasks.csv`, T06 là `Verified`; T01 đang `In Progress`, T02/T03–T05/T07–T09 đang `Pending`, T20 đang `In Progress`. Mã T03/T04/T05 có thể chạy kiểm tra bằng fixture; cần FER2013 gốc để tạo `results/data_check.json` và EDA thật. Không commit file dữ liệu gốc.

## Cấu trúc thư mục

Cây mục tiêu theo mục 4 của kế hoạch. Các tệp mã và cấu hình sẽ xuất hiện khi hoàn thành nhiệm vụ tương ứng.

```text
BTL02/
├── README.md, requirements.txt, PHAN_CONG_NHIEM_VU.md
├── configs/                    baseline.json, E1_arch.json … E7_final.json
├── src/
│   ├── data/                   fer2013.py, loader.py, augment.py, eda.py
│   ├── models/                 cnn.py, transfer.py
│   ├── engine/                 trainer.py, evaluate.py
│   ├── visual/                 plots.py, filters.py
│   ├── utils/                  seed.py, io.py
│   ├── run_experiment.py, make_tables.py, visualize_filters.py
├── tests/
├── data/                       fer2013.csv (chỉ lưu cục bộ, không commit/nộp)
├── results/                    config, log, metrics, history, weights của từng run
├── figures/                    hình sinh từ dữ liệu run thật
├── notebooks/                  notebook nguồn và bằng chứng đã chạy
│   └── executed/               notebook đã chạy và lưu output
├── report/                     báo cáo, slide, nhật ký thực nghiệm
│   └── tables/                 bảng sinh từ kết quả run
└── tracking/                   phân công, kiểm thử, thực nghiệm, nhật ký git
```

TV2 đã khởi tạo các script tracking. Theo ma trận hiện tại, TV4 là R của T01; đây là vai trò nghiệm thu và quản trị nhiệm vụ, khác với tác giả ban đầu của script.

| Thư mục | Nội dung | Phụ trách chính | Nhiệm vụ |
|---|---|---|---|
| `src/data/` | đọc `fer2013.csv`, loader, augmentation, EDA | TV1 | T04, T05 |
| `src/models/` | CNN VGG-style, transfer (ResNet18/VGG16) | TV2 | T06 |
| `src/engine/` | trainer, evaluate, tự giữ/xóa weights | TV2 (trainer), TV3 (evaluate) | T07, T08 |
| `src/visual/` | đường cong, ma trận nhầm lẫn, hình bộ lọc | TV3 (plots), TV4 (filters) | T08, T20 |
| `src/utils/` | seed, io, log | TV2 | T07 |
| `configs/` | cấu hình từng nhóm thực nghiệm (`baseline.json`, E1–E7) | người phụ trách nhóm | T11–T18 |
| `tests/` | pytest: shape, nhãn, không rò rỉ test, luật giữ/xóa weights | người viết module | |
| `data/` | `fer2013.csv` (**không commit, không nộp**) | TV1 | T03 |
| `results/<nhóm>/<exp_id>/` | `config.json`, `run.log`, `history.csv`, `metrics.json`, `best.pt` (chỉ run hợp lệ) | tự sinh khi chạy | |
| `figures/` | hình + file CSV dữ liệu đã vẽ (sinh từ code) | tự sinh | |
| `notebooks/`, `notebooks/executed/` | notebook, bản đã chạy giữ output làm bằng chứng | TV3 | T10 |
| `report/` | báo cáo, slide, nhật ký thực nghiệm, `tables/` | TV4 (ghép), mỗi người một chương | T20–T26 |
| `tracking/` | phân công, thành viên, kiểm thử, thực nghiệm, nhật ký git; script sinh bảng | TV2 (script ban đầu); TV4 (R T01) | T01 |

## Phân công theo tệp và nhiệm vụ

| Tệp/thư mục bàn giao | Phụ trách | Nhiệm vụ chính |
|---|---|---|
| `README.md`, `requirements.txt`, khung thư mục | TV1 – Đoàn Trung Kiên | T02 |
| `data/fer2013.csv`, `results/data_check.json`, `src/data/{fer2013,loader,augment,eda}.py` | TV1 – Đoàn Trung Kiên | T03–T05, T15–T16, T21 |
| `src/models/{cnn,transfer}.py`, `src/engine/trainer.py`, `src/run_experiment.py`, `src/utils/` | TV2 – Lê Phan Nguyên Hải | T06–T07, T11–T13, T17, T22 |
| `src/engine/evaluate.py`, `src/visual/plots.py`, `src/make_tables.py`, `notebooks/` | TV3 – Võ Hoàng Bảo | T08–T10, T14, T18–T19, T23 |
| `src/visual/filters.py`, `src/visualize_filters.py`, `report/` | TV4 – Huỳnh Viễn Thông | T20, T24–T27 |
| `tracking/*.csv`, Google Sheets và nghiệm thu T01 | TV4 – Huỳnh Viễn Thông | T01 (R theo ma trận) |
| `tracking/*.py`, workflow nhật ký hoạt động | TV2 – Lê Phan Nguyên Hải | Script tracking được TV2 khởi tạo; TV2 là C1 ở T01 |
| Đóng gói repo và nộp zip | TV2 – Lê Phan Nguyên Hải | T28 |
| `configs/`, `results/`, `figures/`, `tests/` | Người phụ trách từng nhiệm vụ; ghi trong file và commit | T04–T28 theo ma trận phân công |

Ma trận R/C/T đầy đủ và nội dung C cụ thể nằm trong mục 5 của [`PHAN_CONG_NHIEM_VU.md`](PHAN_CONG_NHIEM_VU.md) và `tracking/tasks.csv`.

## Quy ước bắt buộc

- Thông điệp commit viết tiếng Việt và có mã nhiệm vụ trong ngoặc, ví dụ `feat(T04): thêm loader FER2013`, `fix(T07): sửa lỗi ghi history.csv`, `docs(T02): cập nhật README`. Thiếu mã nhiệm vụ được ghi là “không có mã nhiệm vụ” và tính là thiếu.
- Mỗi commit chỉ chứa một việc nhỏ của một nhiệm vụ; commit sớm, thường xuyên.
- Mỗi file code mở đầu bằng docstring ghi người phụ trách theo mẫu `Phụ trách: TVx – Họ tên`. Ví dụ Python: `"""Phụ trách: TV1 – Đoàn Trung Kiên. Đọc và tách dữ liệu FER2013."""`.
- Không commit `data/fer2013.csv`, file `*.pt`/`*.pth`, hoặc thư mục `.venv/`.
- Trước mỗi lần push chạy `git pull --rebase origin main` để nhận commit nhật ký bot; không tự sửa `tracking/activity_log.csv`.
- Không dùng TensorFlow/Keras. Không đường dẫn tuyệt đối.
- Mọi hình và số trong báo cáo sinh từ code; không số minh họa.
- `fit()` không nhận PrivateTest; PrivateTest chỉ chạy 1 lần ở `--eval-test`.

## Cài đặt

Các lệnh trong README giả định terminal đang ở thư mục `BTL02/`.

Môi trường tham chiếu của nhóm: Python 3.12, Windows 11, PyTorch 2.6.0, torchvision 0.21.0 và CUDA 12.4. Tạo môi trường ảo trong thư mục `BTL02/` bằng PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, chạy lệnh trong Command Prompt (`.venv\Scripts\activate.bat`) hoặc gọi trực tiếp `.venv\Scripts\python.exe -m pip ...`. Danh sách thư viện và phiên bản giữ tại [`requirements.txt`](requirements.txt). Chưa có kết quả cài đặt độc lập từ TV2/TV4; họ cần báo lỗi nếu lệnh không chạy trên máy riêng.

Kiểm tra cài đặt (chạy từ thư mục `BTL02/`, venv đã kích hoạt):

```
python -m pytest -q
```

## Dữ liệu

Tải file `fer2013.csv` từ một trong các nguồn chính thức dưới đây, giải nén nếu tải gói Kaggle, rồi đặt tại `data/fer2013.csv`:

- [Kaggle – Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) (cần tài khoản và chấp nhận quy định cuộc thi).
- [Google Drive do giảng viên cung cấp](https://drive.google.com/open?id=0BxINLo5jshCRbXpRcnl2SjJkdlk).

Không dùng mirror đã gộp `PublicTest` và `PrivateTest`. File khoảng 290 MB gồm 35.887 dòng và các cột `emotion`, `pixels`, `Usage`; các split dự kiến là `Training` 28.709, `PublicTest` 3.589 (validation), `PrivateTest` 3.589 (chỉ đánh giá một lần ở mô hình cuối). Sau khi đặt file, chạy kiểm tra cột, split và SHA-256:

```powershell
python -m src.data.check_fer2013 --csv data/fer2013.csv --output results/data_check.json
```

Nếu kiểm tra đạt, có thể sinh EDA và các CSV dữ liệu cho hình:

```powershell
python -m src.data.eda --csv data/fer2013.csv --output-dir figures
```

Hai lệnh cần file FER2013 gốc tại chỗ; chúng không tải dữ liệu. Không commit hoặc đưa CSV gốc vào gói nộp.

## Lệnh chạy

`src/` đã có các module triển khai từng phần, nhưng `configs/` và FER2013 gốc hiện chưa có; các lệnh huấn luyện dưới đây chưa chạy được cho đến khi cấu hình và dữ liệu sẵn sàng. Không chạy hàng loạt E1–E6 trước khi T04, T06, T07, T08 và T11 được Verified. Lệnh `--eval-test` chỉ chạy một lần trên mô hình cuối, sau khi chọn cấu hình bằng validation:

```powershell
# Baseline và các nhóm E1–E7 (T07 trở đi)
python src/run_experiment.py --config configs/baseline.json
python src/run_experiment.py --config configs/E1_arch.json
python src/run_experiment.py --config configs/E3_optim.json --only adamw_1e-3
python src/run_experiment.py --config configs/E7_final.json --eval-test
python src/make_tables.py
python src/visualize_filters.py
```

Chạy các kiểm tra bằng `python -m pytest -q`. Test dữ liệu dùng fixture tổng hợp nằm trong thư mục tạm của pytest; fixture không dùng cho báo cáo, slide hoặc kết quả thực nghiệm.

PrivateTest chỉ được đọc bởi lệnh đánh giá mô hình cuối đúng một lần; các lệnh khác phải giới hạn dữ liệu ở Training/PublicTest.

Lệnh sinh bảng theo dõi và nhật ký (đã có):

```
python tracking/build_tracking.py     # sinh tasks.csv và tracking.xlsx (không ghi đè members.csv)
python tracking/git_activity.py       # sinh tracking/activity_log.csv từ git (workflow cũng tự chạy sau mỗi push)
python tracking/build_slides.py       # sinh report/Slide_TienDo_BTL02.pptx
```
