# BTL02 – Nhận dạng cảm xúc khuôn mặt FER2013 bằng CNN (PyTorch)

Kế hoạch, phân công và quy chuẩn: [`PHAN_CONG_NHIEM_VU.md`](PHAN_CONG_NHIEM_VU.md). Bảng theo dõi: `tracking/` và Google Sheets (link ở mục 10 của file kế hoạch).

**Trạng thái (2026-11-20):** mới có khung thư mục và hệ thống theo dõi. **Chưa có code, dữ liệu, kết quả.** Các mục "chưa có" dưới đây sẽ được bổ sung khi nhiệm vụ tương ứng xong. Bản khung này do TV2 dựng; TV1 hoàn thiện README ở nhiệm vụ T02.

## Cấu trúc thư mục

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
| `tracking/` | phân công, thành viên, kiểm thử, thực nghiệm, nhật ký git; script sinh bảng | TV2 | T01 |

## Quy ước bắt buộc

- Thông điệp commit mang **mã nhiệm vụ**: `feat(T07): ...`, `fix(T04): ...`, `docs(T01): ...`. Nhật ký hoạt động tự sinh từ đây.
- Mỗi file code mở đầu bằng docstring có dòng `Phụ trách: TVx – Họ tên`.
- Không dùng TensorFlow/Keras. Không đường dẫn tuyệt đối.
- Mọi hình và số trong báo cáo sinh từ code; không số minh họa.
- `fit()` không nhận PrivateTest; PrivateTest chỉ chạy 1 lần ở `--eval-test`.

## Cài đặt

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Kiểm tra cài đặt (chạy từ thư mục `BTL02/`, venv đã kích hoạt):

```
python -m pytest -q
```

## Dữ liệu

Tải `fer2013.csv` (Kaggle hoặc Drive, link ở mục 2.1 của file kế hoạch), đặt vào `data/fer2013.csv`. Lệnh kiểm toàn vẹn: **chưa có** (T03).

## Lệnh chạy

Huấn luyện, đánh giá, sinh bảng, hình: **chưa có** (T07, T08, T09).

Lệnh sinh bảng theo dõi và nhật ký (đã có):

```
python tracking/build_tracking.py     # sinh tasks.csv và tracking.xlsx (không ghi đè members.csv)
python tracking/git_activity.py       # sinh tracking/activity_log.csv từ git (workflow cũng tự chạy sau mỗi push)
python tracking/build_slides.py       # sinh report/Slide_TienDo_BTL02.pptx
```
