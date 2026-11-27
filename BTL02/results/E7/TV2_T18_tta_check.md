# Kiểm độc lập TTA — TV2 (C2 của T18)

- Người kiểm: TV2 – Lê Phan Nguyên Hải, 2026-11-27, RTX 4060 Laptop, tại commit `456a385`.
- Đối tượng: phần TTA (`flip`, `ten_crop`) trong `src/eval_e7.py` / `src/engine/evaluate.py` của TV3 trên 3 checkpoint `E1-014_s0/_s1/_s2` (large_bn) và các file `results/E7/val_predictions_ens_{none,flip,ten_crop}.npz`, `report/tables/E7_tta_ensemble.md`.
- Cách kiểm: script độc lập (không commit, giống quy ước KT-003 của T08), **không** dùng `evaluate.py`, `loader.py` hay `eval_e7.py`; chỉ dùng `src/models/cnn.build_model` để dựng kiến trúc.
  - Tự đọc `data/fer2013.csv` bằng `csv` thuần, lấy 3589 ảnh PublicTest; chuẩn hóa bằng mean/std chỉ tính từ Training (`0.507743` / `0.255009`). Không chạm PrivateTest.
  - Tự cài TTA: `flip` = trung bình softmax của ảnh gốc và ảnh lật ngang; `ten_crop` = `torchvision.transforms.functional.ten_crop(44)` (4 góc + tâm của ảnh gốc và ảnh lật) rồi phóng bilinear về 48, trung bình softmax 10 view. Cách lấy 10 view khác `evaluate._ten_crops` (lật trước rồi cắt, so với cắt rồi lật) nhưng cho cùng tập 10 ảnh.
  - Ensemble = trung bình xác suất 3 seed; tự tính accuracy, macro-F1, NLL.
- Số liệu đầy đủ: `results/E7/TV2_T18_tta_check.json`.

## Kết quả

| Kiểm tra | Kết quả |
|---|---|
| SHA-256 của 3 `best.pt` so với `metrics.json` | khớp cả 3 |
| `none` từng seed so với `metrics.json` (trainer) | s0, s2 khớp tuyệt đối (0.7086, 0.7047); s1 lệch 1 ảnh (0.7044 so với 0.7047) |
| `flip` từng seed so với bảng TV3 | s0 0.7166, s1 0.7122, s2 0.7158: khớp cả 3 |
| `ten_crop` từng seed so với bảng TV3 | s0 0.6885 (TV3 0.6882, lệch 1 ảnh), s1 0.6826, s2 0.6801: khớp trong 1 ảnh |
| Ensemble `none` | 0.7236 / F1 0.7134 / NLL 0.9780 = TV3; nhãn trùng 100%, max\|Δp\| = 0.0021 |
| Ensemble `flip` | 0.7253 (2603/3589) / F1 0.7109 / NLL 0.9402; TV3 0.7250 (2602/3589); nhãn trùng 3588/3589, max\|Δp\| = 0.0011 |
| Ensemble `ten_crop` | 0.6991 / F1 0.6839 / NLL 0.9716 = TV3; nhãn trùng 100%, max\|Δp\| = 0.00075 |
| Trung bình đơn lẻ (mean ± std) | none 0.7059 ± 0.0023, flip 0.7149 ± 0.0024, ten_crop 0.6838 ± 0.0043: khớp TV3 trong 0.0003 |
| `y_true` trong 3 file `.npz` | trùng nhãn PublicTest tự đọc |
| `flip` FP32 so với AMP | nhãn trùng 100% (s0), 3588/3589 (s1), 3587/3589 (s2); accuracy bằng nhau |
| Luật chọn tổ hợp cho T19 (acc → F1 → NLL) | tổ hợp cao nhất vẫn là **ensemble + flip**, đúng như `e7_val_summary.json` |

Các chênh lệch 1 ảnh (≤ 0.0003) đến từ sai số số học của AMP (FP16) và kích thước batch khác nhau giữa hai cách chạy, không đổi thứ hạng tổ hợp nào.

## Nhận xét

- **Pass** phần TTA: tái hiện độc lập được mọi số của `E7_tta_ensemble.md` trong sai số 1 ảnh, và chọn đúng `ensemble + flip` theo luật tiền đăng ký.
- `flip` giúp đều cả 3 seed (+0.008 đến +0.011 so với `none`). `ten_crop` làm **giảm** 0.020–0.025 ở cả 3 seed dù NLL thấp hơn `none`: phóng crop 44 → 48 là phép zoom 1.09× kèm cắt mất mép khuôn mặt, nên không phải lỗi cài đặt (bản tự cài cho cùng kết quả).
- Lưu ý khi viết báo cáo: ensemble `flip` (0.7250) chỉ hơn ensemble `none` (0.7236) 5 ảnh trên 3589 (bản tự chạy: 6 ảnh); McNemar chính xác giữa hai tổ hợp: flip đúng riêng 54 ảnh, none đúng riêng 48 ảnh, p = 0.62, tức **không khác biệt có ý nghĩa**. Việc chọn `flip` vẫn hợp lệ vì luật đã đăng ký trước, nhưng không nên viết là "flip tốt hơn rõ rệt"; phần tăng chính đến từ ensemble (không TTA: +0.018 so với trung bình đơn lẻ; có flip: +0.010).
