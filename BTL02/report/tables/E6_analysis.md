# E6 – Transfer learning: bảng tổng hợp và giải thích VGG16 ở 48 px

_Sinh tự động bằng `python src/e6_analysis.py` từ `tracking/experiments.csv`, `results/E6/*/history.csv`, `metrics.json` và từ việc chạy thật VGG16 torchvision. Không sửa tay. Mỗi biến thể 1 seed (seed 0)._

Baseline (T11, 3 seed): val acc 0.6815 ± 0.0075.

## Bảng E6.1 – Toàn bộ biến thể

| Cấu hình | Run | Val acc | Δ baseline | Val macro-F1 | Best val loss | Best epoch / chạy / tối đa | Train acc cuối | Train − Val | Tham số | Weights (MB) | s/epoch (GPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VGG16 pretrained, fine-tune, 48 px | E6-004_s0 (TV1) | 0.6938 | +0.0123 (cao hơn) | 0.6812 | 0.9373 | 38 / 53 / 100 | 0.9509 | +0.2571 | 134,289,223 | 512.3 | 60.5 (RTX 3050) |
| VGG16 pretrained, fine-tune, 96 px (30 epoch) | E6-001_s0 (TV2) | 0.6977 | +0.0162 (cao hơn) | 0.6818 | 0.9220 | 30 / 30 / 30 | 0.8853 | +0.1876 | 134,289,223 | 512.3 | 71.3 (RTX 4060) |
| ResNet18 từ đầu, 96 px | E6-002_s0 (TV1) | 0.6771 | -0.0045 (< std) | 0.6672 | 1.0006 | 60 / 75 / 100 | 0.9208 | +0.2437 | 11,180,103 | 42.7 | 55.5 (RTX 3050) |
| ResNet18 pretrained, fine-tune, 96 px | E6-007_s0 (TV4) | 0.6874 | +0.0059 (< std) | 0.6698 | 0.9718 | 42 / 57 / 100 | 0.9857 | +0.2983 | 11,180,103 | 42.7 | 79.4 (RTX 3060) |
| ResNet18 pretrained, đóng băng backbone, 96 px | E6-006_s0 (TV4) | 0.4570 | -0.2246 (thấp hơn) | 0.3683 | 1.4507 | 50 / 65 / 100 | 0.4436 | -0.0133 | 11,180,103 | 42.7 | 75.0 (RTX 3060) |

## Bảng E6.2 – ResNet18 96 px: từ đầu so với pretrained

| Cấu hình | Val acc | Val macro-F1 | Best val loss | Val acc epoch 1 | Epoch đầu tiên val ≥ 0.60 | Epoch đầu tiên val ≥ 0.65 | Best epoch | Train − Val |
|---|---|---|---|---|---|---|---|---|---|
| ResNet18 từ đầu, 96 px | 0.6771 | 0.6672 | 1.0006 | 0.2862 | 17 | 45 | 60 | +0.2437 |
| ResNet18 pretrained, fine-tune, 96 px | 0.6874 | 0.6698 | 0.9718 | 0.5612 | 4 | 14 | 42 | +0.2983 |
| ResNet18 pretrained, đóng băng backbone, 96 px | 0.4570 | 0.3683 | 1.4507 | 0.3806 | không đạt | không đạt | 50 | -0.0133 |

## Bảng E6.3 – Kích thước bản đồ đặc trưng VGG16 (đo bằng forward thật)

| Đầu vào | Sau pool1 → pool5 | Bản đồ conv5 vào pool5 | Vùng tiếp nhận conv5_3 (px) | Ô conv5 pool5 dùng tới | Sau AdaptiveAvgPool | Giá trị khác nhau / kênh trong 7×7 | Ô pool5 (thông tin thật) | FC1: đầu vào, tham số | FC1: tham số hiệu dụng |
|---|---|---|---|---|---|---|---|---|---|
| 48×48×3 | 64×24×24 → 128×12×12 → 256×6×6 → 512×3×3 → 512×1×1 | 3×3 | 196 | 4/9 | 512×7×7 | 1 | 1 | 25,088, 102,764,544 | 2,101,248 |
| 96×96×3 | 64×48×48 → 128×24×24 → 256×12×12 → 512×6×6 → 512×3×3 | 6×6 | 196 | 36/36 | 512×7×7 | 25 | 9 | 25,088, 102,764,544 | 18,878,464 |

_AdaptiveAvgPool chỉ lặp/trung bình các ô của pool5 nên là phép tuyến tính; FC1 tương đương Linear(512 × số ô pool5, 4096), tham số hiệu dụng = 512 × số ô pool5 × 4096 + 4096. Ở 48 px cả 49 ô của bản đồ 7×7 là bản sao của một vector 512 chiều, nên FC1 chỉ thấy 512 số thật dù có 25,088 đầu vào._
