# Nhận xét TV1 — T17 / E6

## Kết quả phần TV1

| Run | Cấu hình | Trạng thái | Epoch chạy | Epoch tốt nhất | Best validation accuracy | Best validation macro-F1 | Tham số | Weights |
|---|---|---|---:|---:|---:|---:|---:|---|
| `E6-002_s0` | ResNet18 96 px, scratch | Completed | 75 | 60 | 0.6771 | 0.6672 | 11,180,103 | KEEP |
| `E6-004_s0` | VGG16 48 px, fine-tune pretrained | Completed | 53 | 38 | 0.6938 | 0.6812 | 134,289,223 | KEEP |

Hai lần thử VGG trước đó được ghi riêng, không tính là kết quả Completed: `E6-005_s0` không tạo được thư mục cache mặc định do `PermissionError`; `E6-003_s0` bị sandbox chặn kết nối (`URLError`). `E6-004_s0` thành công sau khi dùng cache tạm có quyền ghi và cho phép tải checkpoint.

## Nguồn pretrained và kích thước feature map

VGG16 dùng `torchvision.models.VGG16_Weights.IMAGENET1K_V1`, tải checkpoint chính thức từ [download.pytorch.org](https://download.pytorch.org/models/vgg16-397923af.pth). Trạng thái pretrained và cấu hình run được lưu trong `results/E6/E6-004_s0/config.json`.

`results/transfer_summary.csv` ghi các kích thước khi đầu vào là 48 px: năm MaxPool lần lượt đưa cạnh ảnh `48 → 24 → 12 → 6 → 3 → 1`; `features.30_pool` có shape `512x1x1`, sau đó `AdaptiveAvgPool` đưa lên `512x7x7` trước classifier 7 lớp. Tổng mô hình có 134,289,223 tham số.

Số liệu validation chỉ lấy từ `metrics.json` của các run Completed; không chạy PrivateTest. Phần C1 của TV1 hoàn tất; T17 vẫn chờ TV2 hoàn thiện phần R và TV3 kiểm thử độc lập trước khi được đặt Verified.
