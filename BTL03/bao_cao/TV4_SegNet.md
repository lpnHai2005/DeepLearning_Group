# Phần báo cáo của TV4 – Huỳnh Viễn Thông: 1 notebook (`5. Others/1_segnet`)

Code: `BTL03/pytorch/5. Others/1_segnet.ipynb` (giữ đúng cấu trúc thư mục giống `examples/`). Môi trường chạy: Python 3.11.4, PyTorch 2.6.0 + CUDA 12.4, `torchvision` 0.21.0 + CUDA 12.4, GPU NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM). Cố định `SEED = 0`, bật `torch.backends.cudnn.deterministic = True` và tắt `benchmark` thống nhất với 5 notebook khuôn của TV2.

Các chuyển đổi chung (chọn `device`, `Conv2d`/`MaxPool2d`, đổi ảnh channels-last → channels-first, `init_like_keras`, vòng lặp `summary`/`fit`/`evaluate`/`predict` tự chứa trong notebook) kế thừa từ mục 1 báo cáo của TV2 (`TV2_5_notebook_khuon.md`).

## 1. Thay đổi riêng của notebook `5. Others/1_segnet`

Đây là bài toán **phân đoạn ngữ nghĩa ảnh chiều sâu bàn tay (semantic segmentation)** kích thước `96×96` thành **7 lớp điểm ảnh** (`0`: lòng bàn tay — palm, `1..5`: 5 ngón tay, `6`: nền — background) dựa trên kiến trúc Encoder–Decoder đối xứng kiểu SegNet (VGG-16 backbone):

| Thành phần Keras gốc | Bản PyTorch | Ghi chú |
|---|---|---|
| Dữ liệu `data_x` dạng `(N, 96, 96, 1)` (`float64`), `data_y` one-hot trải phẳng `(N, 9216, 7)` (`uint8`) | `data_x` channels-first `(N, 1, 96, 96)` (`float32`), `data_y` chỉ số lớp nguyên `(N, 96, 96)` (`int64`, giá trị `0..6`) | Đọc ảnh nhãn an toàn qua `Path(root_path).parent / 'labels' / name.replace('depth', 'labels')`. Các pixel ngoài 7 mức xám chuẩn (`0.03%` tập train, `0.27%` tập test) được gán `-100` để `ignore_index=-100` bỏ qua (tương đương hàng one-hot toàn `0` trong Keras) |
| Khối Encoder: 5 khối (`64×2`, `128×2`, `256×3`, `512×3`, `512×3`), mỗi lớp `Conv2D(f, (3,3), padding='same')` + `BatchNormalization()` + `Activation('relu')` + `MaxPooling2D()` | `self.encoder = nn.Sequential(OrderedDict(...))` gồm `nn.Conv2d(in_c, out_c, 3, padding=1, bias=True)` + `nn.BatchNorm2d(out_c)` + `nn.ReLU(inplace=True)` + `nn.MaxPool2d(2)` | Giữ `bias=True` theo mặc định Keras để bảo toàn 100% cấu trúc và số tham số (`96 → 48 → 24 → 12 → 6 → 3`) |
| Khối Decoder: 5 tầng `UpSampling2D()` xen kẽ các lớp `Conv2D + BN + ReLU`, cuối cùng là `Conv2D(7, (1,1), padding='valid') + BatchNormalization()` | `self.decoder = nn.Sequential(OrderedDict(...))` thay `UpSampling2D()` bằng `nn.Upsample(scale_factor=2, mode='nearest')`, lớp cuối `nn.Conv2d(64, 7, 1, padding=0)` + `nn.BatchNorm2d(7)` | Khôi phục kích thước không gian `3 → 6 → 12 → 24 → 48 → 96`, không có ReLU sau `BatchNorm2d(7)` cuối như bản gốc |
| `Reshape((7, 9216))` → `Permute((2, 1))` → `Activation('softmax')` + `loss='categorical_crossentropy'` | Bỏ `Reshape`/`Permute`/`Softmax`; mô hình trả về logits `(N, 7, 96, 96)` đưa thẳng vào `nn.CrossEntropyLoss(ignore_index=-100)` | `CrossEntropyLoss` của PyTorch hỗ trợ trực tiếp đầu ra phân đoạn 4 chiều `(N, C, H, W)` và đã tích hợp `log_softmax` |
| `Adam(lr=0.0001)`, `batch_size=8`, `epochs=15`, `metrics=['accuracy']` | `torch.optim.Adam(model.parameters(), lr=0.0001)`, `batch_size=8`, `epochs=15`, tự tính **pixel accuracy** trên toàn bộ điểm ảnh hợp lệ (`y != -100`) | Hiển thị thêm biểu đồ lịch sử `Loss` / `Pixel Accuracy` theo 15 epoch |
| `class_to_rgb`: `np.ones(..., dtype=np.float)` và hiển thị 2 ảnh (Depth, Prediction) | Sửa thành `dtype=np.float32` (tương thích NumPy 2.x, tránh lỗi bỏ alias `np.float`) và hiển thị đối chiếu 3 ảnh: **Depth Input – Ground Truth – SegNet Prediction** | Giúp kiểm tra trực quan độ khớp từng vùng lòng bàn tay và 5 ngón tay |

Về số tham số của `SegNet`: tổng tham số bản PyTorch trùng khớp **tuyệt đối từng tham số** với `autoencoder.summary()` bản Keras gốc — **29,442,325** tham số huấn luyện (`trainable`) + **15,886** tham số thống kê chạy của các lớp `BatchNorm2d` (`running_mean` & `running_var`) = **29,458,211** tham số tổng.

## 2. Kết quả so với bản Keras gốc

Đánh giá trên tập test (`1,050` ảnh chiều sâu `96×96`, tương đương ~`9,65` triệu điểm ảnh hợp lệ; tập train `2,450` ảnh).

| Notebook | Cấu hình chạy | Kết quả Keras gốc | Kết quả PyTorch (TV4) | Thời gian Keras gốc | Thời gian PyTorch (RTX 3060) |
|---|---|---|---|---|---|
| `5. Others/1_segnet` | 15 epoch, `batch_size=8`, `Adam(lr=0.0001)` | Test loss `0,6860`; Test pixel accuracy **`0,7732`** (train acc cuối `0,9014`, loss `0,3496`) | Test loss **`0,6808`**; Test pixel accuracy **`0,8138`** (cao nhất `0,8148` tại epoch 12; train acc cuối `0,9315`, loss `0,3483`) | `12,50` phút (`750,02` giây) | **`4,44` phút** (`266,25` giây, ~`17,7` giây/epoch) |

Nhận xét:

- **Hội tụ nhanh và độ chính xác cao hơn rõ rệt (+4,06 điểm phần trăm):** Bản PyTorch đạt **`0,8138`** (81,38% pixel accuracy) và `Test loss = 0,6808` sau 15 epoch, vượt bản Keras gốc (`0,7732`, `loss = 0,6860`) ở cả độ chính xác lẫn độ mất mát. Ngay từ Epoch 1, bản PyTorch đã đạt `val_acc = 0,6956` (so với `0,6190` ở Keras) và duy trì đà tăng ổn định vượt mốc `80%` từ Epoch 5 (`0,8010`).
- **Lý do chênh lệch tích cực so với bản Keras gốc:**
  1. **Tham số `momentum` mặc định của BatchNorm:** Trong PyTorch `nn.BatchNorm2d`, thống kê chạy (`running_mean`/`running_var`) cập nhật theo công thức `new = (1 - 0.1)·old + 0.1·batch` (`momentum = 0,1`), nhanh gấp 10 lần mặc định của Keras (`momentum = 0,99`, tức trọng số batch mới chỉ `0,01`). Với ngân sách ngắn `15` epoch và `batch_size = 8`, thống kê BN ở PyTorch thích nghi nhanh hơn với phân bố thật ở chế độ `eval()`, giúp `val_acc` cao và ổn định hơn.
  2. **Xử lý điểm ảnh nhiễu ngoài bảng màu chuẩn:** Dùng `ignore_index=-100` giúp không tính oan các pixel có mức xám lạ (`51, 102, 153, 204` ở `8/1050` ảnh test, `0,27%` pixel) thành lớp 0 trong khi bản Keras tính `argmax([0,..,0]) = 0` trong `categorical_accuracy`.
- **Trực quan hóa đầu ra (`test #684`):** Bản đồ phân đoạn màu dự đoán của mô hình PyTorch tách rõ ràng phần lòng bàn tay (đen), ngón cái (xanh dương đậm) và các ngón tay còn lại (xanh lá, đỏ sẫm, tím, vàng ô-liu) sát với ảnh nhãn thực (Ground Truth), chỉ còn một ít nhiễu ranh giới mảnh giữa các ngón tay kề nhau do kích thước ảnh nhỏ (`96×96`) và không có kết nối tắt (skip connection / max-unpooling indices).
- **Thời gian huấn luyện:** Bản PyTorch trên RTX 3060 Laptop hoàn tất 15 epoch trong **4,44 phút** (`266,25` giây), nhanh gấp **2,82×** so với bản Keras gốc (`12,50` phút), chủ yếu nhờ phần cứng GPU mới hơn và cài đặt nhân tính toán cuDNN hiện đại.
