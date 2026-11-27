# Phần báo cáo của TV3 – Võ Hoàng Bảo: 2 notebook (`4. Training Utilities/3_Emotion_Reg_CNN` và `5. Others/2_obj_detection`)

Code: `BTL03/pytorch/` (giữ đúng cấu trúc thư mục giống `examples/`). Môi trường chạy: Python 3.12.10, PyTorch 2.6.0 + CUDA 12.4, `torchvision` 0.21.0 + CUDA 12.4, GPU NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM). Cố định `SEED = 0`, bật `torch.backends.cudnn.deterministic = True` và tắt `benchmark` thống nhất với 5 notebook khuôn của TV2.

Các chuyển đổi chung (chọn `device`, `Conv2d`/`MaxPool2d`/`Dropout`/`Flatten`, đổi ảnh channels-last → channels-first, `init_like_keras`, vòng lặp `summary`/`fit`/`evaluate`/`predict` tự chứa trong từng notebook) kế thừa từ mục 1 báo cáo của TV2 (`TV2_5_notebook_khuon.md`).

## 1. Thay đổi riêng từng notebook

### 1.1. `4. Training Utilities/3_Emotion_Reg_CNN`

Kế thừa khuôn `2. CNN + MLP/3_Emotion_Reg_CNN` (đọc thẳng `fer2013.csv`, kiến trúc CNN + MLP giữ nguyên **19,028,287 tham số**, filter `218`, đầu vào `(N, 1, 48, 48)` pixel thô `0–255`, nhãn số nguyên `0..6`, `nn.CrossEntropyLoss()`, `Adam(lr=0.0001)`) và lớp `EarlyStopping` từ khuôn `4. Training Utilities/2_Human_Gender_CNN`. Các thay đổi bổ sung của notebook này:

| Thành phần Keras gốc | Bản PyTorch | Ghi chú |
|---|---|---|
| `ImageDataGenerator(rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)` | `AugmentedEmotionDataset(Dataset)` kết hợp `torchvision.transforms.Compose([RandomHorizontalFlip(p=0.5), RandomAffine(degrees=45, translate=(0.2, 0.2), interpolation=BILINEAR)])` | Áp dụng on-the-fly chỉ trên tập train; thao tác trực tiếp trên tensor `(1, 48, 48)` float32 để giữ nguyên thang pixel `0–255` như bản gốc |
| `model.fit_generator(datagen.flow(...), steps_per_epoch=..., callbacks=[early_stopping])` | `DataLoader(train_ds, batch_size=128, shuffle=True)` + hàm `fit_with_early_stopping` | Mỗi epoch đi qua toàn bộ 28 709 ảnh train (225 batch), đánh giá trên 3 589 ảnh `PublicTest` không augment |
| `keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)` | Lớp `EarlyStopping(patience=5)` tự viết | Dừng huấn luyện khi `val_loss` không giảm trong 5 epoch liên tiếp |

### 1.2. `5. Others/2_obj_detection`

Kế thừa khuôn `2. CNN + MLP/1_MNIST_CNN`. Đây là bài toán phát hiện xe dùng mạng Fully Convolutional Network (FCN) có một số điểm khác biệt so với các bài phân loại đa lớp thông thường:

| Thành phần Keras gốc | Bản PyTorch | Ghi chú |
|---|---|---|
| Đọc 17 760 ảnh `.png` (8 792 `vehicles` nhãn `+1`, 8 968 `non-vehicles` nhãn `-1`) bằng `skimage.io.imread`, chia `70/30` | Dùng `PIL.Image.open(...).convert('RGB')` đọc nhanh gấp ~5 lần trên Windows, `train_test_split(..., test_size=0.30, random_state=0)`, `transpose(0, 3, 1, 2)` sang `(N, 3, 64, 64)` | Giữ nguyên 12 432 mẫu train và 5 328 mẫu test như bản gốc |
| `Lambda(lambda x: x/127.5 - 1.)` | Module `NormalizeInput(nn.Module)` đặt đầu `nn.Sequential(OrderedDict(...))` | Đưa pixel `0–255` về `[-1, 1]` ngay trong mô hình |
| `get_conv()` (FCN gồm `conv1`, `conv2`, `MaxPooling2D(8)`, `Dropout(0.25)`, `dense1` `Conv2D(128, 8×8)`, `Dropout(0.5)`, `dense2` `Conv2D(1, 1×1, activation='tanh')`) + `Flatten()` khi train | `backbone = get_conv()` giữ đúng tên lớp và tổng **83,367 tham số**; bọc thêm `nn.Flatten()` khi train và bỏ `Flatten` khi suy luận `heatmodel` | Chia sẻ trực tiếp trọng số giữa `model` và `backbone` để lưu/nạp khớp khóa `state_dict` |
| `loss='mse'`, `optimizer='adam'`, `metrics=['accuracy']` với nhãn `{-1, +1}` và lớp cuối `tanh` | `nn.MSELoss()`, `torch.optim.Adam(lr=0.001)`, giữ lớp cuối `nn.Tanh()` (không dùng `CrossEntropyLoss`) | Độ chính xác tính theo `torch.round(preds) == y` khớp công thức `K.equal(y_true, K.round(y_pred))` của Keras với MSE |
| `model.save_weights('localize.h5')` / `load_weights('localize.h5')` | `torch.save(backbone.state_dict(), 'localize.pt')` / `load_state_dict(torch.load('localize.pt', ...))` | File `.pt` được `.gitignore` chặn, không đẩy lên repo |
| `heatmap[0, :, :, 0] > 0.99` trên `test.jpg` | Đổi ảnh `test.jpg` sang `(1, 3, H, W)`, lấy `heatmap[0, 0, :, :] > 0.99` và vẽ khung `64×64` bước nhảy `8` (có kẹp biên ảnh) | Hiển thị đủ 3 hình `Heatmap`, `Car Area`, `Detected Cars` |

Số tham số của cả 2 model PyTorch trùng khớp hoàn toàn với `model.summary()` của bản Keras gốc (`19,028,287` và `83,367`).

## 2. Kết quả so với bản Keras gốc

Accuracy và loss đo trên tập test (`PublicTest` với FER2013, 30% tập chia với `car_data`). Thời gian là thời gian huấn luyện ghi nhận trực tiếp trong output notebook.

| Notebook | Epoch | Keras gốc | PyTorch (TV3) | Thời gian Keras gốc | Thời gian PyTorch (RTX 3050) |
|---|---|---|---|---|---|
| `4. Training Utilities/3_Emotion_Reg_CNN` | dừng sớm ở epoch 50 (gốc: 57/100) | 0,6163 (loss 1,0158) | 0,5893 ở epoch cuối; cao nhất 0,5971 tại epoch 45 (loss 1,0862 / tốt nhất 1,0531) | 22,75 phút (GTX 1080) | 25,87 phút |
| `5. Others/2_obj_detection` | 5 | 0,9523 (MSE 0,0642) | 0,9728 (MSE 0,0394) | ~0,20 phút / 12 giây | 0,14 phút / 8,44 giây |

Nhận xét:

- **`4. Training Utilities/3_Emotion_Reg_CNN`**:
  - Nhờ data augmentation, mô hình **không bị overfitting**: ở epoch 50, `train_accuracy = 0,5900` gần như trùng khít `val_accuracy = 0,5893` (trong khi bản không augmentation có khoảng cách train–val lớn hơn).
  - Test accuracy cuối (`0,5893`, tốt nhất `0,5971` ở epoch 45) thấp hơn bản Keras gốc (`0,6163`) khoảng 1,9–2,7 điểm phần trăm do hai nguyên nhân chính: (1) `ImageDataGenerator` của Keras mặc định dùng `fill_mode='nearest'` (lấp vùng biên bằng pixel kề cận khi xoay ±45° / dịch ±20%), còn `torchvision.transforms.RandomAffine` mặc định điền viền đen (`fill=0`) tạo biên nhân tạo trên ảnh pixel `0–255`; (2) `val_loss` đạt đáy cục bộ sớm ở epoch 45 (`1,0531`) rồi dao động nhẹ trong 5 epoch tiếp theo khiến `EarlyStopping(patience=5)` dừng sớm ở epoch 50 (trước bản Keras 7 epoch) và không khôi phục trọng số ở epoch tốt nhất (giống hành vi mặc định bản Keras gốc).
  - Thời gian huấn luyện trên RTX 3050 Laptop (25,87 phút cho 50 epoch, ~31 giây/epoch) lâu hơn một chút so với GTX 1080 (22,75 phút cho 57 epoch, ~24 giây/epoch) do bật `cudnn.deterministic = True` và áp dụng `RandomAffine` tuần tự trên CPU trong `DataLoader(num_workers=0)` trên Windows.
- **`5. Others/2_obj_detection`**:
  - Bản PyTorch đạt **Test accuracy 0,9728** và **MSE loss 0,0394**, tốt hơn rõ so với bản Keras gốc (`0,9523` / `0,0642`), thời gian huấn luyện chỉ **8,44 giây** cho 5 epoch.
  - Khi nạp `localize.pt` vào `heatmodel` và suy luận trên ảnh `test.jpg`, bản đồ nhiệt (`Heatmap`) và vùng xe (`Car Area`, ngưỡng `> 0,99`) phát hiện chính xác cả 2 ô tô (xe đen và xe trắng) trên đường cao tốc, khung xanh bao gọn mục tiêu và không phát sinh báo động giả.
