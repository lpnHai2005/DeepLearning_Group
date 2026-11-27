# Phần báo cáo của TV2 – Lê Phan Nguyên Hải: 5 notebook khuôn

Code: `BTL03/pytorch/` (giữ cấu trúc thư mục giống `examples/`). Môi trường chạy: Python 3.12, PyTorch 2.6.0 + CUDA 12.4, GPU NVIDIA GeForce RTX 4060 Laptop (8 GB). Seed cố định `0`, bật `torch.backends.cudnn.deterministic = True` và tắt `benchmark` để mỗi lần Run All cho cùng kết quả. Lần chạy thử đầu chưa bật chế độ này, notebook Training Utilities chạy lại cùng seed vẫn ra hai số khác nhau (0,8790 và 0,8681), vì cuDNN chọn thuật toán không tất định.

## 1. Những thay đổi chung khi chuyển Keras → PyTorch

Áp dụng cho cả 5 notebook; các notebook còn lại của nhóm chép theo khuôn này.

| Keras | PyTorch | Ghi chú |
|---|---|---|
| Tự dùng GPU nếu có | `device = torch.device('cuda' if ... else 'cpu')`, chuyển model và từng batch sang `device` | |
| `Model(inputs, outputs)` / API hàm | Lớp kế thừa `nn.Module` (MNIST) hoặc `nn.Sequential(OrderedDict(...))` (Gender, Emotion) | `OrderedDict` giữ tên lớp `conv1_1`, `pool1`, `fc1`... như bản gốc |
| `Dense(n, activation='relu')` | `nn.Linear(in, n)` + ReLU | PyTorch phải ghi số chiều vào |
| `Conv2D(f, (3,3), padding='same')` | `nn.Conv2d(in, f, 3, padding=1)` | `padding='valid'` → `padding=0` |
| `MaxPooling2D((2,2))`, `Dropout(r)`, `Flatten()` | `nn.MaxPool2d(2)`, `nn.Dropout(r)`, `nn.Flatten()` | Số chiều sau `Flatten` phải tự tính để khai báo `nn.Linear` |
| Ảnh `(N, H, W, C)` (channels-last) | Ảnh `(N, C, H, W)` (channels-first) | Đổi bằng `reshape` hoặc `transpose(0, 3, 1, 2)` |
| Lớp cuối `softmax` + `loss='categorical_crossentropy'` | Lớp cuối trả về logits + `nn.CrossEntropyLoss()` | Loss của PyTorch đã gồm log-softmax; nhãn là chỉ số lớp, bỏ `to_categorical` / one-hot |
| `SGD(lr)`, `Adam(lr)` | `torch.optim.SGD(params, lr)`, `torch.optim.Adam(params, lr)` | |
| Khởi tạo mặc định `glorot_uniform`, bias 0 | Mặc định `kaiming_uniform` → hàm `init_like_keras` đặt lại `xavier_uniform_` và bias 0 | Để kết quả sát bản gốc, nhất là khi đầu vào là pixel thô 0–255 (Gender, Emotion) |
| `model.fit(x, y, validation_data, batch_size, epochs)` | `DataLoader(TensorDataset(...), shuffle=True)` + hàm `fit` tự viết (vòng lặp `train_one_epoch` + `evaluate`) | `fit` in mỗi epoch giống Keras và trả về `history` |
| `model.evaluate`, `model.predict` | Hàm `evaluate` (`model.eval()`, `torch.no_grad()`), `predict` (softmax) | `model.train()` / `model.eval()` bật/tắt Dropout, Keras tự làm |
| `model.summary()`, `plot_model` / `model_to_dot` | Hàm `summary` dùng forward hook in tên lớp, kích thước đầu ra, số tham số | Không có bản tương đương `plot_model` |
| `model.get_layer('fc2').output` | Hàm `features` / `conv_features` viết trong lớp model | |

Số tham số của cả 5 model PyTorch trùng khớp với `model.summary()` của bản Keras.

## 2. Thay đổi riêng từng notebook

**`1. MLP/MNIST_MLP`** – `keras.datasets.mnist` → `torchvision.datasets.MNIST` (tải về `pytorch/data/`). Model `MnistMLP` có hàm `features` thay cho việc tạo `Model(..., outputs=[model.get_layer('fc2').output, model.output])`.

**`2. CNN + MLP/1_MNIST_CNN`** – Ảnh `(60000, 1, 28, 28)`. Sau 3 lần pool kích thước 28 → 14 → 7 → 3 nên `fc1 = nn.Linear(256·3·3, 512)`. Phần PCA trên đặc trưng `fc2` và `pool3` + `GlobalAveragePooling2D` → `model.features`, `model.conv_features(x).mean((2, 3))`. Ô gốc `model.get_layer()` (gọi thiếu tham số, báo lỗi) được bỏ.

**`2. CNN + MLP/2_Human_Gender_CNN`** – Dữ liệu `.npy` dạng `(N, 64, 64, 1)` float64 giá trị 0–255 → `transpose` sang `(N, 1, 64, 64)`, nhãn one-hot → `argmax`. 5 khối conv lặp lại gom vào hàm `conv_block`.

**`2. CNN + MLP/3_Emotion_Reg_CNN`** – Bản gốc đọc `fer2013_training_onehot.csv` và `fer2013_publictest_onehot.csv`. Hai file này là `fer2013.csv` (Kaggle) đổi định dạng: đối chiếu thấy 3 pixel đầu của ảnh train đầu tiên đều là `70, 80, 82`. Vì vậy bản PyTorch đọc thẳng `fer2013.csv`, lọc `Usage = Training` (28 709 ảnh) và `Usage = PublicTest` (3 589 ảnh), giữ pixel thô như bản gốc. Giữ nguyên số filter 218 của bản gốc.

**`4. Training Utilities/2_Human_Gender_CNN`** – PyTorch không có callback; viết trong hàm `fit_with_callbacks`, cùng thứ tự callback bản gốc:

| Keras callback | PyTorch |
|---|---|
| `ModelCheckpoint(monitor='val_loss', save_best_only=True)` | Lớp `ModelCheckpoint` tự viết, lưu `model.state_dict()` dạng `.pt` vào `checkpoints/`, tên file giống bản gốc |
| `EarlyStopping(patience=5)` | Lớp `EarlyStopping` tự viết |
| `ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-4)` | `torch.optim.lr_scheduler.ReduceLROnPlateau(..., patience=4, threshold=1e-4, threshold_mode='abs')`. PyTorch giảm LR khi số epoch không cải thiện *vượt* `patience`, Keras giảm khi *đạt* `patience`, nên truyền `patience - 1` |
| `TensorBoard(log_dir='./logs')` | `torch.utils.tensorboard.SummaryWriter('./logs')` |
| `model.load_weights('000027-...hdf5')` | `model.load_state_dict(torch.load(<file .pt tốt nhất>))`. File `.hdf5` của bản gốc không có và PyTorch không nạp được, nên nạp checkpoint của chính lần chạy này |

Ảnh `checkpoints.PNG`, `tfboard.PNG` của bản gốc được thay bằng danh sách thư mục `checkpoints/` và biểu đồ các đại lượng đã ghi vào TensorBoard.

Learning rate giữ nguyên 0.001 suốt quá trình huấn luyện: EarlyStopping và ReduceLROnPlateau cùng `patience = 5` trên `val_loss`, nên EarlyStopping dừng đúng epoch mà LR lẽ ra bắt đầu giảm. Bản Keras gốc dùng cùng cấu hình nên cũng vậy. Notebook có ghi chú cách đặt `reduce_lr_after_epochs` nhỏ hơn để thấy LR giảm.

## 3. Kết quả so với bản Keras gốc

Accuracy trên tập test (tập validation của bản gốc). Thời gian là thời gian huấn luyện ghi trong notebook.

| Notebook | Epoch | Keras gốc | PyTorch | Thời gian Keras gốc | Thời gian PyTorch |
|---|---|---|---|---|---|
| `MNIST_MLP` | 30 | 0,9648 | 0,9656 | 1,13 phút (GPU Colab) | 0,47 phút |
| `1_MNIST_CNN` | 30 | 0,9884 | 0,9887 | 3,39 phút (GPU Colab) | 1,54 phút |
| `2_Human_Gender_CNN` | 30 | 0,8264 | 0,8695 | 2,94 phút (GTX 1080) | 1,15 phút |
| `3_Emotion_Reg_CNN` | 15 | 0,5954 | 0,5954 | 6,11 phút (GTX 1080) | 3,92 phút |
| `4. Training Utilities/2_Human_Gender_CNN` | dừng sớm ở epoch 30 (gốc: 32) | 0,8773 (checkpoint epoch 27) | 0,8715 (checkpoint epoch 25) | 3,19 phút (GTX 1080) | 1,15 phút |

Nhận xét:

- 4/5 notebook cho kết quả bằng hoặc cao hơn bản Keras; Training Utilities thấp hơn 0,6 điểm phần trăm (0,8715 so với 0,8773). Chênh lệch đến từ khởi tạo ngẫu nhiên, thứ tự trộn dữ liệu và cách hai thư viện cài đặt phép tính; không có notebook nào lệch đáng kể. Emotion trùng đúng 0,5954, cho thấy cách đọc thẳng `fer2013.csv` tương đương hai file onehot của bản gốc.
- `2_Human_Gender_CNN` cao hơn rõ nhất (0,8695 so với 0,8264). Ở bản Keras gốc, val_loss chững quanh 0,41 từ khoảng epoch 14 rồi tăng lên 0,46 ở các epoch cuối (overfit), val_acc cao nhất chỉ 0,8403. Bản PyTorch val_loss xuống thấp nhất 0,343 (epoch 24) và chỉ dao động trong khoảng 0,34–0,38 đến epoch 30.
- Ở bản gốc, Training Utilities cao hơn rõ notebook Gender thường (0,826 → 0,877); ở bản PyTorch hai notebook gần bằng nhau (0,8695 và 0,8715) vì bản thường không bị overfit như bản Keras. Ngoài ra hai notebook dùng learning rate khác nhau (1e-3 so với 1e-4) và LR không lần nào bị giảm, nên không thể quy toàn bộ mức tăng cho callback; phần đóng góp rõ của callback là dừng sớm và giữ lại checkpoint có val_loss thấp nhất thay vì model ở epoch cuối.
- Thời gian ngắn hơn chủ yếu do GPU mới hơn (RTX 4060 so với GTX 1080 / GPU Colab), không dùng để so sánh hiệu năng hai thư viện.
