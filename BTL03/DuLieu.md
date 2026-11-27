# Dữ liệu cho các notebook BTL03

Gói nộp không kèm dữ liệu (khoảng 750 MB). File này ghi cách lấy dữ liệu và chỗ đặt dữ liệu cho từng notebook. Mọi đường dẫn bên dưới tính từ thư mục `pytorch/`.

## Tóm tắt

| Notebook | Dữ liệu | Cách lấy | Đặt ở đâu |
|---|---|---|---|
| `0. Getting Started/getting_started`, `1. MLP/Func_apxm_MLP` | Không cần | | |
| `1. MLP/MNIST_MLP`, `2. CNN + MLP/1_MNIST_CNN`, `3. CNN + LSTM/MNIST_CNN_LSTM` | MNIST | Tự tải khi chạy | Tự lưu vào `data/MNIST/` |
| `2. CNN + MLP/2_Human_Gender_CNN`, `4. Training Utilities/2_Human_Gender_CNN` | 4 file `.npy` | Link Drive của giảng viên | `data/gender/` hoặc cạnh notebook |
| `2. CNN + MLP/3_Emotion_Reg_CNN`, `4. Training Utilities/3_Emotion_Reg_CNN` | `fer2013.csv` | Kaggle | Cạnh từng notebook |
| `5. Others/1_segnet` | `depth_hand.zip` | Có sẵn trong `examples/5. Others/` của giảng viên | Giải nén cạnh notebook |
| `5. Others/2_obj_detection` | `car_data.zip`, `test.jpg` | Có sẵn trong `examples/5. Others/`; `test.jpg` đã kèm trong gói | Giải nén cạnh notebook |

## Chi tiết

### MNIST

Ba notebook `MNIST_MLP`, `1_MNIST_CNN`, `MNIST_CNN_LSTM` gọi `torchvision.datasets.MNIST(root='../data', download=True)`, nên dữ liệu tự tải về `pytorch/data/MNIST/` ở lần chạy đầu. Máy cần có mạng.

### Human Gender (4 file `.npy`)

Hai notebook `2_Human_Gender_CNN` dùng chung 4 file:

```
64_64_11938_4098_train_x_onehot.npy
64_64_11938_4098_train_y_onehot.npy
64_64_5968_4098_val_x_onehot.npy
64_64_5968_4098_val_y_onehot.npy
```

Tải từ link Drive mà giảng viên ghi ở ô markdown đầu notebook gốc:
https://drive.google.com/drive/folders/0BxINLo5jshCRYW8xODhNSlkyLTQ?resourcekey=0-RNJvvSSE67Q1bMqQi3A1RQ&usp=sharing

Đặt 4 file vào `pytorch/data/gender/` để hai notebook dùng chung. Có thể đặt cạnh từng notebook như bản gốc, vì notebook tìm ở cả hai chỗ.

### FER2013 (`fer2013.csv`)

Tải `fer2013.tar.gz` từ trang dữ liệu của cuộc thi trên Kaggle (cần tài khoản và chấp nhận điều khoản cuộc thi), rồi giải nén lấy `fer2013.csv`:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Chép `fer2013.csv` vào cạnh **từng** notebook:

```
pytorch/2. CNN + MLP/fer2013.csv
pytorch/4. Training Utilities/fer2013.csv
```

Notebook gốc đọc `fer2013_training_onehot.csv` và `fer2013_publictest_onehot.csv`. Hai file này chính là `fer2013.csv` đổi định dạng, nên bản PyTorch đọc thẳng `fer2013.csv` rồi lọc `Usage = Training` (28.709 ảnh) làm tập huấn luyện và `Usage = PublicTest` (3.589 ảnh) làm tập kiểm thử.

### SegNet (`depth_hand.zip`)

Lấy `depth_hand.zip` (khoảng 7 MB) trong `examples/5. Others/` của giảng viên, chép vào `pytorch/5. Others/` rồi giải nén tại đó. Đường dẫn trong file zip đã khớp, sau khi giải nén sẽ có:

```
pytorch/5. Others/data/segnet/train/depth/...
pytorch/5. Others/data/segnet/test/depth/...
```

### Phát hiện ô tô (`car_data.zip`)

Lấy `car_data.zip` (khoảng 120 MB) trong `examples/5. Others/`, chép vào `pytorch/5. Others/` rồi giải nén tại đó, sẽ có:

```
pytorch/5. Others/data/obj_detection/vehicles/...
pytorch/5. Others/data/obj_detection/non-vehicles/...
```

Ảnh thử `test.jpg` đã nằm sẵn trong `pytorch/5. Others/`.

## Kiểm tra trước khi chạy

Sau khi đặt đủ dữ liệu, cây thư mục `pytorch/` có dạng:

```
pytorch/
├── data/
│   ├── MNIST/                      # tự tải
│   └── gender/                     # 4 file .npy
├── 2. CNN + MLP/fer2013.csv
├── 4. Training Utilities/fer2013.csv
└── 5. Others/data/
    ├── segnet/{train,test}/depth/
    └── obj_detection/{vehicles,non-vehicles}/
```

Cài thư viện theo `requirements.txt`, sau đó mở từng notebook và chạy Restart & Run All.
