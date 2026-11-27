# Phần báo cáo của TV1 – Đoàn Trung Kiên: 3 notebook (`0. Getting Started/getting_started`, `1. MLP/Func_apxm_MLP` và `3. CNN + LSTM/MNIST_CNN_LSTM`)

Code: `BTL03/pytorch/` (giữ đúng cấu trúc thư mục giống `examples/`). Môi trường chạy: Python 3.12.9, PyTorch 2.6.0 + CUDA 12.4, `torchvision` 0.21.0 + CUDA 12.4, GPU NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM). Cố định `SEED = 0`, bật `torch.backends.cudnn.deterministic = True` và tắt `benchmark` thống nhất với 5 notebook khuôn của TV2.

Các chuyển đổi chung (chọn `device`, `Conv2d`/`MaxPool2d`/`Dropout`/`Flatten`, đổi ảnh channels-last → channels-first, `init_like_keras`, vòng lặp `summary`/`fit`/`evaluate`/`predict` tự chứa trong từng notebook) kế thừa trực tiếp từ mục 1 báo cáo của TV2 (`TV2_5_notebook_khuon.md`).

## 1. Thay đổi riêng từng notebook

### 1.1. `0. Getting Started/getting_started`

Notebook gốc kiểm tra cài đặt TensorFlow/Keras và liệt kê thiết bị phần cứng phục vụ huấn luyện. Bản PyTorch thay thế tương ứng như sau:

| Thành phần Keras / TensorFlow gốc | Bản PyTorch | Ghi chú |
|---|---|---|
| `import tensorflow as tf` và `from tensorflow import keras` | `import torch` và `import torchvision` | Bỏ hoàn toàn phụ thuộc TensorFlow/Keras |
| `print(tf.__version__)` | In `torch.__version__`, `torchvision.__version__`, `torch.version.cuda` | Kiểm tra đồng thời phiên bản PyTorch và bản CUDA đi kèm |
| `device_lib.list_local_devices()` | `torch.cuda.is_available()`, `torch.cuda.device_count()`, `torch.cuda.get_device_properties(i)` và `torch.device('cuda' if ... else 'cpu')` | Liệt kê tên GPU, Compute Capability, tổng VRAM và chọn `device` chuẩn cho các notebook sau |

### 1.2. `1. MLP/Func_apxm_MLP`

Kế thừa khuôn `1. MLP/MNIST_MLP`, nhưng đây là **bài toán hồi quy xấp xỉ hàm mật độ Gaussian chuẩn 2D** trên miền `[-3, 3] × [-3, 3]` (`32 × 32 = 1 024` điểm train, `10 × 10 = 100` điểm test sinh từ `scipy.stats.multivariate_normal.pdf`) qua 3 khảo sát độc lập theo số mẫu train `A`, số nơ-ron ẩn `B` và số epoch `C`:

| Thành phần Keras gốc | Bản PyTorch | Ghi chú |
|---|---|---|
| `gauss_mlp(B)`: `Input(2)` → `Dense(B, activation='sigmoid')` → `Dense(1, activation='sigmoid')` | Lớp `GaussMLP(nn.Module)`: `nn.Linear(2, B)` → `nn.Sigmoid()` → `nn.Linear(B, 1)` → `nn.Sigmoid()`, `.squeeze(-1)` | Giữ `Sigmoid` ở cả tầng ra như bản gốc (vì đỉnh Gaussian chuẩn 2D `≈ 0,159 < 1`); `.squeeze(-1)` đưa đầu ra về `(N,)` khớp với nhãn `y`, tránh lỗi broadcast `(N, 1)` với `(N,)` trong MSE |
| `loss='mean_squared_error'`, `SGD(lr=0.01)` | `nn.MSELoss()`, `torch.optim.SGD(model.parameters(), lr=0.01)` | Nhãn `y` giữ kiểu số thực `torch.float32`, không ép `torch.long` và không tính accuracy phân lớp |
| `model.fit(..., batch_size=128, epochs=C, verbose=0)` và `model.evaluate(..., verbose=0)` | Hàm `make_loader`, `fit(..., verbose=0)`, `evaluate` tự chứa trong notebook + `init_like_keras` (`xavier_uniform_`, `bias=0`) | Giữ nguyên 3 vòng lặp khảo sát `(4.1) A = 100..1000`, `(4.2) B = 8..512`, `(4.3) C = 10..190` và vẽ trực tiếp đường lỗi `training` / `testing` |

Số tham số của `GaussMLP(B=16)` trùng khớp hoàn toàn với bản Keras gốc (**65 tham số**: `fc` 48 + `gauss_apxm` 17).

### 1.3. `3. CNN + LSTM/MNIST_CNN_LSTM`

Kế thừa kết hợp khuôn `1. MLP/MNIST_MLP` và `2. CNN + MLP/1_MNIST_CNN`. Bài toán ước lượng **tổng giá trị của tối đa 8 ảnh chữ số MNIST** (`maxToAdd = 8`, `60 000` chuỗi train và `10 000` chuỗi test sinh với `np.random.seed(2018)` như bản gốc):

| Thành phần Keras gốc | Bản PyTorch | Ghi chú |
|---|---|---|
| `keras.datasets.mnist.load_data()`, mảng chuỗi `(N, 8, 28, 28, 1)` (`float64` mặc định của `np.zeros`) | `torchvision.datasets.MNIST(root='../data', download=True)`, mảng channels-first `(N, 8, 1, 28, 28)` kiểu `np.float32` + `torch.from_numpy` | Cấp phát trực tiếp `float32` và dùng `from_numpy` giúp tiết kiệm một nửa bộ nhớ RAM (~1,5 GB thay vì ~3,0 GB) và không tạo bản sao dư thừa |
| `TimeDistributed(Conv2D / MaxPooling2D / Dropout / Flatten)` | Gộp chiều thời gian vào batch `x.reshape(B * T, 1, 28, 28)` đi qua 3 khối CNN (`1 → 64 → 128 → 256`, sau 3 lần `MaxPool2d(2)` còn `3×3`, `Flatten` ra `2 304`), sau đó tách lại `x.reshape(B, T, 2304)` | PyTorch không cần lớp bọc `TimeDistributed`; xử lý song song toàn bộ `B × 8` khung ảnh trên GPU |
| `LSTM(128, return_sequences=True)` → `LSTM(128, return_sequences=False)` → `Dropout(0.22)` → `Dense(1)` | `nn.LSTM(2304, 128, batch_first=True)` → `nn.LSTM(128, 128, batch_first=True)` lấy bước cuối `x[:, -1, :]` → `nn.Dropout(0.22)` → `nn.Linear(128, 1)` `.squeeze(-1)` | Lớp cuối bản gốc đặt tên `fc3_10ways_softmax` nhưng không truyền `activation`, thực chất là tầng tuyến tính dự đoán tổng chữ số |
| Khởi tạo mặc định của `LSTM` trong Keras (`glorot_uniform`, `orthogonal`, `unit_forget_bias=True`) | Mở rộng `init_like_keras` cho `nn.LSTM`: `xavier_uniform_` cho `weight_ih`, `orthogonal_` cho `weight_hh`, `bias = 0` riêng cổng quên `bias_ih[H:2H] = 1,0` | Giúp `SGD(lr=0.01)` không có momentum hội tụ nhanh và ổn định trên chuỗi dài 8 bước tương đương bản Keras |
| `loss='mean_absolute_error'`, `SGD(lr=0.01)`, `SVG(model_to_dot(...))` | `nn.L1Loss()`, `torch.optim.SGD(lr=0.01)`, hàm `summary` có xử lý đầu ra dạng `tuple` `(output, (h_n, c_n))` của `nn.LSTM` | Hiển thị ảnh minh họa đổi chỉ số kênh từ `x_test[idx, t, :, :, 0]` sang `x_test[idx, t, 0, :, :]` |

Về số tham số của `MNIST_CNN_LSTM`: các lớp tích chập và lớp tuyến tính cuối trùng khớp tuyệt đối với bản Keras (`conv1`: 640; `conv2`: 73 856; `conv3`: 295 168; `fc`: 129). Riêng mỗi lớp `nn.LSTM` trong PyTorch khai báo tách biệt hai vector độ chệch `bias_ih` và `bias_hh` (`4 × 128 = 512` tham số mỗi vector), trong khi `LSTM` của Keras chỉ dùng một vector `bias` duy nhất, nên tổng tham số bản PyTorch là **1 748 097** so với **1 747 073** của bản Keras (nhiều hơn đúng `2 × 512 = 1 024` tham số độ chệch).

## 2. Kết quả so với bản Keras gốc

Chỉ số đánh giá đo trên tập test theo đúng từng bài toán; thời gian là thời gian huấn luyện ghi nhận trực tiếp trong output của notebook.

| Notebook | Cấu hình chạy | Kết quả Keras gốc | Kết quả PyTorch (TV1) | Thời gian Keras gốc | Thời gian PyTorch (RTX 3050) |
|---|---|---|---|---|---|
| `0. Getting Started/getting_started` | Kiểm tra thư viện & GPU | TensorFlow `2.14.0`, nhận 1 GPU `Tesla T4` (Colab, ~14,4 GB) | PyTorch `2.6.0+cu124`, TorchVision `0.21.0+cu124`, nhận 1 GPU `RTX 3050 Laptop` (4 GB, CC 8.6) | < 0,05 phút | < 0,05 phút |
| `1. MLP/Func_apxm_MLP` | 3 khảo sát `(4.1) A=100..1000`, `(4.2) B=8..512`, `(4.3) C=10..190` (tổng 36 mô hình, 3 600 epoch) | Chỉ lưu 3 biểu đồ đường lỗi (`verbose=0`): MSE giảm dần khi tăng `A`, `B`, `C`; train & test bám sát nhau | Tái hiện đúng cả 3 đường cong; Test MSE cuối: `(4.1)` `0,0935 → 0,0088`; `(4.2)` `0,0199 → 0,0013`; `(4.3)` `0,0158 → 0,0015` | Không in trong bản gốc (`verbose=0`) | 0,78 phút (46,69 giây cho cả 3 khảo sát) |
| `3. CNN + LSTM/MNIST_CNN_LSTM` | 30 epoch, `batch_size=128`, `SGD(lr=0.01)` | Train MAE `0,9357`; Test MAE cuối **`0,6502`** (thấp nhất `0,5768` ở epoch 26) | Train MAE `0,9395`; Test MAE cuối **`0,7033`** (thấp nhất **`0,5728`** ở epoch 26) | 26,63 phút (GTX 1080) / ~808 phút (Core i7) | 19,60 phút |

Nhận xét:

- **`0. Getting Started/getting_started`**: Cả hai bản đều xác nhận chính xác phiên bản framework và nhận diện thành công GPU phục vụ tính toán song song.
- **`1. MLP/Func_apxm_MLP`**: Bản Keras gốc đặt `verbose=0` nên chỉ lưu hình vẽ mà không in số MSE cụ thể. Đối chiếu trực quan 3 đồ thị cho thấy bản PyTorch tái hiện trùng khớp quy luật hội tụ: sai số MSE giảm mạnh khi tăng kích thước mẫu `A` từ 100 lên 400–1 000, khi mở rộng số nút ẩn `B` từ 8 lên 128–512 và khi tăng số epoch `C` từ 10 lên 190. Hai đường `training` và `testing` nằm sát nhau trên cả 3 khảo sát vì dữ liệu sinh từ hàm Gaussian giải tích trơn, không có nhiễu.
- **`3. CNN + LSTM/MNIST_CNN_LSTM`**:
  - Quá trình hội tụ của bản PyTorch bám rất sát bản Keras gốc ngay từ các epoch đầu (`Epoch 1`: val MAE `4,1093` so với `4,6118`; `Epoch 30`: train MAE `0,9395` gần như trùng khít `0,9357` của bản gốc).
  - Trên tập test 10 000 chuỗi, bản PyTorch đạt mức MAE tốt nhất **0,5728** tại epoch 26 (tốt hơn nhẹ mức thấp nhất `0,5768` cũng tại epoch 26 của bản Keras gốc). Ở epoch 30 cuối cùng, Test MAE của PyTorch là `0,7033` so với `0,6502` của Keras (lệch ~0,05 đơn vị tổng chữ số trên thang tổng từ 0 đến 72). Sự dao động nhỏ trong khoảng `0,57 – 0,71` ở 5 epoch cuối xuất hiện ở cả hai bản (bản Keras dao động `0,5768 → 0,6142 → 0,7065 → 0,6012 → 0,6502`) do dùng `SGD` với tốc độ học cố định `lr = 0,01` không có lịch giảm LR hay lưu checkpoint tốt nhất.
  - Thử nghiệm dự đoán trên chuỗi 8 ảnh chữ số bất kỳ của tập test cho kết quả sát giá trị thực (`actual summation = 31`, `estimated summation = 31,801`).
  - Thời gian huấn luyện 30 epoch trên RTX 3050 Laptop (19,60 phút, ~39,2 giây/epoch) nhanh hơn khoảng 26% so với GTX 1080 trong bản gốc (26,63 phút, ~53,3 giây/epoch) nhờ gộp chiều `(B·T, C, H, W)` xử lý song song qua cuDNN và nhân `nn.LSTM` tối ưu trên kiến trúc Ampere.
