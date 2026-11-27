# BTL03 – Chuyển code Keras sang PyTorch — Phân công

Nguồn: `YeuCau.txt`, thư mục `examples/` (notebook Keras của giảng viên), `01_Introduction-to-Python-TensorFlow-and-Keras.pdf`, `02_KerasExamples.pdf`.

## 1. Yêu cầu

- Chuyển code từ Keras sang PyTorch.
- Nộp kèm: file báo cáo + code đã chuyển + các file liên quan.

## 2. Danh sách notebook

`examples/` có 13 notebook. 2 notebook không cần chuyển, 11 notebook còn lại cần chuyển.

| Notebook | Xử lý |
|---|---|
| `0. Getting Started/Iris_Bayesian_decision_making.ipynb` | Không chuyển (chỉ dùng scikit-learn) |
| `colab_getting_started.ipynb` | Không chuyển (hướng dẫn dùng Colab, không dùng Keras) |
| `0. Getting Started/getting_started.ipynb` | Chuyển (kiểm tra GPU bằng `torch.cuda` thay cho TensorFlow) |
| `1. MLP/Func_apxm_MLP.ipynb` | Chuyển |
| `1. MLP/MNIST_MLP.ipynb` | Chuyển |
| `2. CNN + MLP/1_MNIST_CNN.ipynb` | Chuyển |
| `2. CNN + MLP/2_Human_Gender_CNN.ipynb` | Chuyển (dữ liệu: xem mục 3) |
| `2. CNN + MLP/3_Emotion_Reg_CNN.ipynb` | Chuyển (dữ liệu: xem mục 3) |
| `3. CNN + LSTM/MNIST_CNN_LSTM.ipynb` | Chuyển |
| `4. Training Utilities/2_Human_Gender_CNN.ipynb` | Chuyển (checkpoint, early stopping, giảm learning rate, TensorBoard) |
| `4. Training Utilities/3_Emotion_Reg_CNN.ipynb` | Chuyển (data augmentation, early stopping) |
| `5. Others/1_segnet.ipynb` | Chuyển (dữ liệu: `depth_hand.zip`) |
| `5. Others/2_obj_detection.ipynb` | Chuyển (dữ liệu: `car_data.zip`, ảnh thử `test.jpg`) |

## 3. Dữ liệu

| Notebook | Dữ liệu cần | Cách lấy |
|---|---|---|
| `MNIST_MLP`, `1_MNIST_CNN`, `MNIST_CNN_LSTM` | MNIST | `torchvision.datasets.MNIST(download=True)` |
| Hai notebook `2_Human_Gender_CNN` | `64_64_11938_4098_train_x_onehot.npy`, `..._train_y_onehot.npy`, `64_64_5968_4098_val_x_onehot.npy`, `..._val_y_onehot.npy` | Không đưa lên git. Tải từ link Drive của giảng viên ghi ở ô markdown đầu notebook (link vẫn còn, đã kiểm ngày 27/11/2026): https://drive.google.com/drive/folders/0BxINLo5jshCRYW8xODhNSlkyLTQ?resourcekey=0-RNJvvSSE67Q1bMqQi3A1RQ&usp=sharing. Giải nén vào `BTL03/pytorch/data/gender/` (hai notebook Gender dùng chung) hoặc đặt cạnh notebook. |
| Hai notebook `3_Emotion_Reg_CNN` | `fer2013.csv` (bản Kaggle gốc, cột `emotion, pixels, Usage`) | Notebook gốc đọc `fer2013_training_onehot.csv` và `fer2013_publictest_onehot.csv`. Hai file này chính là `fer2013.csv` đổi định dạng (mỗi dòng 2304 pixel thô 0–255 + 7 cột one-hot; đã đối chiếu với output lưu trong notebook). Vì vậy bản PyTorch đọc thẳng `fer2013.csv`: lọc `Usage = Training` (28 709 ảnh) làm tập train, `Usage = PublicTest` (3 589 ảnh) làm tập test. Không cần tải link Drive. File đang có tại `BTL02/data/fer2013.csv` trên máy nhóm trưởng. File bị `.gitignore` nên không có sau khi clone: TV2 gửi trực tiếp cho TV3, hoặc TV3 tải từ Kaggle theo `BTL02/README.md`. |
| `1_segnet` | `data/segnet/{train,test}/depth/...` | Giải nén `5. Others/depth_hand.zip` cạnh notebook (đường dẫn trong zip đã khớp) |
| `2_obj_detection` | `data/obj_detection/{vehicles,non-vehicles}/...`, `test.jpg` | Giải nén `5. Others/car_data.zip` cạnh notebook (đường dẫn trong zip đã khớp) |

Dữ liệu duy nhất phải tải là 4 file `.npy` của Gender. TV2 tải ngay khi bắt đầu giai đoạn 1. Nếu link chết, báo cả nhóm và hỏi giảng viên ngay.

Dữ liệu không đưa lên git: `.npy`, `.csv`, hai file zip và thư mục `data/` đã giải nén. Riêng `car_data.zip` nặng 120 MB, vượt giới hạn 100 MB/file của GitHub nên push sẽ bị từ chối. TV2 gửi dữ liệu cho cả nhóm qua thư mục chung (Drive/Zalo).

## 4. Phân công

Làm theo 2 giai đoạn. TV2 chuyển 5 notebook trước để làm khuôn. 6 notebook còn lại chia cho 3 bạn, mỗi bạn làm theo khuôn tương ứng.

### Giai đoạn 1 – TV2 làm khuôn

| Notebook khuôn | Khuôn cho phần nào |
|---|---|
| `1. MLP/MNIST_MLP` | Cấu trúc notebook chung: chọn `device`, `DataLoader`, model `nn.Module`, vòng lặp train/evaluate thay `model.fit` / `model.evaluate`, lưu lịch sử loss/accuracy, vẽ biểu đồ, đo thời gian |
| `2. CNN + MLP/1_MNIST_CNN` | Lớp `Conv2d` / `MaxPool2d` / `Dropout`, đổi ảnh `(N, H, W, C)` → `(N, C, H, W)`, tính lại kích thước sau `Flatten` |
| `2. CNN + MLP/2_Human_Gender_CNN` | Nạp dữ liệu từ file `.npy` / mảng numpy vào `TensorDataset` |
| `2. CNN + MLP/3_Emotion_Reg_CNN` | Đọc `fer2013.csv`, lọc theo `Usage` |
| `4. Training Utilities/2_Human_Gender_CNN` | Thay callback Keras: lưu checkpoint tốt nhất `.pt`, early stopping, giảm learning rate (`ReduceLROnPlateau`), TensorBoard (`torch.utils.tensorboard`) |

Khi xong giai đoạn 1, TV2 đẩy 5 notebook lên repo (đã chạy, còn output) và báo nhóm. Ba bạn còn lại **không cần chờ để bắt đầu đọc notebook Keras gốc** của mình, nhưng chỉ viết code sau khi có khuôn.

### Giai đoạn 2 – 3 bạn còn lại

| Thành viên | Notebook cần chuyển | Theo khuôn | Việc thêm |
|---|---|---|---|
| TV1 – Đoàn Trung Kiên | `0. Getting Started/getting_started`, `1. MLP/Func_apxm_MLP`, `3. CNN + LSTM/MNIST_CNN_LSTM` | `MNIST_MLP`, `1_MNIST_CNN` (LSTM: thay `TimeDistributed` bằng gộp chiều thời gian vào batch rồi tách lại) | Tạo `requirements.txt` (mục 7). Đọc soát báo cáo cùng TV4. |
| TV3 – Võ Hoàng Bảo | `4. Training Utilities/3_Emotion_Reg_CNN`, `5. Others/2_obj_detection` | `3_Emotion_Reg_CNN` + `4. Training Utilities/2_Human_Gender_CNN` (augmentation thay `ImageDataGenerator` bằng `torchvision.transforms`); `1_MNIST_CNN` | |
| TV4 – Huỳnh Viễn Thông | `5. Others/1_segnet` | `1_MNIST_CNN` (thêm `ConvTranspose2d` / `Upsample`, loss theo từng pixel) | Viết báo cáo, ghép phần của mọi người. Ghép luôn 5 phần của TV2 trong lúc chờ giai đoạn 2. |

TV2 trong giai đoạn 2: viết `DuLieu.md`, hỗ trợ ai bị kẹt, Run All kiểm tra từng notebook, gom file và nộp.

Lưu ý khi chuyển:

- Không dùng file hàm chung. Mỗi notebook tự chứa vòng lặp train/evaluate (chép từ khuôn), chạy được từ đầu đến cuối (Run All) chỉ với dữ liệu ở mục 3.
- `4. Training Utilities/2_Human_Gender_CNN` gốc nạp trọng số Keras `000027-...hdf5`. File này không có, và PyTorch cũng không nạp trực tiếp được. Bản PyTorch huấn luyện lại, lưu checkpoint tốt nhất dạng `.pt` rồi nạp lại file `.pt` đó. Hình `checkpoints.PNG`, `tfboard.PNG` thay bằng ảnh chụp thư mục checkpoint và TensorBoard của bản chạy PyTorch.
- `2_obj_detection` gốc lưu/nạp `localize.h5`. Bản PyTorch dùng `localize.pt`.
- Chạy thử với ít epoch (1–2) cho đến khi hết lỗi, sau đó mới chạy đủ số epoch như notebook gốc.

Quy ước commit BTL03 (repo dùng chung với BTL02):

- Thông điệp dạng `feat(BTL03): chuyển MNIST_MLP sang PyTorch`. **Không dùng mã dạng `T` + 2 chữ số (`T05`) và không bắt đầu bằng `done(`**, vì `BTL02/tracking/task_status.py` đọc mọi commit và sẽ đổi trạng thái nhiệm vụ BTL02 trùng mã.
- Commit BTL03 tách riêng khỏi BTL02, không `git add .`.
- Luôn `git pull --rebase` trước khi push (bot nhật ký BTL02 cũng tự push vào `main`).

## 5. Nội dung mỗi người ghi vào báo cáo

Với mỗi notebook mình chuyển:

1. Những chỗ đổi từ Keras sang PyTorch (lớp, hàm loss, optimizer, thứ tự chiều ảnh, cách huấn luyện, callback).
2. Kết quả chạy lại so với kết quả đã lưu trong notebook Keras gốc (loss/accuracy cuối, thời gian huấn luyện, phần cứng). Nếu khác nhiều thì nêu lý do.

Mỗi người gửi phần của mình cho TV4 dưới dạng file `.docx` hoặc Markdown. TV4 ghép thành một báo cáo. Notebook nào chạy xong thì viết phần báo cáo cho notebook đó luôn, không để dồn đến cuối.

## 6. Lịch làm

### Giai đoạn 1 – TV2 (trước ngày làm chung)

Hạn: (nhóm trưởng điền), chậm nhất tối hôm trước ngày làm chung.

| Việc | Đầu ra |
|---|---|
| Tải dữ liệu Gender, gửi `.npy`, `fer2013.csv`, hai file zip vào thư mục chung | Cả nhóm có đủ dữ liệu trước ngày làm chung |
| Chuyển và chạy đủ epoch 5 notebook khuôn | 5 notebook có output trên repo |
| Viết phần báo cáo của 5 notebook, gửi TV4 | TV4 có sẵn khung báo cáo |
| Báo nhóm: khuôn đã xong, notebook nào theo khuôn nào (bảng giai đoạn 2) | |

### Giai đoạn 2 – ngày làm chung (1 ngày)

Ngày làm: (nhóm trưởng điền).

| Giờ | Việc | Đầu ra |
|---|---|---|
| 8:00–8:30 | TV1, TV3, TV4 cài môi trường PyTorch, kéo repo, giải nén dữ liệu, đọc khuôn | Mở chạy được 5 notebook khuôn |
| 8:30–12:00 | Chuyển code theo khuôn, chạy thử với 1–2 epoch | Mọi notebook chạy hết không lỗi |
| 13:00–16:00 | Chạy đủ epoch, lưu notebook còn output. Viết phần báo cáo của mình. | **16:00**: đẩy notebook lên repo, gửi phần báo cáo cho TV4 |
| 16:00–18:00 | TV4 ghép báo cáo. TV1 làm `requirements.txt` rồi đọc soát báo cáo cùng TV4. TV2 Run All kiểm tra từng notebook, viết `DuLieu.md`. | **18:00**: báo cáo hoàn chỉnh |
| 18:00–19:00 | TV2 rà lại, nén theo cấu trúc mục 7 và nộp | Bài đã nộp |

Nếu một notebook đến 16:00 vẫn chưa chạy được, người làm báo ngay trong nhóm. TV2 hỗ trợ trực tiếp.

## 7. Nộp

Do nhóm trưởng (TV2) nộp, một file nén gồm:

```
BTL03_Nhom/
├── BaoCao_BTL03.pdf            # báo cáo (kèm bản .docx nếu giảng viên cần)
├── pytorch/                    # code đã chuyển, giữ cấu trúc thư mục giống examples/
│   ├── 0. Getting Started/getting_started.ipynb
│   ├── 1. MLP/...
│   ├── 2. CNN + MLP/...
│   ├── 3. CNN + LSTM/...
│   ├── 4. Training Utilities/...  # kèm ảnh checkpoint, TensorBoard của bản PyTorch
│   └── 5. Others/...              # kèm test.jpg
├── requirements.txt            # phiên bản python, torch, torchvision, numpy, scikit-learn, scikit-image, matplotlib, seaborn
└── DuLieu.md                   # cách lấy dữ liệu (nội dung mục 3)
```

Không nén kèm dữ liệu (`.npy`, `.csv`, hai file zip khoảng 127 MB) và checkpoint lớn. Chỉ ghi cách lấy trong `DuLieu.md`, trừ khi giảng viên yêu cầu nộp kèm.
