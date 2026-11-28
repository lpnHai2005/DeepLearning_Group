# Chương 1. Giới thiệu

## 1.1. Bài toán

Nhận dạng cảm xúc khuôn mặt (Facial Expression Recognition, FER) là bài toán phân loại ảnh: cho một ảnh khuôn mặt, mô hình trả về một trong bảy cảm xúc cơ bản. Đề bài yêu cầu dùng bộ dữ liệu FER2013 của cuộc thi Kaggle "Challenges in Representation Learning: Facial Expression Recognition Challenge" (Goodfellow et al., 2013). Mỗi ảnh là ảnh xám 48 × 48 pixel, nhãn thuộc bảy lớp theo đúng thứ tự của bộ dữ liệu: 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral. Đầu vào của mô hình là ảnh 48 × 48, đầu ra là vector bảy chiều biểu diễn xác suất của bảy cảm xúc.

Đây là bài toán khó vì ba lý do. Thứ nhất, ảnh có độ phân giải rất thấp, nhiều ảnh nghiêng, bị che hoặc thiếu sáng. Thứ hai, nhãn có nhiễu: theo nhóm tác giả bộ dữ liệu, độ chính xác của con người trên FER2013 chỉ khoảng 65 ± 5%. Thứ ba, các lớp mất cân bằng mạnh, lớp Happy nhiều hơn lớp Disgust 16.55 lần (Chương 2).

## 1.2. Mục tiêu và mốc so sánh

Đề bài nêu kỷ lục của cuộc thi là 69% trên tập công khai (PublicTest) và 71% trên tập ẩn (PrivateTest). Báo cáo mẫu của giảng viên dùng Keras, các CNN tự thiết kế đạt độ chính xác validation từ 0.64 đến 0.67. Nhóm đặt bốn mục tiêu:

1. Cài đặt toàn bộ quy trình bằng Python và PyTorch, không dùng Keras hay TensorFlow; code mẫu Keras chỉ dùng để tham khảo ý tưởng.
2. Thực hiện nhiều thực nghiệm với nhiều bộ tham số, chia thành bảy nhóm E1 đến E7: kiến trúc (E1), hàm kích hoạt (E2), thuật toán tối ưu (E3), augmentation và điều chuẩn (E4), mất cân bằng lớp (E5), transfer learning (E6) và mô hình cuối (E7).
3. Trực quan hóa các bộ lọc tích chập của lớp conv cuối và giải thích mô hình nhìn vào đâu khi đoán đúng và khi đoán sai.
4. Phân tích, giải thích kết quả theo cách hiểu của nhóm, dựa trên số liệu đo được chứ không dựa trên kỳ vọng.

Kết quả chính: mô hình cuối là ensemble ba seed của kiến trúc `large_bn` cộng TTA lật ngang, đạt 72.50% trên PublicTest và 73.64% trên PrivateTest (macro-F1 73.22%), vượt cả hai mốc 69% và 71% mà không dùng trọng số huấn luyện sẵn hay dữ liệu ngoài (Chương 4, mục 4.4; nguồn `results/E7/TEST_DONE.json`).

## 1.3. Nguyên tắc làm việc và truy vết

Để mọi con số trong báo cáo kiểm tra lại được, nhóm áp dụng các nguyên tắc sau từ đầu dự án (chi tiết trong `PHAN_CONG_NHIEM_VU.md`, mục 6 và 9):

1. Chia tập cố định theo cột `Usage`: Training để huấn luyện, PublicTest làm validation để chọn cấu hình, PrivateTest chỉ đánh giá đúng một lần ở mô hình cuối. Lệnh đánh giá PrivateTest tự ghi file khóa `results/E7/TEST_DONE.json` và từ chối mọi lần chạy thứ hai.
2. Mỗi lần huấn luyện (run) do code tự ghi `config.json`, `run.log`, `history.csv`, `metrics.json` và một dòng trong `tracking/experiments.csv`, gồm người chạy, máy, GPU, commit git, SHA-256 dữ liệu và thời gian. Không ai gõ tay kết quả.
3. Mọi bảng sinh từ `tracking/experiments.csv` bằng `src/make_tables.py`; mọi hình có file CSV dữ liệu đi kèm và được đăng ký trong `tracking/figures_manifest.csv`.
4. Mỗi nhiệm vụ có bốn vai: một người chịu trách nhiệm chính (R), hai người phối hợp (C) và một người kiểm thử độc lập (T) khác R và C. Người kiểm thử làm trên bản clone sạch và ghi kết quả Pass hoặc Fail vào `tracking/test_log.csv`. Trạng thái nhiệm vụ được suy tự động từ commit git và nhật ký kiểm thử.
5. Kết quả chưa có thì ghi "chưa thực hiện", không điền số ước lượng.

## 1.4. Cấu trúc báo cáo

Chương 2 mô tả dữ liệu, tiền xử lý và hai thực nghiệm liên quan trực tiếp đến dữ liệu: augmentation, điều chuẩn (E4) và mất cân bằng lớp (E5). Chương 3 trình bày mô hình CNN, quy trình huấn luyện và các thực nghiệm kiến trúc (E1), hàm kích hoạt (E2), transfer learning (E6). Chương 4 trình bày phương pháp đánh giá, thực nghiệm tối ưu (E3), mô hình cuối (E7), kết quả PrivateTest và phân tích lỗi. Chương 5 tóm tắt mã nguồn. Chương 6 trực quan hóa bộ lọc tích chập cuối. Chương 7 thảo luận tổng hợp, Chương 8 kết luận. Phụ lục A liệt kê lệnh tái lập, Phụ lục B mô tả các file nộp kèm.

## 1.5. Thành viên, phân công và tỷ lệ đóng góp

Nhóm gồm bốn thành viên, nhóm trưởng là TV2. Bảng 1.1 tóm tắt phần việc chính của từng người theo ma trận phân công 28 nhiệm vụ (`tracking/tasks.csv`)

https://docs.google.com/spreadsheets/d/1mgwx1TGTL0fQewJ3Qto-K_HcZEUbgN4_P7Cy2-Z-aFk/edit?gid=840636480#gid=840636480

**Bảng 1.1.** Thành viên và phần việc chính.

| Mã | Họ và tên | MSSV | Phần việc chính (R) |
|---|---|---|---|
| TV1 | Đoàn Trung Kiên | 3123411166 | Khung repo, README, requirements (T02); kiểm tra dữ liệu, loader, augmentation, EDA (T03 đến T05); thực nghiệm E4, E5 (T15, T16); Chương 2 (T21) |
| TV2 | Lê Phan Nguyên Hải | 3123411081 | Nhóm trưởng; mô hình CNN, transfer learning (T06); vòng huấn luyện, ghi log, luật giữ/xóa weights (T07); baseline (T11); thực nghiệm E1, E2, E6 (T12, T13, T17); Chương 3 (T22); đóng gói và nộp bài (T28) |
| TV3 | Võ Hoàng Bảo | 3123411030 | Đánh giá, hình kết quả (T08); bảng tổng hợp (T09); notebook tái lập (T10); thực nghiệm E3 (T14); mô hình cuối E7 và PrivateTest (T18, T19); Chương 4 (T23) |
| TV4 | Huỳnh Viễn Thông | 3123411287 | Khung theo dõi (T01); trực quan hóa bộ lọc (T20); nhật ký thực nghiệm và sơ đồ kiến trúc (T24); ghép báo cáo và Chương 1, 5, 6, 7, 8 (T25); slide (T26); QA tái lập toàn repo (T27) |

Bảng 1.2 tổng hợp các chỉ số thực làm được sinh tự động bằng `src/t25_report_stats.py` từ các bảng theo dõi trong `tracking/`: số nhiệm vụ làm R đã hoàn tất trên tổng số được giao (`tracking/tasks.csv`), số commit có mã nhiệm vụ (`tracking/activity_log.csv`, đã loại commit tự động của bot), số lần kiểm thử độc lập đã thực hiện (`tracking/test_log.csv`), số run huấn luyện chính thức (`tracking/experiments.csv`, bỏ các run hạ tầng `_qa`, `_probe`, `_nb` và dòng đánh giá PrivateTest), cùng số lượt phối hợp C và kiểm thử T theo ma trận phân công. Do cả bốn thành viên đều hoàn thành đầy đủ 100% nhiệm vụ R được giao (28/28), đảm nhận đồng đều 7 lượt kiểm thử độc lập T và 13 đến 15 lượt phối hợp C với khối lượng cân bằng giữa mã nguồn, huấn luyện GPU, báo cáo, slide và QA, cả nhóm thống nhất tỷ lệ đóng góp bằng nhau là 25.0% cho mỗi thành viên (tổng 100.0%).

**Bảng 1.2.** Chỉ số đóng góp và tỷ lệ đóng góp (nguồn `report/tables/T25_contribution.csv`).

<!-- BEGIN T25_contribution (tự sinh bởi src/t25_report_stats.py, không sửa tay) -->
| Mã | Họ và tên | Nhiệm vụ R (xong/tổng) | Commit có mã | Lần kiểm thử | Run chính thức | Lượt C | Lượt T | Đóng góp (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TV1 | Đoàn Trung Kiên | 7/7 | 47 | 9 | 17 | 14 | 7 | 25.0 |
| TV2 | Lê Phan Nguyên Hải | 8/8 | 62 | 8 | 32 | 13 | 7 | 25.0 |
| TV3 | Võ Hoàng Bảo | 7/7 | 55 | 7 | 16 | 14 | 7 | 25.0 |
| TV4 | Huỳnh Viễn Thông | 6/6 | 50 | 7 | 21 | 15 | 7 | 25.0 |
| Tổng |  | 28/28 | 214 | 31 | 86 | 56 | 28 | 100.0 |
<!-- END T25_contribution -->

Lưu ý khi đọc các chỉ số phụ trong Bảng 1.2: một commit lớn và một commit nhỏ được đếm như nhau, một run 100 epoch và một run dừng sớm cũng vậy, nên các cột số đếm được trình bày kèm để đối chiếu minh bạch tiến độ thực làm chứ không làm lệch đánh giá đóng góp đều giữa bốn thành viên.
