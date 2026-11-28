# Chương 8. Kết luận

Nhóm đã xây dựng bằng PyTorch toàn bộ quy trình nhận dạng bảy cảm xúc trên FER2013, từ kiểm tra dữ liệu, huấn luyện, đánh giá đến trực quan hóa, và đáp ứng bốn yêu cầu của đề bài như sau.

1. Tóm tắt mã nguồn: Chương 5 mô tả các module, người phụ trách và các đoạn code cốt lõi; mọi file code ghi rõ người phụ trách, và 127 test tự động đạt trên bản clone sạch.
2. Nhiều thực nghiệm với nhiều bộ tham số: 86 run huấn luyện chính thức chia thành baseline và bảy nhóm E1 đến E7, mỗi biến thể chỉ đổi một yếu tố so với baseline; baseline và mô hình cuối có ba seed để báo cáo mean ± std.
3. Trực quan hóa bộ lọc tích chập cuối: Chương 6 trình bày trọng số, bản đồ đặc trưng, activation maximization và Grad-CAM của lớp `conv4_4`, cho thấy mô hình dựa vào vùng mắt, lông mày, mũi và miệng.
4. Phân tích và giải thích: mỗi nhóm thực nghiệm có phần phân tích riêng, Chương 7 tổng hợp cách hiểu của nhóm, các trường hợp không như kỳ vọng và hạn chế.

Mô hình cuối là ensemble ba seed của CNN 16 lớp Conv có BatchNorm (`large_bn`, 20.4 M tham số mỗi mô hình) cộng TTA lật ngang, huấn luyện từ đầu trên FER2013. Mô hình đạt 72.50% trên PublicTest và 73.64% trên PrivateTest (macro-F1 73.22%), vượt mốc 69% và 71% của cuộc thi lần lượt 3.50 và 2.64 điểm %, và vượt các CNN của báo cáo mẫu (0.64 đến 0.67). PrivateTest chỉ được đánh giá một lần, có file khóa làm bằng chứng.

Ba kết luận chính rút ra từ thực nghiệm: BatchNorm là điều kiện để mạng sâu học được với khởi tạo mặc định của PyTorch; cải thiện đo trên một kiến trúc không tự chuyển sang kiến trúc khác, nên phải kiểm lại từng tổ hợp và so với độ lệch chuẩn giữa các seed; ensemble nhiều seed là cách tăng độ chính xác rẻ và ổn định nhất. Lỗi còn lại tập trung ở cụm Angry, Fear, Sad, Neutral, nơi biểu cảm tương tự nhau và nhãn có nhiễu. Hướng phát triển tiếp theo là điều chuẩn cho mô hình sâu, ensemble khác kiến trúc, căn chỉnh khuôn mặt và xử lý nhiễu nhãn.
