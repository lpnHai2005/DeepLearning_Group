# Kịch bản thuyết trình – Nhóm 10

Sinh tự động từ `src/slide_script.py` (cũng nằm trong phần Notes của `Slide_Nhom10.pptx`). Ngoặc tròn là hành động, không đọc. Tổng khoảng 13–15 phút.

## Slide 1

[TV4 · khoảng 40 giây]

Kính chào thầy và các bạn. Nhóm 10 xin trình bày bài tập lớn số 1: xấp xỉ hàm mật độ Gaussian hai chiều bằng mạng MLP viết trong PyTorch.

Câu hỏi chính của bài là: số mẫu huấn luyện, số nơ-ron của tầng ẩn và số tầng ẩn ảnh hưởng thế nào đến sai số. Ba yếu tố này tương ứng với ba thực nghiệm A, B và C.

Bài trình bày gồm bốn phần: bài toán và dữ liệu, mô hình và cách huấn luyện, kết quả ba thực nghiệm, và cuối cùng là thảo luận kèm demo chạy code.

Nguyên tắc nhóm giữ trong suốt bài là: mọi con số trên slide đều truy được về file log hoặc file CSV trong repo, và tập test chỉ được dùng ở bước đánh giá cuối.

Mời bạn TV1 trình bày phần bài toán.

## Slide 2

[TV1 · khoảng 60 giây]

Hàm cần xấp xỉ là mật độ Gaussian hai chiều, có tâm tại gốc tọa độ và ma trận hiệp phương sai C như trên slide.

(Chỉ vào công thức)
Điểm đầu tiên cần lưu ý: định thức của C bằng 0,35, nên giá trị lớn nhất tại tâm chỉ khoảng 0,269. Nhãn y vì vậy nằm trong khoảng từ 0 đến 0,269, chứ không phải từ 0 đến 1.

(Chỉ vào hình đường đồng mức)
Điểm thứ hai: phần tử ngoài đường chéo của C là âm 0,5, nên hệ số tương quan khoảng âm 0,645. Vì thế "đồi" Gaussian không tròn mà là một elip nghiêng, hẹp theo x1 và dài theo x2, lệch khoảng 15 độ so với trục x2.

Miền lấy mẫu từ âm 2 đến 2 theo x1 và từ âm 4 đến 4 theo x2 chứa 99,52% thể tích của hàm, nên gần như toàn bộ phần có ý nghĩa đều nằm trong miền.

Nhiệm vụ: xây dựng mạng nhận hai đầu vào x1, x2 và cho ra một giá trị y mũ. Đây là bài toán hồi quy.

Nếu bị hỏi "vì sao không chuẩn hóa y về 0 đến 1 bằng min-max": nhóm chia y cho 0,269 là hằng số tính từ đề bài, không ước lượng từ dữ liệu, nên không dùng thông tin của tập validation hay test.

## Slide 3

[TV1 · khoảng 60 giây]

Nhóm sinh ba tập dữ liệu độc lập, mỗi tập dùng một seed riêng.

(Chỉ vào ba thẻ số)
Tập train pool có 2000 điểm, seed 2026. Tập validation có 200 điểm, seed 2027, dùng cho early stopping và chọn cấu hình. Tập test có 100 điểm, seed 2028, chỉ dùng ở bước cuối cùng.

Ở thực nghiệm A, với mỗi giá trị A, nhóm lấy A điểm đầu tiên của train pool. Như vậy các tập lồng nhau: tăng A chỉ thêm điểm mới, không đổi điểm cũ, nên chênh lệch kết quả là do số mẫu, không phải do lấy mẫu khác nhau.

Nhóm lấy điểm đều trên toàn miền, vì mục tiêu là xấp xỉ hàm ở mọi nơi, kể cả vùng đuôi.

(Chỉ vào histogram của y và thẻ 62,6%)
Hệ quả quan trọng: 62,6% nhãn trong tập train nhỏ hơn 0,01, vì đỉnh chỉ chiếm một vùng hẹp. Một mô hình lúc nào cũng đoán giá trị trung bình đã có MSE trên validation khoảng 2,49 nhân 10 mũ trừ 3. Vì vậy MSE tuyệt đối luôn rất nhỏ; nhóm luôn so sánh với mô hình hằng số này và báo cáo thêm R bình phương cùng sai số trên lưới đều.

Tiếp theo, mời bạn TV2 trình bày phần mô hình.

## Slide 4

[TV2 · khoảng 60 giây]

Mô hình được viết bằng nn.Module và nn.Linear.

(Chỉ vào đoạn code)
Lớp MLP nhận một danh sách hidden_sizes, là số nơ-ron của từng tầng ẩn, cùng tên hàm kích hoạt. Vòng for tạo lần lượt từng cặp: một tầng Linear và một hàm kích hoạt. Tầng cuối là Linear ra một giá trị, không có hàm kích hoạt, vì đây là bài toán hồi quy.

Nhờ vậy, một lớp duy nhất dùng được cho cả ba thực nghiệm: A giữ một tầng 10 nơ-ron, B thay đổi số nơ-ron B, C thay đổi số tầng, mỗi tầng 10 nơ-ron.

(Chỉ vào hình sơ đồ)
Baseline là mạng 2, 10, 1 dùng sigmoid, có 41 tham số. Mạng một tầng có B nơ-ron thì có 4B cộng 1 tham số.

Một chi tiết nhỏ nhưng quan trọng: hàm forward gọi squeeze để trả về vectơ N phần tử. Nếu để đầu ra dạng N nhân 1 trong khi nhãn là N phần tử, MSELoss sẽ broadcast thành ma trận N nhân N. Loss vẫn giảm nhưng mô hình học ra một hằng số. Nhóm có thêm assert để chặn lỗi này.

Phần lan truyền ngược được kiểm tra trong tests/test_backprop.py: so gradient của autograd với lời giải tính tay và với đạo hàm số.

## Slide 5

[TV2 đọc đoạn 1–2, TV4 đọc đoạn 3 · khoảng 70 giây]

(TV2) Vòng huấn luyện do nhóm tự viết, theo đúng thứ tự đề yêu cầu. Mỗi epoch, hàm train_one_epoch gọi model.train(), rồi với mỗi batch: zero_grad, forward, tính loss, backward, và optimizer.step.

Sau mỗi epoch, hàm validate chạy trong model.eval() và torch.no_grad() để tính loss trên validation. Nếu loss tốt hơn mức tốt nhất, nhóm lưu một bản sao trọng số và ghi ra file best.pt. Nếu sau 200 epoch liên tiếp không cải thiện thì dừng sớm. Mỗi lần chạy lưu lại cấu hình, lịch sử loss theo epoch và chỉ số.

(TV4, chỉ vào sơ đồ, đi theo mũi tên)
Về tính công bằng: mỗi thực nghiệm chỉ thay đổi một yếu tố, mỗi cấu hình chạy 3 seed, và báo cáo trung bình cộng trừ độ lệch chuẩn. Mọi lựa chọn, từ siêu tham số, checkpoint đến mô hình cuối, đều dựa trên validation. (Chỉ vào mũi tên đỏ) Tập test chỉ đi vào bước đánh giá cuối, khi chạy với cờ eval-test. Có một file test tự động kiểm tra cấu hình được chọn đúng là cấu hình có lỗi validation thấp nhất trong log.

Nếu bị hỏi "các cột test trong bảng A, B, C có làm rò rỉ không": không. Đó là hậu kiểm, tính sau khi đã chọn xong và đóng băng mọi checkpoint, chỉ để trình bày đủ ba đường train, validation, test.

## Slide 6

[TV2 · khoảng 60 giây]

Trước khi chạy ba thực nghiệm, nhóm cần chốt cấu hình baseline.

(Chỉ vào biểu đồ bên trái)
Nhóm thử 3 hàm kích hoạt nhân 3 learning rate, mỗi tổ hợp 3 seed, tổng cộng 27 lần chạy, và chỉ nhìn validation. Sigmoid với learning rate 0,01 có lỗi trung bình thấp nhất, 1,89 nhân 10 mũ trừ 5. Nói thẳng: tanh với learning rate 0,003 đứng thứ hai và khoảng sai số của hai cấu hình chồng nhau, nên đây là chọn theo trung bình, chưa chứng minh sigmoid tốt hơn hẳn. ReLU kém nhất, vì 10 nơ-ron ReLU chỉ tạo được một bề mặt gấp khúc thô.

(Chỉ vào bảng bên phải)
Adam ban đầu được chọn theo thông lệ. Sau đó nhóm kiểm chứng lại bằng 6 optimizer, mỗi optimizer thử 7 learning rate, tổng 126 lần chạy. Adam vẫn tốt nhất; SGD thuần kém khoảng 19 lần. Vì không có optimizer nào tốt hơn, nhóm giữ nguyên baseline.

Batch size 32 và patience 200 cũng được kiểm chứng: patience 200 chỉ kém 0,7% so với không dừng sớm, còn patience 100 kém 31%.

Mời bạn TV1 trình bày thực nghiệm A.

## Slide 7

[TV1 · khoảng 70 giây]

Thực nghiệm A thay đổi số mẫu huấn luyện từ 200 đến 2000, bước nhảy 200, mạng giữ nguyên một tầng 10 nơ-ron.

(Chỉ vào biểu đồ trái)
Lỗi validation giảm từ 7,06 nhân 10 mũ trừ 5 ở A bằng 200 xuống 2,16 nhân 10 mũ trừ 5 ở A bằng 600. Sau đó đường cong chững lại và dao động. Train, validation và test luôn sát nhau, không thấy quá khớp.

Khi đọc log, nhóm phát hiện một điểm chưa công bằng. Batch size cố định 32, nên A bằng 200 chỉ có 7 bước cập nhật mỗi epoch, còn A bằng 1000 có 32 bước. Cùng 2000 epoch, A nhỏ được huấn luyện ít hơn khoảng 4,6 lần.

(Chỉ vào biểu đồ phải, đường đứt nét)
Nhóm chạy lại với cùng ngân sách khoảng 64 nghìn bước cho mọi A. Lỗi ở A bằng 200 giảm thêm 43%. Kết luận thứ nhất: một phần lỗi ở A nhỏ là do huấn luyện chưa đủ, không chỉ do ít dữ liệu. Kết luận thứ hai: sau khi cân bằng số bước, từ A bằng 600 trở đi lỗi gần như không giảm nữa, nghĩa là mạng 41 tham số đã chạm giới hạn dung lượng. Muốn giảm lỗi tiếp, phải tăng kích thước mạng, và đó là nội dung của thực nghiệm B và C.

Mời bạn TV3.

## Slide 8

[TV3 · khoảng 60 giây]

Thực nghiệm B giữ một tầng ẩn, A bằng 1000, và thay đổi số nơ-ron B từ 1 đến 10. Nhóm mở rộng thêm 16, 32, 64 và đến 93.

(Chỉ vào biểu đồ trái)
Lỗi validation giảm đều hơn 120 lần, từ 2,29 nhân 10 mũ trừ 3 ở B bằng 1, gần bằng mô hình hằng số, xuống 1,89 nhân 10 mũ trừ 5 ở B bằng 10.

(Chỉ vào các bề mặt bên phải)
Hình bề mặt giải thích vì sao. Với một nơ-ron, đầu ra chỉ là một "sườn dốc" sigmoid thay đổi theo một hướng, không thể tạo ra đỉnh. Với hai nơ-ron, mạng tạo được một dải chạy dọc trục chính nhưng chưa khép kín hai đầu. Phải từ khoảng năm nơ-ron, nhiều sườn dốc xoay theo các hướng khác nhau mới ghép lại thành đỉnh elip. Đó là hiện tượng thiếu khớp khi B nhỏ.

Khi tiếp tục mở rộng, lỗi vẫn giảm, đến 1,46 nhân 10 mũ trừ 6 ở B bằng 93, và không tăng trở lại. Khoảng cách giữa validation và train tiến về 0. Nghĩa là với 1000 mẫu, không có quá khớp.

## Slide 9

[TV3 · khoảng 60 giây]

Vì không thấy quá khớp ở 1000 mẫu, nhóm lặp lại toàn bộ thực nghiệm B với chỉ 200 mẫu. Ít dữ liệu thì mạng dễ học thuộc hơn.

(Chỉ vào biểu đồ phải, đường đứt nét nằm trên số 0)
Lần này có tín hiệu: từ B bằng 8 trở lên, lỗi validation luôn lớn hơn lỗi train, và tỉ số giữa hai lỗi tăng dần theo B, từ 1,30 ở B bằng 10 lên 1,74 ở B bằng 64. Mạng càng lớn so với số mẫu thì càng khớp tập train tốt hơn tập chưa thấy. Đây là dấu hiệu sớm của quá khớp.

Nhưng lỗi validation vẫn giảm theo B, nên chưa có đường chữ U như hình minh họa trong đề. Nhóm cho rằng có ba lý do: nhãn không có nhiễu, nên khớp đúng điểm train cũng là khớp đúng hàm; hàm Gaussian rất trơn; và early stopping luôn lấy checkpoint tốt nhất theo validation.

Hạn chế: 17 trên 39 lần chạy ở A bằng 200 chạm trần 2000 epoch khi vẫn đang cải thiện, nên khoảng cách này có thể còn rõ hơn nếu huấn luyện lâu hơn.

Mời bạn TV2 trình bày thực nghiệm C.

## Slide 10

[TV2 · khoảng 70 giây]

Thực nghiệm C thay đổi số tầng ẩn từ 1 đến 4, mỗi tầng 10 nơ-ron.

(Chỉ vào biểu đồ trái)
Từ 1 tầng lên 2 tầng, lỗi validation giảm khoảng 180 lần, từ 1,89 nhân 10 mũ trừ 5 xuống 1,05 nhân 10 mũ trừ 7. Từ 2 đến 4 tầng, lỗi ở cùng bậc 10 mũ trừ 7, và khoảng sai số của 3 tầng và 4 tầng chồng nhau. Train, validation và test gần bằng nhau ở mọi độ sâu.

Câu hỏi tự nhiên là: mạng sâu tốt hơn vì sâu, hay chỉ vì có nhiều tham số hơn? Mạng 4 tầng có 371 tham số, gấp 9 lần mạng 1 tầng.

(Chỉ vào biểu đồ phải)
Để trả lời, nhóm chạy thêm mạng một tầng có cùng số tham số, với 38, 65 và 93 nơ-ron. Ở cùng số tham số, mạng sâu vẫn tốt hơn mạng rộng 16 đến 28 lần. Vậy lợi ích đến từ độ sâu.

Cách hiểu trực quan: tầng đầu tạo ra các sườn dốc, tầng sau kết hợp phi tuyến các sườn dốc đó thành vùng bao quanh đỉnh, hiệu quả hơn nhiều so với cộng thật nhiều sườn dốc trong một tầng.

(Chỉ vào biểu đồ thời gian) Về thời gian, mỗi epoch tăng từ 20,2 lên 30,5 mili giây khi đi từ 1 lên 4 tầng, trong khi mạng một tầng 93 nơ-ron vẫn chỉ 20,2 mili giây. Với mạng nhỏ, thời gian phụ thuộc vào số tầng tính tuần tự, không phụ thuộc số tham số.

## Slide 11

[TV2 · khoảng 60 giây]

Báo cáo mẫu của môn kết luận rằng thêm tầng làm lỗi tăng, do vanishing gradient. Kết quả của nhóm ngược lại, nên nhóm đo trực tiếp để kiểm tra.

Đạo hàm của sigmoid không vượt quá 0,25. Mỗi khi lan ngược qua một tầng, gradient bị nhân thêm một hệ số nhỏ, nên tầng càng gần đầu vào thì gradient càng nhỏ.

(Chỉ vào các đường chấm, epoch 1)
Ở epoch đầu tiên, điều này đúng: với 4 tầng ẩn, gradient ở tầng đầu nhỏ hơn tầng ra khoảng 3600 lần. Vậy vanishing gradient có xảy ra.

(Chỉ vào các đường liền, trung vị)
Nhưng trong suốt quá trình huấn luyện, gradient của các tầng ẩn chỉ còn chênh nhau khoảng 2,5 lần, và mạng sâu vẫn đạt lỗi thấp nhất. Giả thuyết của nhóm: Adam chia bước cập nhật cho độ lớn gradient của từng tham số, nên tham số có gradient nhỏ vẫn được cập nhật với bước đủ lớn.

Nhóm nói rõ đây là giả thuyết, vì chưa chạy SGD thuần trên mạng sâu để kiểm chứng. Kết luận an toàn là: với Adam và tối đa 4 tầng, vanishing gradient không cản trở việc huấn luyện.

Mời bạn TV3 trình bày mô hình cuối.

## Slide 12

[TV3 · khoảng 60 giây]

Mô hình cuối được chọn bằng một quy tắc duy nhất: trong 53 cấu hình đã chạy, lấy cấu hình có lỗi validation trung bình thấp nhất. Kết quả là mạng 3 tầng ẩn, mỗi tầng 10 nơ-ron, 261 tham số. Cấu hình được đóng băng trong file final.json, sau đó mới đánh giá trên test, đúng một lần.

(Chỉ vào các thẻ số)
Trên tập test, MSE là 7,13 nhân 10 mũ trừ 8, R bình phương 0,99997. So với baseline một tầng, lỗi test giảm khoảng 320 lần.

Điểm nhóm muốn nhấn mạnh: lỗi test, 7,13 nhân 10 mũ trừ 8, rất sát lỗi validation, 6,02 nhân 10 mũ trừ 8. Như vậy việc chọn theo validation không làm mô hình khớp riêng tập validation.

Vì 100 điểm test có 61 điểm gần bằng 0, nhóm đánh giá thêm trên lưới đều 100 nhân 100 điểm phủ toàn miền: lỗi 7,34 nhân 10 mũ trừ 8, khớp với test.

(Chỉ vào hình y – y mũ, rồi hai bề mặt)
Các điểm test nằm sát đường y mũ bằng y, và bề mặt xấp xỉ gần như trùng bề mặt thật. Sai số lớn nhất trên lưới khoảng 1,07 nhân 10 mũ trừ 3, dưới 0,4% chiều cao đỉnh.

Mời bạn TV4 phần thảo luận.

## Slide 13

[TV4 · khoảng 60 giây]

Nhóm muốn trình bày trung thực cả những chỗ kết quả khác kỳ vọng và những hạn chế.

(Chỉ vào cột 1)
Thứ nhất, nhóm không thấy đường chữ U của quá khớp, vì ba lý do đã nêu: nhãn không nhiễu, hàm trơn, và early stopping theo validation.

(Chỉ vào cột 2)
Thứ hai, các hạn chế. Trần 2000 epoch làm từ 6,7% đến 50% số lần chạy bị cắt khi vẫn đang cải thiện, tùy thí nghiệm. Chỉ có 3 seed, nên nhiều chênh lệch nhỏ hơn độ lệch chuẩn và không kết luận được, ví dụ 3 tầng so với 4 tầng. Siêu tham số được tìm lần lượt từng cái, không thử mọi tổ hợp; Adam, batch 32 và patience 200 là chọn trước rồi mới kiểm chứng. Thời gian của 90 lần chạy bổ sung đo khi chạy song song nên nhóm không dùng để so sánh.

(Chỉ vào cột 3)
Thứ ba, hướng cải tiến: thêm nhiễu vào nhãn hoặc giảm số mẫu xuống 50 đến 100 để thử tìm chữ U; cố định tổng số bước cho mọi thí nghiệm; tăng lên 5 đến 10 seed; và so Adam với SGD trên mạng sâu.

(Chỉ vào dòng cuối) Cả 17 quyết định của nhóm đều được ghi lại kèm lý do và file minh chứng trong nhật ký quyết định.

## Slide 14

[TV4 tóm tắt, TV2 chạy demo · khoảng 60 giây + demo]

Tóm lại, nhóm rút ra bốn kết luận.

Một: thêm dữ liệu giúp giảm lỗi đến khoảng 600 mẫu rồi chững lại, và một phần lỗi ở A nhỏ là do huấn luyện ít bước.
Hai: B nhỏ thì thiếu khớp rõ; với 1000 mẫu không có quá khớp, với 200 mẫu chỉ thấy khoảng cách train và validation tăng dần.
Ba: thêm tầng giảm lỗi khoảng 180 lần, và mạng sâu tốt hơn mạng rộng cùng số tham số 16 đến 28 lần.
Bốn: mô hình cuối 3 tầng đạt lỗi test 7,13 nhân 10 mũ trừ 8, được chọn hoàn toàn bằng validation.

Câu chốt: với hàm này, độ sâu quan trọng hơn độ rộng.

Bây giờ nhóm xin demo chạy code. Mời bạn TV2.

(TV2 mở terminal ở thư mục BTL01, gõ lệnh trên slide, chờ khoảng 10 giây)
(TV2) Lệnh này chạy cấu hình demo: một seed, 300 epoch. Trên màn hình là loss train và validation theo epoch. (Mở thư mục results/demo) Mỗi lần chạy lưu config.json, history.csv chứa loss từng epoch, metrics.json chứa chỉ số, và best.pt là checkpoint tốt nhất theo validation.

(TV4) Phần trình bày của nhóm 10 đến đây là kết thúc. Cảm ơn thầy và các bạn đã lắng nghe. Nhóm rất mong nhận được câu hỏi và góp ý.

Nếu bị hỏi "sao biết test không bị dùng để chọn": chỉ lệnh có cờ eval-test mới tính test; file tests/test_config_evidence.py kiểm tra final.json khớp với cấu hình có lỗi validation thấp nhất trong log.
Nếu bị hỏi "sao chọn sigmoid mà không phải ReLU": theo lưới 27 lần chạy, ReLU kém nhất ở mọi learning rate, vì 10 nơ-ron ReLU chỉ tạo bề mặt gấp khúc thô.
