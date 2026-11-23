"""Sinh các bảng theo dõi BTL02 (CSV + tracking.xlsx để nhập vào Google Sheets).

Phụ trách: TV4 (nhóm trưởng).
Nguồn duy nhất của ma trận phân công là TASKS bên dưới. Các CSV do thành viên điền (test_log,
experiments, weights, figures_manifest) do thành viên điền; script chỉ tạo file nếu chưa có,
KHÔNG ghi đè dữ liệu đã nhập. Chạy: python tracking/build_tracking.py
"""
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
MEMBERS = ["TV1", "TV2", "TV3", "TV4"]
MEMBER_NAMES = {"TV1": "Đoàn Trung Kiên", "TV2": "Lê Phan Nguyên Hải", "TV3": "Võ Hoàng Bảo", "TV4": "Huỳnh Viễn Thông"}
STATUS = ["Pending", "In Progress", "Completed", "Failed", "Verified"]
TODAY = "2026-11-20"

# Quy tắc kiểm thử dùng chung, mỗi nhiệm vụ nối thêm phần riêng.
COMMON = "Làm trên bản clone sạch; không dựa vào lời báo cáo của người thực hiện."

# id, module, giai đoạn, tên, R, việc của 2 người phối hợp (mẫu "{C1}: …; {C2}: …", tên điền khi sinh bảng), bằng chứng bắt buộc, cách tester kiểm tra, phụ thuộc
TASKS = [
    ("T01", "M0 Quản trị & truy vết", "G1", "Phân tích đề, lập kế hoạch và dựng khung theo dõi (Sheets, schema CSV, quy ước log)", "TV4",
     "{C1}: soát schema log huấn luyện và weights (tab Thuc_nghiem); {C2}: soát schema đánh giá và hình (tab Kiem_thu, Hinh_ve), mỗi người ghi góp ý vào commit mã T01",
     "PHAN_CONG_NHIEM_VU.md; tracking/*.csv; link Google Sheets và Slides",
     "Mở Sheets, đối chiếu từng tab với CSV; kiểm mỗi nhiệm vụ có đúng 1 R, 2 C, 1 T và T khác R/C; chạy lại build_tracking.py ra CSV trùng", ""),
    ("T03", "M1 Dữ liệu", "G1", "Tải fer2013.csv và kiểm tra toàn vẹn (số dòng, 3 giá trị Usage, sha256)", "TV1",
     "{C1}: tải fer2013.csv từ Kaggle và tính sha256; {C2}: tải từ Google Drive của giảng viên và tính sha256, hai người so với R",
     "results/data_check.json do script ghi (số dòng theo Usage, sha256, ngày tải)",
     "Tester tự tính sha256 và đếm dòng theo Usage bằng lệnh riêng, so với data_check.json", "T02"),
    ("T04", "M1 Dữ liệu", "G1", "fer2013.py, loader.py, augment.py và tests (nhãn, không rò rỉ test)", "TV1",
     "{C1}: viết test nhãn, split, chuẩn hóa trong tests/; {C2}: dùng thử loader trên máy riêng và báo lỗi",
     "src/data/*.py; log pytest; commit",
     "pytest trên bản clone sạch; tự gọi load_fer2013 xác nhận kích thước 3 tập và mean/std chỉ tính trên Training", "T03"),
    ("T05", "M1 Dữ liệu", "G1", "EDA: phân bố lớp 3 tập, lưới ảnh mẫu, ảnh trung bình, ảnh lỗi, minh họa augmentation", "TV1",
     "{C1}: soát nhãn của lưới ảnh khớp nhãn từng tập; {C2}: soát kiểu hình và ghi vào Hinh_ve",
     "src/data/eda.py; figures/eda_*.png kèm figures/eda_*.csv; dòng trong Hinh_ve",
     "Chạy lại eda.py, so từng cột trong CSV với số đếm tính độc lập; xem nhãn lưới ảnh có khớp nhãn tập tương ứng", "T04"),
    ("T06", "M2 Mô hình & hạ tầng huấn luyện", "G1", "cnn.py và transfer.py (in số tham số, kích thước bản đồ đặc trưng từng khối)", "TV2",
     "{C1}: viết test shape đầu ra và số tham số; {C2}: kiểm tay kích thước feature map 48→24→12→6→3→1",
     "src/models/*.py; results/model_summary.csv sinh từ code",
     "pytest; tự đếm tham số bằng sum(p.numel()) và so với model_summary.csv", "T04"),
    ("T07", "M2 Mô hình & hạ tầng huấn luyện", "G1-G2", "trainer.py và run_experiment.py (history.csv, run.log, ghi experiments.csv, tự quyết giữ/xóa weights theo luật)", "TV2",
     "{C1}: nối loader và chạy thử 1 epoch trên máy riêng; {C2}: nối evaluate/plots vào run_dir",
     "src/engine/trainer.py; 1 run_dir hoàn chỉnh trong results/_qa/ có config.json ghi member/hostname/giờ bắt đầu-kết thúc/tổng giây; test 1 bước train giảm loss",
     "Chạy 2 epoch cùng seed hai lần, so history; tạo run lỗi cố ý (config sai) và kiểm có dòng Failed, không còn best.pt dở", "T04, T06"),
    ("T08", "M2 Mô hình & hạ tầng huấn luyện", "G2", "evaluate.py và plots.py (Train/Val loss theo epoch, acc, confusion matrix, ca đúng/sai)", "TV3",
     "{C1}: viết test metric (dữ liệu giả lập chỉ nằm trong unit test); {C2}: thống nhất style hình cho mọi người dùng",
     "src/engine/evaluate.py; src/visual/plots.py; mỗi hình kèm file CSV dữ liệu vẽ",
     "So metric với sklearn.metrics trên cùng dự đoán; so CSV dữ liệu vẽ với history.csv từng epoch", "T07"),
    ("T09", "M2 Mô hình & hạ tầng huấn luyện", "G2", "make_tables.py: experiments.csv → bảng báo cáo E1–E7, mean ± std", "TV3",
     "{C1}: kiểm công thức mean/std; {C2}: soát định dạng bảng đưa vào báo cáo",
     "src/make_tables.py; report/tables/*.md|csv",
     "Tính lại 3 bảng bất kỳ bằng pandas độc lập, diff với bảng sinh ra", "T07"),
    ("T10", "M2 Mô hình & hạ tầng huấn luyện", "G2", "Notebook tái lập (Restart & Run All, log, output lưu lại)", "TV3",
     "{C1}: chạy notebook Restart & Run All trên máy mình; {C2}: soát lời giải thích từng bước",
     "notebooks/*.ipynb; notebooks/executed/<exp_id>.ipynb; file log liên kết",
     "Kernel Restart & Run All trên máy tester, không lỗi, không dùng biến cũ; số trong output khớp history.csv", "T07, T08"),
    ("T11", "M3 Baseline", "G2", "Baseline 3 seed (0/1/2) và đóng băng configs/baseline.json", "TV2",
     "{C1}: chạy seed 1 trên máy riêng với đúng config; {C2}: chạy seed 2 trên máy riêng với đúng config (R chạy seed 0)",
     "3 run_dir; dòng trong experiments.csv; sha256 baseline.json; mean ± std",
     "Chạy lại một seed rút gọn và so với epoch đầu của history; tính lại mean ± std; kiểm sha256 baseline.json. Dung sai tái lập = std 3 seed (chưa có số trước khi T11 xong)", "T04, T06, T07, T08"),
    ("T12", "M4 Thực nghiệm E1–E6", "G3", "E1 Kiến trúc (SMALL/MEDIUM/MEDIUM_01/LARGE, ×0.5/×2, BN, GAP)", "TV2",
     "{C1}: chạy SMALL, MEDIUM, MEDIUM_01, LARGE; {C2}: chạy ×0.5/×2 kênh, có/không BN, Flatten vs GAP, mỗi người viết nhận xét độc lập cho ít nhất 1 kết quả của mình",
     "run_dir từng cấu hình; experiments.csv; Train/Val loss; mục nhận xét",
     "Tester tự chọn 1 cấu hình (không do R chỉ định) chạy lại và so với metrics.json trong dung sai; tính lại bảng; kiểm nhận xét có khớp số", "T11"),
    ("T13", "M4 Thực nghiệm E1–E6", "G3", "E2 Activation (ReLU, LeakyReLU, ELU, GELU, Tanh, Sigmoid) và chuẩn gradient từng lớp", "TV2",
     "{C1}: chạy ReLU, LeakyReLU, ELU, GELU; {C2}: chạy Tanh, Sigmoid và đo chuẩn gradient từng lớp",
     "run_dir; experiments.csv; file CSV chuẩn gradient; hình sinh từ code",
     "Chạy lại 1 activation do tester chọn; kiểm file chuẩn gradient bằng lệnh riêng", "T11"),
    ("T14", "M4 Thực nghiệm E1–E6", "G3", "E3 Tối ưu (optimizer × lr, lịch lr, batch size)", "TV3",
     "{C1}: chạy lưới optimizer × lr của SGD, SGD+Nesterov, Adam; {C2}: chạy AdamW, RMSprop, lịch lr và batch size",
     "run_dir; experiments.csv; hình so sánh run từ plots.py",
     "Chạy lại 1 cấu hình do tester chọn; tính lại bảng; kiểm các run chỉ đổi 1 yếu tố so với baseline", "T11"),
    ("T15", "M4 Thực nghiệm E1–E6", "G3", "E4 Augmentation và điều chuẩn (dropout, weight decay, label smoothing)", "TV1",
     "{C1}: chạy nhóm augmentation (không aug, flip, flip+affine, +RandomErasing) và dropout; {C2}: chạy weight decay, label smoothing và vẽ đường cong train/val so overfit từ history.csv",
     "run_dir; experiments.csv; hình overfit sinh từ code",
     "Chạy lại 1 cấu hình do tester chọn; kiểm khoảng cách train/val trong CSV khớp hình", "T11"),
    ("T16", "M4 Thực nghiệm E1–E6", "G3", "E5 Mất cân bằng lớp (CE thường / class weight / WeightedRandomSampler)", "TV1",
     "{C1}: chạy CE thường và CE có trọng số lớp; {C2}: chạy WeightedRandomSampler và tính recall từng lớp độc lập",
     "run_dir; experiments.csv; recall từng lớp từ evaluate.py",
     "Tính lại recall Disgust/Fear từ dự đoán lưu trong run_dir", "T11"),
    ("T17", "M4 Thực nghiệm E1–E6", "G3", "E6 Transfer learning (VGG16 ở 48 px, ResNet18 từ đầu/pretrained, đóng băng/fine-tune)", "TV2",
     "{C1}: chạy VGG16 ở 48 px và ResNet18 từ đầu; {C2}: chạy ResNet18 pretrained (đóng băng, fine-tune) và kiểm giải thích kích thước feature map của VGG16",
     "run_dir; experiments.csv; model_summary cho VGG16 48 px",
     "Tester tự tính kích thước feature map VGG16 ở 48 px; chạy lại 1 cấu hình; kiểm weights pretrained tải đúng nguồn", "T11"),
    ("T18", "M5 Mô hình cuối & trực quan hóa", "G4", "E7 Mô hình cuối: cấu hình tốt nhất theo val × 3 seed, TTA, ensemble", "TV3",
     "{C1}: chạy seed 1 và kiểm ensemble 3 seed độc lập; {C2}: chạy seed 2 và kiểm TTA (flip, ten-crop) độc lập (R chạy seed 0)",
     "3 run_dir; experiments.csv; bảng mean ± std; kiểm TTA",
     "Tính lại mean ± std và ensemble từ dự đoán lưu; chạy lại 1 seed rút gọn", "T12, T13, T14, T15, T16, T17"),
    ("T19", "M5 Mô hình cuối & trực quan hóa", "G4", "Đánh giá PrivateTest đúng 1 lần với mô hình cuối", "TV3",
     "{C1}: chứng kiến lệnh --eval-test và ghi nhật ký (giờ, lệnh); {C2}: kiểm TEST_DONE.json được tạo và lệnh chạy lần hai bị từ chối",
     "results/E7/TEST_DONE.json; run.log lệnh test; dòng trong experiments.csv",
     "Xác nhận chỉ có 1 lần gọi test (file khóa và log); tính lại acc/macro-F1 từ dự đoán đã lưu", "T18"),
    ("T20", "M5 Mô hình cuối & trực quan hóa", "G4", "Trực quan hóa bộ lọc conv cuối (trọng số, feature map, activation maximization, Grad-CAM)", "TV4",
     "{C1}: cung cấp hook/lớp conv cuối; {C2}: chọn ca đúng/sai cho Grad-CAM từ dự đoán đã lưu",
     "src/visual/filters.py; figures/filters_*.png kèm CSV/npz đầu vào; Hinh_ve",
     "Chạy lại visualize_filters.py với best.pt, so hình; kiểm tên lớp và shape trọng số", "T18"),
    ("T25", "M6 Báo cáo, slide, QA, nộp", "G5", "Báo cáo Ch.1, 5, 6, 7, 8 và ghép toàn bộ", "TV4",
     "{C1}: review chéo Ch.2–4 và dò từng số về nguồn; {C2}: review chéo Ch.5–8 và dò từng số về nguồn",
     "report/BaoCao_BTL02.docx/.pdf; bảng phân công % đóng góp lấy từ Tien_do",
     "Chọn ngẫu nhiên ít nhất 10 số liệu/hình, truy vết về experiments.csv hoặc Hinh_ve; số không truy vết = Fail", "T20, T21, T22, T23, T24"),
    ("T21", "M6 Báo cáo, slide, QA, nộp", "G5", "Báo cáo Ch.2 Dữ liệu và mục E4, E5", "TV1",
     "{C1}: review chéo Ch.2 và dò số liệu về nguồn; {C2}: review chéo mục E4, E5 và dò số liệu về nguồn", "Chương 2 + bảng/hình sinh từ code",
     "Chọn ngẫu nhiên ít nhất 10 số liệu/hình truy vết về nguồn; số không truy vết = Fail", "T05, T15, T16"),
    ("T22", "M6 Báo cáo, slide, QA, nộp", "G5", "Báo cáo Ch.3 Mô hình & huấn luyện và mục E1, E2, E6", "TV2",
     "{C1}: review chéo Ch.3 và dò số liệu về nguồn; {C2}: review chéo mục E1, E2, E6 và dò số liệu về nguồn", "Chương 3 + bảng/hình sinh từ code",
     "Chọn ngẫu nhiên ít nhất 10 số liệu/hình truy vết về nguồn; số không truy vết = Fail", "T12, T13, T17"),
    ("T23", "M6 Báo cáo, slide, QA, nộp", "G5", "Báo cáo Ch.4 Đánh giá, mục E3, E7 và phân tích lỗi", "TV3",
     "{C1}: review chéo Ch.4 và mục E3, dò số liệu về nguồn; {C2}: review chéo mục E7 và phân tích lỗi, dò số liệu về nguồn", "Chương 4 + bảng/hình sinh từ code",
     "Chọn ngẫu nhiên ít nhất 10 số liệu/hình truy vết về nguồn; số không truy vết = Fail", "T14, T18, T19"),
    ("T26", "M6 Báo cáo, slide, QA, nộp", "G5", "Slide .pptx/.pdf (Google Slides), kịch bản thuyết trình và chuẩn bị demo chạy code", "TV4",
     "{C1}: viết 2–3 slide phần mình phụ trách và dò số liệu về nguồn; {C2}: viết 2–3 slide phần mình phụ trách, cùng R tập demo chạy code",
     "report/Slide_BTL02.pptx/.pdf; link Google Slides; kịch bản; demo chạy được từ bản clone sạch",
     "Dò mọi số/hình trên slide về nguồn; kiểm không còn slide minh họa số giả", "T25"),
    ("T27", "M6 Báo cáo, slide, QA, nộp", "G5", "QA tái lập toàn repo (clone sạch, venv mới, theo README, pytest, chạy notebook)", "TV4",
     "{C1}: chạy clone sạch, venv mới, theo README và pytest trên máy riêng; {C2}: chạy các notebook trên máy riêng, mỗi người báo lỗi cho người phụ trách",
     "Biên bản QA trong Kiem_thu; danh sách lỗi và commit sửa",
     "Tester không tham gia viết code được QA; tự clone và chạy theo README, báo lỗi", "T19, T20"),
    ("T28", "M6 Báo cáo, slide, QA, nộp", "G5", "Đóng gói BTLON02_Emotion_HoTen_MaSV.zip (họ tên, MSSV nhóm trưởng) và nộp bài", "TV2",
     "{C1}: kiểm checklist mục 8 phần code, README, requirements, không có fer2013.csv và checkpoint trung gian; {C2}: kiểm phần report, slide, nhật ký thực nghiệm, tracking/*.csv", "File zip (gồm code, report, slide, nhật ký thực nghiệm, tracking/*.csv + log); checklist mục 8",
     "Giải nén vào thư mục trống, chạy checklist; kiểm không có fer2013.csv và checkpoint trung gian", "T25, T26, T27"),
    ("T02", "M0 Quản trị & truy vết", "G1", "Khung repo: cây thư mục, requirements.txt, README (cài đặt, tải dữ liệu, lệnh chạy, bảng phân công theo file), quy ước docstring Phụ trách và commit", "TV1",
     "{C1}: cài đặt và chạy theo README trên máy riêng, báo chỗ sai; {C2}: cài trên máy thứ hai và kiểm bảng phân công theo file khớp tab Phan_cong",
     "README.md; requirements.txt; cây thư mục theo mục 4; docstring mẫu",
     "Clone sạch, venv mới, pip install -r requirements.txt, chạy lệnh trong README; kiểm mỗi file code có dòng Phụ trách khớp Phan_cong", "T01"),
    ("T24", "M6 Báo cáo, slide, QA, nộp", "G5", "Nhật ký thực nghiệm NHAT_KY_THUC_NGHIEM.md (giả thuyết, nhận xét, quyết định; kết quả tự điền từ experiments.csv) và sơ đồ kiến trúc/pipeline vẽ từ model thật", "TV4",
     "{C1}: viết nhận xét và quyết định cho các run E1, E2, E3; {C2}: viết nhận xét và quyết định cho các run E4, E5, E6, mỗi người đối chiếu số với experiments.csv",
     "report/NHAT_KY_THUC_NGHIEM.md; hình sơ đồ kèm script và dòng Hinh_ve",
     "Chọn ngẫu nhiên ít nhất 10 run trong nhật ký, so từng số với experiments.csv và history.csv; kiểm có nhận xét cho cả run lỗi hoặc metric thấp", "T12, T13, T14, T15, T16, T17, T19, T20"),
]

VIEC_R = {  # việc chính của R, tên điền khi sinh bảng
    "T01": "{R}: đọc đề, viết phân tích và kế hoạch (PHAN_CONG_NHIEM_VU.md), dựng schema các CSV và Google Sheets, soạn quy ước log",
    "T02": "{R}: dựng cây thư mục, requirements.txt, README (cài đặt, tải dữ liệu, lệnh chạy, bảng phân công theo file), viết quy ước docstring Phụ trách và commit",
    "T03": "{R}: tải fer2013.csv, viết script kiểm tra (số dòng theo Usage, sha256) và ghi results/data_check.json",
    "T04": "{R}: viết fer2013.py (EMOTIONS, load_fer2013), loader.py (mean/std chỉ từ Training, nạp RAM), augment.py (HFlip, RandomAffine), sửa theo test và góp ý của hai C",
    "T05": "{R}: viết eda.py: phân bố lớp 3 tập, lưới ảnh mẫu, ảnh trung bình, thống kê độ sáng, đếm ảnh lỗi, minh họa augmentation, mỗi hình kèm CSV",
    "T06": "{R}: viết cnn.py (VGG-style cấu hình được, 4 cấu hình mẫu) và transfer.py (ResNet18/VGG16 thay lớp cuối 7 lớp), in số tham số và kích thước feature map từng khối, ghi model_summary.csv",
    "T07": "{R}: viết trainer.py (train/eval, AMP, scheduler, early stopping, best.pt, history.csv, run.log) và run_experiment.py (đọc config JSON, lặp giá trị × seed, ghi experiments.csv và config.json)",
    "T08": "{R}: viết evaluate.py (accuracy, macro-F1, recall từng lớp, ma trận nhầm lẫn, TTA) và plots.py (đường cong loss/acc, confusion, lưới ca đúng/sai, histogram ca sai), mỗi hình kèm CSV",
    "T09": "{R}: viết make_tables.py đọc experiments.csv và sinh bảng E1–E7 dạng markdown/Word với mean ± std",
    "T10": "{R}: viết notebook tái lập gọi code trong src/, chạy Restart & Run All và lưu bản có output vào notebooks/executed/",
    "T11": "{R}: chạy baseline seed 0 trên máy mình, tổng hợp 3 seed (mean ± std), cùng nhóm chốt cấu hình theo val và đóng băng configs/baseline.json",
    "T12": "{R}: viết configs/E1_arch.json, chia cấu hình cho hai C, gom run của cả ba người vào experiments.csv, tổng hợp bảng và kết luận E1",
    "T13": "{R}: viết configs/E2_activation.json (có và không BN), gom run, tổng hợp bảng, kết luận về vanishing gradient của Sigmoid",
    "T14": "{R}: viết configs/E3_optim.json (optimizer × lr, lịch lr, batch size), gom run, tổng hợp bảng, chọn optimizer và lr tốt nhất theo val",
    "T15": "{R}: viết configs/E4_aug_reg.json, gom run, tổng hợp bảng và kết luận mức giảm overfit của augmentation, dropout, weight decay, label smoothing",
    "T16": "{R}: viết configs/E5_imbalance.json, gom run, tổng hợp bảng, kết luận về recall Disgust/Fear và đánh đổi accuracy",
    "T17": "{R}: viết configs/E6_transfer.json, giải thích \"fail case\" VGG16 bằng kích thước feature map, tổng hợp bảng ResNet18 từ đầu so với pretrained",
    "T18": "{R}: chọn cấu hình tốt nhất theo val từ E1–E6, chạy seed 0, ghép 3 seed và ensemble, tính TTA, ghi mean ± std",
    "T19": "{R}: chạy python src/run_experiment.py --eval-test đúng 1 lần, lưu TEST_DONE.json, so kết quả với mốc 69%/71%",
    "T20": "{R}: viết filters.py và visualize_filters.py (trọng số conv đầu/cuối, feature map, activation maximization, Grad-CAM) và chạy trên best.pt",
    "T21": "{R}: viết Ch.2 Dữ liệu và mục E4, E5 bằng bảng và hình sinh từ code",
    "T22": "{R}: viết Ch.3 Mô hình & huấn luyện và mục E1, E2, E6",
    "T23": "{R}: viết Ch.4 Đánh giá, mục E3, E7 và phân tích lỗi (cặp lớp hay nhầm, ca sai điển hình)",
    "T24": "{R}: lập mẫu NHAT_KY_THUC_NGHIEM.md, viết giả thuyết, nhận xét, quyết định cho run E7, vẽ sơ đồ kiến trúc và pipeline từ model thật, kiểm khớp với experiments.csv",
    "T25": "{R}: viết Ch.1, 5, 6, 7, 8, ghép toàn bộ vào BaoCao_BTL02, xuất .docx/.pdf, điền bảng % đóng góp từ Tien_do",
    "T26": "{R}: làm slide 12–15 trang, kịch bản thuyết trình, bộ câu hỏi Q&A dự kiến, chuẩn bị demo chạy code",
    "T27": "{R}: clone sạch, tạo venv mới, chạy theo README, pytest, chạy notebook, ghi biên bản QA và danh sách lỗi",
    "T28": "{R}: gom file theo checklist mục 8, tạo BTLON02_Emotion_HoTen_MaSV.zip (họ tên, MSSV của nhóm trưởng) và nộp bài",
}

TASKS.sort(key=lambda t: t[0])

# Trạng thái hiện tại chỉ theo bằng chứng thật. Chưa nhiệm vụ nào được Verified.
CURRENT_STATUS = {"T01": "In Progress"}

HEADERS = {
    "Kiem_thu": ["test_id", "thoi_gian", "task_id", "exp_id", "nguoi_kiem_thu", "nguoi_thuc_hien",
                 "phuong_phap", "lenh_da_chay", "so_lieu_goc", "so_lieu_tai_tinh", "ket_qua", "bang_chung", "nhan_xet"],
    "Thuc_nghiem": ["exp_id", "task_id", "thanh_vien", "bat_dau", "ket_thuc", "tong_thoi_gian_s", "epoch_time_s_tb", "may_chay", "config_path", "hyperparams_json",
                    "dataset_sha256", "seed", "git_commit", "n_params", "train_loss_cuoi", "train_acc_cuoi", "val_loss_cuoi", "best_val_loss", "best_epoch", "val_acc",
                    "val_macro_f1", "test_acc", "trang_thai_ky_thuat", "so_voi_baseline_tu_tinh", "weights_path", "weights_sha256", "weights_size_mb", "weights_quyet_dinh", "weights_ma_ly_do", "run_dir",
                    "log_path", "history_path", "curve_path", "notebook_path"],
    "Hinh_ve": ["fig_id", "file_hinh", "file_du_lieu_dau_vao", "script_sinh_hinh", "lenh_chay", "exp_ids",
                "task_id", "thoi_gian_sinh", "sha256_hinh", "dung_o"],
}
LOG_FILES = {"Kiem_thu": "test_log.csv", "Thuc_nghiem": "experiments.csv", "Hinh_ve": "figures_manifest.csv"}


def assign_testers():
    """T xoay vòng: mỗi nhiệm vụ chọn người ít được giao test nhất trong 3 người không phải R."""
    count = {m: 0 for m in MEMBERS}
    rows = []
    for i, t in enumerate(TASKS):
        r = t[4]
        others = [m for m in MEMBERS if m != r]
        order = others[i % 3:] + others[:i % 3]
        tester = min(order, key=lambda m: (count[m], order.index(m)))
        count[tester] += 1
        cs = [m for m in MEMBERS if m not in (r, tester)]
        t = t[:5] + (t[5].format(C1=cs[0], C2=cs[1]),) + t[6:]  # việc của C ghi tên người cụ thể
        rows.append(t + (cs[0], cs[1], tester))
    return rows


def check(rows):
    for t in rows:
        assert len({t[4], t[-3], t[-2], t[-1]}) == 4, f"{t[0]}: R/C/C/T phải là 4 người khác nhau"
    ids = [t[0] for t in rows]
    assert len(ids) == len(set(ids))
    for t in rows:
        for d in [x.strip() for x in t[8].split(",") if x.strip()]:
            assert d in ids and d < t[0], f"{t[0]}: phụ thuộc {d} không hợp lệ"
    t_cnt = {m: sum(1 for t in rows if t[-1] == m) for m in MEMBERS}
    assert max(t_cnt.values()) - min(t_cnt.values()) <= 1, t_cnt


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:  # không BOM: Sheets IMPORTDATA đọc trực tiếp các CSV này
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    rows = assign_testers()
    check(rows)
    pc_header = ["task_id", "module", "giai_doan", "ten_nhiem_vu", "R_chiu_trach_nhiem", "C1", "C2", "T_kiem_thu",
                 "viec_cua_R", "viec_cua_C", "bang_chung_can_co", "cach_tester_kiem_tra", "phu_thuoc", "trang_thai", "ngay_cap_nhat"]
    pc = [[t[0], t[1], t[2], t[3], t[4], t[-3], t[-2], t[-1], VIEC_R[t[0]].format(R=t[4]), t[5], t[6], t[7] + " " + COMMON,
           t[8], CURRENT_STATUS.get(t[0], "Pending"), TODAY] for t in rows]
    write_csv(HERE / "tasks.csv", pc_header, pc)

    for name, fn in LOG_FILES.items():
        p = HERE / fn
        if not p.exists():
            write_csv(p, HEADERS[name], [])

    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="1F4E78")

    def sheet(name, header, data, widths):
        ws = wb.create_sheet(name)
        ws.append(header)
        for r in data:
            ws.append(list(r))
        for c in ws[1]:
            c.font, c.fill = bold, fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for i, w in enumerate(widths):
            ws.column_dimensions[ws.cell(1, i + 1).column_letter].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        return ws

    # Huong_dan
    guide = [
        ("Thanh_vien", "Mã TV1–TV4 ↔ họ tên; MSSV, máy, GPU do từng người điền (run_experiment đối chiếu hostname với tab này)."),
        ("Phan_cong", "Nguồn gốc của ma trận. Mỗi nhiệm vụ đúng 1 R, 2 C, 1 T là 4 người khác nhau; T xoay vòng."),
        ("Ma_tran", "Chiều ngang = module/giai đoạn, chiều dọc = vai trò R/C/T từng người. Công thức lấy từ Phan_cong."),
        ("Tien_do", "Đếm bằng công thức từ Phan_cong và Kiem_thu. Không nhập tay."),
        ("Nhật ký hoạt động", "Không có tab nhập tay. Tự sinh từ git (tracking/git_activity.py; mã nhiệm vụ nằm trong thông điệp commit) và từ run.log/Thuc_nghiem do code tự ghi."),
        ("Kiem_thu", "Tester ghi lệnh đã chạy, số liệu gốc, số tái tính, Pass/Fail. Nhiệm vụ chỉ thành Verified khi có dòng Pass."),
        ("Thuc_nghiem", "Mỗi lần train một dòng, kể cả run lỗi hoặc metric thấp. trang_thai_ky_thuat tách khỏi danh_gia_chat_luong."),
        ("Weights", "Không có tab riêng: code tự quyết giữ/xóa weights theo luật và ghi vào các cột weights_* của Thuc_nghiem (xem PHAN_CONG_NHIEM_VU.md mục 9.3)."),
        ("Hinh_ve", "Mỗi hình kèm script, file dữ liệu đầu vào và lệnh chạy."),
        ("Trạng thái", " / ".join(STATUS)),
        ("Dữ liệu giả", "Cấm. Ô chưa có kết quả để trống hoặc ghi 'chưa thực hiện'."),
    ]
    sheet("Huong_dan", ["Tab", "Ý nghĩa / quy tắc"], guide, [18, 110])

    ws = sheet("Phan_cong", pc_header, pc, [8, 28, 9, 50, 12, 8, 8, 10, 50, 45, 40, 55, 14, 12, 12])
    dv = DataValidation(type="list", formula1='"' + ",".join(STATUS) + '"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add(f"N2:N{len(pc) + 1}")

    # Ma_tran bằng công thức
    ws = sheet("Ma_tran", ["task_id", "module", "giai_doan", "ten_nhiem_vu"] + MEMBERS, [], [8, 28, 9, 55, 7, 7, 7, 7])
    for i in range(len(pc)):
        r = i + 2
        ws.append([f"=Phan_cong!A{r}", f"=Phan_cong!B{r}", f"=Phan_cong!C{r}", f"=Phan_cong!D{r}"] +
                  [f'=IF(Phan_cong!E{r}={c}$1,"R",IF(Phan_cong!H{r}={c}$1,"T","C"))' for c in "EFGH"])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")

    mem_head = ["ma", "ho_ten", "vai_tro", "mssv", "may_chay_chinh (hostname)", "gpu", "git_user"]
    mp = HERE / "members.csv"
    if mp.exists():  # thành viên đã điền MSSV/hostname/GPU/git_user: giữ nguyên, không ghi đè
        with open(mp, encoding="utf-8-sig") as f:
            mem_rows = [[r[h] for h in mem_head] for r in csv.DictReader(f)]
    else:
        mem_rows = [(m, MEMBER_NAMES[m], "Nhóm trưởng" if m == "TV2" else "", "", "", "", "") for m in MEMBERS]
        write_csv(mp, mem_head, mem_rows)
    sheet("Thanh_vien", mem_head, mem_rows, [8, 26, 14, 14, 28, 24, 20])
    sheet("Kiem_thu", HEADERS["Kiem_thu"], [], [8, 12, 8, 10, 14, 14, 30, 30, 20, 20, 10, 30, 30])
    sheet("Thuc_nghiem", HEADERS["Thuc_nghiem"], [], [10, 8, 10] + [14] * 31)
    sheet("Hinh_ve", HEADERS["Hinh_ve"], [], [8, 32, 32, 28, 30, 14, 8, 14, 20, 16])

    # Tien_do: công thức đếm
    mods = []
    for t in TASKS:
        if t[1] not in mods:
            mods.append(t[1])
    ws = sheet("Tien_do", ["Module"] + STATUS + ["Tổng"], [], [36, 10, 12, 12, 10, 10, 8])
    for i, m in enumerate(mods, start=2):
        ws.append([m] + [f'=COUNTIFS(Phan_cong!$B:$B,$A{i},Phan_cong!$N:$N,{c}$1)' for c in "BCDEF"] + [f"=SUM(B{i}:F{i})"])
    n = len(mods) + 1
    ws.append(["TỔNG"] + [f"=SUM({c}2:{c}{n})" for c in "BCDEFG"])
    ws.append([])
    head = ["Thành viên", "Số nhiệm vụ làm R", "Số nhiệm vụ làm C", "Số nhiệm vụ làm T",
            "Dòng kiểm thử Pass (với tư cách tester)", "Dòng kiểm thử Fail"]
    ws.append(head)
    hr = ws.max_row
    for c in ws[hr]:
        c.font, c.fill = bold, fill
        c.alignment = Alignment(wrap_text=True)
    for m in MEMBERS:
        r = ws.max_row + 1
        ws.append([m, f'=COUNTIF(Phan_cong!E:E,A{r})', f'=COUNTIF(Phan_cong!F:F,A{r})+COUNTIF(Phan_cong!G:G,A{r})',
                   f'=COUNTIF(Phan_cong!H:H,A{r})',
                   f'=COUNTIFS(Kiem_thu!E:E,A{r},Kiem_thu!K:K,"Pass")', f'=COUNTIFS(Kiem_thu!E:E,A{r},Kiem_thu!K:K,"Fail")'])
    ws.append([])
    ws.append(["Số run trong Thuc_nghiem", "=COUNTA(Thuc_nghiem!A:A)-1"])
    ws.append(["Số hình trong Hinh_ve", "=COUNTA(Hinh_ve!A:A)-1"])
    wb.move_sheet("Huong_dan", offset=-wb.index(wb["Huong_dan"]))
    wb.save(HERE / "tracking.xlsx")
    print("OK:", len(rows), "nhiệm vụ;", {m: sum(1 for t in rows if t[-1] == m) for m in MEMBERS}, "lượt T")


if __name__ == "__main__":
    main()
