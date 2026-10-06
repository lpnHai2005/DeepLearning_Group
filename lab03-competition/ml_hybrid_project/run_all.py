"""
=================================================================
MAIN RUNNER - Chạy toàn bộ Pipeline Dự Án (House Price Prediction)
=================================================================
Script này chạy tuần tự tất cả các bước của Lab03:
    1. Tiền xử lý dữ liệu (Feature Engineering & Scaling)
    2. Huấn luyện 6 mô hình Scikit-Learn / XGBoost
    3. Đánh giá Scikit-Learn & xuất biểu đồ sklearn.png
    4. Huấn luyện mạng Deep Learning PyTorch MLP
    5. Đánh giá PyTorch MLP & xuất pytorch.html, loss curves
    6. Tạo 6 sơ đồ kiến trúc nghiệm thu (Matplotlib)

Cách sử dụng:
    python run_all.py

Author: SGU Deep Learning Group
"""

import os
import sys
import subprocess

# Đảm bảo UTF-8 encoding trên Windows terminal
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Cấu hình thư mục gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("=" * 70)
print("🏠 HOUSE PRICE PREDICTION - TOÀN BỘ PIPELINE THỰC THI")
print("=" * 70)
print(f"Working directory: {BASE_DIR}\n")


def run_step(step_name, script_rel_path, description):
    """Chạy một bước trong pipeline bằng Python con với UTF-8 mode"""
    print("\n" + "=" * 70)
    print(f"📦 {step_name}")
    print(f"   Mô tả: {description}")
    print(f"   Script: {script_rel_path}")
    print("=" * 70)

    script_path = os.path.join(BASE_DIR, script_rel_path)
    if not os.path.exists(script_path):
        print(f"\n❌ LỖI: Không tìm thấy script tại {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", script_rel_path],
            cwd=BASE_DIR,
            capture_output=False
        )

        if result.returncode == 0:
            print(f"\n✅ {step_name} - HOÀN TẤT THÀNH CÔNG")
            return True
        else:
            print(f"\n❌ {step_name} - THẤT BẠI (Mã lỗi: {result.returncode})")
            return False

    except Exception as e:
        print(f"\n❌ LỖI NGOẠI LỆ: {e}")
        return False


def main():
    """Hàm điều phối toàn bộ luồng chạy"""
    if sys.version_info < (3, 8):
        print("❌ LỖI: Yêu cầu Python 3.8 trở lên!")
        sys.exit(1)

    print(f"✅ Python phiên bản: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Kiểm tra dữ liệu đầu vào
    data_dir = os.path.join(BASE_DIR, "data", "raw")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ LỖI: Không tìm thấy train.csv hoặc test.csv trong data/raw/")
        print("   Vui lòng tải bộ dữ liệu từ Kaggle và đặt vào data/raw/")
        sys.exit(1)

    print("✅ Đã xác nhận dữ liệu gốc: train.csv, test.csv")

    # Danh sách các bước trong pipeline
    steps = [
        ("BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU", "prj/eda/preprocess.py",
         "Làm sạch, xử lý missing values, tạo 14 features mới, chuẩn hóa"),
        ("BƯỚC 2: TRAIN SKLEARN & XGBOOST", "prj/sklearn/train_baseline.py",
         "Huấn luyện 6 mô hình hồi quy với 5-Fold Cross Validation"),
        ("BƯỚC 3: EVALUATE SKLEARN MODELS", "prj/sklearn/evaluate_sklearn.py",
         "Đánh giá chi tiết, phân tích phần dư và tạo biểu đồ sklearn.png"),
        ("BƯỚC 4: TRAIN PYTORCH MLP", "prj/pytorch_mlp/train_mlp.py",
         "Huấn luyện mạng Deep Learning MLP với Early Stopping & Cosine LR"),
        ("BƯỚC 5: EVALUATE PYTORCH MLP", "prj/pytorch_mlp/evaluate_mlp.py",
         "Đánh giá mô hình PyTorch, vẽ loss curve và xuất pytorch.html"),
        ("BƯỚC 6: TẠO SƠ ĐỒ NGHIỆM THU", "prj/model/diagrams.py",
         "Sinh tự động 6 sơ đồ kiến trúc chuẩn đẹp bằng Matplotlib"),
    ]

    results = {}
    for step_name, script_rel_path, desc in steps:
        results[step_name] = run_step(step_name, script_rel_path, desc)

    # ─────────────────────────────────────────────────────────────
    # BÁO CÁO TỔNG HỢP KẾT QUẢ
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📋 BÁO CÁO TỔNG KẾT PIPELINE")
    print("=" * 70)

    for step_name, success in results.items():
        status = "✅ HOÀN TẤT" if success else "❌ THẤT BẠI"
        print(f"  {status:15s} | {step_name}")

    # Kiểm tra các file đầu ra quan trọng trong prj/model/ và prj/model/diagrams/
    model_dir = os.path.join(BASE_DIR, "prj", "model")
    diagrams_dir = os.path.join(model_dir, "diagrams")

    expected_model_files = [
        ("sklearn.png", "Biểu đồ so sánh các mô hình Sklearn"),
        ("pytorch.html", "Trang web trực quan hóa training curves tương tác"),
        ("training_curves.png", "Biểu đồ đường cong hàm mất mát MLP"),
        ("submission_sklearn.csv", "File nộp bài Kaggle từ Best Sklearn Model"),
        ("submission_mlp.csv", "File nộp bài Kaggle từ PyTorch MLP"),
        ("submission_ensemble.csv", "File nộp bài Kaggle kết hợp Ensemble"),
    ]

    expected_diagram_files = [
        ("preprocessing_flowchart.png", "Sơ đồ luồng tiền xử lý dữ liệu"),
        ("mlp_architecture.png", "Sơ đồ kiến trúc mạng nơ-ron MLP"),
        ("training_loop_diagram.png", "Sơ đồ vòng lặp huấn luyện PyTorch"),
        ("evaluation_diagram.png", "Sơ đồ đánh giá kiểm định mô hình"),
        ("full_pipeline_diagram.png", "Sơ đồ luồng tổng thể dự án"),
        ("sklearn_models_diagram.png", "Sơ đồ 6 mô hình Machine Learning"),
    ]

    print("\n📁 Kiểm tra file nghiệm thu tại prj/model/:")
    for fname, desc in expected_model_files:
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  ✅ {fname:28s} ({size_kb:7.1f} KB) - {desc}")
        else:
            print(f"  ❌ {fname:28s} (MISSING)   - {desc}")

    print("\n🖼️  Kiểm tra 6 sơ đồ kiến trúc tại prj/model/diagrams/:")
    for fname, desc in expected_diagram_files:
        fpath = os.path.join(diagrams_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  ✅ {fname:28s} ({size_kb:7.1f} KB) - {desc}")
        else:
            print(f"  ❌ {fname:28s} (MISSING)   - {desc}")

    print("\n" + "=" * 70)
    print("🎉 TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT CHUẨN ĐẸP!")
    print("=" * 70)
    print("\n📝 Hướng dẫn tiếp theo cho nhóm:")
    print("   1. Mở file prj/model/pytorch.html trên trình duyệt để tương tác.")
    print("   2. Mở file prj/model/sklearn.png để xem bảng xếp hạng mô hình.")
    print("   3. Nộp file prj/model/submission_ensemble.csv hoặc submission_xgb.csv lên Kaggle.")
    print("   4. Sử dụng 6 ảnh sơ đồ trong prj/model/diagrams/ để đưa vào Báo Cáo Nghiệm Thu.")


if __name__ == "__main__":
    main()
