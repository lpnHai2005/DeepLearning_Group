# 📋 NGHIỆM THU LAB03 - HOUSE PRICE PREDICTION

## Tổng quan

Dự án **House Price Prediction** là bài toán dự đoán giá nhà sử dụng bộ dữ liệu **Ames Housing Dataset** từ cuộc thi Kaggle. Yêu cầu xây dựng **Hybrid Model** kết hợp Machine Learning (sklearn) và Deep Learning (PyTorch MLP).

---

## 📁 Cấu trúc Project

```
ml_hybrid_project/
│
├── data/
│   ├── raw/                           # Dữ liệu gốc
│   │   ├── train.csv (1460 samples)
│   │   ├── test.csv (1459 samples)
│   │   ├── data_description.txt
│   │   └── sample_submission.csv
│   │
│   └── processed/                     # Dữ liệu đã xử lý
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_train_log.csv
│       ├── scaler_sklearn.pkl
│       ├── scaler_pytorch.pkl
│       └── test_ids.csv
│
├── prj/
│   ├── eda/
│   │   ├── exploratory_data_analysis.ipynb  # EDA chi tiết
│   │   ├── house-price-kaggle.ipynb        # Kaggle notebook
│   │   └── preprocess.py                   # Script tiền xử lý
│   │
│   ├── model/                       # Output nghiệm thu
│   │   ├── diagrams/                # 6 sơ đồ kiến trúc & quy trình nghiệm thu (200 DPI)
│   │   │   ├── preprocessing_flowchart.png
│   │   │   ├── mlp_architecture.png
│   │   │   ├── training_loop_diagram.png
│   │   │   ├── evaluation_diagram.png
│   │   │   ├── full_pipeline_diagram.png
│   │   │   └── sklearn_models_diagram.png
│   │   ├── sklearn.png              # So sánh sklearn models
│   │   ├── pytorch.html            # Training curves tương tác
│   │   ├── training_curves.png
│   │   ├── actual_vs_predicted_*.png
│   │   ├── feature_importance_*.png
│   │   ├── submission_sklearn.csv
│   │   └── submission_mlp.csv
│   │
│   ├── sklearn/
│   │   ├── features.py              # Feature engineering
│   │   ├── train_baseline.py       # Train 6 models
│   │   └── evaluate_sklearn.py      # Evaluate & visualize
│   │
│   └── pytorch_mlp/
│       ├── dataset.py               # Custom Dataset & DataLoader
│       ├── model.py                 # MLP Architecture
│       ├── train_mlp.py             # Training loop
│       └── evaluate_mlp.py          # Evaluate & HTML
│
├── exps/
│   ├── data/
│   └── feature/
│
├── requirements.txt
├── README.md
└── run_all.py
```

---

## ✅ Checklist nghiệm thu

### 1. Folder `model/` chứa ảnh + HTML output


| File                          | Mô tả                                | Trạng thái |
| ----------------------------- | ------------------------------------ | ---------- |
| `sklearn.png`                 | So sánh RMSE/R² các sklearn models   | ✅          |
| `pytorch.html`                | Training curves tương tác (Chart.js) | ✅          |
| `training_curves.png`         | Loss curves                          | ✅          |
| `actual_vs_predicted_*.png`   | Actual vs Predicted plots            | ✅          |
| `feature_importance_*.png`    | Feature importance                   | ✅          |
| `residual_analysis.png`       | Residual analysis                    | ✅          |

### 2. Folder `model/diagrams/` chứa 6 sơ đồ kiến trúc

| File (trong `prj/model/diagrams/`) | Mô tả                                | Trạng thái |
| ---------------------------------- | ------------------------------------ | ---------- |
| `preprocessing_flowchart.png`      | Sơ đồ tiền xử lý dữ liệu             | ✅          |
| `mlp_architecture.png`             | Sơ đồ kiến trúc MLP                  | ✅          |
| `training_loop_diagram.png`        | Sơ đồ training loop                  | ✅          |
| `evaluation_diagram.png`           | Sơ đồ evaluation pipeline            | ✅          |
| `full_pipeline_diagram.png`        | Sơ đồ tổng thể pipeline              | ✅          |
| `sklearn_models_diagram.png`       | Sơ đồ các sklearn models             | ✅          |


### 2. Folder `eda/` có phân tích trực quan


| File                                | Mô tả                  | Trạng thái |
| ----------------------------------- | ---------------------- | ---------- |
| `exploratory_data_analysis.ipynb`   | EDA notebook đầy đủ    | ✅          |
| Missing values analysis             | Bar chart + Heatmap    | ✅          |
| Outliers analysis                   | Scatter plots          | ✅          |
| Correlation analysis                | Heatmap + Top features | ✅          |
| Neighborhood &amp; Quality analysis | Box plots              | ✅          |
| Feature distributions               | Histograms             | ✅          |


### 3. File `README.md` có hướng dẫn chạy


| Nội dung                 | Trạng thái |
| ------------------------ | ---------- |
| Cấu trúc project         | ✅          |
| Yêu cầu cài đặt          | ✅          |
| Hướng dẫn chạy từng bước | ✅          |
| Output nghiệm thu        | ✅          |
| Mô hình sử dụng          | ✅          |


### 4. Module sklearn đầy đủ


| File                  | Mô tả                                           | Trạng thái |
| --------------------- | ----------------------------------------------- | ---------- |
| `features.py`         | Feature engineering module                      | ✅          |
| `train_baseline.py`   | Train Ridge, Lasso, ElasticNet, RF, GB, XGBoost | ✅          |
| `evaluate_sklearn.py` | Evaluate &amp; visualize                        | ✅          |


### 5. Module PyTorch MLP đầy đủ


| File              | Mô tả                            | Trạng thái |
| ----------------- | -------------------------------- | ---------- |
| `dataset.py`      | Custom Dataset &amp; DataLoader  | ✅          |
| `model.py`        | MLP architecture                 | ✅          |
| `train_mlp.py`    | Training loop với Early Stopping | ✅          |
| `evaluate_mlp.py` | Evaluate &amp; tạo pytorch.html  | ✅          |


---

## 📊 Các sơ đồ nghiệm thu (Lưu tại `prj/model/diagrams/`)

### 1. preprocessing_flowchart.png

Sơ đồ quy trình tiền xử lý dữ liệu:

- Input: train.csv, test.csv
- Missing Values → Ordinal/Label Encoding → StandardScaler
- Output: X_train, X_test (chuẩn hóa)

### 2. mlp_architecture.png

Sơ đồ kiến trúc MLP:

- Input (93 features) → 256 → 128 → 64 → 32 → Output (1)
- BatchNorm, ReLU, Dropout sau mỗi layer

### 3. training_loop_diagram.png

Sơ đồ training loop:

- Forward Pass → Loss → Backward Pass → Optimizer Step
- Validation → Early Stopping → LR Scheduler
- Save Best Model → Predict → Submission

### 4. evaluation_diagram.png

Sơ đồ evaluation pipeline:

- Metrics: RMSE, MAE, R², RMSLE
- Visualizations: Actual vs Predicted, Feature Importance
- Submission files

### 5. full_pipeline_diagram.png

Sơ đồ tổng thể 6 bước:

1. Data Loading
2. EDA &amp; Preprocessing
3. Feature Engineering
4. Train Models
5. Evaluate &amp; Select
6. Submit Kaggle

### 6. sklearn_models_diagram.png

Sơ đồ 6 sklearn models:

- Linear: Ridge, Lasso, ElasticNet
- Ensemble: Random Forest, Gradient Boosting, XGBoost

### 7. sklearn.png

Biểu đồ so sánh RMSE/R² của các sklearn models (Chart.js)

### 8. pytorch.html

Trang HTML tương tác hiển thị:

- Training &amp; Validation Loss curves
- Validation RMSE
- Learning Rate schedule

---

## 🚀 Hướng dẫn chạy

### Cách 1: Chạy toàn bộ pipeline

```bash
cd ml_hybrid_project
python run_all.py
```

### Cách 2: Chạy từng bước

```bash
# Bước 1: Tiền xử lý
python prj/eda/preprocess.py

# Bước 2: Train sklearn
python prj/sklearn/train_baseline.py

# Bước 3: Evaluate sklearn
python prj/sklearn/evaluate_sklearn.py

# Bước 4: Train PyTorch
python prj/pytorch_mlp/train_mlp.py

# Bước 5: Evaluate PyTorch
python prj/pytorch_mlp/evaluate_mlp.py

# Bước 6: Tạo sơ đồ nghiệm thu
python prj/model/diagrams.py
```

---

## 📈 Kết quả thực tế & Điểm nộp bài Kaggle


| Model | RMSLE (5-Fold CV) | Kaggle Score (RMSLE) | Đánh giá & Danh hiệu |
| :--- | :---: | :---: | :--- |
| **🏆 Ensemble (Weighted)** | **0.1311** | **0.12088** | 🥇 **QUÁN QUÂN DỰ ÁN (Tốt nhất toàn diện)** |
| **🥇 XGBoost** | **0.1295** | **0.12593** | 🥇 **Mô hình Đơn tốt nhất (Best Single Model)** |
| **🥈 Gradient Boosting** | 0.1343 | **0.12546** | 🥈 Mô hình thuần Scikit-Learn tốt nhất |
| **Random Forest** | 0.1418 | \~0.14 | Khá |
| **Ridge Regression** | 0.1481 | \~0.15 | Tuyến tính L2 Regularization |
| **Lasso Regression** | 0.1492 | \~0.15 | Tuyến tính L1 Regularization |
| **ElasticNet** | 0.1507 | \~0.15 | Tuyến tính kết hợp L1 + L2 |
| **MLP (PyTorch)** | 0.2742 | \~0.27 | Mạng nơ-ron Deep Learning |


> **Quy tắc xếp hạng Kaggle**: Metric chấm bài là RMSLE (Root Mean Squared Logarithmic Error) — **sai số càng thấp thì dự đoán càng chính xác và thứ hạng càng cao**. File `submission_ensemble.csv` đạt điểm **0.12088** là kết quả tốt nhất của nhóm.

---

## 🎯 Phân công công việc


| Thành viên | Phụ trách           | File                                |
| ---------- | ------------------- | ----------------------------------- |
| Trung Kiên | EDA + Preprocessing | `eda/*.ipynb`, `eda/preprocess.py`  |
| Viễn Thông | Sklearn Pipeline    | `sklearn/*.py`                      |
| Nguyên Hải | PyTorch MLP         | `pytorch_mlp/*.py`                  |
| Hoàng Bảo  | Tổng hợp + Diagrams | `model/diagrams.py`, `NGHIEMTHU.md` |


---

## 📝 Ghi chú

1. **Data**: Sử dụng bộ dữ liệu Ames Housing từ Kaggle
2. **Target**: log1p(SalePrice) để xử lý skewed distribution
3. **Validation**: 80% train, 20% validation (random split)
4. **Early Stopping**: Patience = 30 epochs
5. **Kaggle Metric**: RMSLE (Root Mean Squared Logarithmic Error)

---

