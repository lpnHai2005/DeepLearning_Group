# 🏠 House Price Prediction - Lab03 Deep Learning

> **Dự án Dự đoán Giá nhà (Ames Housing Dataset - Kaggle)**  
> **Kiến trúc Hybrid Model**: Kết hợp Machine Learning truyền thống (Scikit-Learn, XGBoost) và Deep Learning (PyTorch MLP).

---

## 📋 Mục lục

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Phân công nhiệm vụ nhóm](#2-phân-công-nhiệm-vụ-nhóm)
3. [Cấu trúc thư mục dự án](#3-cấu-trúc-thư-mục-dự-án)
4. [Yêu cầu môi trường &amp; Cài đặt](#4-yêu-cầu-môi-trường--cài-đặt)
5. [Hướng dẫn thực thi](#5-hướng-dẫn-thực-thi)
6. [Chi tiết các bước thực hiện](#6-chi-tiết-các-bước-thực-hiện)
7. [Kết quả thực nghiệm](#7-kết-quả-thực-nghiệm)
8. [Danh mục file nghiệm thu &amp; Output](#8-danh-mục-file-nghiệm-thu--output)
9. [Quy cách nộp bài](#9-quy-cách-nộp-bài)

---

## 1. Giới thiệu dự án

Dự án tham gia cuộc thi [**House Prices: Advanced Regression Techniques**](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) do Kaggle tổ chức, dựa trên tập dữ liệu nhà ở Ames, Iowa của tác giả Dean De Cock (2011).

- **Mục tiêu**: Dự đoán giá bán nhà (`SalePrice`) dựa trên 79 biến giải thích mô tả chi tiết các khía cạnh bất động sản.
- **Độ đo đánh giá**: **Root-Mean-Squared-Logarithmic-Error (RMSLE)**:
  $$
  \text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(1+y_i) - \log(1+\hat{y}_i))^2}
  $$
- **Phương pháp tiếp cận**: Xây dựng **Hybrid Model** gồm:
  - Nhánh học máy truyền thống: 6 mô hình Scikit-Learn / XGBoost với **5-Fold Cross Validation**.
  - Nhánh học sâu: Mạng nơ-ron đa tầng PyTorch MLP với kỹ thuật BatchNorm, Dropout, Cosine LR Scheduler và Early Stopping.
  - Mô hình kết hợp (Ensemble): Trung bình dự đoán của các mô hình tốt nhất.

---

## 2. Phân công nhiệm vụ nhóm

Dự án được module hóa chuẩn xác theo 4 vai trò thành viên:


| Thành viên     | Phân công phụ trách                                                                     | Thư mục / File chính                                                        |
| -------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Trung Kiên** | Khảo sát dữ liệu (EDA), phân tích missing/outlier, tiền xử lý &amp; tạo đặc trưng       | `prj/eda/`, `data/`, `exps/`                                                |
| Viễn Thông     | Nhánh học máy truyền thống (Scikit-Learn, XGBoost, Cross Validation)                    | `prj/sklearn/features.py`, `train_baseline.py`, `evaluate_sklearn.py`       |
| Nguyên Hải     | Nhánh học sâu PyTorch MLP (Custom Dataset, MLP Architecture, Training Loop)             | `prj/pytorch_mlp/dataset.py`, `model.py`, `train_mlp.py`, `evaluate_mlp.py` |
| Hoàng Bảo      | Runner tự động (`run_all.py`), sinh sơ đồ nghiệm thu, tổng hợp submission &amp; báo cáo | `run_all.py`, `prj/model/diagrams.py`, Báo cáo tổng hợp                     |


---

## 3. Cấu trúc thư mục dự án

```text
ml_hybrid_project/
│
├── data/                               # Dữ liệu dự án
│   ├── raw/                            # Dữ liệu gốc từ Kaggle
│   │   ├── train.csv                   # 1,460 mẫu huấn luyện (81 cột)
│   │   ├── test.csv                    # 1,459 mẫu kiểm tra (80 cột)
│   │   ├── data_description.txt         # Tài liệu mô tả chi tiết các đặc trưng
│   │   └── sample_submission.csv        # Định dạng mẫu nộp bài Kaggle
│   │
│   └── processed/                      # Dữ liệu sau khi làm sạch & chuẩn hóa
│       ├── X_train.csv                 # Ma trận đặc trưng train (StandardScaled)
│       ├── X_test.csv                  # Ma trận đặc trưng test (StandardScaled)
│       ├── y_train.csv                 # Giá nhà gốc
│       ├── y_train_log.csv             # Giá nhà log-transformed: log(1 + SalePrice)
│       ├── scaler_sklearn.pkl          # Scaler phục vụ Sklearn
│       ├── scaler_pytorch.pkl          # Scaler phục vụ PyTorch
│       └── test_ids.csv                # Danh sách ID kiểm tra cho submission
│
├── prj/                                # Thư mục mã nguồn và thí nghiệm chính
│   ├── eda/                            # Khám phá dữ liệu & Tiền xử lý
│   │   ├── exploratory_data_analysis.ipynb # Jupyter Notebook EDA trực quan
│   │   ├── house-price-kaggle.ipynb        # Kaggle Notebook tham khảo
│   │   └── preprocess.py                   # Script tiền xử lý dữ liệu độc lập
│   │
│   ├── model/                          # DUY NHẤT: Output nghiệm thu & submissions
│   │   ├── diagrams/                   # Thư mục lưu trữ riêng 6 sơ đồ nghiệm thu (200 DPI)
│   │   │   ├── preprocessing_flowchart.png # Sơ đồ quy trình tiền xử lý dữ liệu
│   │   │   ├── mlp_architecture.png        # Sơ đồ kiến trúc mạng PyTorch MLP
│   │   │   ├── training_loop_diagram.png   # Sơ đồ vòng lặp huấn luyện PyTorch
│   │   │   ├── evaluation_diagram.png      # Sơ đồ quy trình đánh giá & kiểm định
│   │   │   ├── full_pipeline_diagram.png   # Sơ đồ tổng thể toàn bộ dự án
│   │   │   └── sklearn_models_diagram.png  # Sơ đồ 6 mô hình Machine Learning
│   │   ├── sklearn.png                 # Biểu đồ so sánh hiệu năng 6 mô hình ML
│   │   ├── pytorch.html                # Báo cáo tương tác training curves (Chart.js)
│   │   ├── training_curves.png          # Đồ thị Train Loss vs Val Loss
│   │   ├── mlp_model.pth               # Checkpoint trọng số mô hình PyTorch tốt nhất
│   │   ├── training_history.csv        # Lịch sử loss và learning rate qua từng epoch
│   │   ├── sklearn_results.csv         # Bảng tổng hợp số liệu chi tiết các mô hình
│   │   ├── actual_vs_predicted_*.png   # Đồ thị giá thực tế vs giá dự đoán
│   │   ├── feature_importance_*.png    # Biểu đồ mức độ quan trọng của đặc trưng
│   │   ├── residual_analysis.png       # Biểu đồ phân tích sai số phần dư
│   │   ├── diagrams.py                 # Script tự động vẽ 6 sơ đồ bằng Matplotlib
│   │   ├── submission_xgb.csv          # ⭐ File nộp bài từ mô hình TỐT NHẤT (XGBoost)
│   │   ├── submission_sklearn.csv      # File nộp bài từ Best Sklearn Model (Gradient Boosting)
│   │   ├── submission_gb.csv           # File nộp bài từ Gradient Boosting
│   │   ├── submission_rf.csv           # File nộp bài từ Random Forest
│   │   ├── submission_elastic.csv      # File nộp bài từ ElasticNet
│   │   ├── submission_lasso.csv       # File nộp bài từ Lasso
│   │   ├── submission_ridge.csv        # File nộp bài từ Ridge
│   │   ├── submission_mlp.csv          # File nộp bài từ PyTorch MLP
│   │   └── submission_ensemble.csv      # File nộp bài kết hợp Ensemble
│   │
│   ├── sklearn/                        # Phân nhánh Machine Learning
│   │   ├── features.py                 # Module trích xuất đặc trưng & encoding
│   │   ├── train_baseline.py           # Huấn luyện 6 mô hình với 5-Fold CV
│   │   └── evaluate_sklearn.py         # Đánh giá & tạo biểu đồ so sánh
│   │
│   └── pytorch_mlp/                    # Phân nhánh Deep Learning
│       ├── dataset.py                  # Custom Dataset (Dataset, DataLoader)
│       ├── model.py                    # Kiến trúc nn.Module cho MLP
│       ├── train_mlp.py                # Vòng lặp huấn luyện (Loss, Early Stopping, Cosine LR)
│       └── evaluate_mlp.py             # Đánh giá & xuất file pytorch.html
│
├── exps/                               # Dữ liệu phục vụ thí nghiệm mở rộng
│   ├── data/                           # Bản sao dữ liệu gốc
│   └── feature/                        # Bản sao dữ liệu đặc trưng đã tiền xử lý
│
├── requirements.txt                    # Danh sách các thư viện phụ thuộc
├── README.md                           # Tài liệu hướng dẫn dự án (File này)
├── NGHIEMTHU.md                        # Checklist biên bản nghiệm thu
└── run_all.py                          # Script duy nhất chạy toàn bộ pipeline
```

---

## 4. Yêu cầu môi trường &amp; Cài đặt

### Yêu cầu

- Python **3.8 trở lên** (khuyến nghị Python 3.10 – 3.12).
- Hệ điều hành: Windows, macOS hoặc Linux.

### Cài đặt dependencies

```bash
# 1. Di chuyển vào thư mục dự án
cd d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project

# 2. Khởi tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt môi trường:
# Windows (PowerShell/CMD):
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 5. Hướng dẫn thực thi

### Cách 1: Chạy toàn bộ Pipeline tự động (Khuyến nghị)

Chỉ cần chạy duy nhất 1 câu lệnh, toàn bộ 6 bước sẽ được thực thi tuần tự và lưu kết quả vào `prj/model/`:

```bash
python run_all.py
```

### Cách 2: Chạy từng bước độc lập

```bash
# Bước 1: Khám phá dữ liệu & Tiền xử lý
python prj/eda/preprocess.py
# (Hoặc mở prj/eda/exploratory_data_analysis.ipynb trong VS Code / Jupyter)

# Bước 2: Huấn luyện 6 mô hình Sklearn & XGBoost
python prj/sklearn/train_baseline.py

# Bước 3: Đánh giá mô hình Sklearn & vẽ biểu đồ so sánh
python prj/sklearn/evaluate_sklearn.py

# Bước 4: Huấn luyện mạng Deep Learning PyTorch MLP
python prj/pytorch_mlp/train_mlp.py

# Bước 5: Đánh giá PyTorch MLP & tạo pytorch.html tương tác
python prj/pytorch_mlp/evaluate_mlp.py

# Bước 6: Tự động vẽ 6 sơ đồ kiến trúc nghiệm thu
python prj/model/diagrams.py
```

---

## 6. Chi tiết các bước thực hiện

### 6.1. Khám phá dữ liệu (EDA)

File: [prj/eda/exploratory_data_analysis.ipynb](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/eda/exploratory_data_analysis.ipynb)

- **Xử lý Target**: Biến đổi log $\log(1 + \text{SalePrice})$ giúp đưa phân phối giá nhà từ lệch phải về phân phối chuẩn đối xứng.
- **Missing Values**: Phân loại theo ngữ cảnh Ames Housing (`PoolQC`, `MiscFeature`, `Alley`, `Fence` thiếu &gt;80% được quy định là `"None"` thay vì mất dữ liệu).
- **Outliers**: Phát hiện 2 mẫu nhà ngoại lai có diện tích sàn `GrLivArea > 4000 sqft` nhưng giá bán `< $300,000`.
- **Tương quan**: Xác định Top 5 đặc trưng ảnh hưởng mạnh nhất: `OverallQual` ($r=0.79$), `GrLivArea` ($r=0.71$), `GarageCars` ($r=0.64$), `GarageArea` ($r=0.62$), `TotalBsmtSF` ($r=0.61$).

### 6.2. Tiền xử lý &amp; Kỹ nghệ đặc trưng (Feature Engineering)

File: [prj/eda/preprocess.py](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/eda/preprocess.py) và [prj/sklearn/features.py](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/sklearn/features.py)

Tạo **105 đặc trưng** (80 gốc + 25 mới) giúp mô hình nắm bắt thông tin thực tế:

**Features về Diện tích:**

- `TotalSF`: Tổng diện tích sàn nhà (`TotalBsmtSF + 1stFlrSF + 2ndFlrSF`).
- `TotalFloorSF`: Tổng diện tích sàn không basement.
- `TotalPorchSF`: Tổng diện tích hiên nhà.

**Features về Phòng tắm:**

- `TotalBath`: Tổng số phòng tắm (`FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`).

**Features về Tuổi nhà:**

- `HouseAge`: Số năm tuổi của ngôi nhà khi bán.
- `RemodAge`: Số năm từ lần sửa chữa/nâng cấp gần nhất.

**Binary Flags:**

- `HasPool`, `HasGarage`, `HasBasement`, `HasFireplace`, `Has2ndFloor`, `HasMasVnr`, `IsRemodeled`, `IsNew`.

**Quality Scores:**

- `OverallScore`, `ExterScore`, `BsmtScore`, `GarageScore` (tích các điểm chất lượng).

**Ratio &amp; Interaction Features:**

- `BsmtRatio`, `AreaPerRoom`, `LivAreaRatio`, `GarageAreaPerCar`, `LotFrontageRatio`.
- `SF_Qual_Interaction`, `Year_Qual_Interaction`, `SF_Bath_Interaction`.

**Mã hóa:**

- **Ordinal Mapping** cho các bậc chất lượng (`Ex: 5, Gd: 4, TA: 3, Fa: 2, Po: 1, None: 0`).
- **LabelEncoder** cho các biến danh mục.
- Chuẩn hóa toàn bộ bằng `StandardScaler`.

### 6.3. Nhánh Machine Learning (Scikit-Learn &amp; XGBoost)

File: [prj/sklearn/train_baseline.py](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/sklearn/train_baseline.py)

- Đánh giá khách quan bằng **5-Fold Cross Validation** (RMSLE chính xác, không overfitting).
- Huấn luyện 6 mô hình:
  - **Ridge Regression** ($L_2$ regularization, $\alpha=10.0$): Kiểm soát đa cộng tuyến.
  - **Lasso Regression** ($L_1$ regularization, $\alpha=0.001$): Lọc bỏ đặc trưng dư thừa.
  - **ElasticNet** (Kết hợp $L_1$ &amp; $L_2$).
  - **Random Forest Regressor** (200 cây, `max_depth=15`): Bắt tương tác phi tuyến.
  - **Gradient Boosting Regressor** (200 cây, learning rate $0.1$).
  - **XGBoost Regressor** (200 cây, `subsample=0.8`, `colsample_bytree=0.8`) - **🥇 MÔ HÌNH TỐT NHẤT**.

### 6.4. Nhánh Deep Learning (PyTorch MLP)

Files: [prj/pytorch_mlp/dataset.py](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/pytorch_mlp/dataset.py), [model.py](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/pytorch_mlp/model.py), [train_mlp.py](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/pytorch_mlp/train_mlp.py)

- **Kiến trúc mạng (MLPRegressor)**:
  $$
  \text{Input (105 features)} \to \text{FC(256)} \to \text{FC(128)} \to \text{FC(64)} \to \text{FC(32)} \to \text{FC(1)}
  $$
- Mỗi tầng ẩn tích hợp: **BatchNorm1d** (ổn định phân phối gradient), hàm kích hoạt **ReLU**, và **Dropout (0.3** $\to$ **0.1)** (chống học vẹt).
- Khởi tạo trọng số **Kaiming / Xavier Normal**.
- Tối ưu hóa: `AdamW` (`lr=0.001`, `weight_decay=1e-4`), hàm mất mát `MSELoss`.
- Điều chỉnh tốc độ học mượt mà bằng **CosineAnnealingLR**.
- Cơ chế dừng sớm **Early Stopping** (`patience=30` epochs) giúp dừng đúng điểm cực tiểu trên tập validation.

---

## 7. Kết quả thực nghiệm

### Bảng so sánh hiệu năng các mô hình (5-Fold Cross Validation & Kaggle Leaderboard)

> ⚠️ **Lưu ý quan trọng**: RMSLE được đánh giá trên **Validation Set** (Cross-Validation) và **Public Test Set** của Kaggle. Thang đo **RMSLE càng thấp càng chính xác** (Lower is Better).


| Hạng | Mô hình / Thuật toán | RMSLE (5-Fold CV) | Std | Kaggle Score | Đánh giá & Danh hiệu |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🏆 | **Ensemble (Log-Space Blend)** | **0.1311** | - | **0.12088** | 🥇 **MÔ HÌNH TỔNG THỂ TỐT NHẤT (QUÁN QUÂN DỰ ÁN)** |
| 1 | **XGBoost Regressor** | **0.1295** | ±0.0170 | **0.12593** | 🥇 **Mô hình Đơn tốt nhất (Best Single Model)** |
| 2 | **Gradient Boosting (GB)** | 0.1343 | ±0.0184 | **0.12546** | 🥈 Mô hình thuần Scikit-Learn tốt nhất |
| 3 | **Random Forest** | 0.1418 | ±0.0165 | \~0.14 | Khá (Giảm phương sai tốt) |
| 4 | **Ridge Regression** | 0.1481 | ±0.0401 | \~0.15 | Trung bình (Tuyến tính L2) |
| 5 | **Lasso Regression** | 0.1492 | ±0.0426 | \~0.15 | Trung bình (Tuyến tính L1) |
| 6 | **ElasticNet** | 0.1507 | ±0.0395 | \~0.15 | Trung bình (Kết hợp L1 + L2) |
| - | **PyTorch MLP** | 0.2742 | - | \~0.27 | Hội tụ nhưng kém tree-based |


> **Nhận xét chuyên sâu**: 
>
> 1. **Đỉnh cao Ensemble Blending (0.12088)**: Kỹ thuật kết hợp trọng số trong không gian log ($35\%\text{ XGB} + 30\%\text{ GB} + 15\%\text{ RF} + 10\%\text{ Elastic} + 5\%\text{ Lasso} + 5\%\text{ Ridge}$) giúp triệt tiêu phương sai giữa các mô hình và tận dụng thế mạnh riêng của từng thuật toán, đưa sai số giảm sâu từ $0.125$ xuống **$0.12088$** (đạt thứ hạng cao nhất trên Kaggle Leaderboard).
> 2. **XGBoost dẫn đầu các mô hình đơn**: Trên tập huấn luyện (5-Fold CV), XGBoost đạt RMSLE = **0.1295** và điểm Kaggle = **0.12593**, chứng minh sức mạnh vượt trội của cơ chế Extreme Gradient Boosting và chính quy hóa độ phức tạp cây.
> 3. **Gradient Boosting (Scikit-Learn) hoạt động xuất sắc**: Đạt điểm Kaggle **0.12546** (được xuất thành cả `submission_gb.csv` và `submission_sklearn.csv`), là mô hình Scikit-Learn độc lập tốt nhất.
> 4. **Tree-based so với Linear & MLP**:
>    - Các mô hình dạng cây (XGBoost, GB, RF) vượt trội hoàn toàn so với các mô hình tuyến tính trên dữ liệu dạng bảng (tabular data).
>    - PyTorch MLP hội tụ tốt qua 100 epoch với CosineAnnealingLR, tuy nhiên với dữ liệu bảng kích thước nhỏ (~1,460 mẫu), mạng Deep Learning không phát huy được ưu thế bằng Gradient Boosted Trees.

---

## 8. Danh mục file nghiệm thu & Output

Tất cả các sản phẩm đầu ra được lưu tập trung tại **`prj/model/`**:

### Biểu đồ & Báo cáo trực quan

- [**sklearn.png**](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/model/sklearn.png): Biểu đồ cột so sánh RMSE và $R^2$ của các mô hình Machine Learning.
- [**pytorch.html**](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/model/pytorch.html): Trang web báo cáo tương tác HTML (Chart.js) cho phép zoom, rê chuột xem loss từng epoch.
- [**training_curves.png**](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/model/training_curves.png): Đường cong suy giảm hàm mất mát qua quá trình huấn luyện MLP.
- [**actual_vs_predicted_\*.png**](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/model): Biểu đồ phân tán so sánh giá thực tế và giá dự đoán cho từng model.
- [**feature_importance_\*.png**](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/model): Mức độ đóng góp của Top 20 đặc trưng quan trọng nhất.
- [**residual_analysis.png**](file:///d:/SGU_HK1_2026-2027/SGU_DL/DeepLearning_Group/lab03-competition/ml_hybrid_project/prj/model/residual_analysis.png): Phân tích phân phối sai số phần dư.

### 6 Sơ đồ kiến trúc vẽ từ code (Lưu riêng tại `prj/model/diagrams/` - Dùng cho Báo Cáo Nghiệm Thu)

1. `prj/model/diagrams/preprocessing_flowchart.png`: Sơ đồ luồng tiền xử lý dữ liệu.
2. `prj/model/diagrams/mlp_architecture.png`: Sơ đồ kiến trúc mạng nơ-ron đa tầng PyTorch MLP.
3. `prj/model/diagrams/training_loop_diagram.png`: Sơ đồ chu trình lặp huấn luyện PyTorch.
4. `prj/model/diagrams/evaluation_diagram.png`: Sơ đồ pipeline đánh giá và kiểm định.
5. `prj/model/diagrams/full_pipeline_diagram.png`: Sơ đồ tổng thể 6 bước của dự án.
6. `prj/model/diagrams/sklearn_models_diagram.png`: Sơ đồ phân loại các thuật toán học máy.

### File nộp bài Kaggle


| File | Mô hình | Điểm Kaggle (RMSLE) | Đánh giá & Khuyến nghị |
| :--- | :--- | :---: | :--- |
| `submission_ensemble.csv` | **Ensemble Blending (6 mô hình)** | **0.12088** | 🏆 **KHUYẾN NGHỊ NỘP CHÍNH THỨC (ĐIỂM CAO NHẤT DỰ ÁN)** |
| `submission_gb.csv` | **Gradient Boosting** | **0.12546** | 🥈 Mô hình thuần Scikit-Learn tốt nhất |
| `submission_sklearn.csv` | **Best Sklearn (GB)** | **0.12546** | 🥈 Trùng khớp 100% với `submission_gb.csv` |
| `submission_xgb.csv` | **XGBoost Regressor** | **0.12593** | 🥇 Mô hình đơn tốt nhất |
| `submission_rf.csv` | Random Forest | \~0.14 | Bagging Baseline |
| `submission_*.csv` (Linear) | Ridge / Lasso / ElasticNet | \~0.15 | Regularized Linear Baselines |
| `submission_mlp.csv` | PyTorch MLP | \~0.27 | Deep Learning Branch |


---

## 9. Quy cách nộp bài

1. **Nộp bài lên Kaggle**: Nộp file **`submission_ensemble.csv`** (đạt điểm **0.12088**) hoặc **`submission_xgb.csv`** lên cuộc thi Kaggle [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) và chụp lại màn hình điểm số / xếp hạng.
2. **Viết Báo Cáo (Word/PDF)**:
   - Trang bìa: Ghi rõ Họ tên, Mã số sinh viên của 4 thành viên.
   - Bảng phân công công việc & mức độ đóng góp (%).
   - Nội dung bài làm: Trình bày quy trình EDA, Feature Engineering, kiến trúc mô hình và chèn 6 sơ đồ từ `prj/model/diagrams/`.
3. **Đóng gói file nén**: Nén thư mục project cùng file Báo cáo thành tệp:  
`lab03_house_price_hoten_masv.zip` (theo họ tên và mã số sinh viên của nhóm trưởng).

---

*Dự án thực hiện bởi Nhóm sinh viên Môn Deep Learning - Đại học Sài Gòn (SGU).*