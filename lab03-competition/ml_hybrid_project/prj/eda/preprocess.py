"""
=================================================================
TIỀN XỬ LÝ DỮ LIỆU - HOUSE PRICE PREDICTION
=================================================================
Mục tiêu: Làm sạch, mã hóa và lưu dữ liệu đã xử lý vào data/processed/

Pipeline:
    1. Đọc dữ liệu train.csv và test.csv
    2. Xử lý Missing Values
    3. Tạo Features mới (Feature Engineering)
    4. Mã hóa biến phân loại (One-Hot Encoding)
    5. Chuẩn hóa dữ liệu (StandardScaler)
    6. Lưu vào data/processed/
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

TRAIN_FILE = os.path.join(DATA_RAW_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_RAW_DIR, "test.csv")

print("=" * 60)
print("BƯỚC 1: ĐỌC DỮ LIỆU")
print("=" * 60)

# Đọc dữ liệu
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"✓ Train shape: {train_df.shape}")
print(f"✓ Test shape: {test_df.shape}")

# ─────────────────────────────────────────────
# LƯU TRỮ ID ĐỂ NỘP BÀI KAGGLE
# ─────────────────────────────────────────────
test_ids = test_df['Id'].copy()
train_ids = train_df['Id'].copy()

print("\n" + "=" * 60)
print("BƯỚC 2: TÁCH BIẾN MỤC TIÊU & GỘP DỮ LIỆU")
print("=" * 60)

# Tách biến mục tiêu
y_train = train_df['SalePrice'].copy()

# Drop Id và SalePrice khỏi train
train_features = train_df.drop(['SalePrice', 'Id'], axis=1)
# Drop Id khỏi test
test_features = test_df.drop(['Id'], axis=1)

# Gộp train + test để xử lý đồng bộ
all_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)
print(f"✓ Combined shape: {all_df.shape}")
print(f"  - Train: {len(train_features)} dòng")
print(f"  - Test:  {len(test_features)} dòng")

print("\n" + "=" * 60)
print("BƯỚC 3: XỬ LÝ MISSING VALUES")
print("=" * 60)

# === 3.1. Các cột mà NA có nghĩa là "Không có" ===
# Điền 'None' cho các cột categorical
none_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]
for col in none_cols:
    if col in all_df.columns:
        all_df[col] = all_df[col].fillna('None')

# === 3.2. Các cột số mà NA có nghĩa là "Không có" ===
# Điền 0 cho các cột numeric liên quan
zero_cols = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]
for col in zero_cols:
    if col in all_df.columns:
        all_df[col] = all_df[col].fillna(0)

# === 3.3. LotFrontage: điền theo median của từng khu vực (Neighborhood) ===
if 'LotFrontage' in all_df.columns and 'Neighborhood' in all_df.columns:
    all_df['LotFrontage'] = all_df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # Nếu còn NaN, điền median tổng
    all_df['LotFrontage'] = all_df['LotFrontage'].fillna(all_df['LotFrontage'].median())

# === 3.4. Các cột categorical còn lại: điền mode ===
for col in all_df.select_dtypes(include=['object']).columns:
    if all_df[col].isnull().sum() > 0:
        all_df[col] = all_df[col].fillna(all_df[col].mode()[0])

# === 3.5. Các cột numeric còn lại: điền 0 ===
for col in all_df.select_dtypes(include=[np.number]).columns:
    if all_df[col].isnull().sum() > 0:
        all_df[col] = all_df[col].fillna(0)

# Kiểm tra còn NaN không
remaining_na = all_df.isnull().sum().sum()
print(f"✓ Remaining NaN after preprocessing: {remaining_na}")

print("\n" + "=" * 60)
print("BƯỚC 4: FEATURE ENGINEERING")
print("=" * 60)

# === Tạo các features mới ===

# 4.1. Tổng diện tích
all_df['TotalSF'] = all_df['TotalBsmtSF'] + all_df['1stFlrSF'] + all_df['2ndFlrSF']

# 4.2. Tổng số phòng tắm
all_df['TotalBath'] = (all_df['FullBath'] + 0.5 * all_df['HalfBath'] +
                       all_df['BsmtFullBath'] + 0.5 * all_df['BsmtHalfBath'])

# 4.3. Tổng diện tích porch
all_df['TotalPorchSF'] = (all_df['OpenPorchSF'] + all_df['EnclosedPorch'] +
                          all_df['3SsnPorch'] + all_df['ScreenPorch'])

# 4.4. Tuổi nhà khi bán
all_df['HouseAge'] = all_df['YrSold'] - all_df['YearBuilt']

# 4.5. Số năm từ lần cải tạo cuối
all_df['RemodAge'] = all_df['YrSold'] - all_df['YearRemodAdd']

# 4.6. Cờ: có được cải tạo không
all_df['IsRemodeled'] = (all_df['YearRemodAdd'] != all_df['YearBuilt']).astype(int)

# 4.7. Cờ: nhà mới (bán trong năm xây)
all_df['IsNew'] = (all_df['YrSold'] == all_df['YearBuilt']).astype(int)

# 4.8. Cờ: có garage không
all_df['HasGarage'] = (all_df['GarageArea'] > 0).astype(int)

# 4.9. Cờ: có basement không
all_df['HasBasement'] = (all_df['TotalBsmtSF'] > 0).astype(int)

# 4.10. Cờ: có pool không
all_df['HasPool'] = (all_df['PoolArea'] > 0).astype(int)

# 4.11. Cờ: có fireplace không
all_df['HasFireplace'] = (all_df['Fireplaces'] > 0).astype(int)

# 4.12. Chất lượng tổng hợp = quality * condition
all_df['OverallScore'] = all_df['OverallQual'] * all_df['OverallCond']

# 4.13. Tỷ lệ diện tích basement / diện tích tầng 1
all_df['BsmtRatio'] = all_df['TotalBsmtSF'] / (all_df['1stFlrSF'] + 1)

# 4.14. Diện tích trung bình mỗi phòng
all_df['AreaPerRoom'] = all_df['GrLivArea'] / (all_df['TotRmsAbvGrd'] + 1)

print(f"✓ Created 14 new features")
print(f"✓ Total features now: {all_df.shape[1]}")

print("\n" + "=" * 60)
print("BƯỚC 5: MÃ HÓA BIẾN PHÂN LOẠI")
print("=" * 60)

# === 5.1. Label Encoding cho Ordinal Variables (có thứ tự) ===
ordinal_mappings = {
    'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
    'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
    'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3},
    'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
    'CentralAir': {'N': 0, 'Y': 1},
}

for col, mapping in ordinal_mappings.items():
    if col in all_df.columns:
        all_df[col] = all_df[col].map(mapping).fillna(0)

print(f"✓ Ordinal encoding applied to {len(ordinal_mappings)} columns")

# === 5.2. Label Encoding cho Nominal Variables (không có thứ tự) ===
# Lấy các cột object còn lại
nominal_cols = all_df.select_dtypes(include=['object']).columns.tolist()
print(f"  Nominal columns to encode: {nominal_cols}")

for col in nominal_cols:
    le = LabelEncoder()
    all_df[col] = le.fit_transform(all_df[col].astype(str))

print(f"✓ Nominal encoding applied to {len(nominal_cols)} columns")

print("\n" + "=" * 60)
print("BƯỚC 6: CHUẨN HÓA DỮ LIỆU (StandardScaler)")
print("=" * 60)

# Tách lại train và test
n_train = len(train_features)
X_train = all_df.iloc[:n_train].copy()
X_test = all_df.iloc[n_train:].copy()

# Lưu tên features trước khi scale
feature_names = X_train.columns.tolist()

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Chuyển về DataFrame để dễ đọc
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

print(f"✓ X_train_scaled shape: {X_train_scaled.shape}")
print(f"✓ X_test_scaled shape:  {X_test_scaled.shape}")
print(f"✓ y_train shape:         {y_train.shape}")

print("\n" + "=" * 60)
print("BƯỚC 7: LƯU DỮ LIỆU ĐÃ XỬ LÝ")
print("=" * 60)

# Tạo thư mục nếu chưa có
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Lưu các file
X_train_scaled.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_train.csv"), index=False)
X_test_scaled.to_csv(os.path.join(DATA_PROCESSED_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv"), index=False, header=True)
pd.DataFrame({'Id': test_ids}).to_csv(os.path.join(DATA_PROCESSED_DIR, "test_ids.csv"), index=False)

# Lưu scaler (bằng pickle)
import pickle
with open(os.path.join(DATA_PROCESSED_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

print(f"✓ Saved: {os.path.join(DATA_PROCESSED_DIR, 'X_train.csv')}")
print(f"✓ Saved: {os.path.join(DATA_PROCESSED_DIR, 'X_test.csv')}")
print(f"✓ Saved: {os.path.join(DATA_PROCESSED_DIR, 'y_train.csv')}")
print(f"✓ Saved: {os.path.join(DATA_PROCESSED_DIR, 'test_ids.csv')}")
print(f"✓ Saved: {os.path.join(DATA_PROCESSED_DIR, 'scaler.pkl')}")

# === LOG TRANSFORM TARGET (cho model training) ===
y_train_log = np.log1p(y_train)
y_train_log.to_csv(os.path.join(DATA_PROCESSED_DIR, "y_train_log.csv"), index=False, header=True)
print(f"✓ Saved: {os.path.join(DATA_PROCESSED_DIR, 'y_train_log.csv')}")

print("\n" + "=" * 60)
print("✅ TIỀN XỬ LÝ HOÀN TẤT!")
print("=" * 60)
print(f"Output directory: {DATA_PROCESSED_DIR}")
print(f"Final features:   {X_train_scaled.shape[1]}")
print(f"Train samples:    {X_train_scaled.shape[0]}")
print(f"Test samples:     {X_test_scaled.shape[0]}")
