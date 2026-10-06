"""
=================================================================
PYTORCH CUSTOM DATASET & DATALOADER
=================================================================
Module chứa class Dataset tùy chỉnh và DataLoader cho PyTorch MLP.

Classes:
    - HousePriceDataset: Dataset cho bài toán House Price

DataLoader Config:
    - batch_size: 64
    - shuffle: True (train) / False (test)
    - num_workers: 0 (Windows compatible)


"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


class HousePriceDataset(Dataset):
    """
    Custom Dataset cho bài toán House Price Prediction.

    Args:
        X (np.ndarray or pd.DataFrame): Features
        y (np.ndarray, optional): Target (log-transformed SalePrice)
        scaler (StandardScaler, optional): Scaler để transform
        is_train (bool): True nếu là train set (có target)
    """

    def __init__(self, X, y=None, is_train=True):
        """
        Khởi tạo Dataset.

        Args:
            X: Features (numpy array hoặc DataFrame)
            y: Target (numpy array), None nếu là test set
            is_train: Cờ cho biết có target hay không
        """
        # Chuyển DataFrame -> numpy
        if hasattr(X, 'values'):
            self.X = X.values.astype(np.float32)
        else:
            self.X = X.astype(np.float32)

        self.is_train = is_train

        # Target (đã log-transformed)
        if y is not None:
            if hasattr(y, 'values'):
                self.y = y.values.astype(np.float32)
            else:
                self.y = y.astype(np.float32)
        else:
            self.y = None

        # Feature names (để debug/analysis)
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]

    def __len__(self):
        """Số lượng samples"""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Lấy một sample.

        Returns:
            tuple: (features, target) nếu train, (features,) nếu test
        """
        X = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_train and self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return X, y
        else:
            return X

    def get_feature_names(self):
        """Trả về tên các features"""
        return self.feature_names


def create_dataloaders(batch_size=64, val_split=0.2, random_seed=42):
    """
    Tạo DataLoaders cho train và test.

    Args:
        batch_size: Kích thước batch
        val_split: Tỷ lệ validation split
        random_seed: Random seed cho reproducibility

    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler)
    """
    print("=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)

    # ── Load raw data ──
    train_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "test.csv"))

    print(f"✓ Loaded train: {train_df.shape}")
    print(f"✓ Loaded test:  {test_df.shape}")

    # ── Preprocess ──
    from sklearn.preprocessing import LabelEncoder

    # Lưu IDs
    test_ids = test_df['Id'].copy()

    # Target (log-transformed)
    y_train = np.log1p(train_df['SalePrice'].values).astype(np.float32)
    y_train_original = train_df['SalePrice'].values.astype(np.float32)

    # Features
    train_features = train_df.drop(['SalePrice', 'Id'], axis=1)
    test_features = test_df.drop(['Id'], axis=1)

    # Gộp để xử lý đồng bộ
    all_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    # ── Missing Values ──
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in none_cols:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna('None')

    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna(0)

    if 'LotFrontage' in all_df.columns:
        all_df['LotFrontage'] = all_df['LotFrontage'].fillna(all_df['LotFrontage'].median())

    for col in all_df.select_dtypes(include=['object']).columns:
        if all_df[col].isnull().sum() > 0:
            all_df[col] = all_df[col].fillna(all_df[col].mode()[0])

    all_df = all_df.fillna(0)

    # ── Feature Engineering ──
    all_df['TotalSF'] = all_df['TotalBsmtSF'] + all_df['1stFlrSF'] + all_df['2ndFlrSF']
    all_df['TotalBath'] = (all_df['FullBath'] + 0.5 * all_df['HalfBath'] +
                           all_df['BsmtFullBath'] + 0.5 * all_df['BsmtHalfBath'])
    all_df['HouseAge'] = all_df['YrSold'] - all_df['YearBuilt']
    all_df['RemodAge'] = all_df['YrSold'] - all_df['YearRemodAdd']
    all_df['OverallScore'] = all_df['OverallQual'] * all_df['OverallCond']

    # ── Ordinal Encoding ──
    ordinal_mappings = {
        'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
        'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
        'CentralAir': {'N': 0, 'Y': 1},
    }

    for col, mapping in ordinal_mappings.items():
        if col in all_df.columns:
            all_df[col] = all_df[col].map(mapping).fillna(0)

    # ── Label Encoding cho nominal columns ──
    nominal_cols = all_df.select_dtypes(include=['object']).columns.tolist()
    for col in nominal_cols:
        le = LabelEncoder()
        all_df[col] = le.fit_transform(all_df[col].astype(str))

    # ── Tach train/test ──
    n_train = len(train_features)
    X_train_full = all_df.iloc[:n_train].values.astype(np.float32)
    X_test = all_df.iloc[n_train:].values.astype(np.float32)
    y_train_full = y_train.copy()  # Tao copy de shuffle

    # ── Train/Val Split ──
    np.random.seed(random_seed)
    indices = np.random.permutation(n_train)
    val_size = int(n_train * val_split)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = X_train_full[train_indices]
    X_val = X_train_full[val_indices]
    y_train = y_train_full[train_indices]
    y_val = y_train_full[val_indices]

    # ── Chuẩn hóa ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ X_val shape:   {X_val.shape}")
    print(f"✓ X_test shape:  {X_test.shape}")

    # ── Tạo Datasets ──
    train_dataset = HousePriceDataset(X_train, y_train, is_train=True)
    val_dataset = HousePriceDataset(X_val, y_val, is_train=True)
    test_dataset = HousePriceDataset(X_test, is_train=False)

    # ── Tạo DataLoaders ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\n✓ Created DataLoaders:")
    print(f"  - Train: {len(train_loader)} batches (batch_size={batch_size})")
    print(f"  - Val:   {len(val_loader)} batches")
    print(f"  - Test:  {len(test_loader)} batches")

    # Lưu scaler
    scaler_path = os.path.join(DATA_PROCESSED_DIR, "scaler_pytorch.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved: {scaler_path}")

    # Lưu test_ids
    pd.DataFrame({'Id': test_ids}).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "test_ids.csv"), index=False
    )

    # Luu original target (de danh gia)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_train_original.npy"), y_train_original)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_train_full.npy"), y_train_full)

    return train_loader, val_loader, test_loader, scaler, test_ids


if __name__ == "__main__":
    print("✅ dataset.py loaded successfully")
    print("\nTesting dataloader creation...")

    train_loader, val_loader, test_loader, scaler, test_ids = create_dataloaders(batch_size=64)

    # Test một batch
    print("\n" + "=" * 60)
    print("TESTING ONE BATCH")
    print("=" * 60)

    for batch_X, batch_y in train_loader:
        print(f"✓ Batch X shape: {batch_X.shape}")
        print(f"✓ Batch y shape: {batch_y.shape}")
        print(f"✓ X mean: {batch_X.mean():.4f}, std: {batch_X.std():.4f}")
        print(f"✓ y mean (log): {batch_y.mean():.4f}")
        break
