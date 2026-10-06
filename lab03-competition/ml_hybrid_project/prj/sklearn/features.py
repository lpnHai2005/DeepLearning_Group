"""
=================================================================
FEATURE ENGINEERING MODULE
=================================================================
Module chứa các hàm tạo features mới và xử lý feature cho sklearn

Features được tạo:
    - TotalSF: Tổng diện tích
    - TotalBath: Tổng số phòng tắm
    - HouseAge, RemodAge: Tuổi nhà
    - Binary flags: IsRemodeled, IsNew, HasGarage, HasBasement...
    - OverallScore: Chất lượng tổng hợp
    - BsmtRatio, AreaPerRoom: Tỷ lệ diện tích

Author: Thành viên 2 (sklearn pipeline)
"""

import numpy as np
import pandas as pd


def get_ordinal_mappings() -> dict:
    """
    Trả về dictionary chứa ordinal encoding mappings.

    Returns:
        Dictionary {column_name: {value: encoded_number}}
    """
    return {
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


def handle_missing_for_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý missing values và ordinal encoding cho sklearn.

    Args:
        df: DataFrame cần xử lý

    Returns:
        DataFrame đã xử lý missing values và ordinal encoding
    """
    df = df.copy()

    # Các cột NA có nghĩa là "None"
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # Các cột numeric NA = 0
    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: median imputation
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    # Áp dụng ORDINAL ENCODING
    ordinal_mappings = get_ordinal_mappings()
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Các cột categorical còn lại: mode
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Các cột numeric còn lại: 0
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)

    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo tất cả engineered features từ dataframe đã được encode.

    Args:
        df: DataFrame đã xử lý missing values và ordinal encoding

    Returns:
        DataFrame đã được thêm các features mới
    """
    df = df.copy()

    # ============================================================
    # 1. FEATURES VỀ DIỆN TÍCH
    # ============================================================

    # Tổng diện tích (bao gồm basement)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Tổng diện tích sàn (không basement)
    df['TotalFloorSF'] = df['1stFlrSF'] + df['2ndFlrSF']

    # Tổng diện tích porch
    df['TotalPorchSF'] = (
        df['OpenPorchSF'] +
        df['EnclosedPorch'] +
        df['3SsnPorch'] +
        df['ScreenPorch']
    )

    # ============================================================
    # 2. FEATURES VỀ PHÒNG TẮM
    # ============================================================

    # Tổng phòng tắm (half bath = 0.5)
    df['TotalBath'] = (
        df['FullBath'] +
        0.5 * df['HalfBath'] +
        df['BsmtFullBath'] +
        0.5 * df['BsmtHalfBath']
    )

    # ============================================================
    # 3. FEATURES VỀ TUỔI NHÀ
    # ============================================================

    # Tuổi nhà khi bán
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    # Số năm từ lần cải tạo cuối đến khi bán
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # ============================================================
    # 4. BINARY FLAGS
    # ============================================================

    # Nhà đã được cải tạo
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

    # Nhà mới (bán trong năm xây)
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)

    # Có garage
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

    # Có basement
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)

    # Có pool
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)

    # Có fireplace
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    # Có 2nd floor
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)

    # Có masonry veneer
    df['HasMasVnr'] = (df['MasVnrArea'] > 0).astype(int)

    # ============================================================
    # 5. FEATURES VỀ CHẤT LƯỢNG (đã được ordinal encode)
    # ============================================================

    # Điểm tổng hợp chất lượng
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']

    # Chất lượng exterior tổng hợp
    df['ExterScore'] = df['ExterQual'] * df['ExterCond']

    # Chất lượng basement tổng hợp
    df['BsmtScore'] = df['BsmtQual'] * df['BsmtCond']

    # Chất lượng garage tổng hợp
    df['GarageScore'] = df['GarageQual'] * df['GarageCond']

    # ============================================================
    # 6. RATIO FEATURES
    # ============================================================

    # Tỷ lệ basement / tầng 1
    df['BsmtRatio'] = df['TotalBsmtSF'] / (df['1stFlrSF'] + 1)

    # Diện tích trung bình mỗi phòng
    df['AreaPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)

    # Tỷ lệ diện tích sống / diện tích đất
    df['LivAreaRatio'] = df['GrLivArea'] / (df['LotArea'] + 1)

    # Garage area per car
    df['GarageAreaPerCar'] = df['GarageArea'] / (df['GarageCars'] + 1)

    # Lot frontage ratio
    df['LotFrontageRatio'] = df['LotFrontage'] / (np.sqrt(df['LotArea']) + 1)

    # ============================================================
    # 7. INTERACTION FEATURES
    # ============================================================

    # Diện tích × Chất lượng
    df['SF_Qual_Interaction'] = df['TotalSF'] * df['OverallQual']

    # Năm xây × Chất lượng
    df['Year_Qual_Interaction'] = df['YearBuilt'] * df['OverallQual']

    # Diện tích × Số phòng tắm
    df['SF_Bath_Interaction'] = df['TotalSF'] * df['TotalBath']

    return df


if __name__ == "__main__":
    # Test module
    print("features.py loaded successfully")
    print("Available functions:")
    print("  - create_all_features(df)")
    print("  - handle_missing_for_sklearn(df)")
    print("  - get_ordinal_mappings()")
