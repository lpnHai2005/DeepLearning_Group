"""
=================================================================
TRAIN SKLEARN - TÌM MÔ HÌNH TỐT NHẤT THỰC SỰ
=================================================================
1. Preprocess đầy đủ giống code gốc
2. Train tất cả models trên full data
3. Tạo submission cho TẤT CẢ models
4. So sánh để tìm model tốt nhất

Author: Deep Learning Lab03
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# UTF-8
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "prj", "model")

os.makedirs(MODEL_DIR, exist_ok=True)


def calculate_rmsle(y_true, y_pred):
    """Tính RMSLE - metric chính thức của Kaggle"""
    y_pred = np.maximum(y_pred, 1)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def load_and_preprocess():
    """
    Load và preprocess dữ liệu GIỐNG HỆT code gốc
    """
    print("\n" + "=" * 60)
    print("LOADING & PREPROCESSING DATA")
    print("=" * 60)

    # Load
    train_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "test.csv"))
    print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")

    # Save IDs
    test_ids = test_df['Id'].copy()

    # Target - log transform
    y_original = train_df['SalePrice'].values
    y_log = np.log1p(y_original)

    # Features
    train_features = train_df.drop(['SalePrice', 'Id'], axis=1)
    test_features = test_df.drop(['Id'], axis=1)

    # Combine for consistent processing
    all_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    n_train = len(train_features)
    print(f"✓ Combined: {all_df.shape}")

    # ============================================================
    # HANDLE MISSING VALUES (giống features.py)
    # ============================================================

    # NA có nghĩa là "None"
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    for col in none_cols:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna('None')

    # Numeric NA = 0
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
    for col in zero_cols:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna(0)

    # LotFrontage - median by neighborhood
    if 'LotFrontage' in all_df.columns:
        all_df['LotFrontage'] = all_df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        all_df['LotFrontage'] = all_df['LotFrontage'].fillna(all_df['LotFrontage'].median())

    # Fill remaining
    for col in all_df.columns:
        if all_df[col].isnull().sum() > 0:
            if all_df[col].dtype == 'object':
                all_df[col] = all_df[col].fillna(all_df[col].mode()[0] if len(all_df[col].mode()) > 0 else 'None')
            else:
                all_df[col] = all_df[col].fillna(0)

    print("✓ Missing values handled")

    # ============================================================
    # ORDINAL ENCODING
    # ============================================================
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

    print("✓ Ordinal encoding applied")

    # ============================================================
    # FEATURE ENGINEERING (giống features.py)
    # ============================================================

    # Diện tích
    all_df['TotalSF'] = all_df['TotalBsmtSF'] + all_df['1stFlrSF'] + all_df['2ndFlrSF']
    all_df['TotalFloorSF'] = all_df['1stFlrSF'] + all_df['2ndFlrSF']
    all_df['TotalPorchSF'] = (all_df['OpenPorchSF'] + all_df['EnclosedPorch'] +
                              all_df['3SsnPorch'] + all_df['ScreenPorch'])

    # Phòng tắm
    all_df['TotalBath'] = (all_df['FullBath'] + 0.5 * all_df['HalfBath'] +
                          all_df['BsmtFullBath'] + 0.5 * all_df['BsmtHalfBath'])

    # Tuổi nhà
    all_df['HouseAge'] = all_df['YrSold'] - all_df['YearBuilt']
    all_df['RemodAge'] = all_df['YrSold'] - all_df['YearRemodAdd']

    # Binary flags
    all_df['IsRemodeled'] = (all_df['YearRemodAdd'] != all_df['YearBuilt']).astype(int)
    all_df['IsNew'] = (all_df['YrSold'] == all_df['YearBuilt']).astype(int)
    all_df['HasGarage'] = (all_df['GarageArea'] > 0).astype(int)
    all_df['HasBasement'] = (all_df['TotalBsmtSF'] > 0).astype(int)
    all_df['HasPool'] = (all_df['PoolArea'] > 0).astype(int)
    all_df['HasFireplace'] = (all_df['Fireplaces'] > 0).astype(int)
    all_df['Has2ndFloor'] = (all_df['2ndFlrSF'] > 0).astype(int)
    all_df['HasMasVnr'] = (all_df['MasVnrArea'] > 0).astype(int)

    # Quality scores
    all_df['OverallScore'] = all_df['OverallQual'] * all_df['OverallCond']
    all_df['ExterScore'] = all_df['ExterQual'] * all_df['ExterCond']
    all_df['BsmtScore'] = all_df['BsmtQual'] * all_df['BsmtCond']
    all_df['GarageScore'] = all_df['GarageQual'] * all_df['GarageCond']

    # Ratios
    all_df['BsmtRatio'] = all_df['TotalBsmtSF'] / (all_df['1stFlrSF'] + 1)
    all_df['AreaPerRoom'] = all_df['GrLivArea'] / (all_df['TotRmsAbvGrd'] + 1)
    all_df['LivAreaRatio'] = all_df['GrLivArea'] / (all_df['LotArea'] + 1)
    all_df['GarageAreaPerCar'] = all_df['GarageArea'] / (all_df['GarageCars'] + 1)
    all_df['LotFrontageRatio'] = all_df['LotFrontage'] / (np.sqrt(all_df['LotArea']) + 1)

    # Interactions
    all_df['SF_Qual_Interaction'] = all_df['TotalSF'] * all_df['OverallQual']
    all_df['Year_Qual_Interaction'] = all_df['YearBuilt'] * all_df['OverallQual']
    all_df['SF_Bath_Interaction'] = all_df['TotalSF'] * all_df['TotalBath']

    print(f"✓ Feature engineering: {all_df.shape[1]} features")

    # ============================================================
    # NOMINAL ENCODING (LabelEncoder)
    # ============================================================
    nominal_cols = all_df.select_dtypes(include=['object']).columns.tolist()
    for col in nominal_cols:
        le = LabelEncoder()
        all_df[col] = le.fit_transform(all_df[col].astype(str))

    print(f"✓ Nominal encoding applied to {len(nominal_cols)} columns")

    # ============================================================
    # SPLIT & SCALE
    # ============================================================
    X_all = all_df.iloc[:n_train].values
    X_test = all_df.iloc[n_train:].values

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_all)
    X_test_scaled = scaler.transform(X_test)

    print(f"✓ Final: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}")

    # Save scaler and feature names
    with open(os.path.join(DATA_PROCESSED_DIR, "scaler_sklearn.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    feature_names = all_df.columns.tolist()
    with open(os.path.join(DATA_PROCESSED_DIR, "feature_names.pkl"), 'wb') as f:
        pickle.dump(feature_names, f)

    return X_train_scaled, X_test_scaled, y_log, y_original, test_ids


def cross_validate(model, X, y_log, y_original, n_splits=5):
    """
    Cross-validation với RMSLE đúng
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmsle_scores = []

    for train_idx, val_idx in kfold.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val_log = y_log[train_idx], y_log[val_idx]
        y_val_orig = y_original[val_idx]

        model.fit(X_tr, y_tr)
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)

        rmsle = calculate_rmsle(y_val_orig, y_pred)
        rmsle_scores.append(rmsle)

    return np.mean(rmsle_scores), np.std(rmsle_scores)


def train_and_compare_all(X_train, y_log, y_original, X_test, test_ids):
    """
    Train tất cả models và so sánh bằng Cross-Validation
    """
    print("\n" + "=" * 70)
    print("TRAINING & COMPARING ALL MODELS (5-Fold CV)")
    print("=" * 70)

    models_config = {
        'ridge': Ridge(alpha=10.0, random_state=42),
        'lasso': Lasso(alpha=0.001, random_state=42, max_iter=10000),
        'elastic': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000),
        'rf': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                    min_samples_leaf=2, random_state=42, n_jobs=-1),
        'gb': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                        min_samples_split=5, min_samples_leaf=2, random_state=42),
    }

    # Thử XGBoost
    try:
        from xgboost import XGBRegressor
        models_config['xgb'] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
    except ImportError:
        print("⚠️ XGBoost not installed")

    results = {}
    trained_models = {}

    print(f"\n{'Model':<20} | {'RMSLE Mean':<12} | {'RMSLE Std':<12} | {'Status'}")
    print("-" * 60)

    for name, model in models_config.items():
        print(f"Training {name.upper()}...", end=" ")

        # Cross-validation
        rmsle_mean, rmsle_std = cross_validate(model, X_train, y_log, y_original)

        # Train on full data
        model.fit(X_train, y_log)

        results[name] = {
            'rmsle_mean': rmsle_mean,
            'rmsle_std': rmsle_std
        }
        trained_models[name] = model

        print(f"RMSLE = {rmsle_mean:.4f} ± {rmsle_std:.4f}")

    # Sort by RMSLE
    results_df = pd.DataFrame(results).T.sort_values('rmsle_mean')

    print("\n" + "=" * 70)
    print("KẾT QUẢ SẮP XẾP THEO RMSLE (THẤP NHẤT = TỐT NHẤT)")
    print("=" * 70)
    print(f"\n{'Rank':<6} | {'Model':<20} | {'RMSLE Mean':<12} | {'RMSLE Std':<12}")
    print("-" * 55)

    for i, (name, row) in enumerate(results_df.iterrows(), 1):
        marker = "🏆" if i == 1 else "   "
        print(f"{marker}{i:<3}  | {name.upper():<20} | {row['rmsle_mean']:.4f}     | {row['rmsle_std']:.4f}")

    best_model_name = results_df.index[0]
    print(f"\n🏆 MÔ HÌNH TỐT NHẤT: {best_model_name.upper()}")
    print(f"   RMSLE (CV): {results_df.iloc[0]['rmsle_mean']:.4f}")

    return trained_models, results_df, best_model_name


def create_all_submissions(trained_models, X_test, test_ids):
    """
    Tạo submission cho TẤT CẢ models
    """
    print("\n" + "=" * 60)
    print("CREATING SUBMISSION FILES FOR ALL MODELS")
    print("=" * 60)

    submissions = {}

    for name, model in trained_models.items():
        # Predict
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        # Create submission
        submission = pd.DataFrame({
            'Id': test_ids,
            'SalePrice': y_pred
        })

        # Save
        filename = f"submission_{name}.csv"
        path = os.path.join(MODEL_DIR, filename)
        submission.to_csv(path, index=False)

        # Also save to processed
        submission.to_csv(os.path.join(DATA_PROCESSED_DIR, filename), index=False)

        submissions[name] = path
        print(f"✓ {name.upper()}: {filename}")
        print(f"   Range: ${y_pred.min():,.0f} - ${y_pred.max():,.0f}, Mean: ${y_pred.mean():,.0f}")

    # Create ensemble
    print("\nCreating ensemble (weighted average)...")
    ensemble_pred = np.zeros(len(X_test))
    weights = {'ridge': 0.05, 'lasso': 0.05, 'elastic': 0.10,
               'rf': 0.15, 'gb': 0.30, 'xgb': 0.35}
    total_weight = 0

    for name, model in trained_models.items():
        pred = model.predict(X_test)
        w = weights.get(name, 0.1)
        ensemble_pred += w * pred
        total_weight += w
        print(f"   + {name.upper()}: weight={w}")

    ensemble_pred /= total_weight
    ensemble_pred = np.expm1(ensemble_pred)

    ensemble_sub = pd.DataFrame({'Id': test_ids, 'SalePrice': ensemble_pred})
    ensemble_path = os.path.join(MODEL_DIR, "submission_ensemble.csv")
    ensemble_sub.to_csv(ensemble_path, index=False)
    ensemble_sub.to_csv(os.path.join(DATA_PROCESSED_DIR, "submission_ensemble.csv"), index=False)
    print(f"\n✓ ENSEMBLE: submission_ensemble.csv")
    print(f"   Range: ${ensemble_pred.min():,.0f} - ${ensemble_pred.max():,.0f}, Mean: ${ensemble_pred.mean():,.0f}")

    # Đảm bảo lưu submission_sklearn.csv (từ best pure Scikit-Learn Model: Gradient Boosting, Kaggle: 0.12966)
    if 'gb' in trained_models:
        gb_pred = np.expm1(trained_models['gb'].predict(X_test))
        sub_sk = pd.DataFrame({'Id': test_ids, 'SalePrice': gb_pred})
        sub_sk.to_csv(os.path.join(MODEL_DIR, "submission_sklearn.csv"), index=False)
        sub_sk.to_csv(os.path.join(DATA_PROCESSED_DIR, "submission_sklearn.csv"), index=False)
        print("✓ BEST SKLEARN (GB): submission_sklearn.csv (Kaggle: 0.12966)")

    return submissions


def save_models(trained_models):
    """Lưu tất cả models"""
    print("\n" + "=" * 60)
    print("SAVING ALL MODELS")
    print("=" * 60)

    for name, model in trained_models.items():
        path = os.path.join(DATA_PROCESSED_DIR, f"model_{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved: model_{name}.pkl")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print(" SKLEARN - TÌM MÔ HÌNH TỐT NHẤT THỰC SỰ")
    print(" Sử dụng 5-Fold Cross-Validation để đánh giá RMSLE chính xác")
    print("=" * 70)

    # Step 1: Load & Preprocess
    X_train, X_test, y_log, y_original, test_ids = load_and_preprocess()

    # Step 2: Train & Compare all models
    trained_models, results_df, best_model_name = train_and_compare_all(
        X_train, y_log, y_original, X_test, test_ids
    )

    # Step 3: Create submissions for all models
    submissions = create_all_submissions(trained_models, X_test, test_ids)

    # Step 4: Save all models
    save_models(trained_models)

    # Summary
    print("\n" + "=" * 70)
    print(" ✅ HOÀN TẤT - TỔNG KẾT")
    print("=" * 70)

    print("\n📊 KẾT QUẢ CROSS-VALIDATION:")
    for i, (name, row) in enumerate(results_df.iterrows(), 1):
        marker = "🏆" if i == 1 else "   "
        print(f"   {marker} {i}. {name.upper():<15}: RMSLE = {row['rmsle_mean']:.4f} ± {row['rmsle_std']:.4f}")

    print(f"\n🏆 MÔ HÌNH TỐT NHẤT: {best_model_name.upper()}")
    print(f"   RMSLE (5-Fold CV): {results_df.iloc[0]['rmsle_mean']:.4f}")

    print("\n📁 FILES ĐÃ TẠO:")
    print("   ├─ submission_xgb.csv")
    print("   ├─ submission_gb.csv")
    print("   ├─ submission_rf.csv")
    print("   ├─ submission_elastic.csv")
    print("   ├─ submission_lasso.csv")
    print("   ├─ submission_ridge.csv")
    print("   ├─ submission_sklearn.csv (GB - Kaggle: 0.12966)")
    print("   └─ submission_ensemble.csv")

    print("\n" + "=" * 70)
    print(" Nộp file submission_xgb.csv hoặc submission_ensemble.csv lên Kaggle")
    print("=" * 70)
