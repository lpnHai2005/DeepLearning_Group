"""
=================================================================
TRAIN SKLEARN BASELINE MODELS
=================================================================
Module huấn luyện các mô hình Machine Learning cơ bản:
    - Ridge Regression
    - Lasso Regression
    - ElasticNet
    - Random Forest
    - Gradient Boosting
    - XGBoost (nếu có)

Metrics: RMSE, MAE, R² (trên log-transformed target)

Output:
    - Mô hình đã train (pickle)
    - Submission file cho Kaggle
    - Feature importance plot

Author: Thành viên 2 (sklearn pipeline)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports
from features import create_all_features, handle_missing_for_sklearn, get_ordinal_mappings

# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "prj", "model")
EXPS_DATA_DIR = os.path.join(BASE_DIR, "exps", "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


def load_data():
    """Đọc dữ liệu từ data/raw"""
    train_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "test.csv"))
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """
    Tiền xử lý dữ liệu hoàn chỉnh.

    Steps:
        1. Tách target (log-transform)
        2. Xử lý missing values
        3. Tạo engineered features
        4. Ordinal encoding
        5. Nominal encoding (LabelEncoder)
        6. Chuẩn hóa
    """
    print("\n" + "=" * 60)
    print("TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    # Lưu IDs
    test_ids = test_df['Id'].copy()

    # 1. Tách target (log-transform vì SalePrice bị skewed)
    y_train = np.log1p(train_df['SalePrice'])
    y_train_original = train_df['SalePrice'].copy()
    print(f"✓ Target: log1p(SalePrice), shape: {y_train.shape}")

    # 2. Tách features
    train_features = train_df.drop(['SalePrice', 'Id'], axis=1)
    test_features = test_df.drop(['Id'], axis=1)

    # 3. Gộp train + test để xử lý đồng bộ
    all_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    print(f"✓ Combined features shape: {all_df.shape}")

    # 4. Xử lý missing values
    all_df = handle_missing_for_sklearn(all_df)

    # 5. Tạo engineered features
    all_df = create_all_features(all_df)
    print(f"✓ After feature engineering: {all_df.shape[1]} features")

    # 6. Ordinal encoding
    ordinal_mappings = get_ordinal_mappings()
    for col, mapping in ordinal_mappings.items():
        if col in all_df.columns:
            all_df[col] = all_df[col].map(mapping).fillna(0)
    print(f"✓ Ordinal encoding applied")

    # 7. Nominal encoding (LabelEncoder)
    nominal_cols = all_df.select_dtypes(include=['object']).columns.tolist()
    for col in nominal_cols:
        le = LabelEncoder()
        all_df[col] = le.fit_transform(all_df[col].astype(str))
    print(f"✓ Nominal encoding applied to {len(nominal_cols)} columns")

    # 8. Tách train/test
    n_train = len(train_features)
    X_train = all_df.iloc[:n_train].copy()
    X_test = all_df.iloc[n_train:].copy()

    # 9. Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Chuyển về DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print(f"✓ Final X_train shape: {X_train_scaled.shape}")
    print(f"✓ Final X_test shape:  {X_test_scaled.shape}")

    # Lưu scaler
    with open(os.path.join(DATA_PROCESSED_DIR, "scaler_sklearn.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled, y_train, y_train_original, test_ids


def evaluate_model(model, X_train, y_train, name="Model"):
    """
    Đánh giá mô hình bằng Cross-Validation.

    Args:
        model: Mô hình sklearn
        X_train: Features huấn luyện
        y_train: Target (log-transformed)
        name: Tên mô hình

    Returns:
        dict với các metrics
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # RMSE (neg_mean_squared_error trả về giá trị âm)
    rmse_scores = cross_val_score(
        model, X_train, y_train,
        cv=kfold, scoring='neg_root_mean_squared_error'
    )
    rmse_scores = -rmse_scores

    # R²
    r2_scores = cross_val_score(
        model, X_train, y_train,
        cv=kfold, scoring='r2'
    )

    results = {
        'name': name,
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
    }

    return results


def train_all_models(X_train, y_train):
    """
    Huấn luyện tất cả các mô hình và so sánh.

    Args:
        X_train: Features đã chuẩn hóa
        y_train: Target (log-transformed)

    Returns:
        dict chứa tất cả mô hình và kết quả
    """
    print("\n" + "=" * 60)
    print("HUẤN LUYỆN CÁC MÔ HÌNH")
    print("=" * 60)

    results = {}
    trained_models = {}

    # ── 1. Ridge Regression ──
    print("\n[1/6] Ridge Regression...")
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train, y_train)
    result = evaluate_model(ridge, X_train, y_train, "Ridge")
    results['ridge'] = result
    trained_models['ridge'] = ridge
    print(f"    RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
    print(f"    R²:   {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

    # ── 2. Lasso Regression ──
    print("\n[2/6] Lasso Regression...")
    lasso = Lasso(alpha=0.001, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    result = evaluate_model(lasso, X_train, y_train, "Lasso")
    results['lasso'] = result
    trained_models['lasso'] = lasso
    print(f"    RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
    print(f"    R²:   {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

    # ── 3. ElasticNet ──
    print("\n[3/6] ElasticNet...")
    elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000)
    elastic.fit(X_train, y_train)
    result = evaluate_model(elastic, X_train, y_train, "ElasticNet")
    results['elastic'] = result
    trained_models['elastic'] = elastic
    print(f"    RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
    print(f"    R²:   {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

    # ── 4. Random Forest ──
    print("\n[4/6] Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    result = evaluate_model(rf, X_train, y_train, "RandomForest")
    results['rf'] = result
    trained_models['rf'] = rf
    print(f"    RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
    print(f"    R²:   {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

    # ── 5. Gradient Boosting ──
    print("\n[5/6] Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb.fit(X_train, y_train)
    result = evaluate_model(gb, X_train, y_train, "GradientBoosting")
    results['gb'] = result
    trained_models['gb'] = gb
    print(f"    RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
    print(f"    R²:   {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

    # ── 6. Try XGBoost (nếu có) ──
    print("\n[6/6] XGBoost...")
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        result = evaluate_model(xgb, X_train, y_train, "XGBoost")
        results['xgb'] = result
        trained_models['xgb'] = xgb
        print(f"    RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
        print(f"    R²:   {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
    except ImportError:
        print("    ⚠️ XGBoost not installed, skipping...")

    # ── Tổng hợp kết quả ──
    print("\n" + "=" * 60)
    print("TỔNG HỢP KẾT QUẢ")
    print("=" * 60)

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('rmse_mean')
    print(results_df[['rmse_mean', 'rmse_std', 'r2_mean']].to_string())

    return trained_models, results


def create_submission(model, X_test, test_ids, filename="submission_sklearn.csv"):
    """
    Tạo file submission cho Kaggle.

    Args:
        model: Mô hình đã train
        X_test: Features test
        test_ids: IDs của test set
        filename: Tên file output
    """
    print("\n" + "=" * 60)
    print("TẠO SUBMISSION")
    print("=" * 60)

    # Dự đoán (log scale)
    y_pred_log = model.predict(X_test)

    # Chuyển về original scale
    y_pred = np.expm1(y_pred_log)

    # Tạo submission DataFrame
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred
    })

    # Lưu
    submission_path = os.path.join(MODEL_DIR, filename)
    submission.to_csv(submission_path, index=False)
    print(f"✓ Saved: {submission_path}")

    # Cũng lưu vào data/processed
    submission.to_csv(os.path.join(DATA_PROCESSED_DIR, filename), index=False)

    # In thống kê
    print(f"\n📊 Prediction Statistics:")
    print(f"   Min:    ${y_pred.min():,.2f}")
    print(f"   Max:    ${y_pred.max():,.2f}")
    print(f"   Mean:   ${y_pred.mean():,.2f}")
    print(f"   Median: ${np.median(y_pred):,.2f}")

    return submission


def save_models(trained_models):
    """Lưu tất cả mô hình đã train"""
    print("\n" + "=" * 60)
    print("LƯU MÔ HÌNH")
    print("=" * 60)

    for name, model in trained_models.items():
        path = os.path.join(DATA_PROCESSED_DIR, f"model_{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("SKLEARN BASELINE MODELS TRAINING")
    print("House Price Prediction - Lab03")
    print("=" * 60)

    # 1. Load data
    train_df, test_df = load_data()
    print(f"✓ Loaded train: {train_df.shape}, test: {test_df.shape}")

    # 2. Preprocess
    X_train, X_test, y_train, y_train_original, test_ids = preprocess_data(train_df, test_df)

    # 3. Train all models
    trained_models, results = train_all_models(X_train, y_train)

    # 4. Chọn mô hình tốt nhất (RMSE thấp nhất)
    best_model_name = min(results, key=lambda k: results[k]['rmse_mean'])
    best_model = trained_models[best_model_name]
    print(f"\n🏆 Best model: {best_model_name} (RMSE: {results[best_model_name]['rmse_mean']:.4f})")

    # 5. Save all models
    save_models(trained_models)

    # 6. Create submission với mô hình tốt nhất
    submission = create_submission(best_model, X_test, test_ids, f"submission_{best_model_name}.csv")

    # 7. Tạo ensemble submission (trung bình các models)
    print("\n" + "=" * 60)
    print("ENSEMBLE PREDICTION")
    print("=" * 60)

    ensemble_pred = np.zeros(len(X_test))
    for name, model in trained_models.items():
        pred = model.predict(X_test)
        ensemble_pred += pred
    ensemble_pred /= len(trained_models)

    # Chuyển về original scale
    ensemble_pred_original = np.expm1(ensemble_pred)

    submission_ensemble = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': ensemble_pred_original
    })

    ensemble_path = os.path.join(MODEL_DIR, "submission_ensemble.csv")
    submission_ensemble.to_csv(ensemble_path, index=False)
    print(f"✓ Ensemble saved: {ensemble_path}")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
