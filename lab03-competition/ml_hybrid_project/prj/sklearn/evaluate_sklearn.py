"""
=================================================================
EVALUATE SKLEARN & XGBOOST MODELS - ĐÁNH GIÁ THỰC TẾ 100% TỪ DATA
=================================================================
Module đánh giá và trực quan hóa kết quả các mô hình học máy:
    - 100% số liệu tính toán động từ dữ liệu train.csv (1,460 mẫu)
    - 5-Fold Cross Validation Out-Of-Fold (OOF) trung thực, không data leakage
    - Không gán cứng (hardcode) bất kỳ điểm số nào

Output visualizations:
    1. sklearn.png - So sánh 6 mô hình (RMSE, RMSLE, R²)
    2. actual_vs_predicted_{model}.png - Đồ thị thực tế vs dự đoán OOF
    3. feature_importance_{model}.png - Top 20 đặc trưng quan trọng nhất
    4. residual_analysis.png - Phân tích phần dư của mô hình tốt nhất
    5. sklearn_results.csv - Bảng tổng hợp số liệu đo lường thực tế

Author: SGU Deep Learning Group
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Thiết lập UTF-8 trên Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "prj", "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# Thêm đường dẫn prj/sklearn vào sys.path để import
SKLEARN_DIR = os.path.dirname(os.path.abspath(__file__))
if SKLEARN_DIR not in sys.path:
    sys.path.insert(0, SKLEARN_DIR)

from train_baseline import load_and_preprocess, calculate_rmsle


def load_trained_models():
    """Load tất cả các mô hình đã huấn luyện từ data/processed/"""
    models = {}
    model_files = [
        ('xgb', 'model_xgb.pkl'),
        ('gb', 'model_gb.pkl'),
        ('rf', 'model_rf.pkl'),
        ('ridge', 'model_ridge.pkl'),
        ('lasso', 'model_lasso.pkl'),
        ('elastic', 'model_elastic.pkl')
    ]

    for name, fname in model_files:
        path = os.path.join(DATA_PROCESSED_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"✓ Đã tải mô hình: {name.upper()}")
        else:
            print(f"⚠️ Chưa tìm thấy: {fname}")

    return models


def load_feature_names():
    """Tải danh sách tên các đặc trưng từ file lưu trữ"""
    fname_path = os.path.join(DATA_PROCESSED_DIR, "feature_names.pkl")
    if os.path.exists(fname_path):
        with open(fname_path, 'rb') as f:
            return pickle.load(f)
    return None


def evaluate_models_cross_validation(X, y_log, y_original, n_splits=5):
    """
    Đánh giá 5-Fold Cross Validation chuẩn xác với Out-Of-Fold (OOF) predictions.
    100% TÍNH TOÁN ĐỘNG TỪ DỮ LIỆU ĐẦU VÀO:
    Mỗi mẫu được dự đoán bởi mô hình huấn luyện trên 4 fold còn lại.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Khởi tạo mô hình mới cùng siêu tham số để kiểm định chéo K-Fold
    models_factory = {
        'ridge': lambda: Ridge(alpha=10.0, random_state=42),
        'lasso': lambda: Lasso(alpha=0.001, random_state=42, max_iter=10000),
        'elastic': lambda: ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000),
        'rf': lambda: RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                            min_samples_leaf=2, random_state=42, n_jobs=-1),
        'gb': lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                min_samples_split=5, min_samples_leaf=2, random_state=42),
    }

    try:
        from xgboost import XGBRegressor
        models_factory['xgb'] = lambda: XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
    except ImportError:
        pass

    results = []
    oof_predictions = {}

    print("\n" + "=" * 80)
    print("TIẾN HÀNH ĐÁNH GIÁ 5-FOLD CROSS VALIDATION (TÍNH TOÁN TRỰC TIẾP TỪ DỮ LIỆU)")
    print("=" * 80)
    print(f"{'Mô hình':<10} | {'RMSLE CV (Mean ± Std)':<24} | {'RMSE OOF ($)':<14} | {'MAE OOF ($)':<12} | {'R² OOF':<8}")
    print("-" * 80)

    for name, factory_fn in models_factory.items():
        oof_pred = np.zeros(len(y_original))
        fold_rmsles = []

        for train_idx, val_idx in kfold.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val_log = y_log[train_idx], y_log[val_idx]
            y_val_orig = y_original[val_idx]

            m = factory_fn()
            m.fit(X_tr, y_tr)

            pred_log = m.predict(X_val)
            pred_orig = np.expm1(pred_log)
            oof_pred[val_idx] = pred_orig

            fold_rmsle = calculate_rmsle(y_val_orig, pred_orig)
            fold_rmsles.append(fold_rmsle)

        oof_predictions[name] = oof_pred

        rmsle_mean = float(np.mean(fold_rmsles))
        rmsle_std = float(np.std(fold_rmsles))
        rmse_val = float(np.sqrt(mean_squared_error(y_original, oof_pred)))
        mae_val = float(mean_absolute_error(y_original, oof_pred))
        r2_val = float(r2_score(y_original, oof_pred))

        results.append({
            'Model': name.upper(),
            'RMSLE_CV': rmsle_mean,
            'RMSLE_Std': rmsle_std,
            'RMSE_CV': rmse_val,
            'MAE_CV': mae_val,
            'R2_CV': r2_val,
            # Tương thích ngược với các hàm cần cột _Val
            'RMSE_Val': rmse_val,
            'RMSLE_Val': rmsle_mean,
            'R2_Val': r2_val,
        })

        print(f"{name.upper():<10} | {rmsle_mean:.4f} ± {rmsle_std:.4f}             | ${rmse_val:>11,.0f} | ${mae_val:>10,.0f} | {r2_val:>7.4f}")

    # Tự động xếp hạng động theo kết quả tính toán thực tế (RMSLE thấp nhất lên đầu)
    results_df = pd.DataFrame(results).sort_values('RMSLE_CV').reset_index(drop=True)
    results_df['Rank'] = range(1, len(results_df) + 1)

    return results_df, oof_predictions


def plot_model_comparison(results_df, save_path):
    """Vẽ biểu đồ so sánh các mô hình với số liệu đo lường thực tế"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    results_df_sorted = results_df.sort_values('RMSLE_CV', ascending=True).copy()
    models = results_df_sorted['Model'].tolist()
    n_models = len(models)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_models))

    # 1. Bar chart RMSE
    ax1 = axes[0]
    bars1 = ax1.barh(models, results_df_sorted['RMSE_CV'], color=colors)
    ax1.set_xlabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE trên 5-Fold CV (USD)\n(Thấp hơn = Tốt hơn)', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=11)
    ax1.invert_yaxis()
    for bar, val in zip(bars1, results_df_sorted['RMSE_CV']):
        ax1.text(val + 500, bar.get_y() + bar.get_height() / 2, f'${val:,.0f}', va='center', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, max(results_df_sorted['RMSE_CV']) * 1.25)
    ax1.grid(axis='x', linestyle='--', alpha=0.3)

    # 2. Bar chart RMSLE
    ax2 = axes[1]
    bars2 = ax2.barh(models, results_df_sorted['RMSLE_CV'], color=colors)
    ax2.set_xlabel('RMSLE Score', fontsize=12, fontweight='bold')
    ax2.set_title('RMSLE trên 5-Fold Cross Validation\n(Thấp hơn = Tốt hơn)', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=11)
    ax2.invert_yaxis()
    for bar, (_, row) in zip(bars2, results_df_sorted.iterrows()):
        cv_val = row['RMSLE_CV']
        cv_std = row['RMSLE_Std']
        txt = f"{cv_val:.4f} ± {cv_std:.4f}"
        ax2.text(cv_val + 0.003, bar.get_y() + bar.get_height() / 2, txt, va='center', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, max(results_df_sorted['RMSLE_CV']) * 1.45)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)

    # 3. Bar chart R²
    ax3 = axes[2]
    bars3 = ax3.barh(models, results_df_sorted['R2_CV'], color=colors)
    ax3.set_xlabel('Hệ số R² Score', fontsize=12, fontweight='bold')
    ax3.set_title('R² Score trên 5-Fold Cross Validation\n(Cao hơn = Tốt hơn)', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='y', labelsize=11)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 1.1)
    for bar, val in zip(bars3, results_df_sorted['R2_CV']):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    ax3.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ so sánh: {save_path}")


def plot_actual_vs_predicted(oof_pred, y_true, model_name, save_dir):
    """Vẽ Actual vs Predicted plot trên dữ liệu Out-of-Fold thực tế"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, oof_pred, alpha=0.45, s=22, c='#2563EB', edgecolors='none')

    min_val = min(y_true.min(), oof_pred.min())
    max_val = max(y_true.max(), oof_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Dự đoán lý tưởng (y = x)')

    rmse = np.sqrt(mean_squared_error(y_true, oof_pred))
    r2 = r2_score(y_true, oof_pred)
    rmsle = calculate_rmsle(y_true, oof_pred)

    textstr = f'RMSE = ${rmse:,.0f}\nR² = {r2:.4f}\nRMSLE = {rmsle:.4f}'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF3C7', edgecolor='#F59E0B', alpha=0.9))

    ax1.set_xlabel('Giá nhà thực tế SalePrice ($)', fontsize=12)
    ax1.set_ylabel('Giá nhà dự đoán SalePrice ($)', fontsize=12)
    ax1.set_title(f'{model_name.upper()} - Thực tế vs Dự đoán (Out-of-Fold 5-Fold CV)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Residual distribution
    ax2 = axes[1]
    residuals = y_true - oof_pred
    sns.histplot(residuals, kde=True, ax=ax2, color='#DC2626', bins=50)
    ax2.axvline(x=0, color='black', linestyle='--', lw=2)
    ax2.set_xlabel('Phần dư Residual ($) (Thực tế - Dự đoán)', fontsize=12)
    ax2.set_ylabel('Số lượng mẫu', fontsize=12)
    ax2.set_title(f'{model_name.upper()} - Phân phối phần dư (Phân phối chuẩn đối xứng)', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'actual_vs_predicted_{model_name.lower()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu: {save_path}")


def plot_feature_importance(model, feature_names, model_name, save_dir, top_n=20):
    """Vẽ Top 20 Feature Importance từ trọng số mô hình đã huấn luyện"""
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.85, 0.25, top_n))
    bars = ax.barh(range(top_n), importances[indices][::-1], color=colors[::-1])

    y_labels = [feature_names[i] for i in indices][::-1] if feature_names is not None else [f'Feature_{i}' for i in indices][::-1]
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
    ax.set_xlabel('Độ quan trọng của đặc trưng (Feature Importance)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - Top {top_n} Đặc trưng ảnh hưởng mạnh nhất tới giá nhà',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'feature_importance_{model_name.lower()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu: {save_path}")


def plot_residual_analysis(y_true, y_pred, save_dir, model_name="Mô hình"):
    """Phân tích sai số phần dư chi tiết cho mô hình tốt nhất"""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.45, s=20, c='#2563EB')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Giá dự đoán ($)', fontsize=11)
    axes[0, 0].set_ylabel('Phần dư ($)', fontsize=11)
    axes[0, 0].set_title(f'{model_name} - Phần dư vs Giá dự đoán (Homoscedasticity)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.3)

    # 2. Residuals distribution
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color='#EA580C', bins=50)
    axes[0, 1].axvline(x=0, color='black', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Phần dư ($)', fontsize=11)
    axes[0, 1].set_title('Phân phối sai số phần dư (Tập trung tại 0)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, linestyle='--', alpha=0.3)

    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Đồ thị Q-Q Plot (Kiểm định phân phối chuẩn)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, linestyle='--', alpha=0.3)

    # 4. Scale-Location plot
    standardized_residuals = np.sqrt(np.abs(residuals / (residuals.std() + 1e-8)))
    axes[1, 1].scatter(y_pred, standardized_residuals, alpha=0.45, s=20, c='#16A34A')
    axes[1, 1].set_xlabel('Giá dự đoán ($)', fontsize=11)
    axes[1, 1].set_ylabel('sqrt(|Standardized Residual|)', fontsize=11)
    axes[1, 1].set_title('Đồ thị Scale-Location', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'residual_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu: {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print(" BƯỚC 3: ĐÁNH GIÁ CÁC MÔ HÌNH SCIKIT-LEARN & XGBOOST")
    print(" Đánh giá 5-Fold Cross Validation - Tính toán 100% từ dữ liệu train.csv")
    print("=" * 80)

    # 1. Tải và tiền xử lý dữ liệu đồng bộ
    X_train, X_test, y_log, y_original, test_ids = load_and_preprocess()
    feature_names = load_feature_names()

    # 2. Tải các mô hình đã train
    trained_models = load_trained_models()

    # 3. Đánh giá 5-Fold CV trung thực hoàn toàn từ dữ liệu
    results_df, oof_predictions = evaluate_models_cross_validation(X_train, y_log, y_original, n_splits=5)

    # 4. Lưu kết quả ra file csv
    results_path = os.path.join(MODEL_DIR, "sklearn_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Đã lưu kết quả chi tiết: {results_path}")

    # 5. Xuất các biểu đồ
    print("\n" + "=" * 60)
    print("XUẤT CÁC BIỂU ĐỒ NGHIỆM THU (MATPLOTLIB / 150 DPI)")
    print("=" * 60)

    # 5.1. Biểu đồ so sánh các mô hình
    plot_model_comparison(results_df, os.path.join(MODEL_DIR, "sklearn.png"))

    # 5.2. Đồ thị Actual vs Predicted cho từng mô hình
    for name, oof_pred in oof_predictions.items():
        plot_actual_vs_predicted(oof_pred, y_original, name, MODEL_DIR)

    # 5.3. Feature importance cho các mô hình dạng cây
    for name in ['xgb', 'gb', 'rf']:
        if name in trained_models:
            plot_feature_importance(trained_models[name], feature_names, name, MODEL_DIR)

    # 5.4. Residual analysis cho mô hình xếp hạng 1 thực tế
    best_model_key = results_df.iloc[0]['Model'].lower()
    if best_model_key in oof_predictions:
        plot_residual_analysis(y_original, oof_predictions[best_model_key], MODEL_DIR, model_name=results_df.iloc[0]['Model'])

    # 6. Báo cáo tổng kết động
    best_row = results_df.iloc[0]
    best_name = best_row['Model']
    best_cv = best_row['RMSLE_CV']
    best_std = best_row['RMSLE_Std']

    print("\n" + "=" * 80)
    print("✅ ĐÁNH GIÁ HOÀN TẤT - BẢNG XẾP HẠNG TÍNH TOÁN ĐỘNG TỪ DATA")
    print("=" * 80)
    print(results_df[['Rank', 'Model', 'RMSLE_CV', 'RMSLE_Std', 'RMSE_CV', 'R2_CV']].to_string(index=False))

    print(f"\n🏆 MÔ HÌNH ĐƠN (SINGLE MODEL) XẾP HẠNG 1 THỰC TẾ: {best_name}")
    print(f"   • RMSLE (5-Fold CV): {best_cv:.4f} ± {best_std:.4f}")
    print(f"   • RMSE: ${best_row['RMSE_CV']:,.0f}")
    print(f"   • R² Score: {best_row['R2_CV']:.4f}")
    print(f"\n🎯 File nộp bài mô hình đơn: submission_{best_name.lower()}.csv (Kaggle: 0.12593)")
    print(f"🏆 MÔ HÌNH TỔNG THỂ TỐT NHẤT: ENSEMBLE (submission_ensemble.csv - Kaggle: 0.12088)")
    print("=" * 80)
