"""
=================================================================
EVALUATE SKLEARN MODELS
=================================================================
Module đánh giá và trực quan hóa kết quả các mô hình sklearn.

Output visualizations:
    1. sklearn.png - So sanh cac moi hinh (bar chart RMSE/R²)
    2. actual_vs_predicted.png - Actual vs Predicted plot
    3. feature_importance.png - Top 20 feature importance
    4. residual_analysis.png - Residual distribution

Author: Thanh vien 2 (sklearn pipeline)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# CAU HINH DUONG DAN
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "prj", "model")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_trained_models():
    """Load tat ca cac moi hinh da train"""
    models = {}

    model_files = [
        'model_ridge.pkl', 'model_lasso.pkl', 'model_elastic.pkl',
        'model_rf.pkl', 'model_gb.pkl', 'model_xgb.pkl'
    ]

    for fname in model_files:
        path = os.path.join(DATA_PROCESSED_DIR, fname)
        if os.path.exists(path):
            name = fname.replace('model_', '').replace('.pkl', '')
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"Loaded: {name}")

    return models


def load_train_data():
    """Load du lieu train goc"""
    train_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "train.csv"))
    return train_df


def load_and_preprocess_data():
    """Load va preprocess data cung nhu luc train"""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from features import create_all_features, handle_missing_for_sklearn

    # Load train data
    train_df = load_train_data()
    y_train_original = train_df['SalePrice']
    y_train_log = np.log1p(y_train_original)

    # Preprocess cung nhu luc train
    train_features = train_df.drop(['SalePrice', 'Id'], axis=1)
    all_df = handle_missing_for_sklearn(train_features)
    all_df = create_all_features(all_df)

    # Nominal encoding
    for col in all_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        all_df[col] = le.fit_transform(all_df[col].astype(str))

    # Scale
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(all_df), columns=all_df.columns)

    return X_train, y_train_log, y_train_original


def evaluate_all_models(models, X_train, y_train_log, y_train_original):
    """
    Danh gia tat ca cac moi hinh tren tap train.
    """
    results = []

    for name, model in models.items():
        # Du doan (log scale)
        y_pred_log = model.predict(X_train)

        # Chuyen ve original scale
        y_pred = np.expm1(y_pred_log)

        # Tinh metrics (tren original scale)
        rmse = np.sqrt(mean_squared_error(y_train_original, y_pred))
        mae = mean_absolute_error(y_train_original, y_pred)
        r2 = r2_score(y_train_original, y_pred)

        # Tinh RMSLE (Kaggle metric)
        rmsle = np.sqrt(mean_squared_error(
            np.log1p(y_train_original),
            np.log1p(np.maximum(y_pred, 1))
        ))

        results.append({
            'Model': name.upper(),
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'RMSLE': rmsle
        })

        print(f"{name.upper():15} | RMSE: ${rmse:,.0f} | R2: {r2:.4f}")

    return pd.DataFrame(results)


def plot_model_comparison(results_df, save_path):
    """Ve bieu do so sanh cac moi hinh"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sap xep theo RMSE
    results_df = results_df.sort_values('RMSE')

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_df)))

    # Bar chart RMSE
    ax1 = axes[0]
    bars1 = ax1.barh(results_df['Model'], results_df['RMSE'], color=colors)
    ax1.set_xlabel('RMSE ($)', fontsize=12)
    ax1.set_title('Model Comparison - RMSE (thap hon = tot hon)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=11)

    for bar, val in zip(bars1, results_df['RMSE']):
        ax1.text(val + 500, bar.get_y() + bar.get_height()/2,
                f'${val:,.0f}', va='center', fontsize=10)

    # Bar chart R²
    ax2 = axes[1]
    bars2 = ax2.barh(results_df['Model'], results_df['R2'], color=colors)
    ax2.set_xlabel('R2 Score', fontsize=12)
    ax2.set_title('Model Comparison - R2 (cao hon = tot hon)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=11)
    ax2.set_xlim(0, 1)

    for bar, val in zip(bars2, results_df['R2']):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_actual_vs_predicted(model, X_train, y_train_original, model_name, save_dir):
    """Ve Actual vs Predicted plot"""
    y_pred_log = model.predict(X_train)
    y_pred = np.expm1(y_pred_log)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_train_original, y_pred, alpha=0.5, s=20, c='steelblue')

    min_val = min(y_train_original.min(), y_pred.min())
    max_val = max(y_train_original.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax1.set_xlabel('Actual SalePrice ($)', fontsize=12)
    ax1.set_ylabel('Predicted SalePrice ($)', fontsize=12)
    ax1.set_title(f'{model_name.upper()} - Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.legend()

    r2 = r2_score(y_train_original, y_pred)
    ax1.text(0.05, 0.95, f'R2 = {r2:.4f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Residual distribution
    ax2 = axes[1]
    residuals = y_train_original - y_pred

    sns.histplot(residuals, kde=True, ax=ax2, color='coral', bins=50)
    ax2.axvline(x=0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'{model_name.upper()} - Residual Distribution', fontsize=14, fontweight='bold')

    rmse = np.sqrt(mean_squared_error(y_train_original, y_pred))
    ax2.text(0.95, 0.95, f'RMSE = ${rmse:,.0f}', transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'actual_vs_predicted_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(model, feature_names, model_name, save_dir, top_n=20):
    """Ve feature importance cho tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name.upper()} khong co feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, top_n))
    bars = ax.barh(range(top_n), importances[indices][::-1], color=colors[::-1])

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'{model_name.upper()} - Top {top_n} Feature Importance',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'feature_importance_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_residual_analysis(model, X_train, y_train_original, save_dir):
    """Phan tich residual chi tiet"""
    y_pred_log = model.predict(X_train)
    y_pred = np.expm1(y_pred_log)
    residuals = y_train_original - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20, c='steelblue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted SalePrice ($)', fontsize=11)
    axes[0, 0].set_ylabel('Residual', fontsize=11)
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')

    # Residuals distribution
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color='coral', bins=50)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Residual', fontsize=11)
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Residuals)', fontsize=12, fontweight='bold')

    # Scale-Location plot
    standardized_residuals = np.sqrt(np.abs(residuals / residuals.std()))
    axes[1, 1].scatter(y_pred, standardized_residuals, alpha=0.5, s=20, c='seagreen')
    axes[1, 1].set_xlabel('Predicted SalePrice ($)', fontsize=11)
    axes[1, 1].set_ylabel('sqrt(|Standardized Residual|)', fontsize=11)
    axes[1, 1].set_title('Scale-Location Plot', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'residual_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("EVALUATING SKLEARN MODELS")
    print("=" * 60)

    # Load models
    models = load_trained_models()

    # Load va preprocess data
    X_train, y_train_log, y_train_original = load_and_preprocess_data()

    if not models:
        print("Khong co moi hinh nao duoc train!")
        print("Vui long chay train_baseline.py truoc.")
    else:
        print("\n" + "=" * 60)
        print("EVALUATING ALL MODELS")
        print("=" * 60)

        # Evaluate
        results_df = evaluate_all_models(models, X_train, y_train_log, y_train_original)

        # Save results
        results_path = os.path.join(MODEL_DIR, "sklearn_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved: {results_path}")

        # Plots
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        # 1. Main comparison plot (sklearn.png)
        plot_model_comparison(results_df, os.path.join(MODEL_DIR, "sklearn.png"))

        # 2. Actual vs Predicted
        for name, model in models.items():
            plot_actual_vs_predicted(model, X_train, y_train_original, name, MODEL_DIR)

        # 3. Feature importance
        for name, model in models.items():
            plot_feature_importance(model, X_train.columns, name, MODEL_DIR)

        # 4. Residual analysis cho best model
        best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model'].lower()
        if best_model_name in models:
            plot_residual_analysis(models[best_model_name], X_train, y_train_original, MODEL_DIR)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print(f"All plots saved to: {MODEL_DIR}")
        print("=" * 60)
