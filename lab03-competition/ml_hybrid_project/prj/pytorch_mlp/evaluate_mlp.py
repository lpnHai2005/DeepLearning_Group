"""
=================================================================
EVALUATE PYTORCH MLP
=================================================================
Module danh gia va truc quan hoa ket qua PyTorch MLP.

"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# CAU HINH DUONG DAN
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "prj", "model")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_model():
    """Load trained PyTorch model"""
    model_path = os.path.join(MODEL_DIR, "mlp_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    from model import create_mlp

    config = checkpoint.get('config', {
        'hidden_dims': [256, 128, 64, 32],
        'dropout_rate': 0.3
    })

    input_dim = checkpoint.get('input_dim', 84)

    model = create_mlp(
        input_dim=input_dim,
        config=config
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: {model_path}")

    return model, checkpoint


def load_history():
    """Load training history"""
    history_path = os.path.join(MODEL_DIR, "training_history.csv")

    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        print(f"History loaded: {len(history_df)} epochs")
        return history_df
    else:
        print(f"History file not found: {history_path}")
        return None


def plot_training_curves(history_df, save_path):
    """Ve training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = history_df['epoch']

    # Loss Curves
    ax1 = axes[0]
    ax1.plot(epochs, history_df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history_df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Val RMSE
    ax2 = axes[1]
    ax2.plot(epochs, history_df['val_rmse'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val RMSE ($)')
    ax2.set_title('Validation RMSE (Original Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Learning Rate
    ax3 = axes[2]
    ax3.semilogy(epochs, history_df['lr'], 'purple', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate (log scale)')
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_pytorch_html(history_df, save_path):
    """Tao file HTML tuong tac cho PyTorch training results"""

    # Convert data to JSON-compatible format
    epochs = history_df['epoch'].tolist()
    train_loss = history_df['train_loss'].tolist()
    val_loss = history_df['val_loss'].tolist()
    val_rmse = history_df['val_rmse'].tolist()
    lr = history_df['lr'].tolist()

    best_epoch = int(history_df.loc[history_df['val_loss'].idxmin(), 'epoch'])
    best_val_loss = history_df['val_loss'].min()
    best_val_rmse = history_df.loc[history_df['val_loss'].idxmin(), 'val_rmse']

    # Create HTML using string concatenation to avoid f-string issues
    html_content = '<!DOCTYPE html>\n'
    html_content += '<html lang="vi">\n<head>\n'
    html_content += '    <meta charset="UTF-8">\n'
    html_content += '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    html_content += '    <title>PyTorch MLP Training Results</title>\n'
    html_content += '    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n'
    html_content += '    <style>\n'
    html_content += '        * { margin: 0; padding: 0; box-sizing: border-box; }\n'
    html_content += '        body { font-family: Segoe UI, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }\n'
    html_content += '        .container { max-width: 1200px; margin: 0 auto; }\n'
    html_content += '        h1 { color: white; text-align: center; margin-bottom: 30px; font-size: 2.5rem; }\n'
    html_content += '        .subtitle { color: rgba(255,255,255,0.9); text-align: center; margin-bottom: 40px; }\n'
    html_content += '        .card { background: white; border-radius: 15px; padding: 25px; margin-bottom: 25px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }\n'
    html_content += '        .card h2 { color: #333; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 3px solid #667eea; }\n'
    html_content += '        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }\n'
    html_content += '        .stat-item { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; }\n'
    html_content += '        .stat-label { font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px; }\n'
    html_content += '        .stat-value { font-size: 1.8rem; font-weight: bold; }\n'
    html_content += '        .chart-container { position: relative; height: 400px; margin-bottom: 20px; }\n'
    html_content += '    </style>\n'
    html_content += '</head>\n<body>\n'
    html_content += '    <div class="container">\n'
    html_content += '        <h1>House Price Prediction - PyTorch MLP</h1>\n'
    html_content += '        <p class="subtitle">Lab03 Deep Learning - Training Results</p>\n'
    html_content += '        <div class="card">\n'
    html_content += '            <h2>Training Statistics</h2>\n'
    html_content += '            <div class="stats-grid">\n'
    html_content += '                <div class="stat-item"><div class="stat-label">Total Epochs</div><div class="stat-value">' + str(len(epochs)) + '</div></div>\n'
    html_content += '                <div class="stat-item"><div class="stat-label">Best Epoch</div><div class="stat-value">' + str(best_epoch) + '</div></div>\n'
    html_content += '                <div class="stat-item"><div class="stat-label">Best Val Loss</div><div class="stat-value">' + f'{best_val_loss:.6f}' + '</div></div>\n'
    html_content += '                <div class="stat-item"><div class="stat-label">Best Val RMSE</div><div class="stat-value">$' + f'{best_val_rmse:,.0f}' + '</div></div>\n'
    html_content += '            </div>\n'
    html_content += '        </div>\n'
    html_content += '        <div class="card"><h2>Loss Curves</h2><div class="chart-container"><canvas id="lossChart"></canvas></div></div>\n'
    html_content += '        <div class="card"><h2>Validation RMSE</h2><div class="chart-container"><canvas id="rmseChart"></canvas></div></div>\n'
    html_content += '        <div class="card"><h2>Learning Rate Schedule</h2><div class="chart-container"><canvas id="lrChart"></canvas></div></div>\n'
    html_content += '        <div class="card">\n'
    html_content += '            <h2>Model Architecture</h2>\n'
    html_content += '            <p><strong>Hidden Layers:</strong> [256, 128, 64, 32]</p>\n'
    html_content += '            <p><strong>Dropout:</strong> Progressive (0.3 to 0.1)</p>\n'
    html_content += '            <p><strong>Batch Normalization:</strong> Enabled</p>\n'
    html_content += '            <p><strong>Optimizer:</strong> Adam</p>\n'
    html_content += '            <p><strong>Early Stopping:</strong> Patience = 30</p>\n'
    html_content += '        </div>\n'
    html_content += '    </div>\n'
    html_content += '    <script>\n'

    # JavaScript data
    html_content += '        const epochs = ' + str(epochs) + ';\n'
    html_content += '        const trainLoss = ' + str(train_loss) + ';\n'
    html_content += '        const valLoss = ' + str(val_loss) + ';\n'
    html_content += '        const valRmse = ' + str(val_rmse) + ';\n'
    html_content += '        const lr = ' + str(lr) + ';\n'

    html_content += '        const lossCtx = document.getElementById("lossChart").getContext("2d");\n'
    html_content += '        new Chart(lossCtx, {\n'
    html_content += '            type: "line",\n'
    html_content += '            data: {\n'
    html_content += '                labels: epochs,\n'
    html_content += '                datasets: [\n'
    html_content += '                    { label: "Train Loss", data: trainLoss, borderColor: "rgba(54, 162, 235, 1)", backgroundColor: "rgba(54, 162, 235, 0.1)", fill: true },\n'
    html_content += '                    { label: "Val Loss", data: valLoss, borderColor: "rgba(255, 99, 132, 1)", backgroundColor: "rgba(255, 99, 132, 0.1)", fill: true }\n'
    html_content += '                ]\n'
    html_content += '            },\n'
    html_content += '            options: { responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: "Training & Validation Loss" } } }\n'
    html_content += '        });\n'

    html_content += '        const rmseCtx = document.getElementById("rmseChart").getContext("2d");\n'
    html_content += '        new Chart(rmseCtx, {\n'
    html_content += '            type: "line",\n'
    html_content += '            data: { labels: epochs, datasets: [{ label: "Val RMSE", data: valRmse, borderColor: "rgba(75, 192, 192, 1)", backgroundColor: "rgba(75, 192, 192, 0.1)", fill: true }] },\n'
    html_content += '            options: { responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: "Validation RMSE" } } }\n'
    html_content += '        });\n'

    html_content += '        const lrCtx = document.getElementById("lrChart").getContext("2d");\n'
    html_content += '        new Chart(lrCtx, {\n'
    html_content += '            type: "line",\n'
    html_content += '            data: { labels: epochs, datasets: [{ label: "LR", data: lr, borderColor: "rgba(153, 102, 255, 1)", backgroundColor: "rgba(153, 102, 255, 0.1)", fill: true }] },\n'
    html_content += '            options: { responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: "Learning Rate" } } }\n'
    html_content += '        });\n'

    html_content += '    </script>\n'
    html_content += '</body>\n</html>'

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("EVALUATING PYTORCH MLP")
    print("=" * 60)

    try:
        # Load model
        model, checkpoint = load_model()

        # Load history
        history_df = load_history()

        if history_df is not None:
            # Plot training curves
            plot_training_curves(history_df, os.path.join(MODEL_DIR, "training_curves.png"))

            # Generate HTML
            generate_pytorch_html(history_df, os.path.join(MODEL_DIR, "pytorch.html"))

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print(f"All outputs saved to: {MODEL_DIR}")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run train_mlp.py first to train the model!")
