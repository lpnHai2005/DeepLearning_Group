"""
=================================================================
PYTORCH MLP TRAINING LOOP
=================================================================
Module chứa training loop và các hàm huấn luyện MLP.

Training Features:
    - Early Stopping: dừng sớm khi validation không cải thiện
    - Learning Rate Scheduling: giảm LR khi plateau
    - Gradient Clipping: tránh gradient explosion
    - Checkpoint: lưu model tốt nhất
    - TensorBoard/CSV logging: theo dõi training

Metrics:
    - Train Loss, Val Loss
    - RMSE (original scale)
    - RMSLE (Kaggle metric)


"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
from dataset import create_dataloaders, HousePriceDataset
from model import create_mlp, MLP_DEFAULT_CONFIG

# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "prj", "model")
EXPS_FEATURE_DIR = os.path.join(BASE_DIR, "exps", "feature")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPS_FEATURE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────
TRAINING_CONFIG = {
    # Model
    'input_dim': 79,  # Sẽ được update tự động
    'hidden_dims': [256, 128, 64, 32],
    'dropout_rate': 0.3,

    # Training
    'epochs': 500,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,  # L2 regularization

    # Optimizer
    'optimizer': 'adam',  # adam, sgd, adamw

    # Scheduler
    'use_scheduler': True,
    'scheduler_patience': 15,
    'scheduler_factor': 0.5,
    'min_lr': 1e-6,

    # Early Stopping
    'early_stopping_patience': 30,

    # Gradient
    'grad_clip': 1.0,

    # Validation
    'val_split': 0.2,

    # Random seed
    'seed': 42,
}


class EarlyStopping:
    """
    Early Stopping để tránh overfitting.

    Dừng training khi validation loss không cải thiện
    trong `patience` epochs.
    """

    def __init__(self, patience=30, min_delta=0.0001, mode='min'):
        """
        Args:
            patience: Số epochs không cải thiện trước khi dừng
            min_delta: Ngưỡng cải thiện tối thiểu
            mode: 'min' cho loss, 'max' cho accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        """
        Kiểm tra xem có nên dừng không.

        Args:
            score: Giá trị metric hiện tại
            epoch: Epoch hiện tại

        Returns:
            True nếu nên dừng
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def set_seed(seed: int):
    """Set random seed cho reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính RMSE trên original scale"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính RMSLE (Kaggle metric)"""
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(np.maximum(y_pred, 0))) ** 2))


def train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip=None):
    """
    Train một epoch.

    Args:
        model: PyTorch model
        train_loader: DataLoader cho training
        optimizer: Optimizer
        criterion: Loss function
        device: Device (cpu/cuda)
        grad_clip: Gradient clipping threshold

    Returns:
        Tuple (train_loss, train_rmse)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward
        loss.backward()

        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def validate(model, val_loader, criterion, device, y_original=None):
    """
    Validate model.

    Args:
        model: PyTorch model
        val_loader: DataLoader cho validation
        criterion: Loss function
        device: Device
        y_original: Target trên original scale (để tính RMSE thực)

    Returns:
        Tuple (val_loss, val_rmse_log, val_rmse)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            n_batches += 1

            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / n_batches

    # RMSE trên log scale
    val_rmse_log = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))

    # RMSE trên original scale (nếu có y_original)
    val_rmse = None
    if y_original is not None:
        # Chuyển predictions về original scale
        preds_original = np.expm1(np.array(all_preds))
        targets_original = np.expm1(np.array(all_targets))
        val_rmse = calculate_rmse(targets_original, preds_original)

    return avg_loss, val_rmse_log, val_rmse


def train_model(model, train_loader, val_loader, config, device, y_original=None):
    """
    Full training loop với early stopping và learning rate scheduling.

    Args:
        model: PyTorch model
        train_loader: DataLoader cho training
        val_loader: DataLoader cho validation
        config: Training config dict
        device: Device
        y_original: Target trên original scale

    Returns:
        Tuple (best_model, history)
    """
    # Setup
    criterion = nn.MSELoss()

    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Scheduler
    scheduler = None
    if config['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['scheduler_factor'],
            patience=config['scheduler_patience'],
            min_lr=config['min_lr'],
            verbose=True
        )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        mode='min'
    )

    # History
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_rmse_log': [],
        'val_rmse': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_model_state = None

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config['grad_clip']
        )

        # Validate
        val_loss, val_rmse_log, val_rmse = validate(
            model, val_loader, criterion, device, y_original
        )

        # Current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Log history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse_log'].append(val_rmse_log)
        history['val_rmse'].append(val_rmse)
        history['lr'].append(current_lr)

        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)

        # Print progress
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSE: ${val_rmse:,.0f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model saved! Val Loss: {val_loss:.6f}")

        # Early stopping check
        if early_stopping(val_loss, epoch):
            print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
            print(f"   Best epoch: {early_stopping.best_epoch}")
            print(f"   Best val loss: {early_stopping.best_score:.6f}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def save_model_and_history(model, history, config, save_dir):
    """
    Lưu model và training history.
    """
    # Save model state dict
    model_path = os.path.join(save_dir, "mlp_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_dim': config['input_dim']
    }, model_path)
    print(f"✓ Model saved: {model_path}")

    # Save history as CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(save_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"✓ History saved: {history_path}")

    # Save config
    config_path = os.path.join(save_dir, "training_config.json")
    # Convert non-serializable values
    config_serializable = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                          for k, v in config.items()}
    with open(config_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print(f"✓ Config saved: {config_path}")

    return model_path, history_path


def create_submission_pytorch(model, test_loader, test_ids, device, save_dir):
    """
    Tạo submission file cho Kaggle.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_X in test_loader:
            if isinstance(batch_X, tuple):
                batch_X = batch_X[0]
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())

    # Chuyển về original scale
    predictions = np.expm1(np.array(predictions))

    # Tạo submission
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })

    # Save
    submission_path = os.path.join(save_dir, "submission_mlp.csv")
    submission.to_csv(submission_path, index=False)
    print(f"✓ Submission saved: {submission_path}")

    # Stats
    print(f"\n📊 Prediction Statistics:")
    print(f"   Min:    ${predictions.min():,.2f}")
    print(f"   Max:    ${predictions.max():,.2f}")
    print(f"   Mean:   ${predictions.mean():,.2f}")
    print(f"   Median: ${np.median(predictions):,.2f}")

    return submission


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("PYTORCH MLP TRAINING")
    print("House Price Prediction - Lab03")
    print("=" * 60)

    # Set seed
    set_seed(TRAINING_CONFIG['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataloaders
    train_loader, val_loader, test_loader, scaler, test_ids = create_dataloaders(
        batch_size=TRAINING_CONFIG['batch_size'],
        val_split=TRAINING_CONFIG['val_split'],
        random_seed=TRAINING_CONFIG['seed']
    )

    # Update input_dim
    TRAINING_CONFIG['input_dim'] = train_loader.dataset.X.shape[1]

    # Load original y values for RMSE calculation
    y_original_path = os.path.join(DATA_PROCESSED_DIR, "y_train_original.npy")
    if os.path.exists(y_original_path):
        y_original = np.load(y_original_path)
    else:
        y_original = None

    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = create_mlp(
        input_dim=TRAINING_CONFIG['input_dim'],
        config={
            'hidden_dims': TRAINING_CONFIG['hidden_dims'],
            'dropout_rate': TRAINING_CONFIG['dropout_rate']
        }
    )
    model = model.to(device)

    print(f"✓ Model created:")
    print(f"  - Architecture: {TRAINING_CONFIG['hidden_dims']}")
    print(f"  - Input dim: {TRAINING_CONFIG['input_dim']}")
    print(f"  - Parameters: {model.get_num_params():,}")

    # Train
    model, history = train_model(
        model, train_loader, val_loader, TRAINING_CONFIG, device, y_original
    )

    # Save
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    save_model_and_history(model, history, TRAINING_CONFIG, MODEL_DIR)

    # Create submission
    print("\n" + "=" * 60)
    print("CREATING SUBMISSION")
    print("=" * 60)

    submission = create_submission_pytorch(model, test_loader, test_ids, device, MODEL_DIR)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
