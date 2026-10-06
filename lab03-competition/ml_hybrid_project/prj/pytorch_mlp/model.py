"""
=================================================================
PYTORCH MLP MODEL ARCHITECTURE
=================================================================
Module định nghĩa kiến trúc MLP (Multi-Layer Perceptron) cho bài toán
House Price Prediction.

Architecture:
    Input (n_features)
        ↓
    Linear(n_features → 256) + BatchNorm1d + ReLU + Dropout(0.3)
        ↓
    Linear(256 → 128) + BatchNorm1d + ReLU + Dropout(0.3)
        ↓
    Linear(128 → 64) + BatchNorm1d + ReLU + Dropout(0.2)
        ↓
    Linear(64 → 32) + BatchNorm1d + ReLU + Dropout(0.1)
        ↓
    Linear(32 → 1)  (Output: log(SalePrice))

Features:
    - BatchNorm: ổn định training
    - Dropout: giảm overfitting
    - Xavier/Glorot initialization

"""

import torch
import torch.nn as nn
import torch.nn.init as init


class MLPRegressor(nn.Module):
    """
    MLP Regressor cho House Price Prediction.

    Args:
        input_dim (int): Số lượng features đầu vào
        hidden_dims (list): Danh sách số neurons trong mỗi hidden layer
        dropout rates (float): Dropout rate cho các layers

    Attributes:
        features: Sequential layer chứa các hidden layers
        output: Output layer (linear)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64, 32],
        dropout_rate: float = 0.3
    ):
        super(MLPRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # BatchNorm
            layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation (ReLU)
            layers.append(nn.ReLU(inplace=True))

            # Dropout (giảm dần theo depth)
            current_dropout = dropout_rate * (1 - i * 0.1)
            current_dropout = max(current_dropout, 0.1)  # Tối thiểu 0.1
            layers.append(nn.Dropout(current_dropout))

            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(prev_dim, 1)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Khởi tạo weights sử dụng Xavier/Glorot initialization
        cho better gradient flow.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization
                init.xavier_uniform_(m.weight)

                # Bias = 0
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm: weight = 1, bias = 0
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, n_features)

        Returns:
            Output tensor (batch_size, 1)
        """
        # Flatten nếu input có shape (batch, 1, n_features)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Hidden layers
        x = self.features(x)

        # Output layer
        x = self.output(x)

        # Squeeze để có shape (batch_size,)
        x = x.squeeze(-1)

        return x

    def get_num_params(self):
        """Trả về tổng số parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self):
        """Trả về số parameters có thể train"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """
    Residual Block (optional, để experiment).
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


def create_mlp(input_dim: int, config: dict = None) -> MLPRegressor:
    """
    Factory function để tạo MLP với config.

    Args:
        input_dim: Số features đầu vào
        config: Dictionary chứa hyperparameters

    Returns:
        MLPRegressor instance
    """
    if config is None:
        config = {
            'hidden_dims': [256, 128, 64, 32],
            'dropout_rate': 0.3
        }

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [256, 128, 64, 32]),
        dropout_rate=config.get('dropout_rate', 0.3)
    )

    return model


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════

# Default config - balanced
MLP_DEFAULT_CONFIG = {
    'name': 'MLP_DEFAULT',
    'hidden_dims': [256, 128, 64, 32],
    'dropout_rate': 0.3,
    'description': 'Balanced architecture for general use'
}

# Deep config - deeper network
MLP_DEEP_CONFIG = {
    'name': 'MLP_DEEP',
    'hidden_dims': [512, 256, 128, 64, 32],
    'dropout_rate': 0.4,
    'description': 'Deeper architecture for complex patterns'
}

# Wide config - wider layers
MLP_WIDE_CONFIG = {
    'name': 'MLP_WIDE',
    'hidden_dims': [512, 256, 128],
    'dropout_rate': 0.2,
    'description': 'Wider architecture for more capacity'
}


if __name__ == "__main__":
    # Test model
    print("=" * 60)
    print("TESTING MLP MODEL")
    print("=" * 60)

    # Tạo model với 79 features (House Price dataset)
    input_dim = 79
    model = create_mlp(input_dim, MLP_DEFAULT_CONFIG)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, input_dim)

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"\n✓ Model created successfully:")
    print(f"  - Architecture: {MLP_DEFAULT_CONFIG['hidden_dims']}")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Output dim: {output.shape}")
    print(f"  - Total parameters: {model.get_num_params():,}")
    print(f"  - Trainable parameters: {model.get_trainable_params():,}")

    # Test backward pass
    print("\n✓ Testing backward pass...")
    model.train()
    x = torch.randn(batch_size, input_dim)
    target = torch.randn(batch_size)

    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    print("✓ Backward pass successful!")

    # Print model architecture
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model)
