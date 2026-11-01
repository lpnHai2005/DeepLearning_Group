import numpy as np
import torch
from torch import nn

from src.data.gaussian import Y_SCALE
from src.engine.evaluate import metrics, predict, predict_grid


class FirstCoordinateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x[:, 0] * self.scale


def test_predict_scales_to_original_y_and_restores_mode():
    model = FirstCoordinateModel().train()
    x = np.array([[-2.0, 0.0], [1.0, 2.0]])

    y_hat = predict(model, x)

    assert np.allclose(y_hat, x[:, 0] / 2.0 * Y_SCALE)
    assert model.training


def test_metrics_and_predict_grid():
    y = np.array([0.0, 1.0, 2.0])
    perfect = metrics(y, y.copy())
    assert perfect == {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 1.0}
    assert metrics(y, np.full_like(y, y.mean()))["r2"] == 0.0

    model = FirstCoordinateModel()
    grid_x1, grid_x2, grid_y = predict_grid(model, n=7)
    assert grid_x1.shape == grid_x2.shape == grid_y.shape == (7, 7)
    assert np.isclose(grid_x1.min(), -2.0) and np.isclose(grid_x1.max(), 2.0)
    assert np.isclose(grid_x2.min(), -4.0) and np.isclose(grid_x2.max(), 4.0)
