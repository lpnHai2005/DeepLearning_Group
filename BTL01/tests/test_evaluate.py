import numpy as np
import torch
from torch import nn

from src.data.gaussian import Y_SCALE
from src.engine.evaluate import evaluate_grid, metrics, predict, predict_grid
from src.visual.plots import (
    plot_error_map,
    plot_error_vs_param,
    plot_loss_curve,
    plot_parity,
    plot_surface_comparison,
)


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

    grid_stats = evaluate_grid(model, n=7)
    for key in ("mse", "mae", "rmse", "r2", "max_abs_err"):
        assert key in grid_stats and np.isfinite(grid_stats[key])


def test_plots_generate_files(tmp_path):
    model = FirstCoordinateModel()
    history = [{"epoch": 1, "train_loss": 0.1, "val_loss": 0.12},
               {"epoch": 2, "train_loss": 0.05, "val_loss": 0.06}]
    y_true = np.array([0.01, 0.05, 0.15, 0.25])
    y_pred = np.array([0.012, 0.048, 0.14, 0.24])

    p1 = plot_loss_curve(history, tmp_path / "loss.png")
    p2 = plot_parity(y_true, y_pred, tmp_path / "parity.png")
    p3 = plot_surface_comparison(model, tmp_path / "surface.png", n=10)
    p4 = plot_error_map(model, tmp_path / "errmap.png", n=10, sample_x=np.zeros((3, 2)))
    p5 = plot_error_vs_param([1, 2], [0.01, 0.005], [0.001, 0.001],
                             [0.012, 0.006], [0.001, 0.001],
                             tmp_path / "err_vs_p.png", xlabel="B", baseline_mse=0.02)

    for p in (p1, p2, p3, p4, p5):
        assert p.exists() and p.stat().st_size > 0

