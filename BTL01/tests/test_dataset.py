import numpy as np
import torch

from src.data.dataset import load_split, make_loader, subset
from src.data.gaussian import X_SCALE, Y_SCALE


def test_load_split_and_nested_subset():
    x, y = load_split("train_pool")
    x_small, y_small = subset(x, y, 200)
    x_large, y_large = subset(x, y, 400)

    assert x.shape == (2000, 2)
    assert y.shape == (2000,)
    assert x_small.shape == (200, 2)
    assert y_small.shape == (200,)
    assert np.array_equal(x_small, x_large[:200])
    assert np.array_equal(y_small, y_large[:200])


def test_make_loader_scales_and_shuffles_reproducibly():
    x, y = load_split("train_pool")
    ordered = make_loader(x[:35], y[:35], batch_size=32, shuffle=False, seed=7)
    first_batch_x, first_batch_y = next(iter(ordered))
    assert first_batch_x.shape == (32, 2)
    assert first_batch_y.shape == (32,)
    assert first_batch_x.dtype == torch.float32
    assert first_batch_y.dtype == torch.float32
    assert torch.allclose(first_batch_x[0], torch.tensor(x[0] / X_SCALE, dtype=torch.float32))
    assert torch.isclose(first_batch_y[0], torch.tensor(y[0] / Y_SCALE, dtype=torch.float32))
    assert [len(batch_x) for batch_x, _ in ordered] == [32, 3]

    shuffled_a = torch.cat([batch_x for batch_x, _ in make_loader(x[:35], y[:35], 8, True, 2026)])
    shuffled_b = torch.cat([batch_x for batch_x, _ in make_loader(x[:35], y[:35], 8, True, 2026)])
    assert torch.equal(shuffled_a, shuffled_b)
    assert not torch.equal(shuffled_a, torch.from_numpy((x[:35] / X_SCALE).astype(np.float32)))
