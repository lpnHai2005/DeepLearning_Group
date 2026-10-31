"""T2.4: backprop tính tay (NumPy) cho mạng 2→2→1 tanh khớp với autograd; thêm kiểm tra đạo hàm số."""
import numpy as np
import torch
import torch.nn as nn

from src.model.mlp import MLP


def _setup():
    torch.manual_seed(0)
    model = MLP([2], "tanh").double()
    x = torch.randn(5, 2, dtype=torch.float64)
    y = torch.randn(5, dtype=torch.float64)
    return model, x, y


def test_manual_backprop_matches_autograd():
    model, x, y = _setup()
    l1, l2 = model.net[0], model.net[2]
    nn.MSELoss()(model(x), y).backward()

    X, Y = x.numpy(), y.numpy()
    W1, b1 = l1.weight.detach().numpy(), l1.bias.detach().numpy()
    W2, b2 = l2.weight.detach().numpy(), l2.bias.detach().numpy()
    a1 = np.tanh(X @ W1.T + b1)                      # (N,2)
    y_hat = (a1 @ W2.T + b2).ravel()                 # (N,)
    d_out = 2 / len(Y) * (y_hat - Y)                 # δ_out = (2/N)(ŷ − y)
    d1 = (d_out[:, None] @ W2) * (1 - a1 ** 2)       # δ⁽¹⁾ = (W⁽²⁾ᵀ δ_out) ⊙ tanh′(z⁽¹⁾)
    manual = {l2.weight: d_out[None] @ a1, l2.bias: d_out.sum(keepdims=True),
              l1.weight: d1.T @ X, l1.bias: d1.sum(0)}
    for p, g in manual.items():
        assert torch.allclose(p.grad, torch.from_numpy(g), atol=1e-12)


def test_numeric_gradient():
    model, x, y = _setup()
    params = list(model.parameters())

    def loss(*ps):
        return nn.MSELoss()(torch.func.functional_call(model, dict(zip(dict(model.named_parameters()), ps)), x), y)

    assert torch.autograd.gradcheck(loss, tuple(p.detach().requires_grad_() for p in params))
