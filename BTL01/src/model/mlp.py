"""MLP tùy biến số tầng ẩn / số nơ-ron / hàm kích hoạt. (TV2 – T2.1)

Đầu ra tuyến tính; forward trả về shape (N,) để khớp với y (tránh bẫy broadcast của MSELoss).
"""
import torch.nn as nn

ACTS = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(10,), activation="tanh"):
        super().__init__()
        layers, d = [], 2
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), ACTS[activation]()]
            d = h
        layers.append(nn.Linear(d, 1))                 # đầu ra tuyến tính (hồi quy)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)                 # (N,) cùng shape với y


def n_params(model):
    return sum(p.numel() for p in model.parameters())
