"""MLP tùy biến số tầng ẩn / số nơ-ron / hàm kích hoạt. (TV2 – T2.1)

Đầu ra tuyến tính; forward trả về shape (N,) để khớp với y (tránh bẫy broadcast của MSELoss).
"""
import torch.nn as nn

ACTS = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(10,), activation="tanh"):
        super().__init__()
        raise NotImplementedError("TV2 - T2.1")

    def forward(self, x):
        raise NotImplementedError("TV2 - T2.1")
