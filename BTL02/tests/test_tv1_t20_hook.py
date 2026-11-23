# Phụ trách: TV1 – Đoàn Trung Kiên
"""Kiểm tra hook của TV1 cho lớp Conv cuối trong T20."""

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from src.models.cnn import build_model
from src.visual.filters import _get_target_layer, feature_maps


def test_last_conv_hook_returns_features_from_the_model_last_conv():
    torch.manual_seed(73)
    model = build_model("baseline", blocks=[[8], [16]], fc_units=32).eval()
    x = torch.linspace(-1.0, 1.0, 48 * 48).reshape(1, 1, 48, 48)

    convs = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
    layer, name = _get_target_layer(model, "last")
    assert layer is convs[-1][1]
    assert name == model.last_conv_name == convs[-1][0].removeprefix("features.")

    captured = []
    handle = layer.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach().cpu()))
    fig = None
    try:
        fig, maps = feature_maps(model, x, layer="last", top_k=4)
    finally:
        handle.remove()
        if fig is not None:
            plt.close(fig)

    assert len(captured) == 1
    raw = captured[0][0]
    assert tuple(raw.shape) == (16, 24, 24)
    energy = raw.mean(dim=(1, 2))
    indices = energy.topk(4).indices
    expected = raw[indices].numpy()
    assert maps.shape == (4, 24, 24)
    assert np.isfinite(maps).all()
    np.testing.assert_allclose(maps, expected)
    assert [axis.get_title() for axis in fig.axes] == [
        f"Ch {index.item()}\n({energy[index].item():.2f})" for index in indices
    ]
    assert fig._suptitle.get_text() == "Bản đồ đặc trưng lớp conv2_1 (Top 4/16 kênh mạnh nhất)"
